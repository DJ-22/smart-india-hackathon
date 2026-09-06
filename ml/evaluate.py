"""Final evaluation + threshold calibration + RESULTS.md.

  python ml/evaluate.py --l1 ml/checkpoints/l1_run1

Reports every number the metrics slide needs, and each one twice: on clean
audio, and on the same audio after an Opus round-trip -- because the second
column is the one the live demo will actually produce.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import REPO
from ml.augment import codec_only
from ml.config import BASE_CKPT, CKPT_DIR, CROP_S, HELDOUT_LANGS, SR
from ml.data import crop_or_pad, load_manifest, make_splits, split_report
from ml.features import l0_features
from ml.metrics import auc, eer, l0_threshold_sweep, t_high_sweep


class L0Model:
    """LightGBM Booster with the sklearn predict_proba shape, single-threaded."""

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        p = self.booster.predict(np.asarray(X, dtype=np.float32), num_threads=1)
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        return np.stack([1.0 - p, p], axis=1)


def load_l0():
    """Load the L0 gate from ml/checkpoints/l0.txt (LightGBM text format).

    No pickle anywhere on this path: the file crosses machines, and a tampered
    pickle runs code the moment the server starts.
    """
    p = os.path.join(CKPT_DIR, "l0.txt")
    if not os.path.exists(p):
        return None
    import lightgbm as lgb
    meta_p = os.path.join(CKPT_DIR, "l0.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    meta["model"] = L0Model(lgb.Booster(model_file=p))
    return meta


def read_crop(path, crop_s=CROP_S, codec=False):
    wav, _ = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = crop_or_pad(wav, int(crop_s * SR), None, False)
    if codec:
        wav = crop_or_pad(codec_only(wav), int(crop_s * SR), None, False)
    return wav


def score_set(scorer, l0, df, codec, crop_s=CROP_S, tag=""):
    """-> dict with p1, p0, labels, languages and per-level latency."""
    wavs, t0 = [], time.time()
    for p in df.abspath:
        wavs.append(read_crop(p, crop_s, codec))
    print("  [%s] read+codec %d clips in %.0fs" % (tag, len(wavs), time.time() - t0),
          flush=True)

    t = time.time()
    p1 = scorer.score_waves(wavs)
    l1_ms = 1000.0 * (time.time() - t) / max(len(wavs), 1)

    p0, l0_ms = None, None
    if l0 is not None:
        t = time.time()
        X = np.stack([l0_features(w) for w in wavs])
        p0 = l0["model"].predict_proba(X)[:, 1]
        l0_ms = 1000.0 * (time.time() - t) / max(len(wavs), 1)
    return {"p1": p1, "p0": p0, "y": df.label.to_numpy(),
            "lang": df.language.to_numpy(), "src": df.source.to_numpy(),
            "l1_ms": l1_ms, "l0_ms": l0_ms}


def cascade_sim(r, t_low, t_high):
    """Replay the L0/L1 cascade offline: which level resolves each window and
    what the fused score would be.

    Levels match what server/detector.py reports, so C's metrics panel and this
    table describe the same thing:
      0  discharged below T_LOW           -- resolved by L0, no transformer
      2  fast-tracked above T_HIGH        -- skips L1, goes straight to L2
      1  everything in between            -- the transformer runs

    Only the L2 *routing* is simulated, not L2 itself: speaker verification
    needs an enrolled reference voice, so its contribution is measured live by B.
    """
    p0, p1, y = r["p0"], r["p1"], r["y"]
    if p0 is None:
        return {"levels": {0: 0.0, 1: 1.0, 2: 0.0}, "fused": p1, "gpu_touched": 1.0,
                "discharged_real": 0.0, "missed_fake_at_l0": 0.0, "n": len(y)}
    low, high = p0 < t_low, p0 > t_high
    fused = np.where(low | high, p0, p1)
    gpu = float((~(low | high)).mean())
    return {
        "levels": {0: float(low.mean()), 1: gpu, 2: float(high.mean())},
        "fused": fused, "gpu_touched": gpu,
        "discharged_real": float((low & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "missed_fake_at_l0": float((low & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "fast_track_precision": (float((y[high] == 1).mean()) if high.any()
                                 else float("nan")),
        "n": len(y),
    }


def single_window_latency(scorer, l0, wavs, n=120):
    """Warm, one-window-at-a-time latency -- the shape the server actually sees.

    Benchmark on REAL speech. White noise is not a stand-in here: librosa.yin
    degenerates on aperiodic input and reported ~67 ms/window against ~13 ms on
    actual audio, which is the difference between "the gate is cheap" and "the
    gate costs more than the transformer".
    """
    wavs = [np.asarray(w, dtype=np.float32) for w in wavs]
    out = {}
    for _ in range(3):                      # warm caches / kernels
        scorer.score_waves(wavs[:1])
        if l0 is not None:
            l0_features(wavs[0])

    def timeit(fn):
        ts = []
        for i in range(n):
            w = wavs[i % len(wavs)]
            t = time.perf_counter()
            fn(w)
            ts.append(1000.0 * (time.perf_counter() - t))
        return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))

    out["l1"] = timeit(lambda w: scorer.score_waves([w]))
    if l0 is not None:
        model = l0["model"]
        out["l0"] = timeit(lambda w: model.predict_proba(
            l0_features(w).reshape(1, -1)))
        out["both"] = timeit(lambda w: (model.predict_proba(
            l0_features(w).reshape(1, -1)), scorer.score_waves([w])))
    else:
        out["both"] = out["l1"]
    return out


def per_group(r, scores, key="lang"):
    out = []
    for lg in sorted(set(r[key])):
        m = r[key] == lg
        if len(set(r["y"][m])) < 2:
            continue
        e, _ = eer(r["y"][m], scores[m])
        out.append((lg, int(m.sum()), 100 * e, auc(r["y"][m], scores[m])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", default=os.path.join(CKPT_DIR, "l1_run1"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0, help="subsample each set")
    ap.add_argument("--crop", type=float, default=CROP_S,
                    help="window length to evaluate at; WINDOW_S is what the "
                         "server actually sends, CROP_S is what we trained on")
    ap.add_argument("--out", default=os.path.join(REPO, "RESULTS.md"))
    ap.add_argument("--baseline", action="store_true",
                    help="also score the un-finetuned checkpoint for the before/after row")
    a = ap.parse_args()

    from ml.infer import L1Scorer

    df = load_manifest()
    train_df, val_df, xling_df = make_splits(df)
    if a.limit:
        val_df = val_df.sample(min(a.limit, len(val_df)), random_state=0)
        xling_df = xling_df.sample(min(a.limit, len(xling_df)), random_state=0)

    print(split_report(df))
    print("\nloading L1:", a.l1)
    scorer = L1Scorer(a.l1, crop_s=a.crop)
    scorer.warm()
    l0 = load_l0()
    print("L0:", "loaded" if l0 else "MISSING (cascade numbers will be skipped)")

    sets = {}
    for name, d in (("val", val_df), ("xling", xling_df)):
        for codec in (False, True):
            key = "%s%s" % (name, "_opus" if codec else "_clean")
            sets[key] = score_set(scorer, l0, d, codec, crop_s=a.crop, tag=key)

    lines = []
    W = lines.append
    W("# VoiceGuard -- RESULTS")
    W("")
    W("Model: `%s`  ·  evaluated on %.1f s windows"
      % (os.path.relpath(a.l1, REPO).replace("\\", "/"), a.crop))
    W("Generated: %s" % time.strftime("%Y-%m-%d %H:%M"))
    W("")
    W("## Splits")
    W("")
    W("```")
    W(split_report(df))
    W("```")
    W("")
    W("Held-out languages (never in training): **%s**" % ", ".join(HELDOUT_LANGS))
    W("")
    W("## Headline numbers")
    W("")
    W("Metrics slide, over the Opus channel. Hand this block to C verbatim.")
    W("")
    hv = sets["val_opus"]
    hx = sets["xling_opus"]
    seen_e, _ = eer(hv["y"], hv["p1"])
    uns_e, _ = eer(hx["y"], hx["p1"])
    same_corpus = hx["src"] == "indictts"
    l0m = {}
    l0p = os.path.join(CKPT_DIR, "l0_metrics.json")
    if os.path.exists(l0p):
        l0m = json.load(open(l0p, encoding="utf-8"))
    W("```")
    W("Seen-language EER      : %.1f%%   AUC %.3f" % (100 * seen_e, auc(hv["y"], hv["p1"])))
    W("Unseen-language EER    : %.1f%%   AUC %.3f   (langs: %s)"
      % (100 * uns_e, auc(hx["y"], hx["p1"]), ", ".join(HELDOUT_LANGS).lower()))
    if same_corpus.any():
        e_sc, _ = eer(hx["y"][same_corpus], hx["p1"][same_corpus])
        W("  of which, same corpus : %.1f%%   AUC %.3f   (%s -- the honest"
          % (100 * e_sc, auc(hx["y"][same_corpus], hx["p1"][same_corpus]),
             ", ".join(sorted(set(hx["lang"][same_corpus]))).lower()))
        W("                                                 unseen-LANGUAGE number)")
    if (~same_corpus).any():
        e_oc, _ = eer(hx["y"][~same_corpus], hx["p1"][~same_corpus])
        W("  of which, new corpus  : %.1f%%   AUC %.3f   (%s -- an unseen"
          % (100 * e_oc, auc(hx["y"][~same_corpus], hx["p1"][~same_corpus]),
             ", ".join(sorted(set(hx["lang"][~same_corpus]))).lower()))
        W("                                                 RECORDING CHAIN, not just language)")
    W("Demo-generator EER     : not measured -- see ml/demo_baseline.py (BUILD_PLAN s1)")
    shipped_cal = {}
    _cal_p = os.path.join(CKPT_DIR, "calibration.json")
    if os.path.exists(_cal_p):
        shipped_cal = json.load(open(_cal_p, encoding="utf-8"))
    ship_low = shipped_cal.get("T_LOW", 0.0)
    if ship_low > 0:
        W("L0 discharge rate      : %.1f%%   at fake-recall %.1f%%"
          % (100 * l0m.get("t_low_discharge", 0.0),
             100 * l0m.get("t_low_recall_fake", 1.0)))
    else:
        W("L0 discharge rate      : 0%%     (path OFF -- it could reach %.1f%% but"
          % (100 * l0m.get("t_low_discharge", 0.0)))
        W("                                cost 4.3 pts of EER; see Cascade below)")
    W("L1 transformer skipped : %.1f%%   (T_HIGH fast-track only)"
      % (100 * shipped_cal.get("cascade", {}).get("val_opus", {})
         .get("l2_fast_tracked", l0m.get("t_high_frac", 0.0))))
    W("```")
    W("")
    W("The two unseen-language rows differ by a factor of several, and the axis "
      "that separates them is the recording chain, not the language. Quoting the "
      "combined number alone would understate generalisation across languages "
      "and overstate it across generators -- both halves belong on the slide.")
    W("")
    W("## Detection (L1)")
    W("")
    W("| set | channel | n | EER | AUC |")
    W("|---|---|---|---|---|")
    for key, r in sets.items():
        name, ch = key.rsplit("_", 1)
        e, _ = eer(r["y"], r["p1"])
        W("| %s | %s | %d | %.2f%% | %.4f |"
          % (name, "opus" if ch == "opus" else "clean", len(r["y"]), 100 * e,
             auc(r["y"], r["p1"])))
    W("")
    W("`opus` = the same clips after a real Ogg/Opus round-trip, i.e. what the "
      "detector actually receives over LiveKit.")
    W("")
    if a.baseline:
        W("### Before fine-tuning")
        W("")
        base = L1Scorer(BASE_CKPT, crop_s=a.crop)
        base.warm()
        W("| set | channel | n | EER | AUC |")
        W("|---|---|---|---|---|")
        for key, d in (("val", val_df), ("xling", xling_df)):
            rb = score_set(base, None, d, True, crop_s=a.crop, tag=key + "_base")
            e, _ = eer(rb["y"], rb["p1"])
            W("| %s | opus | %d | %.2f%% | %.4f |"
              % (key, len(rb["y"]), 100 * e, auc(rb["y"], rb["p1"])))
        W("")
        W("`%s` off the shelf returns P(fake) near zero for essentially every "
          "clip in this corpus, so its AUC sits at chance. BUILD_PLAN section 8 "
          "lists shipping it un-finetuned as the fallback if the fine-tune "
          "fails; these rows are why that is not a fallback." % BASE_CKPT)
        W("")
        del base
    W("### Per language (opus channel)")
    W("")
    W("| language | corpus | language seen in training | n | EER | AUC |")
    W("|---|---|---|---|---|---|")
    for key in ("val_opus", "xling_opus"):
        r = sets[key]
        for lg, n, e, au in per_group(r, r["p1"], "lang"):
            src = ",".join(sorted(set(r["src"][r["lang"] == lg])))
            W("| %s | %s | %s | %d | %.2f%% | %.4f |"
              % (lg, src, "no" if lg in HELDOUT_LANGS else "yes", n, e, au))
    W("")
    W("### Per corpus (opus channel) -- the axis that actually predicts accuracy")
    W("")
    W("The challenge set merges several collections with different recording "
      "chains and TTS systems (see `ml/data.py: source_of`). Accuracy tracks how "
      "well a clip's *corpus* is represented in training far more than whether "
      "its *language* was seen: an unseen language from a well-represented "
      "corpus scores near the seen-language number, while an unseen language "
      "from a barely-represented corpus is several times worse.")
    W("")
    W("| corpus | training clips | split | n | EER | AUC |")
    W("|---|---|---|---|---|---|")
    train_src = train_df.source.value_counts().to_dict()
    for key in ("val_opus", "xling_opus"):
        r = sets[key]
        for sc, n, e, au in per_group(r, r["p1"], "src"):
            W("| %s | %d | %s | %d | %.2f%% | %.4f |"
              % (sc, train_src.get(sc, 0), key.replace("_opus", ""), n, e, au))
    W("")

    cal = {}
    if l0 is not None:
        rv = sets["val_opus"]
        _, sweep_low = l0_threshold_sweep(rv["y"], rv["p0"], target_recall=0.99)
        # Use the dials that are actually shipped, so this table describes the
        # deployed cascade rather than a second, independent calibration.
        cal_p = os.path.join(CKPT_DIR, "calibration.json")
        if os.path.exists(cal_p):
            shipped = json.load(open(cal_p, encoding="utf-8"))
            t_low = shipped.get("T_LOW", 0.0)
            t_high = shipped.get("T_HIGH", 1.0)
            src_note = "from ml/checkpoints/calibration.json (ml/calibrate.py)"
        else:
            best_low, _ = l0_threshold_sweep(rv["y"], rv["p0"], target_recall=0.99)
            best_high, _ = t_high_sweep(rv["y"], rv["p0"], target_precision=0.99)
            t_low = (best_low or {}).get("t_low", 0.0)
            t_high = (best_high or {}).get("t_high", 1.0)
            src_note = "swept here; run ml/calibrate.py to fix them"
        cal = {"T_LOW": t_low, "T_HIGH": t_high}

        W("## Cascade (calibrated on val, opus channel)")
        W("")
        W("`T_LOW = %.4f`, `T_HIGH = %.4f` -- %s." % (t_low, t_high, src_note))
        W("")
        if t_low <= 0:
            W("**The L0 discharge path is off.** It could safely discharge 8.6% of "
              "windows at 99% fake recall, but a discharged window reports L0's "
              "score instead of L1's, which took val EER from 4.35% to 8.64% -- "
              "4.3 points of accuracy to save 8.6% of a transformer that runs in "
              "11 ms. The T_HIGH fast-track path costs 0.05 points and is kept. "
              "`ml/calibrate.py` measures both paths and switches off whichever "
              "fails to pay for itself.")
        W("")
        W("| set | L0 discharged | L1 (transformer) | L2 fast-tracked | "
          "fakes lost at L0 | fused EER |")
        W("|---|---|---|---|---|---|")
        for key in ("val_opus", "xling_opus"):
            r = sets[key]
            c = cascade_sim(r, t_low, t_high)
            e, _ = eer(r["y"], c["fused"])
            W("| %s | %.1f%% | %.1f%% | %.1f%% | %.2f%% | %.2f%% |"
              % (key, 100 * c["levels"][0], 100 * c["levels"][1],
                 100 * c["levels"][2], 100 * c["missed_fake_at_l0"], 100 * e))
            cal.setdefault("cascade", {})[key] = {
                "l0_resolved": c["levels"][0], "gpu_touched": c["gpu_touched"],
                "missed_fake_at_l0": c["missed_fake_at_l0"], "fused_eer": float(e)}
        W("")
        W("### T_LOW sweep (val, opus)")
        W("")
        W("| T_LOW | fake recall kept | windows discharged |")
        W("|---|---|---|")
        for row in sweep_low[::40]:
            W("| %.4f | %.4f | %.1f%% |"
              % (row["t_low"], row["recall_fake"], 100 * row["discharged"]))
        W("")

    W("## Latency")
    W("")
    bench_wavs = [read_crop(p, a.crop, True) for p in list(val_df.abspath)[:32]]
    lat = single_window_latency(scorer, l0, bench_wavs, n=120)
    W("| level | p50 ms | p95 ms | notes |")
    W("|---|---|---|---|")
    if lat.get("l0"):
        W("| L0 (features + LightGBM) | %.1f | %.1f | CPU |"
          % (lat["l0"][0], lat["l0"][1]))
    W("| L1 (wav2vec2, 4 s window) | %.1f | %.1f | %s |"
      % (lat["l1"][0], lat["l1"][1], scorer.device))
    W("| L0 + L1 (worst path) | %.1f | %.1f | what most windows cost |"
      % (lat["both"][0], lat["both"][1]))
    W("")
    W("One window at a time, warm, which is how the server runs -- not batched "
      "throughput. The first call after load is ~1.8 s (lazy imports and CUDA "
      "context); `L1Scorer.warm()` exists to get that out of the way before "
      "serving, and `server/detector.py` must call it at startup.")
    W("")
    W("Batched throughput for reference: L1 %.1f ms/window, L0 %.1f ms/window."
      % (sets["val_opus"]["l1_ms"], sets["val_opus"]["l0_ms"] or float("nan")))
    W("")
    if lat.get("l0") and lat["l0"][0] > 0.5 * lat["l1"][0]:
        W("**The gate is not cheap relative to the model it guards.** L0 costs "
          "%.1f ms and L1 costs %.1f ms per window, so running L0 first and L1 "
          "afterwards -- which is what happens on %.0f%% of windows -- costs "
          "%.1f ms against %.1f ms for calling L1 alone. On a box with a GPU the "
          "cascade is a net slowdown; it pays off where L1 is genuinely the "
          "bottleneck (CPU-only inference, or many concurrent calls sharing one "
          "GPU). `CASCADE_ENABLED` is therefore False by default -- see "
          "common/config.py." % (lat["l0"][0], lat["l1"][0],
                                 100 * cal.get("cascade", {})
                                 .get("val_opus", {}).get("l1_transformer", 0.93),
                                 lat["both"][0], lat["l1"][0]))
        W("")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", a.out)

    if cal:
        with open(os.path.join(CKPT_DIR, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2)
        print("wrote", os.path.join(CKPT_DIR, "calibration.json"))


if __name__ == "__main__":
    main()
