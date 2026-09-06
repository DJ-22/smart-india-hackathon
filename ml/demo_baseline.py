"""Score your own audio clips -- the model's verdict on real vs cloned voice.

  python ml/demo_baseline.py --dir <folder>

Put clips in either layout:

  <folder>/                     everything treated as suspected-fake
  <folder>/fake/  <folder>/real/    labelled, so you also get an EER

Any format libsndfile reads (wav, flac, ogg, mp3), any sample rate, mono or
stereo, any length -- everything is converted to 16 kHz mono and scored with the
same sliding window the live server uses, so the verdict here is the verdict the
demo would have produced.

Each clip gets:
  P(fake) mean/max over windows, and the peak of the EMA the risk engine runs
  a GREEN / AMBER / RED band, using the calibrated thresholds from config.py
  the same again after an Opus round-trip, i.e. as it would arrive over LiveKit

Per BUILD_PLAN section 1 this is also the highest-value check in the build: run
it on ~20 clips from whatever cloning tool will be used on stage. Every number
in RESULTS.md is measured against IndicTTS renderings, not against that tool.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (CKPT_DIR, DATA_DIR, EMA_ALPHA, HOP_S, L1_AMBER,
                           L1_RED, REPO, WINDOW_S)
from ml.augment import codec_only
from ml.metrics import auc, eer

AUDIO = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus", "*.m4a", "*.aac")
DEFAULT_L1 = os.path.join(CKPT_DIR, "l1_run3")


def _find(d):
    out = []
    for pat in AUDIO:
        out += glob.glob(os.path.join(d, pat))
        out += glob.glob(os.path.join(d, "**", pat), recursive=True)
    return sorted(set(out))


def band(x):
    return "RED" if x >= L1_RED else "AMBER" if x >= L1_AMBER else "GREEN"


def peak_ema(p, alpha=EMA_ALPHA):
    """The statistic ui/risk.py actually bands on: the running EMA's high-water mark."""
    ema, peak = None, 0.0
    for v in p:
        ema = float(v) if ema is None else alpha * float(v) + (1 - alpha) * ema
        peak = max(peak, ema)
    return peak


def analyse(scorer, path, codec=False):
    from ml.infer import read_audio
    wav = read_audio(path)
    dur = len(wav) / 16000.0
    p = scorer.score_clip_windows(wav, window_s=WINDOW_S, hop_s=HOP_S,
                                  transform=codec_only if codec else None)
    return {"dur_s": dur, "n_windows": int(len(p)), "mean": float(np.mean(p)),
            "max": float(np.max(p)), "peak_ema": peak_ema(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(DATA_DIR, "demo_clips"))
    ap.add_argument("--l1", default=DEFAULT_L1)
    ap.add_argument("--out", default=os.path.join(REPO, "demo_generator_baseline.md"))
    ap.add_argument("--no-opus", action="store_true",
                    help="skip the codec pass (faster, but less like the live call)")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        raise SystemExit(
            "no such folder: %s\n\n"
            "Create it and drop your clips in, either flat or split by label:\n"
            "  %s\\fake\\   cloned / synthetic clips\n"
            "  %s\\real\\   genuine recordings\n"
            % (a.dir, a.dir, a.dir))

    fake = _find(os.path.join(a.dir, "fake"))
    real = _find(os.path.join(a.dir, "real"))
    flat = [p for p in _find(a.dir)
            if p not in set(fake) | set(real)]
    if not fake and not real and flat:
        print("no fake/ or real/ subfolder -- treating all %d clips as suspected-fake"
              % len(flat))
        fake = flat
    elif flat:
        print("ignoring %d clip(s) outside fake/ and real/" % len(flat))
    if not fake and not real:
        raise SystemExit("no audio files found under %s" % a.dir)

    from ml.infer import L1Scorer
    print("loading %s ..." % a.l1)
    s = L1Scorer(a.l1, crop_s=WINDOW_S)
    s.warm()

    rows = []
    for tag, paths, y in (("fake", fake, 1), ("real", real, 0)):
        for p in paths:
            try:
                r = {"file": os.path.basename(p), "truth": tag, "y": y}
                r.update(analyse(s, p, codec=False))
                if not a.no_opus:
                    o = analyse(s, p, codec=True)
                    r["opus_peak_ema"] = o["peak_ema"]
                    r["opus_mean"] = o["mean"]
                rows.append(r)
            except Exception as e:                               # noqa: BLE001
                print("  could not read %s: %r" % (os.path.basename(p), e))

    if not rows:
        raise SystemExit("no clips could be decoded")

    key = "opus_peak_ema" if not a.no_opus else "peak_ema"
    w = max(len(r["file"]) for r in rows)
    print("\n%-*s %-6s %5s %4s %7s %8s %6s" %
          (w, "file", "truth", "dur", "win", "meanP", "peakEMA", "band"))
    for r in rows:
        print("%-*s %-6s %5.1f %4d %7.3f %8.3f %6s"
              % (w, r["file"], r["truth"], r["dur_s"], r["n_windows"],
                 r.get("opus_mean", r["mean"]), r[key], band(r[key])))

    lines = ["# Demo generator baseline", "",
             "Checkpoint: `%s`" % os.path.relpath(a.l1, REPO).replace("\\", "/"),
             "Channel: %s" % ("clean only" if a.no_opus else
                              "Opus round-trip (as it would arrive over LiveKit)"),
             "Bands: AMBER >= %.3f, RED >= %.3f, EMA alpha %.2f, %.0f s window / "
             "%.0f s hop" % (L1_AMBER, L1_RED, EMA_ALPHA, WINDOW_S, HOP_S), "",
             "| file | truth | dur s | windows | mean P(fake) | peak EMA | band |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %.1f | %d | %.3f | %.3f | %s |"
                     % (r["file"], r["truth"], r["dur_s"], r["n_windows"],
                        r.get("opus_mean", r["mean"]), r[key], band(r[key])))

    lines += ["", "## Verdict", ""]
    pf = np.array([r[key] for r in rows if r["y"] == 1])
    pr = np.array([r[key] for r in rows if r["y"] == 0])
    flagged = float(np.mean(pf >= L1_RED)) if len(pf) else float("nan")
    if len(pf):
        lines.append("Cloned clips flagged RED: **%.0f%%** (%d/%d), mean peak EMA %.3f"
                     % (100 * flagged, int((pf >= L1_RED).sum()), len(pf), pf.mean()))
    if len(pr):
        fa = float(np.mean(pr >= L1_RED))
        lines.append("Genuine clips false-alarmed RED: **%.0f%%** (%d/%d), "
                     "mean peak EMA %.3f"
                     % (100 * fa, int((pr >= L1_RED).sum()), len(pr), pr.mean()))
    if len(pf) and len(pr):
        ys = [r["y"] for r in rows]
        ss = [r[key] for r in rows]
        e, _ = eer(ys, ss)
        lines.append("EER on this set: **%.2f%%**, AUC %.4f" % (100 * e, auc(ys, ss)))

    if len(pf):
        lines += ["", "**Action:** " + (
            "this generator is caught -- no extra training data needed."
            if flagged >= 0.8 else
            "this generator is NOT reliably caught. Generate ~200 clips from it "
            "and fold them into L1 training (BUILD_PLAN section 4.7); until then "
            "do not assume the RESULTS.md numbers transfer to it.")]

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("## Verdict"):]))
    print("\nwrote", a.out)
    with open(os.path.join(CKPT_DIR, "demo_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
