"""Level 0 -- the cheap LightGBM gate that keeps the GPU asleep.

  python ml/train_l0.py

Two things about this model are deliberate and easy to get wrong:

  * It is fit on CODEC-AUGMENTED audio only. Rolloff and high-band energy look
    spectacular on clean 16 kHz studio clips and carry almost nothing after
    Opus. Fit it clean and the gate discharges everything on a live call.

  * Its threshold is tuned for safe DISCHARGE, not for F1. T_LOW is the largest
    value that still keeps >=99% of fakes in play; whatever discharge rate that
    buys is what we take.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.augment import codec_only
from ml.config import CKPT_DIR, CROP_S, SR
from ml.data import crop_or_pad, load_manifest, make_splits, split_report
from ml.features import N_FEATURES, l0_features
from ml.metrics import auc, eer, l0_threshold_sweep, t_high_sweep

CACHE = os.path.join(CKPT_DIR, "l0_features.npz")


def _one(args):
    path, codec, crop_s = args
    try:
        wav, _ = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = crop_or_pad(wav, int(crop_s * SR), None, False)
        if codec:
            wav = codec_only(wav)
            wav = crop_or_pad(wav, int(crop_s * SR), None, False)
        return l0_features(wav)
    except Exception:                                            # noqa: BLE001
        return np.zeros(N_FEATURES, dtype=np.float32)


def featurize(paths, codec=True, crop_s=CROP_S, workers=None, tag=""):
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, v in enumerate(ex.map(_one, [(p, codec, crop_s) for p in paths],
                                     chunksize=8)):
            out.append(v)
            if (i + 1) % 500 == 0:
                print("  %s %d/%d  %.0fs" % (tag, i + 1, len(paths),
                                             time.time() - t0), flush=True)
    print("  %s done %d clips in %.0fs" % (tag, len(paths), time.time() - t0),
          flush=True)
    return np.stack(out) if out else np.zeros((0, N_FEATURES), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="fit on clean audio too, for the ablation line in RESULTS.md")
    ap.add_argument("--workers", type=int, default=0,
                    help="feature-extraction processes; leave headroom if L1 is training")
    a = ap.parse_args()
    workers = a.workers or None

    os.makedirs(CKPT_DIR, exist_ok=True)
    df = load_manifest()
    train_df, val_df, xling_df = make_splits(df)
    print(split_report(df))

    # Cache key: the exact rows this manifest produced. Downloading more clips
    # changes the splits, and silently reusing features for a different set of
    # rows would train L0 on one dataset and threshold it on another.
    key = np.array(["%d|%d|%d" % (len(train_df), len(val_df), len(xling_df))]
                   + sorted(df.id.astype(str).tolist()))
    cached = None
    if os.path.exists(CACHE) and not a.no_cache:
        z = np.load(CACHE, allow_pickle=False)     # numeric + unicode only; never object
        if "key" in z and z["key"].shape == key.shape and (z["key"] == key).all():
            cached = z
        else:
            print("manifest changed since the feature cache was built "
                  "-- re-extracting", flush=True)

    if cached is not None:
        z = cached
        Xtr, ytr, Xva, yva, Xxl, yxl = (z["Xtr"], z["ytr"], z["Xva"],
                                        z["yva"], z["Xxl"], z["yxl"])
        Xtr_clean = z["Xtr_clean"] if "Xtr_clean" in z else None
        Xva_clean = z["Xva_clean"] if "Xva_clean" in z else None
        print("loaded cached features", Xtr.shape, Xva.shape, Xxl.shape)
    else:
        print("extracting L0 features (codec-augmented)...")
        Xtr = featurize(list(train_df.abspath), True, workers=workers, tag="train")
        Xva = featurize(list(val_df.abspath), True, workers=workers, tag="val")
        Xxl = featurize(list(xling_df.abspath), True, workers=workers, tag="xling")
        ytr = train_df.label.to_numpy()
        yva = val_df.label.to_numpy()
        yxl = xling_df.label.to_numpy()
        print("extracting clean-audio features (ablation)...")
        Xtr_clean = featurize(list(train_df.abspath), False, workers=workers, tag="train-clean")
        Xva_clean = featurize(list(val_df.abspath), False, workers=workers, tag="val-clean")
        np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                            Xxl=Xxl, yxl=yxl, Xtr_clean=Xtr_clean,
                            Xva_clean=Xva_clean, key=key)
        print("cached ->", CACHE)

    from lightgbm import LGBMClassifier

    def fit(X, y):
        clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.06,
                             subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                             min_child_samples=20, random_state=0, n_jobs=-1,
                             verbose=-1)
        clf.fit(X, y)
        return clf

    clf = fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_xl = clf.predict_proba(Xxl)[:, 1]

    e_va, _ = eer(yva, p_va)
    e_xl, _ = eer(yxl, p_xl)
    print("\nL0 (codec-fit)  val EER=%.2f%% AUC=%.4f | xling EER=%.2f%% AUC=%.4f"
          % (100 * e_va, auc(yva, p_va), 100 * e_xl, auc(yxl, p_xl)))

    ablation = None
    if Xtr_clean is not None and Xva_clean is not None:
        clf_clean = fit(Xtr_clean, ytr)
        p_clean_on_codec = clf_clean.predict_proba(Xva)[:, 1]
        e_clean, _ = eer(yva, p_clean_on_codec)
        ablation = {"clean_fit_eer_on_codec_val": float(e_clean)}
        print("L0 (clean-fit) evaluated on codec val: EER=%.2f%%  <- why we augment"
              % (100 * e_clean))

    best_low, sweep_low = l0_threshold_sweep(yva, p_va, target_recall=0.99)
    best_high, _ = t_high_sweep(yva, p_va, target_precision=0.99)
    print("\nT_LOW sweep (fake-recall >= 0.99):")
    for r in sweep_low[::40]:
        print("  t=%.4f  recall_fake=%.4f  discharged=%.1f%%"
              % (r["t_low"], r["recall_fake"], 100 * r["discharged"]))
    print("chosen T_LOW :", best_low)
    print("chosen T_HIGH:", best_high)

    # A gate that cannot skip L1 is worse than no gate: it adds latency to every
    # window and buys nothing. Say so loudly rather than shipping a guessed
    # threshold that quietly drops fakes. Two paths skip the transformer --
    # discharge below T_LOW, and fast-track above T_HIGH -- so judge the cascade
    # on their sum, not on discharge alone.
    discharge = (best_low or {}).get("discharged", 0.0)
    fast = (best_high or {}).get("frac_fast_tracked", 0.0)
    skipped = discharge + fast
    print("\nL1 transformer skipped on %.1f%% of windows "
          "(%.1f%% discharged at T_LOW, %.1f%% fast-tracked above T_HIGH)"
          % (100 * skipped, 100 * discharge, 100 * fast))
    if best_low is None:
        print("!! No T_LOW reaches 99% fake recall -- the discharge path is dead.")
    if discharge < 0.15:
        print("!! Discharge is %.1f%%, not the 40-60%% BUILD_PLAN section 4.6 "
              "assumed. Report the measured number, not the planned one."
              % (100 * discharge))
    if skipped < 0.10:
        print("!! The cascade saves almost nothing here -- recommend "
              "CASCADE_ENABLED = False and keep L0 as a roadmap item.")

    # Serve single-threaded. Training wants every core, but inference is one
    # 76-feature row at a time, where 28 threads cost more in synchronisation
    # than the tree traversal costs outright -- and the server has torch's
    # thread pool resident alongside it.
    clf.set_params(n_jobs=1)

    # LightGBM's own text format, not pickle. This file is handed to B and to the
    # server out-of-band; a pickle is arbitrary code execution for whoever loads
    # it, a model_file is a list of trees.
    out = os.path.join(CKPT_DIR, "l0.txt")
    clf.booster_.save_model(out)
    with open(os.path.join(CKPT_DIR, "l0.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "lightgbm-text", "crop_s": CROP_S, "sr": SR,
                   "n_features": N_FEATURES, "num_threads": 1}, f, indent=2)
    meta = {
        "val_eer": float(e_va), "val_auc": float(auc(yva, p_va)),
        "xling_eer": float(e_xl), "xling_auc": float(auc(yxl, p_xl)),
        # 0.0 / 1.0 are the *safe* fallbacks: a gate that discharges nothing and
        # fast-tracks nothing, i.e. every window goes to L1.
        "t_low": (best_low or {}).get("t_low", 0.0),
        "t_low_discharge": (best_low or {}).get("discharged", 0.0),
        "t_low_recall_fake": (best_low or {}).get("recall_fake", 1.0),
        "t_low_found": best_low is not None,
        "t_high": (best_high or {}).get("t_high", 1.0),
        "t_high_precision": (best_high or {}).get("precision", 1.0),
        "t_high_frac": (best_high or {}).get("frac_fast_tracked", 0.0),
        "t_high_found": best_high is not None,
        "l1_skipped_frac": float((best_low or {}).get("discharged", 0.0)
                                 + (best_high or {}).get("frac_fast_tracked", 0.0)),
        "recommend_cascade_enabled": bool(
            (best_low or {}).get("discharged", 0.0)
            + (best_high or {}).get("frac_fast_tracked", 0.0) >= 0.10),
        "ablation": ablation,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_xling": int(len(yxl)),
    }
    with open(os.path.join(CKPT_DIR, "l0_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
