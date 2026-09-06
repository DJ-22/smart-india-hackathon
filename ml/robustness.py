"""Channel robustness: how the detector holds up outside studio conditions.

  python ml/robustness.py --l1 ml/checkpoints/l1_run3 ml/checkpoints/l1_run4

The corpus is close-mic studio audio for BOTH classes, which makes it silent
about the one thing a live demo guarantees: a person speaking in a room. This
sweeps the degradations a real call applies and reports EER per condition, so a
regression like "reverb flips genuine speech to fake" shows up as a number
instead of as a surprise on stage.

Reported per condition:
  EER          equal error rate over the split
  P(fake) real mean score on genuine clips -- the false-alarm side, which is
               what a demo audience actually notices
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CKPT_DIR, SR, WINDOW_S
from ml.augment import codec_only, reverb
from ml.data import load_manifest, make_splits
from ml.infer import read_audio
from ml.metrics import auc, eer


def _noise(w, snr_db, rng):
    sig = float(np.sqrt((w ** 2).mean())) + 1e-9
    n = rng.standard_normal(len(w)).astype(np.float32)
    return np.clip(w + n * (sig / (10 ** (snr_db / 20.0))), -1, 1).astype(np.float32)


def conditions(rng):
    """name -> transform. Ordered from studio to worst-case call."""
    return [
        ("clean", lambda w: w),
        ("opus", codec_only),
        ("reverb 0.15s", lambda w: reverb(w, 0.15, rng)),
        ("reverb 0.30s", lambda w: reverb(w, 0.30, rng)),
        ("reverb 0.60s", lambda w: reverb(w, 0.60, rng)),
        ("noise 20dB", lambda w: _noise(w, 20, rng)),
        ("noise 10dB", lambda w: _noise(w, 10, rng)),
        ("room + opus", lambda w: codec_only(reverb(w, 0.25, rng))),
        ("room + opus + noise",
         lambda w: _noise(codec_only(reverb(w, 0.25, rng)), 20, rng)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", nargs="+", default=[os.path.join(CKPT_DIR, "l1_run3")])
    ap.add_argument("--split", default="val", choices=["val", "xling"])
    ap.add_argument("--n", type=int, default=300, help="clips per split (balanced)")
    a = ap.parse_args()

    from ml.infer import L1Scorer

    df = load_manifest()
    _, val_df, xling_df = make_splits(df)
    import pandas as pd
    d = val_df if a.split == "val" else xling_df
    # explicit per-class sample: groupby.apply drops the grouping column in
    # pandas 3, which silently costs you the label you are about to evaluate on
    d = pd.concat([g.sample(min(len(g), a.n // 2), random_state=0)
                   for _, g in d.groupby("label")], ignore_index=True)
    wavs = [read_audio(p) for p in d.abspath]
    y = d.label.to_numpy()
    print("%s: %d clips (%d real / %d fake)"
          % (a.split, len(y), (y == 0).sum(), (y == 1).sum()))

    results = {}
    for ck in a.l1:
        s = L1Scorer(ck, crop_s=WINDOW_S)
        s.warm()
        rng = np.random.default_rng(0)
        rows = []
        for name, fn in conditions(rng):
            p = np.array([float(s.score_clip_windows(fn(w), window_s=WINDOW_S).mean())
                          for w in wavs])
            e, _ = eer(y, p)
            rows.append((name, 100 * e, auc(y, p), p[y == 0].mean(), p[y == 1].mean()))
        results[os.path.basename(ck)] = rows
        del s

    names = list(results)
    print("\n%-22s %s" % ("condition",
                          "  ".join("%-26s" % n for n in names)))
    print("%-22s %s" % ("", "  ".join("%-26s" % "EER    P(fake)real  fake"
                                      for _ in names)))
    for i, (cname, *_rest) in enumerate(results[names[0]]):
        cells = []
        for n in names:
            _, e, _auc, pr, pf = results[n][i]
            cells.append("%5.2f%%  %8.3f %6.3f   " % (e, pr, pf))
        print("%-22s %s" % (cname, "  ".join(cells)))

    print("\nThe column that matters for the demo is P(fake) on *real* clips "
          "under reverb: that is a genuine speaker being called a deepfake.")


if __name__ == "__main__":
    main()
