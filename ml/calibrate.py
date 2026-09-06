"""H8:00 threshold calibration -- produces the table, then writes common/config.py.

  python ml/calibrate.py --l1 ml/checkpoints/l1_run1            # table only
  python ml/calibrate.py --l1 ml/checkpoints/l1_run1 --write    # commit the dials

Everything is calibrated on the val split *after a real Opus round-trip*, since
that is the channel the live pipeline actually sees. Four dials come out:

  T_LOW    largest L0 gate value keeping >=99% of fakes in play
  T_HIGH   smallest L0 value that is >=99% precise, so L1 can be skipped
  L1_RED   largest band catching >=98% of spoofed CALLS  (BUILD_PLAN section 7.2)
  L1_AMBER the same at >=99.5%, as an early-warning band

The two band dials are calibrated on simulated calls, not on single windows.
C's risk engine applies them to an EMA over a whole call, and a detector at ~4%
window EER simply does not reach 98% recall on every individual 4-second slice
-- asking it to would drive L1_RED near zero and make the gauge strobe. See
simulate_sessions() for how the calls are built.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import EMA_ALPHA, REPO
from ml.config import CKPT_DIR
from ml.data import load_manifest, make_splits
from ml.evaluate import cascade_sim, load_l0, score_set
from ml.metrics import eer, l0_threshold_sweep, t_high_sweep

CONFIG = os.path.join(REPO, "common", "config.py")


def band_sweep(y, s, recall_target):
    """Largest threshold whose fake-recall still meets the target.

    Largest, not smallest: among thresholds that catch enough fakes we want the
    one that raises the fewest false alarms, so we walk down from 1.0 and stop
    at the first threshold that clears the bar.
    """
    for t in np.arange(0.99, -0.0001, -0.005):
        t = max(float(t), 0.0)
        rec = float(((s >= t) & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
        fpr = float(((s >= t) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
        if rec >= recall_target:
            return {"t": round(t, 3), "recall": rec, "false_alarm": fpr}
    return None


def simulate_sessions(y, scores, groups, n_sessions=600, windows=12,
                      alpha=EMA_ALPHA, seed=0):
    """Turn independent clips into synthetic calls, then score them the way C's
    risk engine will.

    The bands are not applied to a single window -- they are applied to an EMA
    over a whole call. Calibrating them on per-window scores asks the detector
    to be right about every 4-second slice of speech, which no detector at 4%
    EER can be, and produces a RED threshold so low the UI would strobe.

    Each synthetic session draws `windows` clips from one speaker at one label,
    runs the same EMA C uses, and takes the running peak as the session score.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)

    buckets = {}
    for i, (g, lab) in enumerate(zip(groups, y)):
        buckets.setdefault((g, int(lab)), []).append(i)
    buckets = {k: np.array(v) for k, v in buckets.items() if len(v) >= 4}
    if not buckets:
        return np.zeros(0), np.zeros(0)

    keys = list(buckets)
    sess_y, sess_s = [], []
    for k in range(n_sessions):
        key = keys[k % len(keys)]
        idx = rng.choice(buckets[key], size=windows,
                         replace=len(buckets[key]) < windows)
        ema, peak = None, 0.0
        for p in scores[idx]:
            ema = p if ema is None else alpha * p + (1 - alpha) * ema
            peak = max(peak, ema)
        sess_y.append(key[1])
        sess_s.append(peak)
    return np.array(sess_y), np.array(sess_s)


def write_config(vals):
    src = open(CONFIG, encoding="utf-8").read()
    for k, v in vals.items():
        pat = re.compile(r"^(%s\s*=\s*)([-\d.]+)" % re.escape(k), re.M)
        if not pat.search(src):
            print("  ! %s not found in config.py, skipping" % k)
            continue
        lit = repr(float(v))                     # a number, never arbitrary text
        src = pat.sub(lambda m, lit=lit: "%s%s" % (m.group(1), lit), src, count=1)
    open(CONFIG, "w", encoding="utf-8").write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", default=os.path.join(CKPT_DIR, "l1_run1"))
    ap.add_argument("--write", action="store_true",
                    help="rewrite common/config.py with the chosen values")
    a = ap.parse_args()

    from ml.infer import L1Scorer

    df = load_manifest()
    _, val_df, xling_df = make_splits(df)
    scorer = L1Scorer(a.l1)
    scorer.warm()
    l0 = load_l0()
    if l0 is None:
        raise SystemExit("no ml/checkpoints/l0.pkl -- run ml/train_l0.py first")

    print("scoring val over the Opus channel...")
    rv = score_set(scorer, l0, val_df, codec=True, tag="val_opus")
    print("scoring unseen languages over the Opus channel...")
    rx = score_set(scorer, l0, xling_df, codec=True, tag="xling_opus")

    best_low, sweep_low = l0_threshold_sweep(rv["y"], rv["p0"], target_recall=0.99)
    best_high, _ = t_high_sweep(rv["y"], rv["p0"], target_precision=0.99)
    t_low = (best_low or {}).get("t_low", 0.0)    # safe default: discharge nothing
    t_high = (best_high or {}).get("t_high", 1.0)  # safe default: fast-track nothing

    # Each cascade shortcut has to pay for itself in accuracy, not just in FLOPs.
    # A discharged window reports L0's score in place of L1's, so every fake the
    # gate lets through lands near zero instead of near one. Measure both paths
    # against flat L1 and keep only the ones that are close to free.
    flat_eer, _ = eer(rv["y"], rv["p1"])
    paths = {}
    for name, lo, hi in (("both", t_low, t_high),
                         ("discharge_only", t_low, 1.01),
                         ("fast_track_only", 0.0, t_high)):
        c = cascade_sim(rv, lo, hi)
        e, _ = eer(rv["y"], c["fused"])
        paths[name] = {"eer": float(e), "gpu": c["gpu_touched"]}
    print("\n=== does each cascade shortcut pay for itself? (val, opus) ===")
    print("%-18s %-10s %s" % ("config", "EER", "L1 invoked"))
    print("%-18s %-10.2f %.1f%%" % ("flat L1", 100 * flat_eer, 100.0))
    for k, v in paths.items():
        print("%-18s %-10.2f %.1f%%" % (k, 100 * v["eer"], 100 * v["gpu"]))

    TOL = 0.005          # half a point of EER is the most a shortcut may cost
    if paths["discharge_only"]["eer"] > flat_eer + TOL:
        print("\ndischarge costs %.2f points of EER to save %.1f%% of the GPU "
              "-- disabling it (T_LOW = 0.0). The transformer runs in ~11 ms; "
              "accuracy is the scarcer resource here, not compute."
              % (100 * (paths["discharge_only"]["eer"] - flat_eer),
                 100 * (1 - paths["discharge_only"]["gpu"])))
        t_low = 0.0
    if paths["fast_track_only"]["eer"] > flat_eer + TOL:
        print("\nfast-track costs %.2f points of EER -- disabling it (T_HIGH = 1.0)."
              % (100 * (paths["fast_track_only"]["eer"] - flat_eer)))
        t_high = 1.0

    cv = cascade_sim(rv, t_low, t_high)
    cx = cascade_sim(rx, t_low, t_high)

    # Bands are calibrated on simulated calls, not single windows -- see
    # simulate_sessions. Speaker is the session identity: a call is one voice.
    sy, ss = simulate_sessions(rv["y"], cv["fused"], val_df.speaker.to_numpy())
    red = band_sweep(sy, ss, 0.98)
    amber = band_sweep(sy, ss, 0.995) or red
    win_red = band_sweep(rv["y"], cv["fused"], 0.98)
    if red is None:
        raise SystemExit("session simulation produced no usable RED threshold")

    print("\n=== L0 gate sweep (val, opus) ===")
    print("%-8s %-16s %s" % ("T_LOW", "fake recall", "discharged"))
    for r in sweep_low[::40]:
        print("%-8.4f %-16.4f %.1f%%"
              % (r["t_low"], r["recall_fake"], 100 * r["discharged"]))

    print("\n=== chosen dials ===")
    if t_low > 0:
        print("T_LOW    = %.4f (keeps %.2f%% of fakes, discharges %.1f%% of windows)"
              % (t_low, 100 * (best_low or {}).get("recall_fake", 1.0),
                 100 * (best_low or {}).get("discharged", 0.0)))
    else:
        print("T_LOW    = 0.0000 (discharge path OFF -- it did not pay for itself)")
    if t_high < 1.0:
        print("T_HIGH   = %.4f (L0 %.1f%% precise above it, fast-tracks %.1f%%)"
              % (t_high, 100 * (best_high or {}).get("precision", 1.0),
                 100 * (best_high or {}).get("frac_fast_tracked", 0.0)))
    else:
        print("T_HIGH   = 1.0000 (fast-track path OFF -- it did not pay for itself)")
    print("L1_RED   = %.3f  (simulated calls: %.1f%% of spoofed calls flagged, "
          "%.1f%% of genuine calls false-alarm)"
          % (red["t"], 100 * red["recall"], 100 * red["false_alarm"]))
    print("L1_AMBER = %.3f  (%.1f%% caught, %.1f%% false alarm)"
          % (amber["t"], 100 * amber["recall"], 100 * amber["false_alarm"]))
    print("n simulated calls: %d over %d-window EMA (alpha=%.2f)"
          % (len(sy), 12, EMA_ALPHA))
    if win_red is None:
        print("note: no PER-WINDOW threshold reaches 98%% fake recall; that bar is "
              "only reachable once the EMA aggregates a call, which is exactly "
              "what C's risk engine does.")

    ev, _ = eer(rv["y"], cv["fused"])
    ex, _ = eer(rx["y"], cx["fused"])
    print("\ncascade fused EER: val %.2f%%  unseen-languages %.2f%%"
          % (100 * ev, 100 * ex))
    print("GPU touched on %.1f%% of val windows, %.1f%% of unseen-language windows"
          % (100 * cv["gpu_touched"], 100 * cx["gpu_touched"]))

    out = {"T_LOW": t_low, "T_HIGH": t_high,
           "L1_RED": red["t"], "L1_AMBER": amber["t"],
           "band_basis": "simulated 12-window EMA calls, alpha=%.2f" % EMA_ALPHA,
           "red_session_recall": red["recall"],
           "red_session_false_alarm": red["false_alarm"],
           "n_sessions": int(len(sy)),
           "flat_l1_eer": float(flat_eer),
           "paths": paths,
           "val_fused_eer": float(ev), "xling_fused_eer": float(ex),
           "cascade": {"val_opus": {"l0_discharged": cv["levels"][0],
                                    "l1_transformer": cv["levels"][1],
                                    "l2_fast_tracked": cv["levels"][2],
                                    "gpu_touched": cv["gpu_touched"]},
                       "xling_opus": {"l0_discharged": cx["levels"][0],
                                      "l1_transformer": cx["levels"][1],
                                      "l2_fast_tracked": cx["levels"][2],
                                      "gpu_touched": cx["gpu_touched"]}}}
    with open(os.path.join(CKPT_DIR, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.join(CKPT_DIR, "calibration.json"))

    if a.write:
        write_config({"T_LOW": t_low, "T_HIGH": t_high,
                      "L1_RED": red["t"], "L1_AMBER": amber["t"]})
        print("updated", CONFIG, "-- commit this and nobody touches it again")
    else:
        print("(re-run with --write to commit these into common/config.py)")


if __name__ == "__main__":
    main()
