"""EER / AUC and the threshold sweeps used at H8:00 calibration."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer(y_true, scores):
    """Equal error rate. scores = P(fake), y_true = 1 for fake."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def auc(y_true, scores):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(scores, dtype=float)))


def summarize(name, y_true, scores):
    e, t = eer(y_true, scores)
    a = auc(y_true, scores)
    return {"name": name, "n": int(len(y_true)), "eer": e, "eer_thr": t, "auc": a}


def fmt(row):
    return ("%-22s n=%-5d EER=%5.2f%%  AUC=%.4f  (thr@EER=%.3f)"
            % (row["name"], row["n"], 100 * row["eer"], row["auc"], row["eer_thr"]))


def l0_threshold_sweep(y_true, p0, target_recall=0.99, max_discharge=0.60,
                       n_points=400):
    """Pick T_LOW: the largest gate threshold that still keeps `target_recall`
    of the fakes in play. Optimising L0 for F1 is the classic mistake -- L0 is
    tuned for safe DISCHARGE, and every fake it discharges is a miss the rest of
    the cascade can never recover.

    The sweep walks quantiles of the score distribution, not a fixed 0.01 grid:
    a well-separated LightGBM piles most of its mass below 0.01, so a linear
    grid steps straight over every usable threshold and reports "no safe gate
    exists" when one does.

    Returns (best, rows). `best` is None only when no threshold at all meets the
    recall target -- callers must then disable the gate, never fall back to a
    guessed constant, which would silently discharge real fakes.
    """
    y_true = np.asarray(y_true)
    p0 = np.asarray(p0, dtype=float)
    n_fake = max(int((y_true == 1).sum()), 1)
    grid = np.unique(np.quantile(p0, np.linspace(0.0, max_discharge, n_points)))
    rows, best = [], None
    for t in grid:
        kept = p0 >= t
        recall_fake = float(((p0 >= t) & (y_true == 1)).sum()) / n_fake
        discharged = float(1.0 - kept.mean())
        rows.append({"t_low": float(t), "recall_fake": recall_fake,
                     "discharged": discharged})
        if recall_fake >= target_recall:
            best = rows[-1]
    return best, rows


def t_high_sweep(y_true, p0, target_precision=0.99, min_frac=0.005, n_points=400):
    """Pick T_HIGH: the smallest threshold above which L0 is (almost) never wrong
    about a clip being fake, so we can skip L1 and go straight to attribution.

    Quantile-swept for the same reason as T_LOW, and it ignores thresholds that
    fast-track fewer than `min_frac` of windows -- a threshold that is 100%
    precise on three clips is noise, not an operating point.
    """
    y_true = np.asarray(y_true)
    p0 = np.asarray(p0, dtype=float)
    grid = np.unique(np.quantile(p0, np.linspace(1.0, 0.4, n_points)))[::-1]
    rows, best = [], None
    for t in grid:
        sel = p0 > t
        if sel.mean() < min_frac:
            continue
        prec = float((y_true[sel] == 1).mean())
        rows.append({"t_high": float(t), "precision": prec,
                     "frac_fast_tracked": float(sel.mean())})
        if prec >= target_precision:
            best = rows[-1]
    return best, rows
