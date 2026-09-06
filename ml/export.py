"""Write ml/checkpoints/MANIFEST.json -- the single place B's detector.py looks.

  python ml/export.py --l1 ml/checkpoints/l1_run2

Never hardcode a checkpoint path in two files; if the model moves, it moves here.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import REPO
from ml.config import CKPT_DIR, CROP_S, SR


def _rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", default=os.path.join(CKPT_DIR, "l1_run1"))
    ap.add_argument("--version", default=None)
    a = ap.parse_args()

    l0 = os.path.join(CKPT_DIR, "l0.txt")
    cal_p = os.path.join(CKPT_DIR, "calibration.json")
    cal = json.load(open(cal_p, encoding="utf-8")) if os.path.exists(cal_p) else {}

    val_p = os.path.join(a.l1, "val_metrics.json")
    val = json.load(open(val_p, encoding="utf-8")) if os.path.exists(val_p) else {}

    version = a.version or ("%s+l0-v1" % os.path.basename(a.l1))
    man = {
        "l1_path": _rel(a.l1),
        "l0_path": _rel(l0) if os.path.exists(l0) else None,
        "l0_format": "lightgbm-text" if os.path.exists(l0) else None,
        "model_version": version,
        "sr": SR,
        "crop_s": CROP_S,
        "fake_label_index": 1,
        "t_low": cal.get("T_LOW", 0.15),
        "t_high": cal.get("T_HIGH", 0.85),
        "val_metrics": val.get("metrics", {}),
        "cascade": cal.get("cascade", {}),
    }
    out = os.path.join(CKPT_DIR, "MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(json.dumps(man, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
