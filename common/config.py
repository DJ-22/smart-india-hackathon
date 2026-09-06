"""Single source of truth for every threshold and constant in VoiceGuard.

Owned jointly (A+B+C). Values below the CALIBRATED marker are set at H8:00
threshold calibration and must not be edited afterwards.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------- CALIBRATED
# Set by `python ml/calibrate.py --l1 ml/checkpoints/l1_run3 --write` on the val
# split over a real Opus round-trip. Do not hand-edit -- rerun the script.
#
# T_LOW is 0.0 on purpose: the discharge path works (8.6% of windows at 99% fake
# recall) but a discharged window reports L0's score instead of L1's, which cost
# 2.0 points of val EER (4.66% -> 6.63%) to save 8.6% of a 14 ms transformer.
# The T_HIGH fast-track costs 0.05 points and is kept. calibrate.py re-measures
# both paths every run and switches off whichever fails to pay for itself.
T_LOW = 0.0        # L0 below this -> resolved real, stop (0.0 = discharge off)
T_HIGH = 0.9974    # L0 above this -> skip L1, go straight to L2
# Bands apply to the EMA-smoothed score in ui/risk.py, and were calibrated on
# simulated 12-window calls: RED flags 98.0% of spoofed calls at a 0.7% false
# alarm rate on genuine ones; AMBER 99.7% at 1.3%.
L1_AMBER = 0.645
L1_RED = 0.825
EMA_ALPHA = 0.35
HYSTERESIS_CLEAN_WINDOWS = 10
MIN_SPEECH_RATIO = 0.30    # below this, emit nothing
# 4.0, not the 3.0 BUILD_PLAN started from: the L1 crop is 4 s, and measured on
# the val split a 3 s window costs 1.7 points of EER (9.30% vs 7.64%) purely
# from the train/serve length mismatch. HOP_S is unchanged, so the UI still gets
# a fresh score every second -- only the lookback grows.
WINDOW_S = 4.0
HOP_S = 1.0

# False = L1 always-on, flat pipeline. False is the measured default, not a
# retreat: L0 costs ~13 ms/window and L1 ~14 ms on this GPU, so gating the
# transformer behind the gate costs ~27 ms to save ~1 ms of transformer on the
# 7.4% of windows T_HIGH fast-tracks. The cascade is the right architecture
# where L1 is genuinely the bottleneck -- CPU-only inference, or many concurrent
# calls sharing one GPU -- and both levels stay wired so flipping this to True
# is a one-line demo. See the Latency section of RESULTS.md.
CASCADE_ENABLED = False
L2_ENABLED = True          # kill switch for speaker verification

# ------------------------------------------------------------------- LAYOUT
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training-side constants -- the dataset cache, the base checkpoint, the crop
# length, the held-out languages -- live in ml/config.py. They describe how the
# model was built, not how it is served, and nothing outside ml/ needs them.
# `sr` and `crop_s` reach the server through ml/checkpoints/MANIFEST.json.
