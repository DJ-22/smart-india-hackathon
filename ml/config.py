"""Training-side constants -- owned by A alone, read by nothing outside ml/.

Split out of common/config.py, which is the three-way contract between A, B and
C and should only hold values the *running server* needs. Where the dataset
cache lives, which base checkpoint we fine-tuned from and which languages were
held out are facts about how the model was built, not about how it is served.

The one thing B needs from this file is the pair (sr, crop_s), and B does not
import it: `ml/export.py` copies both into `ml/checkpoints/MANIFEST.json`, which
is the single place `server/detector.py` looks.
"""
import os

from common.config import REPO

# ------------------------------------------------------------------ TRAINING
SR = 16000
CROP_S = 4.0               # training crop length fed to the L1 transformer
HELDOUT_LANGS = ["Tamil", "Gujarati", "Manipuri"]   # never trained on; see RESULTS.md
# Chosen to span three language families the model never sees in training:
# Tamil (Dravidian), Gujarati (Indo-Aryan), Manipuri (Tibeto-Burman).

# ------------------------------------------------------------------- LAYOUT
# Importing REPO above is what loads .env (common/config.py calls load_dotenv at
# import time), so VG_DATA_DIR is already in the environment by the time we read
# it here. Do not "tidy" that import away and hardcode the parent directory.
DATA_DIR = os.environ.get("VG_DATA_DIR") or os.path.join(REPO, "data")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
INDEX_JSONL = os.path.join(DATA_DIR, "index.jsonl")
CKPT_DIR = os.path.join(REPO, "ml", "checkpoints")

HF_DATASET = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
BASE_CKPT = "MelodyMachine/Deepfake-audio-detection-V2"
