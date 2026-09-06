# VoiceGuard — 10-Hour Build Plan

**Problem statement:** SIH26104 — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks
**Team:** 3 people · **Budget:** 10 hours · **Dataset:** IndicTTS-Deepfake-Challenge-Data only
**Architecture:** 3-level cascade (Screening → Detection → Attribution). No language identification anywhere.

---

## 0. Role assignment

| Role | Owns | Assign to |
|---|---|---|
| **A — Model** | Data, training, thresholds, all checkpoints | strongest ML person |
| **B — Pipeline** | LiveKit, audio chunking, FastAPI, cascade orchestration | strongest backend person |
| **C — Product** | Streamlit, risk engine, alerting, metrics, deck | strongest frontend/comms person |

B has the highest-variance task (WebRTC). If someone finishes early, they help B, not A.

---

## 1. Pre-work — do this TODAY, before hour zero

Skipping any of these costs you an hour tomorrow.

**Everyone**
- [ ] GitHub repo created, all three have push access, `main` protected off (no PR reviews during a sprint)
- [ ] Shared Google Drive folder for checkpoints and cached data
- [ ] Colab Pro or a machine with a GPU confirmed working (`nvidia-smi` returns something)

**A**
- [ ] HF account + read token in `~/.cache/huggingface/token`
- [ ] Run this and confirm it returns rows:
  ```python
  from datasets import load_dataset
  ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", split="train", streaming=True)
  print(next(iter(ds)).keys())
  ```
- [ ] Cache 3,000 clips to Drive as 16 kHz WAV + `manifest.csv` (see §4.1). **Do this tonight.** Streaming 3,000 audio rows takes 20–40 min and there is no reason to spend hackathon time on it.
- [ ] Download `MelodyMachine/Deepfake-audio-detection-V2` to Drive

**B**
- [ ] LiveKit Cloud free account, project created, `LIVEKIT_URL` / `API_KEY` / `API_SECRET` in a shared `.env`
- [ ] `pip install livekit livekit-agents fastapi uvicorn soundfile numpy scipy` in a clean venv, no import errors
- [ ] Two devices (laptop + phone) confirmed able to join a LiveKit sample room over the venue-equivalent network

**C**
- [ ] `pip install streamlit plotly requests` working
- [ ] Deck skeleton created with empty slides titled: Problem · Threat model · Architecture · Cascade metrics · Detection metrics · Demo · Roadmap

**Critical — do this tonight, it can invalidate the plan:**
- [ ] Pick the voice-cloning tool you will demo with (XTTS-v2, F5-TTS, ElevenLabs, whatever)
- [ ] Generate **20 clips** with it
- [ ] Run them through the un-finetuned `MelodyMachine` checkpoint
- [ ] Record the scores in the repo as `demo_generator_baseline.md`

If that checkpoint already flags them as fake, your demo is nearly free. If it calls them real, A must generate ~200 clips from that tool and fold them into training — and you need to know that at hour zero, not hour nine.

---

## 2. The interface contract — agreed in the first 30 minutes, changed never

Everything below is built against these two objects. B ships a stub server returning random values within 20 minutes so A and C are never blocked.

### `common/schema.py`

```python
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class WindowScore:
    session_id: str
    window_id: int          # monotonic, starts at 0
    t_start: float          # seconds since call start
    t_end: float
    prob_fake: float        # 0.0-1.0, raw model output, NOT smoothed
    level_resolved: int     # 0, 1, or 2 — which level made the call
    speech_ratio: float     # 0.0-1.0 from VAD; UI hides windows < 0.3
    speaker_sim: Optional[float] = None   # cosine vs enrolled print, L2 only
    latency_ms: int = 0
    model_version: str = "stub"

    def json(self): return asdict(self)
```

**Rules, non-negotiable:**
- The server returns `prob_fake` **raw**. All smoothing, hysteresis and banding happens in C's risk engine. One place, one owner.
- `level_resolved` is always populated, even in the stub (return `0` randomly 55% of the time). C's metrics panel depends on it.
- No field is ever removed. Adding optional fields is fine.

### HTTP surface (B owns)

```
POST /score              body: raw 16kHz mono WAV bytes
                         headers: X-Session-Id
                         -> WindowScore JSON

GET  /session/{sid}/windows?since={window_id}
                         -> {"windows": [WindowScore, ...]}

POST /session/{sid}/enroll   body: WAV bytes (reference voice)
                             -> {"ok": true, "duration_s": 8.2}

GET  /health             -> {"ok": true, "model_version": "...", "levels_active": [0,1,2]}
```

C polls `/session/{sid}/windows` every 500 ms. Do not build websockets. Polling is fine at this scale and cannot break on stage.

### Repo layout

```
voiceguard/
├── common/
│   ├── schema.py         # A+B+C agree, B owns file
│   └── config.py         # thresholds, all constants, single source of truth
├── ml/                   # A owns entirely
│   ├── fetch_data.py
│   ├── augment.py
│   ├── train_l1.py
│   ├── train_l0.py
│   ├── evaluate.py
│   └── checkpoints/
├── server/               # B owns entirely
│   ├── app.py            # FastAPI
│   ├── detector.py       # cascade orchestration, loads A's checkpoints
│   ├── chunker.py
│   └── livekit_agent.py
├── ui/                   # C owns entirely
│   ├── dashboard.py
│   └── risk.py
└── data/                 # gitignored
```

Nobody edits a file outside their directory. `common/` changes require all three in the room.

### `common/config.py` — the shared dials

```python
# Set to real values at H8:00 calibration. Placeholders until then.
T_LOW  = 0.15      # L0 below this -> resolved real, stop
T_HIGH = 0.85      # L0 above this -> skip L1, go straight to L2
L1_AMBER = 0.45    # smoothed prob bands
L1_RED   = 0.75
EMA_ALPHA = 0.35
HYSTERESIS_CLEAN_WINDOWS = 10
MIN_SPEECH_RATIO = 0.30    # below this, emit nothing
WINDOW_S = 3.0
HOP_S = 1.0

CASCADE_ENABLED = True     # False = L1 always-on, flat pipeline
L2_ENABLED = True          # kill switch for speaker verification
```

Both flags exist so that at H9:00 a broken level is one boolean away from disappearing.

---

## 3. Timeline at a glance

| Time | A (Model) | B (Pipeline) | C (Product) |
|---|---|---|---|
| 0:00–0:30 | **All three: contract + repo + stub agreed** | | |
| 0:30–2:30 | Data prep, splits, zero-shot baseline | LiveKit room → audio frames → WAV chunks | Streamlit shell on stub |
| 2:30–5:00 | L1 fine-tune run 1 | Chunker + VAD + FastAPI real endpoint | Risk engine + alert UI |
| **5:00–6:30** | **INTEGRATION 1 — hard cutoff** | | |
| 5:00–5:45 | L0 gate train | | |
| 5:45–6:30 | L1 run 2 (more data) | Cascade wiring | Metrics panel |
| 6:30–7:30 | Held-out language eval | L2 / ECAPA wiring | Enrollment UI |
| 7:30–8:00 | **SCOPE FREEZE** | | |
| 8:00–8:30 | **All three: threshold calibration** | | |
| 8:30–9:00 | End-to-end rehearsal on real devices | | |
| 9:00–10:00 | Deck, buffer, no code changes | | |

---

## 4. Person A — Model

### 4.1 (Tonight) Cache the data — `ml/fetch_data.py`

```python
import io, os, csv, soundfile as sf, numpy as np
from datasets import load_dataset
from scipy.signal import resample_poly

OUT = "data/clips"; os.makedirs(OUT, exist_ok=True)
N = 3000
ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data",
                  split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=5000).take(N)

rows = []
for i, r in enumerate(ds):
    a = r["audio"]
    wav, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_poly(wav, 16000, sr); sr = 16000
    p = f"{OUT}/{i:05d}.wav"
    sf.write(p, wav, sr)
    rows.append({
        "path": p,
        "label": int(r["is_tts"]),          # 1 = fake/TTS, 0 = real
        "language": r.get("language", "unk"),
        "speaker": r.get("speaker_id", f"unk_{i}"),
        "dur_s": round(len(wav) / sr, 2),
    })
    if i % 200 == 0: print(i, flush=True)

with open("data/manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

**Inspect the real column names first** — `is_tts`, `language`, `speaker_id` are what the community repos report, but confirm against `next(iter(ds)).keys()` and adjust. Do not let a KeyError at row 2,900 waste the run; wrap the row build in a try/except that prints and skips.

Then check and write down:
- class balance (`label.value_counts()`)
- duration distribution — if median < 3 s you must shorten `WINDOW_S`
- per-language counts

### 4.2 (H0:30–1:15) Splits — get this right or every number is fake

```python
import pandas as pd
df = pd.read_csv("data/manifest.csv")

HELDOUT_LANGS = ["tamil", "odia", "gujarati"]   # pick across families, adjust to actual values
xling = df[df.language.isin(HELDOUT_LANGS)]     # NEVER trained on
pool  = df[~df.language.isin(HELDOUT_LANGS)]

# speaker-disjoint split within the pool
spk = pool.speaker.unique()
rng = np.random.default_rng(0); rng.shuffle(spk)
val_spk = set(spk[:int(0.15 * len(spk))])
val   = pool[pool.speaker.isin(val_spk)]
train = pool[~pool.speaker.isin(val_spk)]
```

Two rules:
1. **Speaker-disjoint.** If the same speaker appears in train and val, your EER is meaningless — the model memorizes voices.
2. **Held-out languages never enter training.** This is the slide that answers "what about languages you didn't train on."

If `speaker_id` isn't in the dataset, fall back to a random split but **say so on the slide**. Don't quietly report an inflated number.

### 4.3 (H1:15–2:00) Zero-shot baseline — your "before" number

```python
import torch, numpy as np, soundfile as sf
from transformers import pipeline
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

pipe = pipeline("audio-classification",
                model="MelodyMachine/Deepfake-audio-detection-V2", device=0)

def score_file(p):
    out = pipe(p)
    return next(d["score"] for d in out if "fake" in d["label"].lower())
```

Run over `val` and over `xling`. Record both EER and AUC in `RESULTS.md`. Also run it over your 20 demo-generator clips. **Post these three numbers to the team chat immediately** — if the demo-generator number is bad, B and C need to know now.

### 4.4 (H2:00–2:30) Codec augmentation — `ml/augment.py`

LiveKit uses Opus. Opus destroys the high-frequency artifacts the detector relies on. Train without this and the live demo will underperform your slides by a wide margin.

```python
import torch, torchaudio, torchaudio.functional as F, random

def opus_roundtrip(wav_t, sr=16000):
    try:
        return F.apply_codec(wav_t, sr, format="ogg", encoder="opus")
    except Exception:
        # fallback: 8k downsample + back, approximates band-limiting
        d = torchaudio.transforms.Resample(sr, 8000)(wav_t)
        return torchaudio.transforms.Resample(8000, sr)(d)

def augment(wav_t, sr=16000):
    if random.random() < 0.40: wav_t = opus_roundtrip(wav_t, sr)
    if random.random() < 0.30: wav_t = wav_t + 0.004 * torch.randn_like(wav_t)
    if random.random() < 0.20: wav_t = wav_t * random.uniform(0.5, 1.4)
    return wav_t.clamp(-1, 1)
```

Verify `apply_codec` actually works in your environment **before** the training run. If it silently no-ops you lose the whole benefit.

### 4.5 (H2:30–5:00) L1 fine-tune run 1 — `ml/train_l1.py`

```python
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          TrainingArguments, Trainer)

CKPT = "MelodyMachine/Deepfake-audio-detection-V2"
fe = AutoFeatureExtractor.from_pretrained(CKPT)
model = AutoModelForAudioClassification.from_pretrained(CKPT, num_labels=2)
model.freeze_feature_encoder()          # CNN frozen, transformer trains

args = TrainingArguments(
    output_dir="ml/checkpoints/l1_run1",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=25,
    report_to="none",
)
```

Dataset collator: random 4-second crop (pad if shorter), apply `augment()` on train only, `fe(wav, sampling_rate=16000, return_tensors="pt")`.

Expect ~25–40 min for 3,000 clips × 3 epochs on a T4. **Set a 45-minute alarm.** If it hasn't converged, kill it and ship epoch 2 — a mediocre checkpoint that exists beats a good one that doesn't.

The moment run 1 saves: **copy it to Drive and post the path.** This is the guaranteed fallback demo model. Everything after this is upside.

### 4.6 (H5:00–5:45) L0 gate — `ml/train_l0.py`

Cheap features you already have, tuned for **discharge**, not accuracy.

```python
import librosa, numpy as np
def feats(wav, sr=16000):
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
    sc = librosa.feature.spectral_centroid(y=wav, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.95)
    zcr = librosa.feature.zero_crossing_rate(wav)
    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=sr)
    return np.concatenate([m.mean(1), m.std(1),
                           [sc.mean(), sc.std(), ro.mean(), ro.std(),
                            zcr.mean(), np.nanstd(f0), np.nanmean(np.abs(np.diff(f0)))]])
```

Fit `LGBMClassifier(n_estimators=300, num_leaves=31)`.

**Critical:** fit L0 on **codec-augmented audio only**. Rolloff features look brilliant on clean training clips and carry nothing after Opus. Train it clean and your gate will discharge everything on the live call.

Threshold sweep — pick `T_LOW` as the largest value where fake-class recall stays **≥ 0.99** on val:

```python
for t in np.arange(0.02, 0.40, 0.01):
    kept = p >= t
    recall_fake = ((p >= t) & (y == 1)).sum() / (y == 1).sum()
    print(f"{t:.2f}  recall_fake={recall_fake:.4f}  discharged={1-kept.mean():.2%}")
```

Take whatever discharge rate that buys — 40–60% is normal and plenty. Do **not** optimise L0 for F1.

### 4.7 (H5:45–6:30) L1 run 2

Same recipe, plus: the 200 demo-generator clips if §1 showed you need them, LR 1.5e-5, 4 epochs. **Only swap it in if val EER beats run 1.** Compare on the identical val set, no exceptions.

### 4.8 (H6:30–7:30) Held-out language eval + export

Run the final checkpoint over `xling`. Write to `RESULTS.md`:

```
Seen-language EER   : X.X%   AUC .XXX
Unseen-language EER : Y.Y%   AUC .YYY   (langs: tamil, odia, gujarati — never trained)
Demo-generator EER  : Z.Z%   (n=20, <tool name>)
L0 discharge rate   : NN%    at fake-recall 99%
```

That block is your metrics slide. Hand it to C verbatim.

Then export for B:

```python
# ml/checkpoints/MANIFEST.json
{"l1_path": "ml/checkpoints/l1_run2", "l0_path": "ml/checkpoints/l0.pkl",
 "model_version": "l1-run2+l0-v1", "t_low": 0.14, "t_high": 0.88}
```

B's `detector.py` reads this file. Never hardcode paths in two places.

---

## 5. Person B — Pipeline

### 5.1 (H0:30–0:50) Ship the stub FIRST

Nothing else you do matters if A and C are blocked. Twenty minutes, then move on.

```python
# server/app.py
import random, time
from fastapi import FastAPI, Request, Header
from common.schema import WindowScore

app = FastAPI()
STORE = {}   # session_id -> [WindowScore]

@app.post("/score")
async def score(request: Request, x_session_id: str = Header(...)):
    _ = await request.body()
    lvl = 0 if random.random() < 0.55 else (1 if random.random() < 0.8 else 2)
    w = WindowScore(
        session_id=x_session_id,
        window_id=len(STORE.get(x_session_id, [])),
        t_start=0.0, t_end=3.0,
        prob_fake=random.betavariate(2, 5),
        level_resolved=lvl,
        speech_ratio=random.uniform(0.4, 1.0),
        latency_ms=random.randint(5, 120),
    )
    STORE.setdefault(x_session_id, []).append(w)
    return w.json()

@app.get("/session/{sid}/windows")
def windows(sid: str, since: int = -1):
    return {"windows": [w.json() for w in STORE.get(sid, []) if w.window_id > since]}

@app.get("/health")
def health(): return {"ok": True, "model_version": "stub", "levels_active": [0,1,2]}
```

`uvicorn server.app:app --reload --port 8000`. Post the URL in chat. Now go.

### 5.2 (H0:50–2:30) LiveKit → audio frames

This is the riskiest hour of the entire build. Timebox it.

1. Token generation endpoint (`livekit.api.AccessToken`) so a browser can join a room.
2. Minimal browser page or LiveKit Meet sample joining that room, publishing mic.
3. A Python agent (`livekit-agents`) that joins the same room, subscribes to the remote audio track, and iterates frames.
4. Frame → `np.frombuffer(frame.data, dtype=np.int16)` → float32 `/32768` → resample 48000→16000 with `scipy.signal.resample_poly(x, 1, 3)`.
5. **Prove it:** write 10 seconds to `test.wav` and listen to it. Do not proceed until it sounds like speech. Silent or garbled WAV here means everything downstream is debugging noise.

**Hard rule: if audio frames are not landing by H2:30, switch to the fallback** — a Streamlit `st.audio_input` / file-upload path feeding the same `/score` endpoint. You demo "uploaded call recording, analysed in near-real-time." Judges accept this. A dead WebRTC demo at hour 9 is unrecoverable. Set the alarm now.

### 5.3 (H2:30–3:30) Chunker + VAD — `server/chunker.py`

```python
class RingChunker:
    def __init__(self, sr=16000, window_s=3.0, hop_s=1.0):
        self.sr, self.w, self.h = sr, int(sr*window_s), int(sr*hop_s)
        self.buf = np.zeros(0, dtype=np.float32); self.n = 0
    def push(self, chunk):
        self.buf = np.concatenate([self.buf, chunk])
        while len(self.buf) >= self.w:
            yield self.n, self.buf[:self.w].copy()
            self.buf = self.buf[self.h:]; self.n += 1
```

Silero VAD in front of everything:

```python
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
# speech_ratio = fraction of 30ms sub-frames flagged as speech
```

If `speech_ratio < MIN_SPEECH_RATIO`, return a `WindowScore` with `level_resolved=0` and `prob_fake=0.0` and **never touch the model**. Typically 30–50% of live call audio is silence, hold, or crosstalk — this is free saving before any classifier.

### 5.4 (H3:30–5:00) Cascade orchestration — `server/detector.py`

```python
class Cascade:
    def score(self, wav, sid):
        t0 = time.time()
        sr_ratio = vad_speech_ratio(wav)
        if sr_ratio < MIN_SPEECH_RATIO:
            return mk(0.0, 0, sr_ratio, t0)

        if not CASCADE_ENABLED:
            return mk(self.l1(wav), 1, sr_ratio, t0)

        p0 = self.l0(wav)                      # LightGBM, ~4ms
        if p0 < T_LOW:
            return mk(p0, 0, sr_ratio, t0)     # discharged
        if p0 > T_HIGH and L2_ENABLED:
            sim = self.l2(wav, sid)            # skip L1, go to attribution
            return mk(max(p0, 1 - sim), 2, sr_ratio, t0, speaker_sim=sim)

        p1 = self.l1(wav)                      # wav2vec2, ~80ms
        if L2_ENABLED and self.session_is_hot(sid):
            sim = self.l2(wav, sid)
            return mk(combine(p1, sim), 2, sr_ratio, t0, speaker_sim=sim)
        return mk(p1, 1, sr_ratio, t0)
```

Escalation stickiness (`session_is_hot`) lives here: once a session reaches L2 it stays until `HYSTERESIS_CLEAN_WINDOWS` consecutive clean windows pass.

L2 = `speechbrain/spkrec-ecapa-voxceleb`, pretrained, no training. Cosine similarity between the window embedding and the enrolled reference embedding. `combine(p1, sim)` can be as simple as `max(p1, 1 - sim)` — do not overthink it.

Load models **once at startup**, never per request. Warm both with a dummy 3-second tensor before serving, so the first real window isn't a 4-second outlier in your latency chart.

### 5.5 (H5:00–6:30) Integration + latency logging

Log per level: p50, p95, count. C needs this for the metrics panel.

### 5.6 (H6:30–7:30) L2 + enrollment endpoint

`POST /session/{sid}/enroll` stores the ECAPA embedding of an 8–10 second reference clip in memory. **Behind `L2_ENABLED`.** If it isn't clean by 7:30, flip the flag off and move on — see §8.

---

## 6. Person C — Product

### 6.1 (H0:30–2:30) Streamlit shell against the stub

Build the whole layout with fake data. No blocking calls, no waiting on B.

```
┌─────────────────────────────────────────────────────┐
│  ● LIVE   Session a3f9   00:42                      │
├──────────────────┬──────────────────────────────────┤
│  RISK   [ 0.71 ] │  Risk over time (line + bands)   │
│   ▲ AMBER        │  green/amber/red shaded regions  │
│                  │                                  │
│  Caller: unknown │  Level resolved  L0 ██████ 62%   │
│  Enrolled: ✓     │                  L1 ███    28%   │
│  Voice match .42 │                  L2 █      10%   │
├──────────────────┴──────────────────────────────────┤
│  ⚠ ELEVATED RISK — recommend call-back verification │
│  [ Request call-back ]  [ Escalate ]  [ Dismiss ]   │
├─────────────────────────────────────────────────────┤
│  Event log                                          │
│  00:38  risk crossed AMBER  (0.48)                  │
│  00:41  escalated to L2                             │
└─────────────────────────────────────────────────────┘
```

Poll `/session/{sid}/windows?since=N` every 500 ms in a `st.fragment(run_every=0.5)`. Keep all state in `st.session_state`.

### 6.2 (H2:30–4:00) Risk engine — `ui/risk.py`

This is the component that turns a classifier into the "dynamic impersonation risk score" the problem statement asks for. It is entirely yours and depends on nothing from A or B.

```python
class RiskEngine:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.ema = None
        self.band = "GREEN"
        self.clean_streak = 0
        self.events = []

    def update(self, w: dict, ctx: dict) -> dict:
        if w["speech_ratio"] < MIN_SPEECH_RATIO:
            return self.state()                      # ignore, don't decay

        p = w["prob_fake"]
        self.ema = p if self.ema is None else self.alpha*p + (1-self.alpha)*self.ema

        risk = min(1.0, self.ema * self._ctx_multiplier(ctx))

        new = "RED" if risk >= L1_RED else "AMBER" if risk >= L1_AMBER else "GREEN"

        # hysteresis: escalate instantly, de-escalate slowly
        if self._rank(new) > self._rank(self.band):
            self._log(f"risk crossed {new} ({risk:.2f})")
            self.band, self.clean_streak = new, 0
        elif new == "GREEN":
            self.clean_streak += 1
            if self.clean_streak >= HYSTERESIS_CLEAN_WINDOWS:
                self.band = "GREEN"
        else:
            self.clean_streak = 0
        return self.state()

    def _ctx_multiplier(self, ctx):
        m = 1.0
        if ctx.get("unknown_caller"):        m *= 1.25
        if ctx.get("high_value_txn"):        m *= 1.35
        if ctx.get("off_hours"):             m *= 1.10
        if ctx.get("first_contact"):         m *= 1.15
        return m
```

Contextual inputs come from a sidebar of toggles — "Unknown caller ID", "₹50L transfer pending", "Outside business hours". Faking them with checkboxes is entirely legitimate for a demo and it directly demonstrates the PS's contextual-enrichment requirement. Show the multiplier live: `0.58 × 1.35 = 0.78 → RED`.

Escalate instantly, de-escalate slowly. Without that asymmetry the gauge strobes and looks broken.

### 6.3 (H4:00–5:00) Alerting layer

On RED: modal recommending secondary verification (call-back, MFA, supervisor escalation), a logged event, and a disabled-looking "Approve transfer" button with a tooltip. That single disabled button communicates the entire value proposition faster than any slide.

Privacy panel: a small card reading "Audio retained: 0 s · Features logged: 47 windows · Inference: on-premise." That covers the PS's privacy module for the cost of three lines.

### 6.4 (H5:45–6:30) Metrics panel — the judge magnet

A second Streamlit tab, populated from A's `RESULTS.md` and B's latency log:

- Detection: seen-language EER/AUC, **unseen-language EER/AUC**, demo-generator EER
- Cascade: % windows resolved per level, mean latency per level, mean end-to-end latency, "GPU touched on N% of audio"
- One sentence: *"62% of windows resolved at Level 0 in 4 ms; the transformer ran on 31% of audio; median end-to-end latency 90 ms."*

Most teams show a confusion matrix. A cascade-efficiency panel is a systems result and almost nobody else will have one.

### 6.5 (H9:00–10:00) Deck

Seven slides. Threat model first (a 12-second cloned-CEO audio clip if you have one), architecture second, metrics third, live demo fourth. Roadmap slide: IndicSynth mimicry subset for generator diversity, IndicVoices and Vaani for accent and channel robustness, telecom SDK integration. Naming datasets you deliberately deferred, with reasons, reads as judgment rather than omission.

---

## 7. Joint checkpoints — all three stop and sync

**H0:00–0:30 — Contract.** Agree §2 verbatim. B pushes the stub. Nobody leaves until `curl /health` works from all three laptops.

**H5:00–6:30 — Integration 1.** Hard cutoff. Whatever exists gets wired. Real model behind real chunker behind real UI, end to end, once. Bugs found here are the only bugs you have time to fix.

**H7:30 — Scope freeze.** No new features. Only bug fixes on the demo path.

**H8:00–8:30 — Threshold calibration, together.** Do not do this by feel.

1. A runs the final cascade over the held-out set and produces a table of (`T_LOW`, `T_HIGH`, `L1_AMBER`, `L1_RED`) → (fake recall, false-alarm rate, L0 discharge %).
2. Pick the row where **fake recall ≥ 0.98** and false alarms are tolerable.
3. Write them into `common/config.py`. Commit. Nobody touches them again.
4. Replay your 20 demo-generator clips through the live pipeline with the final numbers. If they don't trip RED, lower `L1_RED` — the demo must work.

**H8:30–9:00 — Rehearsal.** On the actual devices, on venue-equivalent wifi, with the actual person who will speak. Run it three times. Time it. Whoever presents does all three.

**H9:00 — Code freeze.** Tag the commit. Nobody merges. Remaining hour is deck and sleep.

---

## 8. Fallback matrix — decide these now, not at hour 9

| If this breaks | By when | Do this |
|---|---|---|
| LiveKit / WebRTC | H2:30 | File-upload path in Streamlit, same `/score` endpoint. Demo = "recorded call analysis." |
| L1 fine-tune run 1 | H5:00 | Ship zero-shot `MelodyMachine` checkpoint. Report the honest number. |
| L1 run 2 worse than run 1 | H6:30 | Keep run 1. No discussion. |
| L0 gate misbehaving | any | `CASCADE_ENABLED = False`. Flat L1 pipeline. Cascade becomes a roadmap slide. |
| L2 / ECAPA not wired | H7:30 | `L2_ENABLED = False`. Keep the enrollment button. Describe L2 as implemented-not-wired. |
| Everything on fire | H8:30 | Recorded screen capture of the last working run. Record one at H8:30 regardless. |

**Record a working demo video at H8:30 no matter how well things are going.** It costs three minutes and it is the only thing that survives a venue wifi failure.

---

## 9. The five things that actually decide the outcome

1. **The stub server exists in the first 20 minutes.** Everything else is downstream of this.
2. **Test your demo cloning tool against the baseline tonight.** The generator gap is a bigger risk than anything about languages.
3. **Codec-augment training data.** Opus will silently eat your detector otherwise.
4. **Speaker-disjoint splits, held-out languages.** Honest numbers are more persuasive than high ones, and judges probe splits.
5. **Two kill switches, tested.** `CASCADE_ENABLED` and `L2_ENABLED` flipped off and back on at least once before H8:00, so you know they work when you need them.
