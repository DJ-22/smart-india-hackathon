# VoiceGuard — real-time voice-clone (audio deepfake) detection

**SIH26104.** Detect AI-generated / cloned voices during a live call and show an
operator a rising **impersonation-risk** score — early enough to stop a
fraudulent transfer before it is approved.

Audio arrives over LiveKit WebRTC, is scored one short window at a time by a
three-level detection cascade, and the raw per-window scores are smoothed into a
GREEN / AMBER / RED risk band on an operator dashboard. On the shipped model the
detector reaches **AUC 0.99** on seen languages and **0.87** on our own live
demo generator; a genuine speaker in a room stays green while a clone climbs to
red within a few seconds of speech.

---

## How it works

```
 caller (browser / phone)
        │  WebRTC audio
        ▼
 LiveKit room ───────────────► server/livekit_agent.py   (subscribe-only bridge)
                                  │  per-participant RingChunker
                                  │  4 s windows, 1 s hop  →  WAV bytes
                                  ▼  POST /score  (non-blocking queue)
                        server/app.py  (FastAPI scoring server)
                                  │
                                  ▼
                        server/detector.py  — the cascade
                          VAD gate (speech_ratio ≥ 0.30, else "no info")
                          ├─ L0  LightGBM acoustic gate      (features + trees)
                          ├─ L1  wav2vec2 P(fake)            (the workhorse)
                          └─ L2  ECAPA speaker verify        (optional)
                                  │  WindowScore (raw prob_fake, never smoothed)
                                  ▼
                        GET /session/{id}/windows?since=N   (polled every 1 s)
                                  │
                                  ▼
                        ui/dashboard.py  (Streamlit)
                          risk.py: EMA smoothing → context multipliers →
                                   GREEN / AMBER / RED with hysteresis
```

Key design rules the whole repo agrees on:

- **The server emits raw `prob_fake` and never smooths.** All smoothing, banding,
  and hysteresis live in the UI (`ui/risk.py`). This is stated in the contract
  file `common/schema.py`.
- **The VAD gate is independent of the cascade.** Any window with
  `speech_ratio < MIN_SPEECH_RATIO` (0.30) returns `prob_fake = 0.0,
  level_resolved = 0` — meaning *no information*, not *genuine*. Gated windows
  never reach a model and never enter the UI's EMA or clean-streak.
- **Windows are 4.0 s** (`WINDOW_S`) to match the L1 model's crop length, with a
  **1.0 s hop** (`HOP_S`) so the UI updates once per second.
- **`CASCADE_ENABLED` is `False` by default** — a measured trade-off, not a
  missing feature. On a GPU the L0 gate costs more than the L1 transformer it
  guards, so the pipeline runs flat (L1 always-on). Both levels stay wired and
  calibrated; flipping the flag to `True` is a one-line change. See `RESULTS.md`.

---

## Repository layout

| Path | What it is |
|---|---|
| `common/` | Shared contract. `config.py` (all tunables, loads `.env` once) and `schema.py` (the `WindowScore` dataclass — the A/B/C interface). |
| `server/` | Audio pipeline + inference API. `app.py` (FastAPI), `detector.py` (cascade), `vad.py`, `chunker.py` (sliding window), `livekit_agent.py` (WebRTC→server bridge), `tokens.py` (LiveKit JWTs), `capture_test.py`, and `test_*.py`. |
| `ui/` | Streamlit operator dashboard. `dashboard.py` (app + 1 s auto-refresh), `risk.py` (risk engine), `alerts.py`, `livekit/` (embedded browser call client). |
| `ml/` | Model training, evaluation, calibration, and export. Produces the shipped checkpoint `l1_run4`. See `ml/README.md`. |
| `ml/checkpoints/` | Small committed artifacts the server reads (`MANIFEST.json`, `l0.txt`, `calibration.json`). The heavy L1 weights are **gitignored** and shared out of band. |
| `fixtures/` | Small committed test WAVs (16 kHz mono) shared by server/ml/ui tests. |
| `RESULTS.md` | Measured accuracy, cascade, and latency numbers. |

---

## Setup

Requires **Python 3.12** and, for real (non-stub) detection, an NVIDIA GPU is
strongly recommended.

```powershell
# from the repo root, Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU inference, install the CUDA build of PyTorch from the PyTorch index (a
plain `pip install torch` is CPU-only — see the note at the end of
`requirements.txt`).

Then create your local environment file:

```powershell
Copy-Item .env.example .env
# edit .env: fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
```

`.env` is gitignored. Precedence is **shell env var > `.env` > built-in default**
(`common/config.py`). LiveKit credentials have no defaults and are only needed
for the live-audio path (agent / tokens / capture); the server and UI run
without them.

**Model weights.** The committed L0 gate and manifest are enough for the L0 path,
but the L1 transformer weights (`ml/checkpoints/l1_run4/`, ~361 MB) are
gitignored and must be placed there for real L1 scoring. Without them the server
logs a warning and falls back to the stub scorer rather than crashing — so you
can always bring the UI up.

---

## Running

Run everything from the repo root with the venv active.

### 1. Scoring server

```powershell
python server\app.py
# or:  python -m uvicorn server.app:app --port 8000
```

Modes are selected by `STUB_MODE` in `.env`:

- `STUB_MODE=0` (default) — real cascade. On startup it prints a banner with the
  active levels, thresholds, kill-switch states, and a two-sided **wiring probe**
  (one genuine + one cloned clip asserted against expected P(fake)), so an
  upside-down checkpoint is caught before the demo, not during it.
- `STUB_MODE=1` — plausible random `WindowScore`s, zero ML imports, instant
  startup. Unblocks the UI when models aren't present.

### 2. LiveKit agent (live audio → server)

```powershell
python server\livekit_agent.py --room demo
python server\livekit_agent.py --selftest       # no LiveKit needed
```

Subscribe-only; keeps a separate `RingChunker` and `session_id` per participant
identity, so two speakers produce two independent score streams.

### 3. Operator dashboard

```powershell
$env:VG_SERVER_URL = "http://127.0.0.1:8000"   # optional, this is the default
$env:VG_SESSION_ID = "daksh"                    # the session/identity to monitor
.\.venv\Scripts\streamlit.exe run ui\dashboard.py
```

The dashboard polls `GET /session/{id}/windows?since=N` once per second inside a
Streamlit fragment, so the risk value, chart, call timer, and event log update
live while the embedded LiveKit call stays connected. It reconnects and rebuilds
if the backend restarts.

### 4. Tests / self-tests

```powershell
python common\config.py         # prints resolved config + self-test
python common\schema.py         # WindowScore contract self-test
python server\test_detector.py  # cascade routing, VAD gate, kill switches
python server\test_chunker.py   # sliding-window arithmetic
python server\vad.py            # VAD self-test
```

Model training / evaluation lives entirely in `ml/` — see `ml/README.md`.

---

## HTTP API

| Method & path | Purpose |
|---|---|
| `POST /score` | Score one WAV window. Headers: `X-Session-Id`, optional `X-Window-Id`, `X-T-Start`. Body: WAV bytes. Returns a `WindowScore`. |
| `GET /session/{sid}/windows?since=N` | All windows with `window_id > N`. The UI's incremental poll. |
| `POST /session/{sid}/enroll` | Enroll a reference voiceprint (WAV body) for L2. |
| `POST /session/{sid}/reset` | Clear a session's history (keeps enrollment). |
| `POST /session/{sid}/unenroll` | Drop a session's voiceprint. |
| `GET /health` | `{ok, model_version, levels_active, stub_mode, probe_ok}`. |
| `GET /metrics` | Per-level counts, stage latency p50/p95, discharge rate. |

### The `WindowScore` contract (`common/schema.py`)

```
session_id: str          window_id: int          # monotonic per session, from 0
t_start: float           t_end: float            # seconds since call start
prob_fake: float         # 0.0–1.0, RAW model output, never smoothed
level_resolved: int      # 0, 1 or 2 — which cascade level made the call
speech_ratio: float      # 0.0–1.0 from VAD
speaker_sim: float|None  latency_ms: int         model_version: str
```

Do not rename or remove fields; new optional fields may only be appended.

---

## Configuration reference

All values live in `common/config.py` and are overridable by an env var of the
same name (via `.env` or the shell). Calibrated demo-day defaults:

| Key | Default | Meaning |
|---|---|---|
| `STUB_MODE` | `0` | `1` = random-score stub, no ML imports. |
| `WINDOW_S` / `HOP_S` | `4.0` / `1.0` | Window length (= L1 crop) and stride. |
| `MIN_SPEECH_RATIO` | `0.30` | Below this a window is VAD-gated. |
| `T_LOW` / `T_HIGH` | `0.0` / `0.9974` | L0 discharge (off) / L2 fast-track cut-offs. |
| `L1_AMBER` / `L1_RED` | `0.645` / `0.825` | UI risk bands (on the smoothed score). |
| `EMA_ALPHA` | `0.35` | UI smoothing factor. |
| `HYSTERESIS_CLEAN_WINDOWS` | `10` | Clean windows needed to de-escalate. |
| `CASCADE_ENABLED` | `False` | `True` = run L0 gate before L1 (see `RESULTS.md`). |
| `L2_ENABLED` | `True` | Speaker-verification level on/off. |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | — | LiveKit creds (no defaults; live-audio path only). |
| `VG_SERVER_URL` / `VG_SESSION_ID` | `http://127.0.0.1:8000` / `demo-session` | UI: backend URL and session to monitor. |

---

## Results

Headline numbers on the shipped model (`ml/checkpoints/l1_run4`, 4.0 s windows,
Opus channel). Full tables — per language, per corpus, cascade, and latency — are
in **`RESULTS.md`**; how they were produced is in **`ml/README.md`**.

| Metric | Value |
|---|---|
| Seen-language EER / AUC | 4.7% / 0.990 |
| Unseen-language EER / AUC | 15.4% / 0.927 |
| Demo-generator EER / AUC (our own tool) | 6.8% / 0.873 |
| L1 latency (wav2vec2, 4 s window, GPU) | 14 ms p50 |
| Genuine laptop recordings false-alarmed RED | 0 / 5 |

---

## Module ownership

- **`ml/`** — models, training, evaluation, calibration, export.
- **`server/`** — audio pipeline, VAD, chunker, cascade, FastAPI inference server, LiveKit bridge.
- **`ui/`** — Streamlit operator dashboard and risk engine.
- **`common/`** — the shared `WindowScore` contract and central config that all three import.
