"""FastAPI scoring server.

Two modes, selected by env var STUB_MODE:
  STUB_MODE=1  -> plausible random WindowScores, zero ML imports, instant
                  startup. Unblocks the UI while models don't exist.
  STUB_MODE=0  -> real Cascade (server/detector.py). If the cascade can't
                  be constructed (missing deps/checkpoints), the server
                  logs a warning and falls back to stub instead of dying.

Run from repo root:
  python -m uvicorn server.app:app --port 8000
or standalone:
  python server\\app.py
"""
from __future__ import annotations

import io
import logging
import os
import random
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

# allow `python server\app.py` to find the `common` package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from common import config
from common.schema import WindowScore

log = logging.getLogger("server.app")

# Ensure our INFO logs (notably the startup banner) are visible even under
# `uvicorn server.app:app`, which configures only its own loggers and leaves
# the root logger without an INFO handler. No-op if logging is already set up.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Read from config (which loaded .env once, repo-root explicit) rather than a
# per-terminal env var - so STUB_MODE=0 in .env is enough, no shell var needed.
STUB_MODE: bool = config.STUB_MODE


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------
class StubScorer:
    """Plausible random scores. No ML imports, no audio decoding."""

    model_version = "stub"

    def __init__(self) -> None:
        # Reflect the kill switches so rehearsal matches real-mode routing.
        self.cascade_enabled = config.CASCADE_ENABLED
        self.l2_enabled = config.L2_ENABLED
        self.levels_active = (
            ([0] if self.cascade_enabled else []) + [1] + ([2] if self.l2_enabled else [])
        )

    def score(self, wav_bytes: bytes, session_id: str, window_id: int,
              t_start: float) -> WindowScore:
        r = random.random()
        level = 0 if r < 0.55 else (1 if r < 0.90 else 2)
        if level == 0 and not self.cascade_enabled:
            level = 1  # flat pipeline: L0 bypassed, L1 always-on
        if level == 2 and not self.l2_enabled:
            level = 1  # speaker verification disabled
        if level == 0:                    # discharged by L0
            prob_fake = random.uniform(0.0, config.T_LOW)
            latency = max(1, int(random.gauss(4, 1)))
            speaker_sim: Optional[float] = None
        elif level == 1:                  # resolved by L1
            prob_fake = random.betavariate(2, 5)
            latency = max(20, int(random.gauss(85, 15)))
            speaker_sim = None
        else:                             # escalated to L2
            prob_fake = random.betavariate(2, 2)
            latency = max(60, int(random.gauss(180, 30)))
            speaker_sim = random.uniform(0.3, 0.9)
        return WindowScore(
            session_id=session_id,
            window_id=window_id,
            t_start=t_start,
            t_end=t_start + config.WINDOW_S,
            prob_fake=round(prob_fake, 4),
            level_resolved=level,
            speech_ratio=round(random.uniform(0.35, 1.0), 3),
            speaker_sim=None if speaker_sim is None else round(speaker_sim, 4),
            latency_ms=latency,
            model_version=self.model_version,
        )

    def enroll(self, session_id: str, wav_bytes: bytes) -> float:
        return _wav_duration_s(wav_bytes)

    def reset_session(self, session_id: str) -> None:
        pass

    def unenroll(self, session_id: str) -> None:
        pass

    def metrics_extra(self) -> Dict[str, Any]:
        return {}


def _build_scorer() -> Any:
    """Return the active scorer; degrade to stub rather than crash.
    A ConfigError (incoherent thresholds) is the exception: it stops the
    server loudly instead of degrading to random-number stub mode."""
    if STUB_MODE:
        log.info("STUB_MODE=1 - serving random scores, no models loaded")
        return StubScorer()
    from server.detector import ConfigError  # cheap; heavy imports are deferred below
    try:
        from server.detector import CascadeScorer  # heavy imports live there
        scorer = CascadeScorer()
        log.info("real mode: cascade ready (model_version=%s)", scorer.model_version)
        return scorer
    except ConfigError:
        raise  # never mask a config typo behind random stub scores
    except Exception as exc:  # noqa: BLE001 - degrade, never crash
        log.warning("real mode unavailable (%s); falling back to stub", exc)
        return StubScorer()


def _wav_duration_s(wav_bytes: bytes) -> float:
    """Duration of a WAV payload; falls back to assuming raw 16k mono s16."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return len(wav_bytes) / (2.0 * config.TARGET_SR)


# --------------------------------------------------------------------------
# in-memory session store + metrics
# --------------------------------------------------------------------------
class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _get(self, sid: str) -> Dict[str, Any]:
        return self._sessions.setdefault(sid, {"scores": [], "next_window_id": 0})

    def next_window(self, sid: str) -> tuple[int, float]:
        with self._lock:
            s = self._get(sid)
            wid = s["next_window_id"]
            s["next_window_id"] = wid + 1
            return wid, wid * config.HOP_S

    def add(self, score: WindowScore) -> None:
        with self._lock:
            self._get(score.session_id)["scores"].append(score)

    def windows_since(self, sid: str, since: int) -> List[Dict[str, Any]]:
        with self._lock:
            s = self._sessions.get(sid)
            if not s:
                return []
            return [w.json() for w in s["scores"] if w.window_id > since]

    def reset(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: Dict[int, List[int]] = {0: [], 1: [], 2: []}

    def observe(self, score: WindowScore) -> None:
        with self._lock:
            self._latencies.setdefault(score.level_resolved, []).append(score.latency_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counts = {f"level_{lvl}": len(v) for lvl, v in self._latencies.items()}
            total = sum(counts.values())
            lat: Dict[str, Any] = {}
            for lvl, vals in self._latencies.items():
                if vals:
                    lat[f"level_{lvl}"] = {
                        "p50_ms": int(statistics.median(vals)),
                        "p95_ms": int(sorted(vals)[max(0, int(len(vals) * 0.95) - 1)]),
                    }
            return {
                "windows_scored": total,
                "per_level_counts": counts,
                "latency": lat,
                "discharge_rate": round(len(self._latencies.get(0, [])) / total, 3) if total else None,
            }


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
app = FastAPI(title="voice-clone detection scoring server")
app.add_middleware(  # Streamlit UI polls from another port
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _log_startup_banner(active_scorer: Any, probe_result: Optional[Dict[str, Any]]) -> None:
    """One glanceable block so you know exactly what's running without curling."""
    is_stub = isinstance(active_scorer, StubScorer)
    cascade = getattr(active_scorer, "cascade", None)
    if cascade is not None:  # real mode: report the cascade's effective values
        versions = [cascade.l0.version, cascade.l1.version, cascade.l2.version]
        t_low, t_high = cascade.t_low, cascade.t_high
        min_speech, hyst = cascade.min_speech_ratio, cascade.hysteresis_clean_windows
        cascade_enabled, l2_enabled = cascade.cascade_enabled, cascade.l2_enabled
    else:  # stub mode: no cascade, fall back to config defaults
        versions = ["stub", "stub", "stub"]
        t_low, t_high = config.T_LOW, config.T_HIGH
        min_speech, hyst = config.MIN_SPEECH_RATIO, config.HYSTERESIS_CLEAN_WINDOWS
        cascade_enabled, l2_enabled = config.CASCADE_ENABLED, config.L2_ENABLED
    def src(name: str, effective: Any, cfg: Any) -> str:
        # a manifest can override a threshold in real mode; otherwise the value
        # came from the shell env, .env, or the built-in default
        return "manifest" if effective != cfg else config.source_of(name)

    # Two-sided wiring probe (ml/fixtures/probe.json): one genuine + one cloned
    # clip, each asserted against its expected P(fake). Shows up here so an
    # upside-down checkpoint is caught in the banner, not during the demo.
    sanity_line = ""
    if probe_result and probe_result.get("clips"):
        by_role = {c["role"]: c for c in probe_result["clips"]}
        parts: List[str] = []
        for role in ("real", "fake"):
            c = by_role.get(role)
            if c is None:
                continue
            if c["skipped"]:
                parts.append(f"{role} p_fake=n/a [SKIP]")
            else:
                verdict = "PASS" if c["pass"] else "FAIL"
                parts.append(f"{role} p_fake={c['prob_fake']:.6f} [{verdict}]")
        if parts:
            sanity_line = "  wiring probe  : " + "  ".join(parts) + "\n"

    bar = "=" * 66
    log.info("\n%s\n"
             "  VOICE-CLONE DETECTION SERVER  |  MODE: %s (%s)\n"
             "  levels        : L0=%s  L1=%s  L2=%s\n"
             "  thresholds    : T_LOW=%s (%s)  T_HIGH=%s (%s)  MIN_SPEECH_RATIO=%s (%s)\n"
             "  hysteresis    : HYSTERESIS_CLEAN_WINDOWS=%s (%s)\n"
             "  kill switches : CASCADE_ENABLED=%s (%s)  L2_ENABLED=%s (%s)\n"
             "%s"
             "  source key    : shell env > .env > default (manifest overrides thresholds)\n"
             "%s",
             bar, "STUB (random scores)" if is_stub else "REAL", config.source_of("STUB_MODE"),
             versions[0], versions[1], versions[2],
             t_low, src("T_LOW", t_low, config.T_LOW),
             t_high, src("T_HIGH", t_high, config.T_HIGH),
             min_speech, src("MIN_SPEECH_RATIO", min_speech, config.MIN_SPEECH_RATIO),
             hyst, src("HYSTERESIS_CLEAN_WINDOWS", hyst, config.HYSTERESIS_CLEAN_WINDOWS),
             cascade_enabled, config.source_of("CASCADE_ENABLED"),
             l2_enabled, config.source_of("L2_ENABLED"), sanity_line, bar)


scorer = _build_scorer()
store = SessionStore()
metrics = Metrics()

# Exception types that map to HTTP 400. Imported only in real mode so stub
# mode keeps its zero-ML-import startup; StubScorer never raises these anyway.
_AUDIO_400_ERRORS: tuple = ()
if not isinstance(scorer, StubScorer):
    from server.detector import AudioDecodeError, EnrollmentError
    _AUDIO_400_ERRORS = (AudioDecodeError, EnrollmentError)

# Run the wiring probe once at startup; /health and /metrics expose probe_ok.
# None  = did not run (stub mode, or no probe.json / missing clip)
# True  = ran and every clip landed within tolerance
# False = ran and a clip failed its assertion or was skipped (e.g. sha mismatch)
_cascade = getattr(scorer, "cascade", None)
probe_result: Optional[Dict[str, Any]] = _cascade.sanity_probe() if _cascade is not None else None
probe_ok: Optional[bool] = probe_result.get("ok") if probe_result else None

_log_startup_banner(scorer, probe_result)


@app.post("/score")
async def score(request: Request,
                x_session_id: Optional[str] = Header(default=None),
                x_window_id: Optional[int] = Header(default=None),
                x_t_start: Optional[float] = Header(default=None)) -> Dict[str, Any]:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="missing X-Session-Id header")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body; expected WAV bytes")
    # Prefer the agent's true window id/time so a dropped POST does not shift
    # every later timestamp; fall back to the arrival counter when absent.
    if x_window_id is not None:
        window_id = x_window_id
        t_start = x_t_start if x_t_start is not None else x_window_id * config.HOP_S
    else:
        window_id, t_start = store.next_window(x_session_id)
    try:
        ws: WindowScore = scorer.score(body, x_session_id, window_id, t_start)
    except _AUDIO_400_ERRORS as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store.add(ws)
    metrics.observe(ws)
    return ws.json()


@app.get("/session/{sid}/windows")
def windows(sid: str, since: int = -1) -> Dict[str, Any]:
    return {"windows": store.windows_since(sid, since)}


@app.post("/session/{sid}/enroll")
async def enroll(sid: str, request: Request) -> Dict[str, Any]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body; expected WAV bytes")
    try:
        duration = scorer.enroll(sid, body)
    except _AUDIO_400_ERRORS as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "duration_s": round(float(duration), 2)}


@app.post("/session/{sid}/reset")
def reset(sid: str) -> Dict[str, Any]:
    store.reset(sid)
    scorer.reset_session(sid)  # clears history + hot flag, KEEPS the voiceprint
    return {"ok": True}


@app.post("/session/{sid}/unenroll")
def unenroll(sid: str) -> Dict[str, Any]:
    scorer.unenroll(sid)
    return {"ok": True}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_version": scorer.model_version,
        "levels_active": scorer.levels_active,
        "stub_mode": isinstance(scorer, StubScorer),
        "probe_ok": probe_ok,  # None = probe did not run at startup
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    snap = metrics.snapshot()
    snap.update(scorer.metrics_extra())
    snap["probe_ok"] = probe_ok  # None = probe did not run at startup
    return snap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
