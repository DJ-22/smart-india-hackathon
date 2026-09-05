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

STUB_MODE: bool = os.getenv("STUB_MODE", "1").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------
class StubScorer:
    """Plausible random scores. No ML imports, no audio decoding."""

    model_version = "stub"
    levels_active = [0, 1, 2]

    def score(self, wav_bytes: bytes, session_id: str, window_id: int,
              t_start: float) -> WindowScore:
        r = random.random()
        if r < 0.55:                      # discharged by L0
            level = 0
            prob_fake = random.uniform(0.0, config.T_LOW)
            latency = max(1, int(random.gauss(4, 1)))
            speaker_sim: Optional[float] = None
        elif r < 0.90:                    # resolved by L1
            level = 1
            prob_fake = random.betavariate(2, 5)
            latency = max(20, int(random.gauss(85, 15)))
            speaker_sim = None
        else:                             # escalated to L2
            level = 2
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

    def metrics_extra(self) -> Dict[str, Any]:
        return {}


def _build_scorer() -> Any:
    """Return the active scorer; degrade to stub rather than crash."""
    if STUB_MODE:
        log.info("STUB_MODE=1 - serving random scores, no models loaded")
        return StubScorer()
    try:
        from server.detector import CascadeScorer  # heavy imports live there
        scorer = CascadeScorer()
        log.info("real mode: cascade ready (model_version=%s)", scorer.model_version)
        return scorer
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

scorer = _build_scorer()
store = SessionStore()
metrics = Metrics()


@app.post("/score")
async def score(request: Request,
                x_session_id: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="missing X-Session-Id header")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body; expected WAV bytes")
    window_id, t_start = store.next_window(x_session_id)
    ws: WindowScore = scorer.score(body, x_session_id, window_id, t_start)
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
    duration = scorer.enroll(sid, body)
    return {"ok": True, "duration_s": round(float(duration), 2)}


@app.post("/session/{sid}/reset")
def reset(sid: str) -> Dict[str, Any]:
    store.reset(sid)
    scorer.reset_session(sid)
    return {"ok": True}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_version": scorer.model_version,
        "levels_active": scorer.levels_active,
        "stub_mode": isinstance(scorer, StubScorer),
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    snap = metrics.snapshot()
    snap.update(scorer.metrics_extra())
    return snap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
