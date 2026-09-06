"""Shared score contract between server, ML, and UI.

CONTRACT FILE — field set agreed with Daksh (ml/) and Tilika (ui/).
Do not rename or remove fields. New Optional fields with defaults may be
appended at the end only.

Smoothing / EMA / risk banding is deliberately NOT done here or anywhere
in the server: prob_fake is the raw model output. The UI owns smoothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class WindowScore:
    session_id: str
    window_id: int          # monotonic per session, starts at 0
    t_start: float          # seconds since call start
    t_end: float
    prob_fake: float        # 0.0-1.0, RAW model output, never smoothed
    level_resolved: int     # 0, 1 or 2 - which cascade level made the call
    speech_ratio: float     # 0.0-1.0 from VAD
    speaker_sim: Optional[float] = None  # cosine vs enrolled print, L2 only
    latency_ms: int = 0
    model_version: str = "stub"

    def json(self) -> Dict[str, Any]:
        return asdict(self)


if __name__ == "__main__":
    import json

    sample = WindowScore(
        session_id="selftest",
        window_id=0,
        t_start=0.0,
        t_end=4.0,
        prob_fake=0.42,
        level_resolved=1,
        speech_ratio=0.87,
    )
    d = sample.json()
    assert d["speaker_sim"] is None
    assert d["model_version"] == "stub"
    assert set(d) == {
        "session_id", "window_id", "t_start", "t_end", "prob_fake",
        "level_resolved", "speech_ratio", "speaker_sim", "latency_ms",
        "model_version",
    }
    print("WindowScore self-test PASS")
    print(json.dumps(d, indent=2))
