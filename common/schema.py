from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class WindowScore:
    session_id: str
    window_id: int          # monotonic, starts at 0
    t_start: float          # seconds since call start
    t_end: float
    prob_fake: float        # 0.0-1.0, raw model output, NOT smoothed
    level_resolved: int     # 0, 1, or 2 -- which level made the call
    speech_ratio: float     # 0.0-1.0 from VAD; UI hides windows < 0.3
    speaker_sim: Optional[float] = None   # cosine vs enrolled print, L2 only
    latency_ms: int = 0
    model_version: str = "stub"

    def json(self):
        return asdict(self)
