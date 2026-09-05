"""Central config. Every constant is overridable via an env var of the
same name; defaults below are the agreed demo-day values.

CASCADE_ENABLED=0  -> flat pipeline: L1 always-on, L0 bypassed (kill switch)
L2_ENABLED=0       -> speaker verification fully disabled (kill switch)
"""
from __future__ import annotations

import os

try:  # .env is optional; config must work without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


T_LOW: float = _env_float("T_LOW", 0.15)
T_HIGH: float = _env_float("T_HIGH", 0.85)
EMA_ALPHA: float = _env_float("EMA_ALPHA", 0.35)  # for the UI's use; server never smooths
HYSTERESIS_CLEAN_WINDOWS: int = _env_int("HYSTERESIS_CLEAN_WINDOWS", 10)
MIN_SPEECH_RATIO: float = _env_float("MIN_SPEECH_RATIO", 0.30)
WINDOW_S: float = _env_float("WINDOW_S", 3.0)
HOP_S: float = _env_float("HOP_S", 1.0)
TARGET_SR: int = _env_int("TARGET_SR", 16000)
CASCADE_ENABLED: bool = _env_bool("CASCADE_ENABLED", True)
L2_ENABLED: bool = _env_bool("L2_ENABLED", True)
CHECKPOINT_MANIFEST: str = os.getenv("CHECKPOINT_MANIFEST", "ml/checkpoints/MANIFEST.json")


if __name__ == "__main__":
    names = [
        "T_LOW", "T_HIGH", "EMA_ALPHA", "HYSTERESIS_CLEAN_WINDOWS",
        "MIN_SPEECH_RATIO", "WINDOW_S", "HOP_S", "TARGET_SR",
        "CASCADE_ENABLED", "L2_ENABLED", "CHECKPOINT_MANIFEST",
    ]
    for n in names:
        print(f"{n} = {globals()[n]!r}")
    assert T_LOW < T_HIGH, "T_LOW must be below T_HIGH"
    assert 0 < HOP_S <= WINDOW_S, "HOP_S must be positive and <= WINDOW_S"
    print("config self-test PASS")
