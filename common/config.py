"""Central config. Every constant is overridable via an env var of the
same name; defaults below are the agreed demo-day values.

CASCADE_ENABLED=0  -> flat pipeline: L1 always-on, L0 bypassed (kill switch)
L2_ENABLED=0       -> speaker verification fully disabled (kill switch)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Capture which keys came from the real shell environment BEFORE loading .env,
# so we can report provenance in the banner. (Precedence is enforced by
# load_dotenv(override=False), not by this set.)
_SHELL_KEYS = set(os.environ)

# Load .env ONCE here, from the repo root explicitly, before any os.getenv()
# below runs. config.py is imported by app/detector/vad/livekit_agent, so this
# one call covers the whole codebase - do not scatter load_dotenv() elsewhere.
# override=False keeps standard precedence: a real shell var still wins over .env.
try:  # .env is optional; config must work without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def source_of(name: str) -> str:
    """Where a config value came from: 'shell' env, '.env' file, or 'default'."""
    if name in _SHELL_KEYS:
        return "shell"
    if os.getenv(name) not in (None, ""):
        return ".env"
    return "default"


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
# Real mode by default (stub is opt-in); .env or a shell var can override.
STUB_MODE: bool = _env_bool("STUB_MODE", False)
# Resolved against the repo root so uvicorn can be started from any directory.
_manifest_raw = os.getenv("CHECKPOINT_MANIFEST", "ml/checkpoints/MANIFEST.json")
CHECKPOINT_MANIFEST: str = str(REPO_ROOT / _manifest_raw) if not os.path.isabs(_manifest_raw) else _manifest_raw
# ECAPA speaker-model download/cache dir (used by RealL2); also repo-root relative.
ECAPA_SAVEDIR: str = str(REPO_ROOT / "ml" / "checkpoints" / "ecapa")

# LiveKit credentials (used by server/tokens.py). No defaults - these are
# secrets that must come from .env or the shell. None means "not set"; the
# consumer is responsible for a clear error naming the missing variable.
LIVEKIT_URL: str | None = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY: str | None = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET: str | None = os.getenv("LIVEKIT_API_SECRET")


if __name__ == "__main__":
    names = [
        "STUB_MODE", "T_LOW", "T_HIGH", "EMA_ALPHA", "HYSTERESIS_CLEAN_WINDOWS",
        "MIN_SPEECH_RATIO", "WINDOW_S", "HOP_S", "TARGET_SR",
        "CASCADE_ENABLED", "L2_ENABLED", "CHECKPOINT_MANIFEST",
    ]
    for n in names:
        print(f"{n} = {globals()[n]!r}  (source: {source_of(n)})")
    assert T_LOW < T_HIGH, "T_LOW must be below T_HIGH"
    assert 0 < HOP_S <= WINDOW_S, "HOP_S must be positive and <= WINDOW_S"
    print("config self-test PASS")
