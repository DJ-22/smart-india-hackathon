"""Tests for the cascade with ZERO checkpoints on disk (all stub levels).
Run with pytest, or standalone:  python server\\test_detector.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import config
from server.detector import Cascade, CascadeScorer, combine_l2, decode_wav

SR = config.TARGET_SR
NO_MANIFEST = "ml/checkpoints/DOES_NOT_EXIST.json"  # forces stubs even if real manifest appears


def _silence() -> np.ndarray:
    return np.zeros(int(3 * SR), dtype=np.float32)


def _speech() -> np.ndarray:
    """Real captured speech from the committed fixture if present, else a
    synthetic voiced-ish signal loud enough to pass the energy-fallback VAD."""
    p = Path(__file__).resolve().parent.parent / "fixtures" / "win3s.wav"
    if p.exists():
        with wave.open(str(p)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()),
                                  dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) >= 3 * SR:
            return audio[: 3 * SR]
    t = np.arange(3 * SR) / SR
    voiced = 0.3 * np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return voiced.astype(np.float32)


def _noise() -> np.ndarray:
    return (np.random.default_rng(1).standard_normal(3 * SR) * 0.3).astype(np.float32)


class _FixedL0:
    """Pin L0's output so routing tests are deterministic regardless of
    what audio the stub heuristic would score."""

    version = "l0-fixed"

    def __init__(self, p: float) -> None:
        self.p = p

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        return self.p


def _check_shape(ws) -> None:
    assert 0.0 <= ws.prob_fake <= 1.0
    assert ws.level_resolved in (0, 1, 2)
    assert 0.0 <= ws.speech_ratio <= 1.0
    assert ws.latency_ms >= 0
    assert ws.t_end == ws.t_start + config.WINDOW_S


def test_all_stub_startup_and_shapes() -> None:
    c = Cascade(manifest_path=NO_MANIFEST)  # zero checkpoints: must not crash
    ws = c.score(_speech(), "s", 0.0, 0)
    _check_shape(ws)
    assert ws.model_version == "stub"


def test_silence_gate_never_touches_models() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True)
    ws = c.score(_silence(), "s", 0.0, 0)
    assert ws.prob_fake == 0.0
    assert ws.level_resolved == 0
    assert ws.speech_ratio < config.MIN_SPEECH_RATIO


def test_kill_switch_cascade_disabled_runs_flat() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=False)
    # even silence must hit L1: L0 (incl. the speech gate) is bypassed
    for wav in (_silence(), _speech(), _noise()):
        ws = c.score(wav, "s", 0.0, 0)
        assert ws.level_resolved == 1, "flat mode must resolve at L1"
    assert 0 not in c.levels_active


def test_kill_switch_l2_disabled() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=False)
    c.enroll("s", _speech())  # even enrolled, L2 must never run
    c._session("s").hot = True
    for wav in (_speech(), _noise()):
        ws = c.score(wav, "s", 0.0, 0)
        assert ws.level_resolved != 2
        assert ws.speaker_sim is None
    assert 2 not in c.levels_active


def test_l2_runs_when_hot_and_enrolled() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _speech())
    c._session("s").hot = True
    c.l0 = _FixedL0(0.5)  # mid-band: route L1, then L2 because hot+enrolled
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 2
    assert ws.speaker_sim is not None
    assert 0.0 <= ws.speaker_sim <= 1.0
    # same audio vs its own enrollment should look similar
    assert ws.speaker_sim > 0.5


def test_straight_to_l2_on_obvious_artifact() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _speech())
    c.l0 = _FixedL0(0.95)  # p0 > T_HIGH: skip L1, go straight to identity check
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 2
    assert ws.speaker_sim is not None
    assert c._session("s").hot, "reaching L2 must make the session sticky-hot"


def test_no_enrollment_means_no_l2() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c._session("s").hot = True  # hot but NOT enrolled
    c.l0 = _FixedL0(0.95)  # would go straight to L2 if a voiceprint existed
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 1
    assert ws.speaker_sim is None


def test_hysteresis_cooldown() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _speech())
    c._session("s").hot = True
    c.score(_speech(), "s", 0.0, 0)  # runs L2, stays hot
    assert c._session("s").hot
    for i in range(config.HYSTERESIS_CLEAN_WINDOWS):
        c.score(_silence(), "s", float(i), i)  # gated windows: prob 0.0 = clean
    assert not c._session("s").hot, "session must cool after N clean windows"


def test_reset_session() -> None:
    c = Cascade(manifest_path=NO_MANIFEST)
    c.enroll("s", _speech())
    c._session("s").hot = True
    c.reset_session("s")
    st = c._session("s")
    assert not st.hot and st.embedding is None


def test_combine_rule() -> None:
    assert combine_l2(0.9, 0.9) == 0.9          # artifact dominates
    assert combine_l2(0.1, 0.2) == 0.8          # identity mismatch dominates
    assert combine_l2(0.0, 1.0) == 0.0


def test_metrics_shape() -> None:
    c = Cascade(manifest_path=NO_MANIFEST)
    c.score(_speech(), "s", 0.0, 0)
    m = c.metrics()
    assert set(m) >= {"per_level_counts", "latency", "discharge_rate",
                      "hot_sessions", "level_versions"}
    assert sum(m["per_level_counts"].values()) == 1


def test_bytes_adapter_roundtrip() -> None:
    import io as _io

    audio = _speech()
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    decoded = decode_wav(buf.getvalue())
    assert len(decoded) == len(audio)
    assert np.max(np.abs(decoded - audio)) < 1e-3

    scorer = CascadeScorer()
    ws = scorer.score(buf.getvalue(), "s", 0, 0.0)
    _check_shape(ws)
    dur = scorer.enroll("s", buf.getvalue())
    assert abs(dur - 3.0) < 0.01


def run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} PASS")
    print(f"all {len(tests)} detector tests PASS")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    run_all()
