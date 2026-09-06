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
WIN_N = int(config.WINDOW_S * SR)  # one full window of audio, tracks WINDOW_S (4 s)
NO_MANIFEST = "ml/checkpoints/DOES_NOT_EXIST.json"  # forces stubs even if real manifest appears


def _silence() -> np.ndarray:
    return np.zeros(WIN_N, dtype=np.float32)


def _speech() -> np.ndarray:
    """One WINDOW_S window of real captured speech from the committed fixture
    if present (tiled up if the fixture is shorter than a window), else a
    synthetic voiced-ish signal loud enough to pass the energy-fallback VAD."""
    p = Path(__file__).resolve().parent.parent / "fixtures" / "win3s.wav"
    if p.exists():
        with wave.open(str(p)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()),
                                  dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) > 0:
            reps = -(-WIN_N // len(audio))  # ceil-divide: enough copies for a window
            return np.tile(audio, reps)[:WIN_N]
    t = np.arange(WIN_N) / SR
    voiced = 0.3 * np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return voiced.astype(np.float32)


def _noise() -> np.ndarray:
    return (np.random.default_rng(1).standard_normal(WIN_N) * 0.3).astype(np.float32)


def _enroll_audio() -> np.ndarray:
    p = Path(__file__).resolve().parent.parent / "fixtures" / "enroll8s.wav"
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()),
                             dtype=np.int16).astype(np.float32) / 32768.0


class _CountingLevel:
    """Wraps a level scorer, counting calls and remembering the last output,
    so tests can PROVE a level ran (or was skipped) rather than infer it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.version = inner.version
        self.calls = 0
        self.last: float | None = None

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        self.calls += 1
        self.last = self.inner.prob_fake(wav, sr)
        return self.last


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
    # flat mode bypasses the L0 CLASSIFIER, but the VAD gate is independent of
    # the cascade and still runs: real speech flows through to L1, while silence
    # is gated at L0 (proven separately in test_silence_gated_in_flat_mode).
    assert c.score(_speech(), "s", 0.0, 0).level_resolved == 1, "speech must resolve at L1"
    assert c.score(_silence(), "s", 0.0, 0).level_resolved == 0, "silence gated even in flat mode"
    assert 0 not in c.levels_active  # the L0 classifier is inactive; the gate is not L0


def test_silence_gated_in_flat_mode() -> None:
    # Regression: the VAD gate must run in the flat (kill-switch) path too. A
    # silent window scored with CASCADE_ENABLED=False must resolve at L0 with
    # prob_fake=0.0 and NEVER reach L1 (which returns ~0.27 on silence, and up
    # to ~0.9 on near-silence live windows).
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=False)
    c.l1 = l1 = _CountingLevel(c.l1)
    ws = c.score(_silence(), "s", 0.0, 0)
    assert ws.level_resolved == 0
    assert ws.prob_fake == 0.0
    assert ws.speech_ratio < config.MIN_SPEECH_RATIO
    assert l1.calls == 0, "L1 must never be touched on a gated window, flat mode included"


def test_kill_switch_l2_disabled() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=False)
    c.enroll("s", _enroll_audio())  # even enrolled, L2 must never run
    c._session("s").hot = True
    for wav in (_speech(), _noise()):
        ws = c.score(wav, "s", 0.0, 0)
        assert ws.level_resolved != 2
        assert ws.speaker_sim is None
    assert 2 not in c.levels_active


def test_l2_runs_when_hot_and_enrolled() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c._session("s").hot = True
    c.l0 = _FixedL0(0.5)  # mid-band: route L1, then L2 because hot+enrolled
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 2
    assert ws.speaker_sim is not None
    assert 0.0 <= ws.speaker_sim <= 1.0
    # same speaker vs its own enrollment should look similar
    assert ws.speaker_sim > 0.5


def test_straight_to_l2_on_obvious_artifact() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c.l0 = _FixedL0(0.999)  # p0 > T_HIGH (0.9974): skip L1, go straight to identity check
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


def test_hot_latches_with_zero_t_low() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c._session("s").hot = True
    # Cooldown only counts windows scoring below T_LOW, and the calibrated
    # default is T_LOW=0.0 (discharge off), which no score can fall under. So a
    # hot session never cools on its own -- it stays hot for the rest of the call.
    for i in range(config.HYSTERESIS_CLEAN_WINDOWS + 2):
        c.score(_speech(), "s", float(i), i)
        assert c._session("s").hot, f"hot must latch under T_LOW=0.0 (window {i})"


def test_gated_silence_does_not_cool() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c._session("s").hot = True
    c._session("s").clean_streak = 3
    # a suspect going silent must NOT de-escalate: gated windows leave the
    # streak and hot flag untouched, no matter how many
    for i in range(2 * config.HYSTERESIS_CLEAN_WINDOWS):
        ws = c.score(_silence(), "s", float(i), i)
        assert ws.level_resolved == 0 and ws.speech_ratio < c.min_speech_ratio
    assert c._session("s").hot, "gated silence must not cool a hot session"
    assert c._session("s").clean_streak == 3, "gated windows must not touch the streak"


def test_reset_session() -> None:
    c = Cascade(manifest_path=NO_MANIFEST)
    c.enroll("s", _enroll_audio())
    c._session("s").hot = True
    c._session("s").clean_streak = 5
    c.reset_session("s")
    st = c._session("s")
    assert not st.hot and st.clean_streak == 0
    assert st.embedding is not None, "reset must KEEP the voiceprint (not disable L2)"
    c.unenroll("s")  # dropping the voiceprint is a separate, explicit action
    assert c._session("s").embedding is None


def test_enroll_validation() -> None:
    from server.detector import ENROLL_MIN_S, EnrollmentError

    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    # too short (3s < ENROLL_MIN_S 4s), too long (40s), and silence must all be
    # rejected. _speech() is now a full 4s window, so slice it for the short case.
    for bad, why in ((_speech()[: int(3 * SR)], "too short"),
                     (np.tile(_enroll_audio(), 6), "too long"),
                     (np.zeros(int(8 * SR), dtype=np.float32), "silence")):
        try:
            c.enroll("s", bad)
            assert False, f"expected rejection for {why}"
        except EnrollmentError:
            pass
    assert c._session("s").embedding is None, "no bad clip should have enrolled"
    # a valid clip enrolls
    dur = c.enroll("s", _enroll_audio())
    assert dur >= ENROLL_MIN_S
    assert c._session("s").embedding is not None


def test_reenroll_replaces_and_warns(capfd=None) -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    first = c._session("s").embedding
    c.enroll("s", _enroll_audio())  # second enroll: must warn (see logs) + replace
    assert c._session("s").embedding is not None
    # replacement occurred without error; embedding still present
    assert first is not None


def test_decode_rejects_garbage_and_bad_wav() -> None:
    import io as _io

    from server.detector import AudioDecodeError

    # non-RIFF JSON: raw fallback yields too few samples -> rejected
    try:
        decode_wav(b'{"session": "oops", "not": "audio"}')
        assert False, "JSON body should be rejected"
    except AudioDecodeError:
        pass
    # RIFF but 24-bit: must reject, not reinterpret as raw
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)  # 24-bit
        w.setframerate(SR)
        w.writeframes(b"\x00\x01\x02" * SR)
    try:
        decode_wav(buf.getvalue())
        assert False, "24-bit WAV should be rejected"
    except AudioDecodeError:
        pass
    # a valid 16-bit mono WAV still decodes fine (no regression)
    good = _io.BytesIO()
    with wave.open(good, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((_speech() * 32767).astype(np.int16).tobytes())
    decoded = decode_wav(good.getvalue())
    assert len(decoded) == len(_speech())


def test_decode_stereo_and_resample_still_work() -> None:
    import io as _io

    # stereo 16-bit at 48k: valid, must downmix + resample, not be rejected
    n = 48000
    stereo = np.zeros((n, 2), dtype=np.int16)
    stereo[:, 0] = (np.sin(2 * np.pi * 200 * np.arange(n) / 48000) * 8000).astype(np.int16)
    stereo[:, 1] = stereo[:, 0]
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(stereo.reshape(-1).tobytes())
    decoded = decode_wav(buf.getvalue())
    assert abs(len(decoded) - SR) < 100, "48k->16k of 1s should be ~16000 samples"


def test_window_id_headers_prevent_timeline_shift() -> None:
    import io as _io

    from fastapi.testclient import TestClient

    from server import app as appmod

    client = TestClient(appmod.app)
    good = _io.BytesIO()
    with wave.open(good, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((_speech() * 32767).astype(np.int16).tobytes())
    wav = good.getvalue()

    def post(window_id):
        return client.post("/score", content=wav, headers={
            "X-Session-Id": "drift", "X-Window-Id": str(window_id),
            "X-T-Start": repr(window_id * config.HOP_S)}).json()

    r0 = post(0)          # window 0 (t=0)
    # window 1 "dropped" (never posted); window 2 arrives next
    r2 = post(2)          # window 2 (t=2s), NOT the 2nd arrival
    assert r0["window_id"] == 0 and r0["t_start"] == 0.0
    assert r2["window_id"] == 2, "server must honor the agent's window id"
    assert r2["t_start"] == 2 * config.HOP_S, "audio from t=2s must be labeled t=2s"

    # without headers, the server falls back to its arrival counter
    r_a = client.post("/score", content=wav, headers={"X-Session-Id": "count"}).json()
    r_b = client.post("/score", content=wav, headers={"X-Session-Id": "count"}).json()
    assert (r_a["window_id"], r_b["window_id"]) == (0, 1)


def test_malformed_manifest_falls_back_to_defaults(tmp_path=None) -> None:
    import json as _json
    import tempfile

    # valid JSON, bad threshold value: must NOT crash the cascade
    d = tempfile.mkdtemp()
    mpath = Path(d) / "MANIFEST.json"
    mpath.write_text(_json.dumps({"thresholds": {"t_low": "abc"}, "model_version": "x"}))
    c = Cascade(manifest_path=str(mpath))
    assert c.t_low == config.T_LOW, "bad t_low must fall back to config default"
    assert c.model_version == "x", "the rest of the manifest still applies"
    # thresholds as a non-object: also tolerated
    mpath.write_text(_json.dumps({"thresholds": "0.2"}))
    c2 = Cascade(manifest_path=str(mpath))
    assert c2.t_low == config.T_LOW


def test_combine_rule() -> None:
    assert combine_l2(0.9, 0.9) == 0.9          # artifact dominates
    assert combine_l2(0.1, 0.2) == 0.8          # identity mismatch dominates
    assert combine_l2(0.0, 1.0) == 0.0


def test_metrics_shape() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True)
    c.score(_speech(), "s", 0.0, 0)
    m = c.metrics()
    assert set(m) >= {"per_level_counts", "latency", "stage_latency",
                      "discharge_rate", "hot_sessions", "level_versions"}
    assert sum(m["per_level_counts"].values()) == 1
    # VAD and L0 both ran on a speech window and are timed SEPARATELY
    assert "vad" in m["stage_latency"] and "l0" in m["stage_latency"]
    assert m["stage_latency"]["vad"]["calls"] == 1


def test_metrics_splits_gated_from_discharged() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True)
    c.score(_speech(), "s", 0.0, 0)   # speech -> scored at L1 (T_LOW=0.0: no L0 discharge)
    c.score(_silence(), "s", 1.0, 1)  # silence -> VAD-gated
    m = c.metrics()
    assert m["vad_gated"] == 1
    # T_LOW=0.0 closes the L0 discharge route, so nothing discharges
    assert m["l0_discharged"] == 0
    # discharge_rate is over NON-gated windows only: 0 discharged / 1 non-gated
    assert m["discharge_rate"] == 0.0


# ---------------------------------------------------------------------------
# route-forcing tests: every cascade path must actually execute
# ---------------------------------------------------------------------------
def test_route_t_low_zero_nothing_discharges() -> None:
    c = Cascade(manifest_path=NO_MANIFEST)
    c.t_low = 0.0  # no p0 can be < 0: discharge route closed
    c.l1 = l1 = _CountingLevel(c.l1)
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 1
    assert l1.calls == 1, "L1 must actually run when discharge is impossible"
    assert abs(ws.prob_fake - l1.last) < 1e-3


def test_route_t_high_zero_skips_l1_straight_to_l2() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c.t_high = 0.0  # every p0 exceeds it: obvious-artifact route
    c.t_low = -1.0  # and the discharge route is closed
    c.l0 = l0 = _CountingLevel(c.l0)
    c.l1 = l1 = _CountingLevel(c.l1)
    ws = c.score(_speech(), "s", 0.0, 0)
    assert ws.level_resolved == 2
    assert l0.calls == 1
    assert l1.calls == 0, "L1 must be SKIPPED on the straight-to-L2 route"
    assert ws.speaker_sim is not None
    # the combine rule actually fired: prob = max(p0, 1 - sim)
    assert abs(ws.prob_fake - combine_l2(l0.last, ws.speaker_sim)) < 1e-3


def test_route_t_high_zero_without_enrollment_falls_to_l1() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, l2_enabled=True)
    c.t_high = 0.0
    c.t_low = -1.0
    c.l1 = l1 = _CountingLevel(c.l1)
    ws = c.score(_speech(), "s", 0.0, 0)  # no voiceprint enrolled
    assert ws.level_resolved == 1, "no enrollment: L2 skipped cleanly, not an error"
    assert l1.calls == 1
    assert ws.speaker_sim is None


def test_hot_stickiness_survives_clean_windows() -> None:
    c = Cascade(manifest_path=NO_MANIFEST, cascade_enabled=True, l2_enabled=True)
    c.enroll("s", _enroll_audio())
    c.t_high = 0.0
    c.t_low = -1.0  # close the discharge route so p0 reaches the T_HIGH check
    ws = c.score(_speech(), "s", 0.0, 0)  # forces the session onto L2
    assert ws.level_resolved == 2
    assert c._session("s").hot
    c.t_high = config.T_HIGH  # restore: subsequent windows score normally
    c.t_low = config.T_LOW
    # Under the calibrated default (T_LOW=0.0) cooldown can never fire, so the
    # session stays hot through every subsequent clean window -- no cooling.
    for i in range(config.HYSTERESIS_CLEAN_WINDOWS + 2):
        c.score(_speech(), "s", float(i + 1), i + 1)
        assert c._session("s").hot, f"must stay hot under T_LOW=0.0 (window {i + 1})"


# ---------------------------------------------------------------------------
# kill switches, flipped FOR REAL via env vars in a subprocess (exercises
# the env -> common.config -> Cascade default plumbing end to end)
# ---------------------------------------------------------------------------
def _run_probe(env_overrides: dict, code: str) -> dict:
    import json
    import os
    import subprocess

    env = os.environ.copy()
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert out.returncode == 0, f"probe crashed:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_kill_switch_env_cascade_disabled() -> None:
    r = _run_probe({"CASCADE_ENABLED": "0"}, r"""
import json, sys, wave
sys.path.insert(0, ".")
import numpy as np
from common import config
assert config.CASCADE_ENABLED is False, "env var did not reach config"
from server.detector import Cascade
with wave.open("fixtures/win3s.wav") as w:
    speech = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
c = Cascade(manifest_path="ml/checkpoints/DOES_NOT_EXIST.json")
ws = c.score(speech, "s", 0.0, 0)  # real speech in flat mode -> straight to L1
print(json.dumps({"level": ws.level_resolved, "levels_active": c.levels_active,
                  "stages": sorted(c.metrics()["stage_latency"])}))
""")
    assert r["level"] == 1, "flat mode: speech must resolve at L1 (L0 classifier bypassed)"
    assert 0 not in r["levels_active"]
    assert "l0" not in r["stages"], "the L0 classifier must never have executed in flat mode"


def test_kill_switch_env_l2_disabled() -> None:
    r = _run_probe({"L2_ENABLED": "0"}, r"""
import json, sys, wave
sys.path.insert(0, ".")
import numpy as np
from common import config
assert config.L2_ENABLED is False, "env var did not reach config"
from server.detector import Cascade
with wave.open("fixtures/enroll8s.wav") as w:
    enroll = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
with wave.open("fixtures/win3s.wav") as w:
    speech = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
c = Cascade(manifest_path="ml/checkpoints/DOES_NOT_EXIST.json")
c.enroll("s", enroll)          # even enrolled...
c._session("s").hot = True     # ...and hot...
c.t_high = 0.0                 # ...and on the obvious-artifact route
c.t_low = -1.0
ws = c.score(speech, "s", 0.0, 0)
print(json.dumps({"level": ws.level_resolved, "sim": ws.speaker_sim,
                  "levels_active": c.levels_active,
                  "stages": sorted(c.metrics()["stage_latency"])}))
""")
    assert r["level"] != 2, "L2 must never run with L2_ENABLED=0"
    assert r["sim"] is None
    assert 2 not in r["levels_active"]
    assert "l2" not in r["stages"], "L2 must never have executed"


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
    # enroll needs a valid-length clip (>=4s); roundtrip the 8s enroll fixture
    enroll_buf = _io.BytesIO()
    enroll_audio = _enroll_audio()
    with wave.open(enroll_buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((enroll_audio * 32767).astype(np.int16).tobytes())
    dur = scorer.enroll("s", enroll_buf.getvalue())
    assert abs(dur - len(enroll_audio) / SR) < 0.05


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
