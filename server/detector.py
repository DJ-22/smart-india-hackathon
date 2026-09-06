"""3-level detection cascade.

  L0 screening   - VAD gate + (LightGBM over MFCC/prosody | flatness stub)
  L1 detection   - (fine-tuned wav2vec2 P(fake)         | deterministic stub)
  L2 attribution - (SpeechBrain ECAPA speaker verify    | spectrum-cosine stub)

Every level loads independently; a missing manifest/checkpoint/dependency
logs a warning and swaps in that level's stub. The server must start and
serve correct-shaped responses with an EMPTY ml/checkpoints/ directory.

Standalone self-test:  python server\\detector.py
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import statistics
import sys
import threading
import time
import wave as wave_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import config
from common.schema import WindowScore
from server import vad

log = logging.getLogger("server.detector")

SR = config.TARGET_SR


class ConfigError(ValueError):
    """An incoherent config (e.g. T_LOW > T_HIGH). Must stop the server
    loudly - never degrade to stub, which would serve random numbers."""


class EnrollmentError(ValueError):
    """A rejected enrollment (too short/long, or silence). Maps to HTTP 400."""


class AudioDecodeError(ValueError):
    """Body could not be decoded as usable audio. Maps to HTTP 400."""


ENROLL_MIN_S = 4.0
ENROLL_MAX_S = 30.0
_EMBED_EPS = 1e-6


def combine_l2(p_detect: float, speaker_sim: float) -> float:
    """L2 fusion: flag on artifact evidence OR identity mismatch.
    Deliberately simple and in one place so it's easy to change."""
    return max(p_detect, 1.0 - speaker_sim)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


# --------------------------------------------------------------------------
# feature extraction (used by real L0; must match Daksh's training features)
# --------------------------------------------------------------------------
def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(h: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(np.array(0.0)), hz_to_mel(np.array(sr / 2.0)), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def mfcc_prosody_features(wav: np.ndarray, sr: int = SR,
                          n_mfcc: int = 13, n_mels: int = 26) -> np.ndarray:
    """13 MFCC mean+std + zcr/rms mean+std -> 30-dim vector.
    COORDINATE WITH DAKSH: the LightGBM checkpoint must be trained on
    exactly this function's output."""
    from scipy.fftpack import dct

    frame, hop, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    n = 1 + max(0, (len(wav) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx] * np.hamming(frame)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = np.log(spec @ _mel_filterbank(sr, n_fft, n_mels).T + 1e-10)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    zcr = np.mean(np.abs(np.diff(np.sign(frames + 1e-12), axis=1)) > 0, axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return np.concatenate(
        [mfcc.mean(0), mfcc.std(0), [zcr.mean(), zcr.std(), rms.mean(), rms.std()]]
    ).astype(np.float32)


# --------------------------------------------------------------------------
# per-level scorers: each has a stub and (lazy) real implementation
# --------------------------------------------------------------------------
class StubL0:
    """Spectral-flatness heuristic: real speech is tonal (low flatness ->
    low p0 -> discharged); noise/synthetic buzz is flatter."""

    version = "l0-stub-flatness"

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        spec = np.abs(np.fft.rfft(wav)) ** 2 + 1e-12
        flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        return float(np.clip(0.05 + 0.95 * flatness, 0.0, 1.0))


class RealL0:
    version = "l0-lightgbm"

    def __init__(self, model_path: str) -> None:
        import lightgbm  # optional dep; guarded by caller

        self._booster = lightgbm.Booster(model_file=model_path)

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        feats = mfcc_prosody_features(wav, sr)
        return float(self._booster.predict(feats[None, :])[0])


class StubL1:
    """Deterministic per-window pseudo-score, beta(2,5)-shaped so the
    distribution looks like a real detector's."""

    version = "l1-stub"

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        h = hashlib.blake2b(wav.tobytes(), digest_size=8).digest()
        return random.Random(int.from_bytes(h, "big")).betavariate(2, 5)


class RealL1:
    version = "l1-wav2vec2"

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self._torch = torch
        self._fe = AutoFeatureExtractor.from_pretrained(model_path)
        self._model = AutoModelForAudioClassification.from_pretrained(model_path).eval()
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._fake_idx = next(
            (int(i) for i, lbl in id2label.items()
             if any(k in str(lbl).lower() for k in ("fake", "spoof", "synth"))),
            1,  # convention with Daksh: index 1 = fake if labels are unnamed
        )

    def prob_fake(self, wav: np.ndarray, sr: int = SR) -> float:
        inputs = self._fe(wav, sampling_rate=sr, return_tensors="pt")
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        return float(self._torch.softmax(logits, dim=-1)[self._fake_idx])


class StubL2:
    """Cheap voiceprint: pooled log-magnitude spectrum, cosine-compared.
    Crude, but genuinely speaker-sensitive enough for wiring tests."""

    version = "l2-stub-spectrum"

    def embed(self, wav: np.ndarray, sr: int = SR, bins: int = 32) -> np.ndarray:
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        edges = np.linspace(0, len(spec), bins + 1).astype(int)
        v = np.array([np.log1p(spec[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])])
        n = float(np.linalg.norm(v))
        return (v / n if n > 0 else v).astype(np.float32)


class RealL2:
    version = "l2-ecapa"

    def __init__(self, source: str) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self._enc = EncoderClassifier.from_hparams(
            source=source, savedir=config.ECAPA_SAVEDIR
        )

    def embed(self, wav: np.ndarray, sr: int = SR) -> np.ndarray:
        t = self._torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        with self._torch.no_grad():
            e = self._enc.encode_batch(t).squeeze().cpu().numpy()
        n = float(np.linalg.norm(e))
        return (e / n if n > 0 else e).astype(np.float32)


# --------------------------------------------------------------------------
# cascade
# --------------------------------------------------------------------------
@dataclass
class _SessionState:
    hot: bool = False
    clean_streak: int = 0
    embedding: Optional[np.ndarray] = None


class Cascade:
    def __init__(self,
                 manifest_path: str = config.CHECKPOINT_MANIFEST,
                 cascade_enabled: Optional[bool] = None,
                 l2_enabled: Optional[bool] = None) -> None:
        self.cascade_enabled = config.CASCADE_ENABLED if cascade_enabled is None else cascade_enabled
        self.l2_enabled = config.L2_ENABLED if l2_enabled is None else l2_enabled

        manifest = self._read_manifest(manifest_path)
        thresholds = manifest.get("thresholds", manifest)
        if not isinstance(thresholds, dict):
            log.warning("manifest 'thresholds' is not an object (%s); using config defaults",
                        type(thresholds).__name__)
            thresholds = {}

        def _num(key: str, default: Any, cast: type) -> Any:
            # A hand-edited manifest with a bad value must fall back to the
            # config default (naming the key), not crash the whole cascade.
            try:
                return cast(thresholds.get(key, default))
            except (TypeError, ValueError):
                log.warning("manifest threshold %r invalid (%r); using default %r",
                            key, thresholds.get(key), default)
                return default

        self.t_low = _num("t_low", config.T_LOW, float)
        self.t_high = _num("t_high", config.T_HIGH, float)
        self.min_speech_ratio = _num("min_speech_ratio", config.MIN_SPEECH_RATIO, float)
        self.hysteresis_clean_windows = _num(
            "hysteresis_clean_windows", config.HYSTERESIS_CLEAN_WINDOWS, int)
        self._validate_thresholds()

        self.l0 = self._load_level("L0", manifest.get("l0_path"), RealL0, StubL0)
        self.l1 = self._load_level("L1", manifest.get("l1_path"), RealL1, StubL1)
        # L2 is pretrained (no teammate checkpoint); only attempt the real
        # (heavy, downloads voxceleb weights) path if the manifest asks.
        l2_source = manifest.get("l2_source")
        if self.l2_enabled and l2_source:
            self.l2 = self._load_level("L2", l2_source, RealL2, StubL2)
        else:
            if self.l2_enabled:
                log.warning("L2: no l2_source in manifest; using stub voiceprint")
            self.l2 = StubL2()

        self.model_version = str(manifest.get("model_version", "stub"))
        self.levels_active: List[int] = (
            ([0] if self.cascade_enabled else []) + [1] + ([2] if self.l2_enabled else [])
        )

        self._lock = threading.Lock()
        self._sessions: Dict[str, _SessionState] = {}
        self._latencies: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        # level-0 splits into two very different outcomes; count them apart
        self._gated = 0          # VAD-gated: too little speech to score
        self._l0_discharged = 0  # actually reached L0 and cleared as real
        # per-stage wall time, so VAD cost is visible separately from the
        # L0 classifier (the ~4ms architecture budget is for L0 alone)
        self._stage_latencies: Dict[str, List[float]] = {
            "vad": [], "l0": [], "l1": [], "l2": []
        }
        self._warmup()

    # -- construction helpers ------------------------------------------------
    def _validate_thresholds(self) -> None:
        """Stop the server loudly on an incoherent config rather than
        silently flatlining the dashboard (e.g. MIN_SPEECH_RATIO=1.5)."""
        if not (0.0 <= self.t_low <= self.t_high <= 1.0):
            raise ConfigError(
                f"threshold invariant violated: need 0 <= T_LOW <= T_HIGH <= 1, "
                f"got T_LOW={self.t_low}, T_HIGH={self.t_high}")
        if not (0.0 <= self.min_speech_ratio <= 1.0):
            raise ConfigError(
                f"MIN_SPEECH_RATIO must be in [0, 1], got {self.min_speech_ratio}")
        if self.hysteresis_clean_windows <= 0:
            raise ConfigError(
                f"HYSTERESIS_CLEAN_WINDOWS must be > 0, got {self.hysteresis_clean_windows}")

    @staticmethod
    def _read_manifest(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            log.warning("manifest %s not found; all levels fall back to stubs", path)
            return {}
        try:
            manifest = json.loads(p.read_text())
            log.info("loaded manifest %s: %s", path, sorted(manifest))
            return manifest
        except Exception as exc:  # noqa: BLE001
            log.warning("manifest %s unreadable (%s); using stubs", path, exc)
            return {}

    @staticmethod
    def _load_level(name: str, path: Optional[str], real_cls: type, stub_cls: type) -> Any:
        if path and (name == "L2" or Path(path).exists()):
            try:
                lvl = real_cls(path)
                log.info("%s: loaded real model from %s", name, path)
                return lvl
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                log.warning("%s: load failed (%s); using stub", name, exc)
        else:
            log.warning("%s: checkpoint missing (%r); using stub", name, path)
        return stub_cls()

    def _warmup(self) -> None:
        """Run every level once on a dummy 3 s window so the first real
        request isn't a latency outlier."""
        rng = np.random.default_rng(0)
        dummy = (rng.standard_normal(int(config.WINDOW_S * SR)) * 0.05).astype(np.float32)
        t0 = time.perf_counter()
        for label, fn in (
            ("vad", lambda: vad.speech_ratio(dummy, SR)),
            ("l0", lambda: self.l0.prob_fake(dummy, SR)),
            ("l1", lambda: self.l1.prob_fake(dummy, SR)),
            ("l2", lambda: self.l2.embed(dummy, SR)),
        ):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup step %s failed: %s", label, exc)
        log.info("cascade warmup done in %.0f ms (levels: %s/%s/%s)",
                 (time.perf_counter() - t0) * 1000,
                 self.l0.version, self.l1.version, self.l2.version)

    # -- session state -------------------------------------------------------
    def _session(self, sid: str) -> _SessionState:
        with self._lock:
            return self._sessions.setdefault(sid, _SessionState())

    def reset_session(self, sid: str) -> None:
        """Clear runtime state (hot flag, clean streak) but KEEP the enrolled
        voiceprint - a UI 'clear chart' must not silently disable L2. Use
        unenroll() to drop the voiceprint on purpose."""
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.hot = False
                s.clean_streak = 0

    def unenroll(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.embedding = None

    def enroll(self, sid: str, wav: np.ndarray, sr: int = SR) -> float:
        wav = np.asarray(wav, dtype=np.float32).ravel()
        duration = len(wav) / sr
        if duration < ENROLL_MIN_S:
            raise EnrollmentError(
                f"clip too short: {duration:.1f}s, need {ENROLL_MIN_S:.0f}-{ENROLL_MAX_S:.0f}s")
        if duration > ENROLL_MAX_S:
            raise EnrollmentError(
                f"clip too long: {duration:.1f}s, need {ENROLL_MIN_S:.0f}-{ENROLL_MAX_S:.0f}s")
        emb = self.l2.embed(wav, sr)
        if float(np.linalg.norm(emb)) < _EMBED_EPS:
            raise EnrollmentError("clip appears to be silence (no usable voiceprint)")
        if self._session(sid).embedding is not None:
            log.warning("re-enrolling session %s: replacing existing voiceprint", sid)
        self._session(sid).embedding = emb
        log.info("enrolled voiceprint for session %s (%.1fs of audio)", sid, duration)
        return duration

    def _l2_ready(self, sess: _SessionState) -> bool:
        return self.l2_enabled and sess.embedding is not None

    # -- scoring -------------------------------------------------------------
    def score(self, wav: np.ndarray, session_id: str,
              t_start: float, window_id: int) -> WindowScore:
        t0 = time.perf_counter()
        wav = np.asarray(wav, dtype=np.float32).ravel()
        sess = self._session(session_id)
        stage_ms: Dict[str, float] = {}

        def _timed(stage: str, fn):
            t = time.perf_counter()
            result = fn()
            stage_ms[stage] = (time.perf_counter() - t) * 1000
            return result

        speech = _timed("vad", lambda: vad.speech_ratio(wav, SR))
        sim: Optional[float] = None
        gated = False
        discharged = False

        if self.cascade_enabled:
            if speech < self.min_speech_ratio:
                prob, level = 0.0, 0  # not enough speech: never touch a model
                gated = True
            else:
                p0 = _timed("l0", lambda: self.l0.prob_fake(wav, SR))
                if p0 < self.t_low:
                    prob, level = p0, 0  # discharged as obviously real
                    discharged = True
                elif p0 > self.t_high and self._l2_ready(sess):
                    # obvious artifact: skip L1, go straight to the identity check
                    sim = _timed("l2", lambda: self._verify(wav, sess))
                    prob, level = combine_l2(p0, sim), 2
                else:
                    p1 = _timed("l1", lambda: self.l1.prob_fake(wav, SR))
                    if sess.hot and self._l2_ready(sess):
                        sim = _timed("l2", lambda: self._verify(wav, sess))
                        prob, level = combine_l2(p1, sim), 2
                    else:
                        prob, level = p1, 1
        else:
            # kill switch: flat pipeline, L0 bypassed, L1 always-on
            p1 = _timed("l1", lambda: self.l1.prob_fake(wav, SR))
            if sess.hot and self._l2_ready(sess):
                sim = _timed("l2", lambda: self._verify(wav, sess))
                prob, level = combine_l2(p1, sim), 2
            else:
                prob, level = p1, 1

        # VAD-gated windows carry no evidence about the speaker, so they must
        # NOT move the hysteresis streak - a suspect going silent must not cool.
        if not gated:
            self._update_hysteresis(sess, prob, level)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        with self._lock:
            self._latencies[level].append(latency_ms)
            if gated:
                self._gated += 1
            elif discharged:
                self._l0_discharged += 1
            for stage, ms in stage_ms.items():
                self._stage_latencies[stage].append(ms)

        return WindowScore(
            session_id=session_id,
            window_id=window_id,
            t_start=t_start,
            t_end=t_start + config.WINDOW_S,
            prob_fake=round(float(prob), 4),
            level_resolved=level,
            speech_ratio=round(float(speech), 3),
            speaker_sim=None if sim is None else round(float(sim), 4),
            latency_ms=latency_ms,
            model_version=self.model_version,
        )

    def _verify(self, wav: np.ndarray, sess: _SessionState) -> float:
        assert sess.embedding is not None
        return cosine_sim(self.l2.embed(wav, SR), sess.embedding)

    def _update_hysteresis(self, sess: _SessionState, prob: float, level: int) -> None:
        if level == 2:
            sess.hot = True  # sticky: reaching L2 keeps the session hot
        if prob > self.t_high:
            sess.hot = True
            sess.clean_streak = 0
        elif sess.hot:
            if prob < self.t_low:
                sess.clean_streak += 1
                if sess.clean_streak >= self.hysteresis_clean_windows:
                    sess.hot = False
                    sess.clean_streak = 0
                    log.info("session cooled down after %d clean windows",
                             self.hysteresis_clean_windows)
            else:
                sess.clean_streak = 0

    # -- metrics -------------------------------------------------------------
    @staticmethod
    def _pctl_summary(vals: List[float]) -> Dict[str, Any]:
        return {
            "p50_ms": round(statistics.median(vals), 2),
            "p95_ms": round(sorted(vals)[max(0, int(len(vals) * 0.95) - 1)], 2),
            "calls": len(vals),
        }

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            counts = {f"level_{lvl}": len(v) for lvl, v in self._latencies.items()}
            total = sum(counts.values())
            gated = self._gated
            discharged = self._l0_discharged
            lat = {f"level_{lvl}": self._pctl_summary(vals)
                   for lvl, vals in self._latencies.items() if vals}
            stage_lat = {stage: self._pctl_summary(vals)
                         for stage, vals in self._stage_latencies.items() if vals}
            hot = sum(1 for s in self._sessions.values() if s.hot)
        non_gated = total - gated  # windows that actually reached L0
        return {
            "per_level_counts": counts,
            "latency": lat,
            "stage_latency": stage_lat,  # VAD cost vs each classifier, separately
            "vad_gated": gated,          # too little speech to score
            "l0_discharged": discharged,  # reached L0 and cleared as real
            # discharge rate over NON-gated windows only (the stat Daksh tunes
            # the >=0.99-recall gate against); gated windows no longer inflate it
            "discharge_rate": round(discharged / non_gated, 3) if non_gated else None,
            "hot_sessions": hot,
            "level_versions": [self.l0.version, self.l1.version, self.l2.version],
            "cascade_enabled": self.cascade_enabled,
            "l2_enabled": self.l2_enabled,
        }


# --------------------------------------------------------------------------
# bytes-level adapter used by server/app.py real mode
# --------------------------------------------------------------------------
MIN_DECODE_SAMPLES = int(0.1 * SR)  # below ~0.1s a /score body is garbage, not a window


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    """WAV bytes -> float32 mono at TARGET_SR.

    A RIFF/WAV body that cannot be parsed to 16-bit PCM (24-bit, float,
    corrupt) is REJECTED with AudioDecodeError, never silently reinterpreted
    as raw samples. Only a non-RIFF body falls back to raw 16k mono int16
    (kept for backward compat). Either way, a result too short to be a real
    window is rejected - this is what turns a stray JSON body into a 400."""
    if wav_bytes[:4] == b"RIFF":
        try:
            with wave_mod.open(io.BytesIO(wav_bytes)) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            if width != 2:
                raise ValueError(f"{width * 8}-bit samples (need 16-bit PCM)")
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                audio = audio.reshape(-1, ch).mean(axis=1)
        except Exception as exc:  # noqa: BLE001 - a bad RIFF is a client error
            raise AudioDecodeError(f"unparseable WAV: {exc}") from exc
    else:
        audio = np.frombuffer(wav_bytes[: len(wav_bytes) // 2 * 2],
                              dtype=np.int16).astype(np.float32) / 32768.0
        sr = SR
    if audio.size < MIN_DECODE_SAMPLES:
        raise AudioDecodeError(
            f"decoded audio too short: {audio.size} samples "
            f"({audio.size / max(sr, 1):.3f}s at {sr} Hz), "
            f"need >= {MIN_DECODE_SAMPLES / SR:.2f}s")
    if sr != SR:
        from scipy.signal import resample_poly

        g = int(np.gcd(int(sr), SR))
        audio = resample_poly(audio, SR // g, int(sr) // g).astype(np.float32)
    return audio


class CascadeScorer:
    """The interface server/app.py expects, over raw WAV bytes."""

    def __init__(self) -> None:
        self.cascade = Cascade()
        self.model_version = self.cascade.model_version
        self.levels_active = self.cascade.levels_active

    def score(self, wav_bytes: bytes, session_id: str,
              window_id: int, t_start: float) -> WindowScore:
        return self.cascade.score(decode_wav(wav_bytes), session_id, t_start, window_id)

    def enroll(self, session_id: str, wav_bytes: bytes) -> float:
        return self.cascade.enroll(session_id, decode_wav(wav_bytes))

    def reset_session(self, session_id: str) -> None:
        self.cascade.reset_session(session_id)

    def unenroll(self, session_id: str) -> None:
        self.cascade.unenroll(session_id)

    def metrics_extra(self) -> Dict[str, Any]:
        return {"cascade": self.cascade.metrics()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from server.test_detector import run_all

    run_all()
