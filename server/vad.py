"""Silero VAD wrapper with an energy-based fallback.

The Silero model is loaded ONCE (lazy singleton, thread-safe), never per
call. If torch.hub can't fetch it (offline), we degrade to a simple
energy VAD with a warning - the pipeline must never crash because VAD is
unavailable.

Standalone self-test:  python server\\vad.py
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.config import TARGET_SR

log = logging.getLogger("server.vad")

# Silero VAD v4/v5 hard-requires these chunk sizes (asserted inside the
# model's forward); verified against the hub repo snakers4/silero-vad.
_CHUNK_SAMPLES = {16000: 512, 8000: 256}
_SPEECH_PROB_THRESHOLD = 0.5


def _energy_speech_ratio(wav: np.ndarray, sr: int) -> float:
    """Fallback VAD: fraction of 30 ms frames with RMS above a floor."""
    frame = max(1, int(0.03 * sr))
    n = len(wav) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1))
    thr = max(0.01, 0.15 * float(rms.max()))
    return float(np.mean(rms > thr))


class _SileroVAD:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._torch = None
        self._load_attempted = False
        self.using_fallback = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            try:
                import torch

                torch.set_num_threads(1)
                model, _utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", trust_repo=True
                )
                model.eval()
                self._model = model
                self._torch = torch
                log.info("silero VAD loaded (torch.hub)")
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                log.warning(
                    "silero VAD unavailable (%s); using energy-based fallback VAD",
                    exc,
                )
                self.using_fallback = True
            finally:
                self._load_attempted = True

    def speech_ratio(self, wav: np.ndarray, sr: int = 16000) -> float:
        """Fraction of the window flagged as speech, 0.0-1.0."""
        wav = np.asarray(wav, dtype=np.float32).ravel()
        if wav.size == 0:
            return 0.0
        if sr not in _CHUNK_SAMPLES:
            from scipy.signal import resample_poly

            g = int(np.gcd(int(sr), TARGET_SR))
            wav = resample_poly(wav, TARGET_SR // g, int(sr) // g).astype(np.float32)
            sr = TARGET_SR
        self._ensure_loaded()
        if self.using_fallback:
            return _energy_speech_ratio(wav, sr)
        try:
            return self._silero_ratio(wav, sr)
        except Exception as exc:  # noqa: BLE001
            log.warning("silero inference failed (%s); energy fallback for this window", exc)
            return _energy_speech_ratio(wav, sr)

    def _silero_ratio(self, wav: np.ndarray, sr: int) -> float:
        chunk = _CHUNK_SAMPLES[sr]
        n = wav.size // chunk
        if n == 0:
            return 0.0
        torch = self._torch
        assert self._model is not None
        with self._lock, torch.no_grad():
            if hasattr(self._model, "reset_states"):
                self._model.reset_states()
            speech = 0
            for i in range(n):
                t = torch.from_numpy(wav[i * chunk : (i + 1) * chunk])
                if float(self._model(t, sr).item()) > _SPEECH_PROB_THRESHOLD:
                    speech += 1
        return speech / n


_vad = _SileroVAD()


def speech_ratio(wav: np.ndarray, sr: int = 16000) -> float:
    return _vad.speech_ratio(wav, sr)


def warmup() -> None:
    """Trigger model load + one inference so the first real window is fast."""
    _vad.speech_ratio(np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR)


def using_fallback() -> bool:
    return _vad.using_fallback


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sr = TARGET_SR
    rng = np.random.default_rng(0)

    silence = np.zeros(3 * sr, dtype=np.float32)
    # speech-like: noise amplitude-modulated at a syllabic ~4 Hz rate
    t = np.arange(3 * sr) / sr
    speechy = (rng.standard_normal(3 * sr) * 0.2 *
               (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    half = np.concatenate([silence[: sr * 3 // 2], speechy[: sr * 3 // 2]])

    r_sil = speech_ratio(silence)
    r_spe = speech_ratio(speechy)
    r_half = speech_ratio(half)
    print(f"fallback_mode={using_fallback()}")
    print(f"speech_ratio(silence)     = {r_sil:.3f}")
    print(f"speech_ratio(speech-like) = {r_spe:.3f}")
    print(f"speech_ratio(half/half)   = {r_half:.3f}")
    assert r_sil < 0.1, "silence must score near zero"

    # exercise the energy fallback path explicitly, whatever loaded above
    assert _energy_speech_ratio(silence, sr) == 0.0
    e_half = _energy_speech_ratio(half, sr)
    assert 0.2 < e_half < 0.8, f"energy VAD on half-silence gave {e_half}"
    print("vad self-test PASS")
