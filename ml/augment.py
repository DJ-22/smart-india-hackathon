"""Channel augmentation -- the difference between slide numbers and demo numbers.

LiveKit carries audio over Opus. Opus is a perceptual codec: it discards exactly
the high-frequency detail that a spectrogram-based deepfake detector leans on.
Train on clean 16 kHz studio audio and the detector quietly loses most of its
signal the moment it meets a real call.

libsndfile >= 1.2 ships an Opus encoder, so we can do a *real* Opus round-trip
in memory (~11 ms/clip) with no ffmpeg and no torchaudio.  Verify with:

    python ml/augment.py
"""
import io
import os
import random
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter, resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import SR

_OPUS_OK = None


def opus_available():
    """True if libsndfile can actually encode Opus here (checked once)."""
    global _OPUS_OK
    if _OPUS_OK is None:
        try:
            b = io.BytesIO()
            sf.write(b, np.zeros(SR, dtype=np.float32), SR, format="OGG", subtype="OPUS")
            b.seek(0)
            sf.read(b, dtype="float32")
            _OPUS_OK = True
        except Exception:                                        # noqa: BLE001
            _OPUS_OK = False
    return _OPUS_OK


def opus_roundtrip(wav, sr=SR):
    """Encode to Ogg/Opus and decode back. Falls back to band-limiting."""
    if opus_available():
        try:
            b = io.BytesIO()
            sf.write(b, wav, sr, format="OGG", subtype="OPUS")
            b.seek(0)
            out, out_sr = sf.read(b, dtype="float32", always_2d=False)
            if out.ndim > 1:
                out = out.mean(axis=1)
            if out_sr != sr:
                g = int(np.gcd(int(out_sr), sr))
                out = resample_poly(out, sr // g, out_sr // g).astype(np.float32)
            return out.astype(np.float32)
        except Exception:                                        # noqa: BLE001
            pass
    # fallback: 8 kHz down/up approximates the band-limiting Opus applies
    d = resample_poly(wav, 1, 2)
    return resample_poly(d, 2, 1).astype(np.float32)


def synthetic_rir(t60, sr=SR, rng=np.random):
    """Exponentially-decaying noise burst -- a crude but effective room impulse."""
    n = max(int(t60 * sr), 8)
    t = np.arange(n) / sr
    h = (rng.standard_normal(n) * np.exp(-6.9 * t / t60)).astype(np.float32)
    h[0] = 1.0                                   # keep the direct path dominant
    return h / (np.abs(h).max() + 1e-9)


def reverb(wav, t60, rng=np.random):
    """Put the speaker in a room.

    This is the single most important augmentation in the file, and the one the
    original plan had no equivalent of. IndicTTS-Deepfake is close-mic studio
    audio for BOTH classes, so 'reverberant' never appears next to a real label.
    Measured on the un-reverbed model, a T60 of just 0.15 s pushed genuine speech
    from P(fake) 0.00 to 0.84, and 0.30 s to 0.97 -- i.e. anyone speaking into a
    laptop in a normal room was confidently flagged as a deepfake.
    """
    peak = float(np.abs(wav).max()) + 1e-9
    y = fftconvolve(wav, synthetic_rir(t60, sr=SR, rng=rng))[:len(wav)]
    y = y.astype(np.float32)
    return np.clip(y / (np.abs(y).max() + 1e-9) * peak, -1.0, 1.0).astype(np.float32)


def _telephone_band(wav):
    """Crude 300-3400 Hz IIR shaping -- the PSTN leg of a spoofed call."""
    b = np.array([1.0, -1.0], dtype=np.float32)          # DC / rumble removal
    a = np.array([1.0, -0.97], dtype=np.float32)
    y = lfilter(b, a, wav).astype(np.float32)
    return resample_poly(resample_poly(y, 1, 2), 2, 1).astype(np.float32)


def augment(wav, sr=SR, rng=random):
    """Apply the live-call channel to a clean training clip.

    Room first, then codec, then noise -- the order a real call degrades in.
    """
    if rng.random() < 0.55:
        wav = reverb(wav, rng.uniform(0.10, 0.60))
    if rng.random() < 0.45:
        wav = opus_roundtrip(wav, sr)
    if rng.random() < 0.15:
        wav = _telephone_band(wav)
    if rng.random() < 0.35:
        amp = rng.choice([0.002, 0.006, 0.015])      # quiet room .. noisy room
        wav = wav + np.float32(amp) * np.random.randn(len(wav)).astype(np.float32)
    if rng.random() < 0.20:
        wav = wav * np.float32(rng.uniform(0.5, 1.4))
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def codec_only(wav, sr=SR):
    """Deterministic Opus pass -- used to build the L0 training features and to
    build the 'as it arrives over LiveKit' evaluation set."""
    return np.clip(opus_roundtrip(wav, sr), -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import time
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR)).astype(np.float32)
    print("opus_available:", opus_available())
    t = time.time()
    y = opus_roundtrip(x)
    print("roundtrip: in=%d out=%d  %.1f ms" % (len(x), len(y), (time.time() - t) * 1e3))
    # a real codec pass must actually change the signal
    n = min(len(x), len(y))
    d = float(np.abs(x[:n] - y[:n]).mean())
    print("mean abs delta: %.5f  %s" % (d, "OK (codec is doing something)"
                                        if d > 1e-4 else "WARNING: no-op!"))
    print("augment sample:", augment(x).shape)
    r = reverb(x, 0.3)
    print("reverb T60=0.3: len=%d  changed=%s" % (len(r), bool(np.abs(r - x).mean() > 1e-4)))
