"""Shared L1 inference: load a wav2vec2 audio classifier, score waveforms.

Used by the zero-shot baseline, by evaluate.py, and (via detector.py) by the
server -- so the fake-class index is resolved in exactly one place.
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CROP_S, HOP_S, SR


def read_audio(path, sr=SR, max_s=600.0):
    """Load any clip as 16 kHz mono float32, refusing anything over `max_s`.

    The duration check reads only the header, so a multi-gigabyte WAV is
    rejected before a single sample is decoded. Anything that accepts audio
    from outside -- B's /score endpoint above all -- should keep this cap or set
    a tighter one; libsndfile is a C parser and every byte it decodes is
    attack surface.

    Resampling is not optional. Every clip in data/clips is already 16 kHz
    because fetch_data.py resampled it, so a scorer that ignores the file's rate
    looks correct on our own data and silently mis-scores anything a user brings
    in: hand wav2vec2 a 44.1 kHz waveform and it hears a voice pitched down to
    a third of speed.
    """
    if max_s is not None:
        info = sf.info(path)
        if info.duration > max_s:
            raise ValueError("%s is %.0f s long; cap is %.0f s"
                             % (os.path.basename(str(path)), info.duration, max_s))
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        g = int(np.gcd(int(file_sr), int(sr)))
        wav = resample_poly(wav, sr // g, file_sr // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def fake_index(id2label):
    """Which logit is the 'fake' class? Never assume index 1."""
    for i, lab in id2label.items():
        if any(k in str(lab).lower() for k in ("fake", "spoof", "tts", "synth")):
            return int(i)
    return 1


class L1Scorer:
    def __init__(self, ckpt, device=None, crop_s=CROP_S, batch_size=16):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(ckpt)
        self.model = AutoModelForAudioClassification.from_pretrained(ckpt)
        self.model.to(self.device).eval()
        self.fake_i = fake_index(self.model.config.id2label)
        self.n = int(crop_s * SR)
        self.batch_size = batch_size
        self.ckpt = ckpt

    def warm(self):
        self.score_waves([np.zeros(self.n, dtype=np.float32)])

    def _prep(self, wav):
        from ml.data import crop_or_pad
        return crop_or_pad(np.asarray(wav, dtype=np.float32), self.n, None, False)

    def score_waves(self, wavs):
        """-> np.array of P(fake), one per waveform."""
        torch = self.torch
        out = []
        for i in range(0, len(wavs), self.batch_size):
            chunk = [self._prep(w) for w in wavs[i:i + self.batch_size]]
            enc = self.fe(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, self.fake_i]
            out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def score_files(self, paths, transform=None, progress=None):
        wavs = []
        scores = []
        for k, p in enumerate(paths):
            w = read_audio(p)
            if transform is not None:
                w = transform(w)
            wavs.append(w)
            if len(wavs) >= self.batch_size:
                scores.append(self.score_waves(wavs))
                wavs = []
                if progress and (k + 1) % (self.batch_size * 10) == 0:
                    progress(k + 1, len(paths))
        if wavs:
            scores.append(self.score_waves(wavs))
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)

    def score_clip_windows(self, wav, window_s=None, hop_s=HOP_S, transform=None):
        """Slide the server's window over a whole clip -> per-window P(fake).

        A single centre crop throws away most of a 30-second recording. This is
        the same windowing server/chunker.py does, so a verdict here matches
        what the live pipeline would have produced for the same audio.
        """
        window_s = window_s or (self.n / SR)
        n = int(window_s * SR)
        hop = max(1, int(hop_s * SR))
        wav = np.asarray(wav, dtype=np.float32)
        if transform is not None:
            wav = np.asarray(transform(wav), dtype=np.float32)
        if len(wav) <= n:
            return self.score_waves([wav])
        starts = list(range(0, len(wav) - n + 1, hop))
        return self.score_waves([wav[s:s + n] for s in starts])
