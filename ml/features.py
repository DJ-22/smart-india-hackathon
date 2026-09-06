"""Cheap spectral features for the Level 0 gate.

Everything here must run in single-digit milliseconds on a 3 s window -- L0's
whole job is to discharge obvious-real windows before the transformer is ever
touched. Imported by both ml/train_l0.py and server/detector.py, so the feature
order can never drift between training and serving.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import SR

FEATURE_NAMES = (
    ["mfcc%d_mean" % i for i in range(20)]
    + ["mfcc%d_std" % i for i in range(20)]
    + ["dmfcc%d_mean" % i for i in range(20)]
    + ["centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
       "bandwidth_mean", "flatness_mean", "flatness_std",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std",
       "hf_ratio_mean", "hf_ratio_std",
       "f0_std", "f0_absdiff_mean", "f0_voiced_frac"]
)
N_FEATURES = len(FEATURE_NAMES)


def l0_features(wav, sr=SR, with_f0=True):
    import librosa
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < 512:
        return np.zeros(N_FEATURES, dtype=np.float32)

    S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)) + 1e-10
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64)
    m = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    dm = librosa.feature.delta(m) if m.shape[1] > 8 else np.zeros_like(m)

    sc = librosa.feature.spectral_centroid(S=S, sr=sr)
    ro = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    fl = librosa.feature.spectral_flatness(S=S)
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(S=S, frame_length=1024)

    # energy above 6 kHz over total -- the band Opus mauls and vocoders fake badly
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hi = S[freqs >= 6000].sum(axis=0) / (S.sum(axis=0) + 1e-10)

    if with_f0:
        try:
            f0 = librosa.yin(wav, fmin=60, fmax=400, sr=sr,
                             frame_length=1024, hop_length=512)
            f0 = np.asarray(f0, dtype=np.float64)
            voiced = np.isfinite(f0) & (f0 > 60) & (f0 < 400)
            f0v = f0[voiced]
            f0_std = float(np.std(f0v)) if f0v.size > 2 else 0.0
            f0_d = float(np.mean(np.abs(np.diff(f0v)))) if f0v.size > 2 else 0.0
            f0_frac = float(voiced.mean())
        except Exception:                                        # noqa: BLE001
            f0_std = f0_d = f0_frac = 0.0
    else:
        f0_std = f0_d = f0_frac = 0.0

    v = np.concatenate([
        m.mean(1), m.std(1), dm.mean(1),
        [sc.mean(), sc.std(), ro.mean(), ro.std(), bw.mean(),
         fl.mean(), fl.std(), zcr.mean(), zcr.std(), rms.mean(), rms.std(),
         hi.mean(), hi.std(), f0_std, f0_d, f0_frac],
    ]).astype(np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
