"""Splits and the training dataset.

Split policy (get this wrong and every number downstream is fiction):

  1. HELD-OUT LANGUAGES never enter training at all.  The labelled train split
     carries 16 languages; Tamil (Dravidian), Gujarati (Indo-Aryan) and Manipuri
     (Tibeto-Burman) are held out, so the unseen-language number spans three
     language families the model has never heard.  The other 13 form the pool.

  2. SPEAKER-DISJOINT val.  Each language has at most two speakers, recoverable
     from the clip id (see fetch_data.speaker_of).  Val takes one speaker from
     each of four pool languages -- BEN_F, BRX_M, hi_m, te_f -- covering
     Indo-Aryan, Tibeto-Burman and Dravidian and both genders, while every pool
     language still keeps a speaker in train.  These are pseudo-speakers, not
     true speaker ids; RESULTS.md says so out loud.

  3. The dataset is *paired*: every utterance exists as a real recording and as
     a TTS render of the same text by the same speaker.  Both members of a pair
     therefore land on the same side of the split automatically, since the split
     is by speaker.  Verified in split_report().
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import CROP_S, DATA_DIR, HELDOUT_LANGS, MANIFEST, SR

VAL_SPEAKERS = ["BEN_F", "BRX_M", "hi_m", "te_f"]


def source_of(clip_id):
    """Which sub-corpus a clip came from, inferred from its id convention.

    The challenge set is a merge of at least four collections, and they differ
    in recording chain and TTS system, not just language:

        indictts   ASM_F_ANGER_00342           10 languages, incl. held-out Tamil
        lowercase  te_m_00123                  English, Hindi, Telugu
        trainset   train_gujaratifemale_01049  Gujarati, Manipuri, some Malayalam
        textid     text1517                    Odia

    This matters more than language does: see RESULTS.md, where an unseen
    language from a *seen* corpus scores far better than an unseen language
    from an unseen corpus.
    """
    parts = str(clip_id).split("_")
    if parts[0] in ("train", "test", "val") and len(parts) >= 3:
        return "trainset"
    if len(parts) == 1:
        return "textid"
    if len(parts) >= 2 and parts[0].isupper():
        return "indictts"
    return "lowercase"


def load_manifest(path=MANIFEST):
    df = pd.read_csv(path)
    df["abspath"] = [os.path.join(DATA_DIR, p) for p in df["path"]]
    df["source"] = [source_of(i) for i in df["id"]]
    return df


def make_splits(df):
    """-> (train, val, xling) dataframes."""
    xling = df[df.language.isin(HELDOUT_LANGS)].copy()
    pool = df[~df.language.isin(HELDOUT_LANGS)].copy()
    val = pool[pool.speaker.isin(VAL_SPEAKERS)].copy()
    train = pool[~pool.speaker.isin(VAL_SPEAKERS)].copy()
    return train, val, xling


def split_report(df):
    train, val, xling = make_splits(df)
    lines = []
    for name, d in (("train", train), ("val", val), ("xling", xling)):
        if len(d) == 0:
            lines.append("%-6s EMPTY" % name)
            continue
        lines.append("%-6s n=%-5d real=%-5d fake=%-5d langs=%s speakers=%s"
                     % (name, len(d), (d.label == 0).sum(), (d.label == 1).sum(),
                        ",".join(sorted(d.language.unique())),
                        ",".join(sorted(d.speaker.unique()))))
    leak_s = set(train.speaker) & set(val.speaker)
    leak_l = set(train.language) & set(xling.language)
    leak_id = set(train.id) & set(val.id)
    lines.append("leak check: shared speakers=%s shared langs(train/xling)=%s shared ids=%d"
                 % (sorted(leak_s) or "none", sorted(leak_l) or "none", len(leak_id)))
    return "\n".join(lines)


# --------------------------------------------------------------------- torch

def crop_or_pad(wav, n, rng=None, train=True):
    if len(wav) >= n:
        if train and rng is not None:
            s = int(rng.integers(0, len(wav) - n + 1))
        else:
            s = max(0, (len(wav) - n) // 2)
        return wav[s:s + n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(wav)] = wav
    return out


class Augmenter:
    """Module-level callable so DataLoader workers can pickle it.

    A lambda here costs you every dataloader worker on Windows (spawn start
    method), which in turn costs you the GPU -- augmentation is ~10 ms/clip, so
    doing it serially in the training process leaves the GPU idle most of the
    time.
    """

    def __init__(self, seed=0):
        self.seed = seed
        self._rng = None

    def __call__(self, wav):
        import random as _random

        from ml.augment import augment
        if self._rng is None:
            self._rng = _random.Random(self.seed + os.getpid())
        return augment(wav, rng=self._rng)


class ClipDataset:
    """Returns raw float32 waveforms; the feature extractor runs in the collator."""

    def __init__(self, df, train=True, crop_s=CROP_S, augment_fn=None, seed=0):
        self.paths = list(df["abspath"])
        self.labels = list(df["label"])
        self.train = train
        self.n = int(crop_s * SR)
        self.augment_fn = augment_fn
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import soundfile as sf
        wav, _ = sf.read(self.paths[i], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = crop_or_pad(wav, self.n, self.rng, self.train)
        if self.train and self.augment_fn is not None:
            wav = self.augment_fn(wav)
            wav = crop_or_pad(wav, self.n, self.rng, False)
        return {"wav": wav, "labels": int(self.labels[i])}


class Collator:
    """Also module-level, and for the same reason as Augmenter: a closure here
    cannot be pickled to a spawned dataloader worker."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, batch):
        import torch
        wavs = [b["wav"] for b in batch]
        enc = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return enc


def make_collator(feature_extractor):
    return Collator(feature_extractor)


if __name__ == "__main__":
    d = load_manifest()
    print("manifest rows:", len(d))
    print(d.groupby(["language", "label"]).size().unstack(fill_value=0))
    print("duration: median=%.2fs mean=%.2fs min=%.2f max=%.2f"
          % (d.dur_s.median(), d.dur_s.mean(), d.dur_s.min(), d.dur_s.max()))
    print()
    print(split_report(d))
