"""Sliding-window chunker: arbitrary-size incoming audio chunks in,
fixed WINDOW_S windows at HOP_S stride out.

Standalone self-test:  python server\\chunker.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.config import HOP_S, TARGET_SR, WINDOW_S


class RingChunker:
    """Buffers incoming float32 audio and yields overlapping windows.

    Window ids are monotonic from 0. Window i covers samples
    [i * hop, i * hop + window) of the stream, i.e. t_start = i * HOP_S.
    """

    def __init__(self, window_s: float = WINDOW_S, hop_s: float = HOP_S,
                 sr: int = TARGET_SR) -> None:
        self.window_samples = int(round(window_s * sr))
        self.hop_samples = int(round(hop_s * sr))
        if not 0 < self.hop_samples <= self.window_samples:
            raise ValueError("hop must be positive and <= window")
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_id = 0

    def push(self, chunk: np.ndarray) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Feed a chunk of any size; yields (window_id, window) for every
        window completed by this chunk. Windows are copies - safe to hold."""
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        self._buf = np.concatenate([self._buf, chunk]) if self._buf.size else chunk.copy()
        while self._buf.size >= self.window_samples:
            yield self._next_id, self._buf[: self.window_samples].copy()
            self._next_id += 1
            self._buf = self._buf[self.hop_samples:]

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_id = 0

    @property
    def next_window_id(self) -> int:
        return self._next_id


if __name__ == "__main__":
    from server.test_chunker import run_all

    run_all()
