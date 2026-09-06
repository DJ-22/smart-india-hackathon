"""Unit tests for RingChunker. Run with pytest, or standalone:
  python server\\test_chunker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import config
from server.chunker import RingChunker

SR = 16000
WIN = int(config.WINDOW_S * SR)   # tracks WINDOW_S (4 s) -> 64000
HOP = int(config.HOP_S * SR)      # tracks HOP_S (1 s)    -> 16000


def _feed(chunker: RingChunker, audio: np.ndarray, chunk_size: int):
    out = []
    for i in range(0, len(audio), chunk_size):
        out.extend(chunker.push(audio[i:i + chunk_size]))
    return out


def _ramp(n: int) -> np.ndarray:
    # each sample equals its stream index, so window content is checkable
    return np.arange(n, dtype=np.float32)


def test_chunks_smaller_than_hop() -> None:
    c = RingChunker(config.WINDOW_S, config.HOP_S, SR)
    n = 10 * SR
    expected = 1 + (n - WIN) // HOP  # windowing is independent of chunk slicing
    wins = _feed(c, _ramp(n), chunk_size=4000)  # 4000 < 16000 hop
    assert len(wins) == expected
    for wid, w in wins:
        assert len(w) == WIN
        assert w[0] == wid * HOP  # window i starts at stream sample i*hop
    assert [wid for wid, _ in wins] == list(range(expected))


def test_chunk_larger_than_window() -> None:
    c = RingChunker(config.WINDOW_S, config.HOP_S, SR)
    wins = list(c.push(_ramp(WIN + 2 * HOP)))  # one chunk longer than a window
    assert len(wins) == 3  # windows starting at 0, 1*hop, 2*hop
    assert wins[2][1][0] == 2 * HOP


def test_exact_boundary_chunk() -> None:
    c = RingChunker(config.WINDOW_S, config.HOP_S, SR)
    wins = list(c.push(_ramp(WIN)))  # exactly one full window
    assert len(wins) == 1
    assert wins[0][0] == 0
    assert len(wins[0][1]) == WIN
    # next hop's worth completes exactly one more window
    wins2 = list(c.push(_ramp(WIN)[:HOP] + WIN))
    assert len(wins2) == 1
    assert wins2[0][0] == 1


def test_window_count_for_known_length() -> None:
    # N samples with N >= win -> 1 + (N - win) // hop windows,
    # regardless of how the input is sliced into chunks
    n = 160000  # 10 s
    expected = 1 + (n - WIN) // HOP
    for chunk_size in (1000, HOP, WIN, n):
        c = RingChunker(config.WINDOW_S, config.HOP_S, SR)
        wins = _feed(c, _ramp(n), chunk_size)
        assert len(wins) == expected, (chunk_size, len(wins))


def test_reset() -> None:
    c = RingChunker(config.WINDOW_S, config.HOP_S, SR)
    list(c.push(_ramp(WIN + HOP)))  # exactly two windows
    assert c.next_window_id == 2
    c.reset()
    assert c.next_window_id == 0
    wins = list(c.push(_ramp(WIN)))
    assert wins[0][0] == 0


def run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} PASS")
    print(f"all {len(tests)} chunker tests PASS")


if __name__ == "__main__":
    run_all()
