"""Unit tests for RingChunker. Run with pytest, or standalone:
  python server\\test_chunker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from server.chunker import RingChunker

SR = 16000
WIN = 3 * SR   # 48000
HOP = 1 * SR   # 16000


def _feed(chunker: RingChunker, audio: np.ndarray, chunk_size: int):
    out = []
    for i in range(0, len(audio), chunk_size):
        out.extend(chunker.push(audio[i:i + chunk_size]))
    return out


def _ramp(n: int) -> np.ndarray:
    # each sample equals its stream index, so window content is checkable
    return np.arange(n, dtype=np.float32)


def test_chunks_smaller_than_hop() -> None:
    c = RingChunker(3.0, 1.0, SR)
    wins = _feed(c, _ramp(10 * SR), chunk_size=4000)  # 4000 < 16000 hop
    assert len(wins) == 8  # 1 + (160000 - 48000) // 16000
    for wid, w in wins:
        assert len(w) == WIN
        assert w[0] == wid * HOP  # window i starts at stream sample i*hop
    assert [wid for wid, _ in wins] == list(range(8))


def test_chunk_larger_than_window() -> None:
    c = RingChunker(3.0, 1.0, SR)
    wins = list(c.push(_ramp(5 * SR)))  # one 5 s chunk > 3 s window
    assert len(wins) == 3  # windows starting at 0s, 1s, 2s
    assert wins[2][1][0] == 2 * HOP


def test_exact_boundary_chunk() -> None:
    c = RingChunker(3.0, 1.0, SR)
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
    expected = 1 + (n - WIN) // HOP  # 8
    for chunk_size in (1000, HOP, WIN, n):
        c = RingChunker(3.0, 1.0, SR)
        wins = _feed(c, _ramp(n), chunk_size)
        assert len(wins) == expected, (chunk_size, len(wins))


def test_reset() -> None:
    c = RingChunker(3.0, 1.0, SR)
    list(c.push(_ramp(4 * SR)))
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
