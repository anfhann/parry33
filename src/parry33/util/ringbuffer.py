"""Fixed-capacity frame ring buffer.

Single-producer / single-consumer. The producer writes into preallocated slots
and bumps a monotonic sequence counter; the consumer reads backwards from it.
No allocation, no locks -- the GIL makes the int store atomic, and the consumer
only reads slots the producer has already passed.

Sized for Phase 2/3: hold the last N frames so that when a hit lands we can dump
the preceding wind-up frames to disk for labelling.
"""

from __future__ import annotations

import numpy as np


class FrameRing:
    __slots__ = ("_frames", "_ts", "_seqs", "_cap", "_seq")

    def __init__(self, capacity, shape, dtype=np.uint8) -> None:
        self._cap = int(capacity)
        self._frames = np.zeros((self._cap, *shape), dtype=dtype)
        self._ts = np.zeros(self._cap, dtype=np.int64)
        self._seqs = np.zeros(self._cap, dtype=np.int64)
        self._seq = 0

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def count(self) -> int:
        return min(self._seq, self._cap)

    @property
    def nbytes(self) -> int:
        return self._frames.nbytes

    def slot(self) -> np.ndarray:
        """Writable view of the next slot -- copy a frame straight into this."""
        return self._frames[self._seq % self._cap]

    def commit(self, t_ns: int, seq=None) -> int:
        """Publish the slot returned by the last slot() call."""
        i = self._seq % self._cap
        self._ts[i] = t_ns
        self._seqs[i] = self._seq if seq is None else seq
        self._seq += 1
        return self._seq

    def write(self, frame: np.ndarray, t_ns: int, seq=None) -> int:
        np.copyto(self.slot(), frame)
        return self.commit(t_ns, seq)

    def latest(self, n: int = 1):
        """Last n frames oldest-first, plus their capture timestamps.

        Returns a fancy-indexed copy -- safe to hand to a worker thread while the
        producer keeps writing.
        """
        n = min(n, self.count)
        if n == 0:
            return self._frames[:0], self._ts[:0]
        end = self._seq
        idx = np.arange(end - n, end) % self._cap
        return self._frames[idx], self._ts[idx]

    def stack_channel(self, n: int, channel: int = 1) -> np.ndarray:
        """Temporal stack of one channel -- the input a small 2.5D/3D net wants."""
        frames, _ = self.latest(n)
        return frames[..., channel] if frames.ndim == 4 else frames
