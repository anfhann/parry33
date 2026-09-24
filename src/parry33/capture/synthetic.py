"""Synthetic frame source.

Lets the whole pipeline (ring buffer, trigger, injector, benches) run with no
GPU, no game and no bettercam install -- useful for CI and for developing the
Phase 3 trigger logic on a laptop.

Emits a bright bar sweeping left-to-right at a fixed rate so frame-difference
and ROI-brightness triggers have something real to fire on.
"""

from __future__ import annotations

import numpy as np

from ..clock import PreciseTimer, now_ns
from .base import CaptureBackend, Frame


class SyntheticCapture(CaptureBackend):
    name = "synthetic"

    def __init__(self, region=None, fps: int = 144, channels: int = 4,
                 sweep_period_s: float = 1.5) -> None:
        super().__init__(region or (0, 0, 960, 540))
        self.fps = fps
        self.channels = channels
        self.sweep_period_s = sweep_period_s
        self._period_ns = int(1e9 / max(fps, 1))
        self._buf = None
        self._timer = None
        self._next_ns = 0

    def start(self) -> None:
        w, h = self.size
        self._buf = np.zeros((h, w, self.channels), dtype=np.uint8)
        self._timer = PreciseTimer(spin_margin_us=200)
        self._next_ns = now_ns()

    def grab(self, timeout_ms: float = 100.0):
        self._timer.sleep_until_ns(self._next_ns)
        self._next_ns += self._period_ns
        t = now_ns()
        w, h = self.size
        phase = (t / 1e9) % self.sweep_period_s / self.sweep_period_s
        x = int(phase * w)
        bar = max(8, w // 24)
        self._buf[:] = 16
        self._buf[:, x:min(w, x + bar)] = 240
        self.seq += 1
        return Frame(self._buf, t, self.seq)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.close()
            self._timer = None
        self._buf = None
