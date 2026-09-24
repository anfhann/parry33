"""Capture backend interface.

Every backend hands back a Frame whose `t_ns` is stamped as close to the moment
the pixels became available to us as we can manage. That timestamp is the anchor
for the whole latency budget -- everything downstream is measured against it.

Note what t_ns does NOT include: the game's own render+present latency and the
DWM/DXGI delivery hop. Those are only observable end-to-end, which is what
`bench loop` exists to measure.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    data: np.ndarray   # (H, W, C) uint8. May alias a reused buffer -- copy to retain.
    t_ns: int          # perf_counter_ns immediately after the grab returned
    seq: int           # monotonic frame counter from this backend


class CaptureBackend(abc.ABC):
    name = "base"

    def __init__(self, region=None) -> None:
        self.region = tuple(region) if region else None
        self.seq = 0

    @property
    def size(self):
        """(width, height) of the captured area, or None until started."""
        if self.region is None:
            return None
        left, top, right, bottom = self.region
        return (right - left, bottom - top)

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def grab(self, timeout_ms: float = 100.0):
        """Block until a NEW frame is available. Returns Frame or None on timeout.

        'New' is load-bearing: Desktop Duplication only produces a frame when the
        desktop actually changes, so a static screen yields nothing at all rather
        than duplicates. A None return is normal, not an error.
        """

    @abc.abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def build(cfg) -> CaptureBackend:
    """Construct the backend named by a CaptureConfig."""
    if cfg.backend == "dxgi":
        from .dxgi import DXGICapture
        return DXGICapture(
            monitor=cfg.monitor, gpu=cfg.gpu, region=cfg.region, color=cfg.color,
            mode=cfg.mode, target_fps=cfg.target_fps, poll_sleep_us=cfg.poll_sleep_us)
    if cfg.backend == "synthetic":
        from .synthetic import SyntheticCapture
        return SyntheticCapture(region=cfg.region, fps=cfg.target_fps or 144)
    raise ValueError(f"unknown capture backend: {cfg.backend!r}")
