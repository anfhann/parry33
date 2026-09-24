"""DXGI Desktop Duplication capture.

Backed by bettercam (maintained fork) or dxcam. Both are thin Cython/ctypes
wrappers over IDXGIOutputDuplication::AcquireNextFrame -> ID3D11DeviceContext::
CopyResource -> Map -> memcpy into a numpy buffer.

Two modes, and the difference is real:

  poll   -- we call grab() ourselves in a spin. AcquireNextFrame returns the
            instant the desktop updates, so this is the lowest-latency path.
            Costs one saturated core.
  thread -- the library runs its own capture thread and signals an event. One
            extra hop and an event wake (~0.1-1 ms of jitter), much lower CPU.

Measured cost, contrary to the obvious guess: ROI area barely matters. A sweep
from 320x180 to 1280x720 held ~144 fps with zero drops and 6+ ms/frame of
downstream headroom at every size. The region is passed down into the copy
rather than cropped in numpy afterwards, but the per-frame cost is dominated by
fixed overhead, not bytes -- do not size the ROI to save copy time. Size it for
whatever the model needs to see, and spend the rest on inference.

Run `parry33 bench capture --sweep` to reconfirm on new hardware.

Known failure modes:
  * True exclusive fullscreen can bypass the desktop compositor. Run the game
    borderless/windowed-fullscreen.
  * The adapter index must match the GPU the game renders on, or you duplicate
    an output that never updates.
  * Protected/DRM content returns black frames by design.
"""

from __future__ import annotations

import time

import numpy as np

from ..clock import PreciseTimer, now_ns
from .base import CaptureBackend, Frame

_IMPL = None
_dx = None
try:
    import bettercam as _dx
    _IMPL = "bettercam"
except ImportError:
    try:
        import dxcam as _dx
        _IMPL = "dxcam"
    except ImportError:
        pass


class DXGICapture(CaptureBackend):
    name = "dxgi"

    def __init__(self, monitor: int = 0, gpu: int = 0, region=None, color: str = "BGRA",
                 mode: str = "poll", target_fps: int = 0, poll_sleep_us: float = 0.0) -> None:
        super().__init__(region)
        if _dx is None:
            raise RuntimeError(
                "no DXGI backend installed. pip install bettercam  (or dxcam)")
        if mode not in ("poll", "thread"):
            raise ValueError(f"mode must be poll|thread, got {mode!r}")
        self.impl = _IMPL
        self.monitor = monitor
        self.gpu = gpu
        self.color = color
        self.mode = mode
        self.target_fps = target_fps
        self._poll_sleep_s = poll_sleep_us / 1e6
        self._cam = None
        self._timer = None
        self._started = False

    @staticmethod
    def available() -> bool:
        return _dx is not None

    @staticmethod
    def describe_devices() -> str:
        if _dx is None:
            return "(no DXGI backend installed)"
        try:
            return str(_dx.device_info()) + "\n" + str(_dx.output_info())
        except Exception as e:  # noqa: BLE001 - diagnostics only
            return f"(device enumeration failed: {e})"

    def start(self) -> None:
        if self._started:
            return
        self._cam = _dx.create(
            device_idx=self.gpu, output_idx=self.monitor,
            output_color=self.color, max_buffer_len=2)
        if self._cam is None:
            raise RuntimeError(
                f"DXGI create() failed for gpu={self.gpu} monitor={self.monitor}")
        self.region = self._validate_region(
            self.region, self._cam.width, self._cam.height)
        self._timer = PreciseTimer(spin_margin_us=200)
        if self.mode == "thread":
            self._cam.start(region=self.region, target_fps=self.target_fps,
                            video_mode=bool(self.target_fps))
        else:
            # Prime the duplicator so the first real grab is not paying setup cost.
            for _ in range(3):
                self._cam.grab(region=self.region)
                time.sleep(0.005)
        self._started = True

    @staticmethod
    def _validate_region(region, out_w: int, out_h: int):
        """Clamp an ROI to the output bounds, loudly.

        The common cause of an out-of-bounds region is a config written for a
        different desktop resolution -- e.g. a 4K-centred box still in place
        after dropping to 1080p. Without this the failure surfaces deep inside
        the copy as an opaque error, or silently returns the wrong pixels.
        """
        if region is None:
            return None
        left, top, right, bottom = region
        cl = (max(0, min(left, out_w)), max(0, min(top, out_h)),
              max(0, min(right, out_w)), max(0, min(bottom, out_h)))
        if cl[2] - cl[0] < 8 or cl[3] - cl[1] < 8:
            w, h = min(960, out_w), min(540, out_h)
            cx, cy = out_w // 2, out_h // 2
            fallback = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            print(f"  WARNING: region {region} does not intersect the "
                  f"{out_w}x{out_h} output. Falling back to centred {w}x{h} "
                  f"{fallback}. Fix [capture] region in config/local.toml.")
            return fallback
        if cl != tuple(region):
            print(f"  WARNING: region {region} exceeds the {out_w}x{out_h} "
                  f"output; clamped to {cl}. This usually means the config was "
                  f"written for a different desktop resolution.")
        return cl

    def grab(self, timeout_ms: float = 100.0):
        if not self._started:
            raise RuntimeError("call start() first")
        if self.mode == "thread":
            arr = self._cam.get_latest_frame()
            if arr is None:
                return None
            t = now_ns()
            self.seq += 1
            return Frame(arr, t, self.seq)

        deadline = now_ns() + int(timeout_ms * 1e6)
        sleep_s = self._poll_sleep_s
        cam_grab = self._cam.grab
        region = self.region
        while True:
            arr = cam_grab(region=region)
            if arr is not None:
                t = now_ns()
                self.seq += 1
                return Frame(arr, t, self.seq)
            if now_ns() >= deadline:
                return None
            if sleep_s:
                self._timer.sleep(sleep_s)

    def stop(self) -> None:
        if self._cam is not None:
            try:
                if self.mode == "thread":
                    self._cam.stop()
            finally:
                try:
                    self._cam.release()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
                self._cam = None
        if self._timer is not None:
            self._timer.close()
            self._timer = None
        self._started = False

    def __repr__(self) -> str:
        w, h = self.size or (0, 0)
        return (f"<DXGICapture {self.impl} {self.mode} gpu={self.gpu} out={self.monitor} "
                f"{w}x{h} {self.color}>")


def bytes_per_frame(size, color: str = "BGRA") -> int:
    w, h = size
    return w * h * (4 if len(color) == 4 else 3)
