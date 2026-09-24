"""High-resolution timing.

`time.perf_counter_ns()` maps to QueryPerformanceCounter (~100 ns) and is fine.
`time.sleep()` is not: the default Windows timer granularity is 15.6 ms, and even
under `timeBeginPeriod(1)` you only get ~1-2 ms of accuracy with a long tail. At
144 Hz one display frame is 6.94 ms, so a 15 ms sleep overshoot costs two frames.

We use CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (Win10 1803+, ~0.5 ms granularity,
no global timer-resolution side effects) for the bulk of a wait and busy-spin the
last few hundred microseconds.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

now_ns = time.perf_counter_ns

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)

_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003
_INFINITE = 0xFFFFFFFF

_k32.CreateWaitableTimerExW.restype = wintypes.HANDLE
_k32.CreateWaitableTimerExW.argtypes = (
    wintypes.LPVOID, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD)
_k32.SetWaitableTimer.restype = wintypes.BOOL
_k32.SetWaitableTimer.argtypes = (
    wintypes.HANDLE, ctypes.POINTER(ctypes.c_int64), wintypes.LONG,
    wintypes.LPVOID, wintypes.LPVOID, wintypes.BOOL)
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
_k32.CloseHandle.argtypes = (wintypes.HANDLE,)


def spin_until_ns(deadline_ns: int) -> int:
    """Busy-wait until `deadline_ns`. Returns the actual wake time."""
    t = now_ns()
    while t < deadline_ns:
        t = now_ns()
    return t


class PreciseTimer:
    """Hybrid waitable-timer + spin sleeper. Not thread-safe; one per thread."""

    __slots__ = ("_h", "_spin_ns", "supported")

    def __init__(self, spin_margin_us: float = 400.0) -> None:
        self._spin_ns = int(spin_margin_us * 1_000)
        h = _k32.CreateWaitableTimerExW(
            None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, _TIMER_ALL_ACCESS)
        if not h:  # pre-1803 fallback: plain timer + timeBeginPeriod(1)
            h = _k32.CreateWaitableTimerExW(None, None, 0, _TIMER_ALL_ACCESS)
            self.supported = False
            _begin_period(1)
        else:
            self.supported = True
        self._h = h

    def sleep_ns(self, duration_ns: int) -> int:
        """Sleep for `duration_ns`. Returns the actual wake time (perf ns)."""
        deadline = now_ns() + duration_ns
        return self.sleep_until_ns(deadline)

    def sleep_until_ns(self, deadline_ns: int) -> int:
        coarse = deadline_ns - self._spin_ns - now_ns()
        if coarse > 0 and self._h:
            # negative due-time == relative, in 100 ns units
            due = ctypes.c_int64(-(coarse // 100))
            if _k32.SetWaitableTimer(self._h, ctypes.byref(due), 0, None, None, False):
                _k32.WaitForSingleObject(self._h, _INFINITE)
        return spin_until_ns(deadline_ns)

    def sleep(self, seconds: float) -> int:
        return self.sleep_ns(int(seconds * 1e9))

    def close(self) -> None:
        if self._h:
            _k32.CloseHandle(self._h)
            self._h = None

    def __enter__(self) -> "PreciseTimer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_winmm = None


def _begin_period(ms: int) -> None:
    global _winmm
    try:
        _winmm = _winmm or ctypes.WinDLL("winmm")
        _winmm.timeBeginPeriod(ms)
    except OSError:
        pass


_default_timer: PreciseTimer | None = None


def precise_sleep(seconds: float) -> int:
    """Process-wide convenience sleeper. Prefer an owned PreciseTimer in hot loops."""
    global _default_timer
    if _default_timer is None:
        _default_timer = PreciseTimer()
    return _default_timer.sleep(seconds)


def ms(ns: int | float) -> float:
    return ns / 1e6
