"""Process/thread scheduling knobs and DPI awareness.

DPI matters for capture: without per-monitor-v2 awareness Windows lies about
screen coordinates on a scaled display and your ROI lands in the wrong place.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
from ctypes import wintypes

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_u32 = ctypes.WinDLL("user32", use_last_error=True)

# argtypes are mandatory here, not cosmetic: GetCurrentProcess/Thread return the
# pseudo-handle (HANDLE)-1. Without a declared restype ctypes hands back a C int,
# which is passed to the next call as a 32-bit -1 in a 64-bit handle slot. The
# call then fails with ERROR_INVALID_HANDLE and the priority is never applied --
# silently, because nobody checks the return value.
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.GetCurrentProcess.argtypes = ()
_k32.GetCurrentThread.restype = wintypes.HANDLE
_k32.GetCurrentThread.argtypes = ()
_k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
_k32.SetPriorityClass.restype = wintypes.BOOL
_k32.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
_k32.SetThreadPriority.restype = wintypes.BOOL

HIGH_PRIORITY_CLASS = 0x00000080
REALTIME_PRIORITY_CLASS = 0x00000100
THREAD_PRIORITY_TIME_CRITICAL = 15
THREAD_PRIORITY_HIGHEST = 2
_DPI_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


_AWARENESS = {0: "unaware", 1: "system-aware", 2: "per-monitor-aware"}


def dpi_awareness() -> str:
    """Read the process's ACTUAL DPI awareness.

    Never infer this from SetProcessDpiAwarenessContext's return value: it
    returns FALSE when awareness is already established (by an application
    manifest, for instance), which is indistinguishable from a real failure.
    Python's own manifest sets it on some builds, so the setter routinely
    "fails" on a process that is already correctly configured.
    """
    try:
        _u32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        _u32.GetAwarenessFromDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        _u32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        ctx = _u32.GetThreadDpiAwarenessContext()
        return _AWARENESS.get(_u32.GetAwarenessFromDpiAwarenessContext(ctx), "unknown")
    except (AttributeError, OSError):
        return "unknown"


def set_dpi_aware() -> bool:
    """Request per-monitor-v2 awareness. Returns True if we end up aware.

    Reports the resulting state, not whether this particular call did the work.
    """
    try:
        _u32.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        _u32.SetProcessDpiAwarenessContext(_DPI_PER_MONITOR_AWARE_V2)
    except (AttributeError, OSError):
        with contextlib.suppress(Exception):
            _u32.SetProcessDPIAware()
    return dpi_awareness() == "per-monitor-aware"


def set_process_priority(realtime: bool = False) -> bool:
    cls = REALTIME_PRIORITY_CLASS if realtime else HIGH_PRIORITY_CLASS
    return bool(_k32.SetPriorityClass(_k32.GetCurrentProcess(), cls))


def set_thread_priority(time_critical: bool = True) -> bool:
    pri = THREAD_PRIORITY_TIME_CRITICAL if time_critical else THREAD_PRIORITY_HIGHEST
    return bool(_k32.SetThreadPriority(_k32.GetCurrentThread(), pri))


_k32.SetThreadAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
_k32.SetThreadAffinityMask.restype = ctypes.c_size_t


def pin_thread_to_cpu(cpu: int) -> bool:
    """Keep the capture thread off a core the game scheduler is fighting over."""
    return bool(_k32.SetThreadAffinityMask(_k32.GetCurrentThread(), 1 << cpu))


@contextlib.contextmanager
def frozen_gc():
    """Disable GC and promote existing objects out of the collectable set.

    A gen-2 collection mid-loop is a multi-millisecond stall — enough to miss a
    parry window outright. Preallocate everything, then run under this.
    """
    gc.collect()
    gc.freeze()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.unfreeze()


def apply(cfg) -> dict:
    """Apply the [timing] block of a Config. Returns what actually took effect."""
    return {
        "dpi_aware": set_dpi_aware(),
        "high_priority": set_process_priority() if cfg.high_priority else False,
        "time_critical": set_thread_priority() if cfg.time_critical else False,
    }
