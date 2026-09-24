"""Global keystroke timestamping for ground-truth labelling.

Why a hook rather than polling: the recorder loop turns over once per video
frame (~16.7 ms at 60 fps), so polling GetAsyncKeyState there would quantise
every keystroke to +/-16.7 ms. Against a 150 ms parry window that is 11% of the
thing we are trying to measure, thrown away for no reason. A WH_KEYBOARD_LL hook
fires on the keystroke itself and we stamp the clock inside the callback.

Scope, deliberately: the hook sees every key on the system, but the callback
filters against an explicit watchlist and stores nothing else. This records game
inputs, not keystrokes in general. Widening the watchlist is a decision, not an
accident.

Injected input is ignored (LLKHF_INJECTED). In Phase 3 the bot will be pressing
the parry key itself, and a labeller that cannot distinguish the human from the
bot would silently poison its own training data.

The hook needs a message loop on the thread that installs it, so this owns a
dedicated thread. The callback is a few microseconds -- append a tuple -- so it
holds the GIL for a negligible slice and does not add input latency.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

from ..clock import now_ns

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105
LLKHF_INJECTED = 0x10

LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

_u32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE,
                                   wintypes.DWORD)
_u32.SetWindowsHookExW.restype = ctypes.c_void_p
_u32.CallNextHookEx.argtypes = (ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM,
                                wintypes.LPARAM)
_u32.CallNextHookEx.restype = LRESULT
_u32.UnhookWindowsHookEx.argtypes = (ctypes.c_void_p,)
# Same trap as GetCurrentProcess in util/prio.py: this returns a 64-bit HMODULE,
# and without a declared restype ctypes truncates it to a C int. The truncated
# handle makes SetWindowsHookEx fail with no useful error.
_k32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
_k32.GetModuleHandleW.restype = wintypes.HMODULE
_k32.GetCurrentThreadId.restype = wintypes.DWORD
_u32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM)

# Names -> virtual-key codes for the keys worth watching.
VK = {
    "q": 0x51, "e": 0x45, "r": 0x52, "f": 0x46, "space": 0x20,
    "shift": 0x10, "lshift": 0xA0, "ctrl": 0x11,
    "plus": 0xBB,        # VK_OEM_PLUS, the '=/+' key
    "numplus": 0x6B,     # VK_ADD, numpad '+'
    "minus": 0xBD, "numminus": 0x6D,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38,
    # Numpad digits, for labelling attacks live while the game has focus. The
    # top-row digits usually do something in combat (switch character, use an
    # item); the numpad is normally unbound, so it annotates without acting.
    "num1": 0x61, "num2": 0x62, "num3": 0x63,
    "num4": 0x64, "num5": 0x65, "num6": 0x66,
}
_VK_NAME = {v: k for k, v in VK.items()}


class KeyWatcher:
    """Timestamps presses of a watchlist of keys, system-wide.

    events is a list of (t_ns, name, is_down). Read it from the owning thread;
    the hook thread only appends.
    """

    def __init__(self, watch=("q", "e", "plus", "numplus")) -> None:
        self.watch = {VK[k]: k for k in watch}
        self.events: list[tuple[int, str, bool]] = []
        self._hook = None
        self._thread = None
        self._tid = 0
        self._ready = threading.Event()
        self._proc = None      # must outlive the hook
        self._err = 0
        self.ignored_injected = 0

    def _callback(self, nCode, wParam, lParam):
        if nCode == 0:
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            name = self.watch.get(kb.vkCode)
            if name is not None:
                if kb.flags & LLKHF_INJECTED:
                    # Our own synthetic presses, not the human's.
                    self.ignored_injected += 1
                else:
                    down = wParam in (WM_KEYDOWN, WM_SYSKEYDOWN)
                    self.events.append((now_ns(), name, down))
        return _u32.CallNextHookEx(None, nCode, wParam, lParam)

    def _run(self):
        self._tid = _k32.GetCurrentThreadId()
        self._proc = HOOKPROC(self._callback)
        # hMod may be NULL for a low-level hook whose procedure lives in this
        # process; fall back to the real module handle if the driver disagrees.
        self._hook = _u32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hook:
            self._err = ctypes.get_last_error()
            self._hook = _u32.SetWindowsHookExW(
                WH_KEYBOARD_LL, self._proc, _k32.GetModuleHandleW(None), 0)
        if not self._hook:
            self._err = ctypes.get_last_error()
        self._ready.set()
        if not self._hook:
            return
        msg = wintypes.MSG()
        while _u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            _u32.TranslateMessage(ctypes.byref(msg))
            _u32.DispatchMessageW(ctypes.byref(msg))

    def start(self) -> "KeyWatcher":
        self._thread = threading.Thread(target=self._run, name="keywatch",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)
        if not self._hook:
            raise RuntimeError(
                f"SetWindowsHookEx failed (error {self._err}). If the game runs "
                f"elevated, this process must be elevated too or the hook sees "
                f"nothing.")
        return self

    def stop(self) -> None:
        if self._hook:
            _u32.UnhookWindowsHookEx(self._hook)
            self._hook = None
        if self._tid:
            _u32.PostThreadMessageW(self._tid, 0x0012, 0, 0)   # WM_QUIT
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def drain(self):
        """Take everything recorded so far and clear the buffer."""
        out, self.events = self.events, []
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
