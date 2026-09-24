"""Win32 SendInput keyboard/mouse injection.

Scancodes, not virtual keys. Games built on DirectInput or Raw Input frequently
read the scancode and ignore wVk entirely, so a VK-only injection silently does
nothing while looking perfectly fine in Notepad. KEYEVENTF_SCANCODE it is.

Every INPUT struct is built once at bind() time and reused, so the hot path is a
single SendInput syscall with zero Python allocation. Measured cost of the call
itself is ~5-30 us; that is not the latency that matters, the game's input poll
is. See bench/loop.py.

dwExtraInfo carries a tag so our own synthetic events are identifiable later
(needed in Phase 3 if we ever hook input to avoid reacting to ourselves).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .base import InputBackend

_u32 = ctypes.WinDLL("user32", use_last_error=True)

INJECT_TAG = 0x50525233  # 'PRR3'

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 0x0001, 0x0002
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0

_MOUSE_FLAGS = {
    "left":   (0x0002, 0x0004, 0),
    "right":  (0x0008, 0x0010, 0),
    "middle": (0x0020, 0x0040, 0),
    "x1":     (0x0080, 0x0100, 1),
    "x2":     (0x0080, 0x0100, 2),
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_u32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_u32.SendInput.restype = wintypes.UINT
_u32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
_u32.MapVirtualKeyW.restype = wintypes.UINT

_SIZEOF_INPUT = ctypes.sizeof(INPUT)

# Virtual-key table. Scancodes are derived from these via MapVirtualKeyW so the
# active keyboard layout is respected (the game reads the same layout).
VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "shift": 0x10,
    "ctrl": 0x11, "alt": 0x12, "pause": 0x13, "capslock": 0x14, "esc": 0x1B,
    "escape": 0x1B, "space": 0x20, "pgup": 0x21, "pgdn": 0x22, "end": 0x23,
    "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "delete": 0x2E, "lshift": 0xA0, "rshift": 0xA1,
    "lctrl": 0xA2, "rctrl": 0xA3, "lalt": 0xA4, "ralt": 0xA5,
}
VK.update({chr(c): c for c in range(0x41, 0x5B)})            # a-z (upper VK codes)
VK.update({chr(c).lower(): c for c in range(0x41, 0x5B)})
VK.update({str(d): 0x30 + d for d in range(10)})             # 0-9
VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})         # f1-f12
VK.update({f"num{d}": 0x60 + d for d in range(10)})          # numpad

# VKs that require KEYEVENTF_EXTENDEDKEY (E0-prefixed scancodes).
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E,
            0x90, 0x2C, 0x6F, 0xA3, 0xA5}


class _KeyToken:
    __slots__ = ("down", "up", "label")

    def __init__(self, down, up, label):
        self.down, self.up, self.label = down, up, label


def _key_inputs(vk: int):
    scan = _u32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0)
    down = (INPUT * 1)()
    down[0].type = INPUT_KEYBOARD
    down[0].ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0,
                            dwExtraInfo=INJECT_TAG)
    up = (INPUT * 1)()
    up[0].type = INPUT_KEYBOARD
    up[0].ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags | KEYEVENTF_KEYUP,
                          time=0, dwExtraInfo=INJECT_TAG)
    return down, up, scan


def _mouse_inputs(button: str):
    downf, upf, xdata = _MOUSE_FLAGS[button]
    down = (INPUT * 1)()
    down[0].type = INPUT_MOUSE
    down[0].mi = MOUSEINPUT(0, 0, xdata, downf, 0, INJECT_TAG)
    up = (INPUT * 1)()
    up[0].type = INPUT_MOUSE
    up[0].mi = MOUSEINPUT(0, 0, xdata, upf, 0, INJECT_TAG)
    return down, up


class SendInputBackend(InputBackend):
    """Bindings look like 'key:space', 'key:lshift', 'mouse:right'."""

    name = "sendinput"

    def bind(self, spec: str) -> _KeyToken:
        kind, _, what = spec.partition(":")
        kind, what = kind.strip().lower(), what.strip().lower()
        if kind == "key":
            if what not in VK:
                raise ValueError(f"unknown key {what!r}; known: {sorted(VK)[:12]}...")
            vk = VK[what]
            down, up, scan = _key_inputs(vk)
            if scan == 0:
                raise ValueError(f"key {what!r} has no scancode on this layout")
            return _KeyToken(down, up, f"key:{what}(vk=0x{vk:02X},sc=0x{scan:02X})")
        if kind == "mouse":
            if what not in _MOUSE_FLAGS:
                raise ValueError(f"unknown mouse button {what!r}")
            down, up = _mouse_inputs(what)
            return _KeyToken(down, up, f"mouse:{what}")
        raise ValueError(f"binding must be key:<name> or mouse:<button>, got {spec!r}")

    def _press(self, token: _KeyToken) -> None:
        _u32.SendInput(1, token.down, _SIZEOF_INPUT)

    def _release(self, token: _KeyToken) -> None:
        _u32.SendInput(1, token.up, _SIZEOF_INPUT)
