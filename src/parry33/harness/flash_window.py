"""Visual responder window for closed-loop latency measurement.

A borderless always-on-top window that flips black -> white the instant it sees
an input, and back to black on release. Point the capture ROI at it, inject a
key or pad button, and the time from "SendInput returned" to "the capture
pipeline saw white pixels" is the real round-trip latency of everything we
control: injection -> OS input stack -> app wake -> GDI paint -> DWM compose ->
DXGI Desktop Duplication -> our grab -> our threshold check.

That number is the floor. In the game the same chain exists, but with the game's
own input poll and render pipeline in place of a GDI paint, so expect the real
figure to be higher by roughly one game frame plus its render latency.

Three modes:
  raw     -- Raw Input with RIDEV_INPUTSINK (default). Same delivery path a UE5
             game uses, but works without keyboard focus, so it is reliable when
             the harness is driven from a script.
  key     -- event-driven, GetMessage + WM_KEYDOWN. Most faithful to a focused
             app, but the window must actually hold focus or it sees nothing.
  pad     -- PeekMessage loop polling XInputGetState. Measures the ViGEm path,
             including the XInput polling interval itself.
  animate -- toggles at a fixed rate with no input at all. Desktop Duplication
             emits nothing on a still screen, so this gives `bench capture`
             a paced source of desktop change to sample.

Run standalone:
  python -m parry33.harness.flash_window --mode key --x 200 --y 200 --size 400
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_g32 = ctypes.WinDLL("gdi32", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_DESTROY, WM_PAINT, WM_CLOSE, WM_QUIT = 0x0002, 0x000F, 0x0010, 0x0012
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0100, 0x0101, 0x0104, 0x0105
WS_POPUP, WS_VISIBLE = 0x80000000, 0x10000000
WS_EX_TOPMOST = 0x00000008
SW_SHOW, PM_REMOVE = 5, 0x0001
VK_ESCAPE = 0x1B
ERROR_CLASS_ALREADY_EXISTS = 1410

LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [("hdc", wintypes.HDC), ("fErase", wintypes.BOOL),
                ("rcPaint", wintypes.RECT), ("fRestore", wintypes.BOOL),
                ("fIncUpdate", wintypes.BOOL), ("rgbReserved", ctypes.c_byte * 32)]


WM_INPUT = 0x00FF
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1
RI_KEY_BREAK = 0x01
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_KEYBOARD = 0x06


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [("MakeCode", wintypes.USHORT), ("Flags", wintypes.USHORT),
                ("Reserved", wintypes.USHORT), ("VKey", wintypes.USHORT),
                ("Message", wintypes.UINT), ("ExtraInformation", wintypes.ULONG)]


class RAWINPUT_KB(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", wintypes.WORD), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", XINPUT_GAMEPAD)]


XINPUT_BUTTON = {
    "dup": 0x0001, "ddown": 0x0002, "dleft": 0x0004, "dright": 0x0008,
    "start": 0x0010, "back": 0x0020, "ls": 0x0040, "rs": 0x0080,
    "lb": 0x0100, "rb": 0x0200, "a": 0x1000, "b": 0x2000, "x": 0x4000, "y": 0x8000,
}

for _fn, _at, _rt in (
    ("DefWindowProcW",
     (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM), LRESULT),
    ("BeginPaint", (wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)), wintypes.HDC),
    ("EndPaint", (wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)), wintypes.BOOL),
    ("FillRect",
     (wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH), ctypes.c_int),
    ("InvalidateRect", (wintypes.HWND, ctypes.c_void_p, wintypes.BOOL), wintypes.BOOL),
    ("UpdateWindow", (wintypes.HWND,), wintypes.BOOL),
    ("GetClientRect", (wintypes.HWND, ctypes.POINTER(wintypes.RECT)), wintypes.BOOL),
):
    getattr(_u32, _fn).argtypes = _at
    getattr(_u32, _fn).restype = _rt

_u32.CreateWindowExW.argtypes = (
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID)
_u32.CreateWindowExW.restype = wintypes.HWND
_u32.RegisterRawInputDevices.argtypes = (
    ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT)
_u32.RegisterRawInputDevices.restype = wintypes.BOOL
_u32.GetRawInputData.argtypes = (
    ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
    ctypes.POINTER(wintypes.UINT), wintypes.UINT)
_u32.GetRawInputData.restype = wintypes.UINT
_g32.CreateSolidBrush.argtypes = (wintypes.DWORD,)
_g32.CreateSolidBrush.restype = wintypes.HBRUSH


def _load_xinput():
    for dll in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            lib = ctypes.WinDLL(dll)
            lib.XInputGetState.argtypes = (wintypes.DWORD, ctypes.POINTER(XINPUT_STATE))
            lib.XInputGetState.restype = wintypes.DWORD
            return lib
        except OSError:
            continue
    return None


class FlashWindow:
    """Owns the window. run() blocks until Esc or close."""

    def __init__(self, x=200, y=200, width=400, height=400, vk=0x20,
                 mode="key", pad_button="rb", pad_index=0, animate_hz=144.0):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.vk, self.mode = vk, mode
        self.pad_mask = XINPUT_BUTTON.get(pad_button, 0x0200)
        self.pad_index = pad_index
        self.animate_hz = animate_hz
        # Raw Input reports scancodes; derive ours from the VK so `raw` mode and
        # the scancode-based injector are talking about the same physical key.
        self.scan = _u32.MapVirtualKeyW(vk, 0)
        self._raw = RAWINPUT_KB()
        self.state = 0
        self.hwnd = None
        self._running = True
        self._black = _g32.CreateSolidBrush(0x000000)
        self._white = _g32.CreateSolidBrush(0xFFFFFF)
        self._proc = WNDPROC(self._wndproc)   # must outlive the window
        self._ps = PAINTSTRUCT()
        self._rect = wintypes.RECT()

    @property
    def region(self):
        """Absolute desktop ROI covering this window -- feed straight to capture."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_PAINT:
            hdc = _u32.BeginPaint(hwnd, ctypes.byref(self._ps))
            _u32.GetClientRect(hwnd, ctypes.byref(self._rect))
            _u32.FillRect(hdc, ctypes.byref(self._rect),
                          self._white if self.state else self._black)
            _u32.EndPaint(hwnd, ctypes.byref(self._ps))
            return 0
        if msg == WM_INPUT and self.mode == "raw":
            size = wintypes.UINT(ctypes.sizeof(RAWINPUT_KB))
            if _u32.GetRawInputData(ctypes.c_void_p(lparam), RID_INPUT,
                                    ctypes.byref(self._raw), ctypes.byref(size),
                                    ctypes.sizeof(RAWINPUTHEADER)) != 0xFFFFFFFF:
                if self._raw.header.dwType == RIM_TYPEKEYBOARD:
                    kb = self._raw.keyboard
                    if kb.MakeCode == self.scan or kb.VKey == self.vk:
                        down = 0 if (kb.Flags & RI_KEY_BREAK) else 1
                        if down != self.state:
                            self._set(down)
            return _u32.DefWindowProcW(hwnd, msg, wparam, lparam)
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN) and self.mode == "key":
            if wparam == VK_ESCAPE:
                self._running = False
                _u32.PostQuitMessage(0)
            elif wparam == self.vk and not self.state:
                self._set(1)
            return 0
        if msg in (WM_KEYUP, WM_SYSKEYUP) and self.mode == "key":
            if wparam == self.vk and self.state:
                self._set(0)
            return 0
        if msg in (WM_DESTROY, WM_CLOSE):
            self._running = False
            _u32.PostQuitMessage(0)
            return 0
        return _u32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _set(self, state: int) -> None:
        self.state = state
        _u32.InvalidateRect(self.hwnd, None, False)
        _u32.UpdateWindow(self.hwnd)   # synchronous WM_PAINT, no queue round trip

    def create(self):
        cls = WNDCLASSW()
        cls.style = 0x0003                      # CS_HREDRAW | CS_VREDRAW
        cls.lpfnWndProc = self._proc
        cls.hInstance = _k32.GetModuleHandleW(None)
        cls.hbrBackground = self._black
        cls.lpszClassName = "Parry33FlashWindow"
        if not _u32.RegisterClassW(ctypes.byref(cls)):
            err = ctypes.get_last_error()
            if err != ERROR_CLASS_ALREADY_EXISTS:
                raise ctypes.WinError(err)
        self.hwnd = _u32.CreateWindowExW(
            WS_EX_TOPMOST, "Parry33FlashWindow", "parry33 flash",
            WS_POPUP | WS_VISIBLE, self.x, self.y, self.width, self.height,
            None, None, cls.hInstance, None)
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        if self.mode == "raw":
            rid = RAWINPUTDEVICE(HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_KEYBOARD,
                                 RIDEV_INPUTSINK, self.hwnd)
            if not _u32.RegisterRawInputDevices(ctypes.byref(rid), 1,
                                                ctypes.sizeof(RAWINPUTDEVICE)):
                raise ctypes.WinError(ctypes.get_last_error())
        _u32.ShowWindow(self.hwnd, SW_SHOW)
        _u32.SetForegroundWindow(self.hwnd)     # required for WM_KEYDOWN in key mode
        _u32.UpdateWindow(self.hwnd)
        return self

    def pump(self) -> None:
        """Drain pending messages once. For driving the window from another loop."""
        msg = wintypes.MSG()
        while _u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if msg.message == WM_QUIT:
                self._running = False
            _u32.TranslateMessage(ctypes.byref(msg))
            _u32.DispatchMessageW(ctypes.byref(msg))

    def run(self) -> None:
        if self.mode in ("key", "raw"):
            msg = wintypes.MSG()
            while self._running and _u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                _u32.TranslateMessage(ctypes.byref(msg))
                _u32.DispatchMessageW(ctypes.byref(msg))
            return
        if self.mode == "animate":
            self._run_animate()
            return
        xi = _load_xinput()
        if xi is None:
            raise RuntimeError("no XInput DLL found; pad mode unavailable")
        st = XINPUT_STATE()
        while self._running:
            self.pump()
            if xi.XInputGetState(self.pad_index, ctypes.byref(st)) == 0:
                pressed = 1 if (st.Gamepad.wButtons & self.pad_mask) else 0
                if pressed != self.state:
                    self._set(pressed)

    def _run_animate(self, hz: float = 0.0) -> None:
        """Toggle at a fixed rate so Desktop Duplication has something to emit.

        DDA only produces a frame when the desktop actually changes -- on a still
        screen `bench capture` measures nothing at all. This gives it a known,
        paced source of change to sample.
        """
        from ..clock import PreciseTimer, now_ns
        hz = hz or self.animate_hz
        period = int(1e9 / max(hz, 1.0))
        timer = PreciseTimer(spin_margin_us=200)
        nxt = now_ns()
        while self._running:
            self.pump()
            nxt += period
            timer.sleep_until_ns(nxt)
            self._set(0 if self.state else 1)
        timer.close()

    def destroy(self) -> None:
        if self.hwnd:
            _u32.DestroyWindow(self.hwnd)
            self.hwnd = None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="parry33 flash responder window")
    p.add_argument("--x", type=int, default=200)
    p.add_argument("--y", type=int, default=200)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--mode", choices=("key", "raw", "pad", "animate"),
                   default="raw")
    p.add_argument("--hz", type=float, default=144.0,
                   help="toggle rate in animate mode")
    p.add_argument("--vk", type=lambda s: int(s, 0), default=0x20,
                   help="VK code to react to in key mode (default 0x20 = space)")
    p.add_argument("--pad-button", default="rb")
    args = p.parse_args(argv)

    from ..util.prio import set_dpi_aware
    set_dpi_aware()
    w = FlashWindow(args.x, args.y, args.size, args.size, vk=args.vk,
                    mode=args.mode, pad_button=args.pad_button,
                    animate_hz=args.hz).create()
    print(f"flash window at {w.region} mode={args.mode} -- Esc to quit", flush=True)
    w.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
