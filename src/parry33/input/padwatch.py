"""Controller input timestamping via XInput.

Exists because the player is materially better on a controller than on the
keyboard, and better play produces more landed parries -- which is the resource
every analysis in docs/phase2-findings.md ran short of. Recording someone
playing badly on an unfamiliar input device costs more in label quality than the
convenience of a keyboard hook is worth.

XInput has no event API, so this polls. The poll runs on its own thread at ~1 ms
rather than in the recorder loop, because that loop turns over once per video
frame (~16.7 ms) and would quantise every button press to 11% of the parry
window. Polling is cheap: XInputGetState is a memcpy from a driver-maintained
struct, and dwPacketNumber lets us skip the comparison entirely when nothing
changed.

Unlike the keyboard hook, this records every button. A gamepad carries no
incidental typing, so there is nothing to filter out, and we do not yet know
which buttons the player has bound to parry and dodge.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

from ..clock import PreciseTimer, now_ns


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", wintypes.WORD), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", XINPUT_GAMEPAD)]


BUTTONS = {
    0x0001: "dup", 0x0002: "ddown", 0x0004: "dleft", 0x0008: "dright",
    0x0010: "start", 0x0020: "back", 0x0040: "ls", 0x0080: "rs",
    0x0100: "lb", 0x0200: "rb", 0x1000: "a", 0x2000: "b",
    0x4000: "x", 0x8000: "y",
}
TRIGGER_ON, TRIGGER_OFF = 40, 25      # hysteresis, 0-255


def _load():
    for dll in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            lib = ctypes.WinDLL(dll)
            lib.XInputGetState.argtypes = (wintypes.DWORD,
                                           ctypes.POINTER(XINPUT_STATE))
            lib.XInputGetState.restype = wintypes.DWORD
            return lib
        except OSError:
            continue
    return None


class PadWatcher:
    """Timestamps controller button and trigger transitions.

    events is a list of (t_ns, name, is_down), same shape as KeyWatcher so the
    recorder can consume either or both.
    """

    def __init__(self, index: int = 0, poll_ms: float = 1.0) -> None:
        self.index = index
        self.poll_s = poll_ms / 1000.0
        self.events: list[tuple[int, str, bool]] = []
        self.connected = False
        self._xi = None
        self._thread = None
        self._running = False
        self._ready = threading.Event()

    @staticmethod
    def available() -> bool:
        return _load() is not None

    def _run(self):
        st = XINPUT_STATE()
        timer = PreciseTimer(spin_margin_us=200)
        prev_buttons = 0
        prev_lt = prev_rt = False
        prev_packet = None
        self._ready.set()
        while self._running:
            if self._xi.XInputGetState(self.index, ctypes.byref(st)) == 0:
                self.connected = True
                if st.dwPacketNumber != prev_packet:
                    prev_packet = st.dwPacketNumber
                    t = now_ns()
                    g = st.Gamepad
                    changed = g.wButtons ^ prev_buttons
                    for mask, name in BUTTONS.items():
                        if changed & mask:
                            self.events.append((t, name, bool(g.wButtons & mask)))
                    prev_buttons = g.wButtons
                    # Triggers are analogue; hysteresis stops a resting finger
                    # from emitting a stream of phantom presses.
                    lt = g.bLeftTrigger > (TRIGGER_OFF if prev_lt else TRIGGER_ON)
                    rt = g.bRightTrigger > (TRIGGER_OFF if prev_rt else TRIGGER_ON)
                    if lt != prev_lt:
                        self.events.append((t, "lt", lt))
                        prev_lt = lt
                    if rt != prev_rt:
                        self.events.append((t, "rt", rt))
                        prev_rt = rt
            else:
                self.connected = False
            timer.sleep(self.poll_s)
        timer.close()

    def start(self) -> "PadWatcher":
        self._xi = _load()
        if self._xi is None:
            raise RuntimeError("no XInput DLL found; is a controller driver present?")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="padwatch",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def drain(self):
        out, self.events = self.events, []
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
