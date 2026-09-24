"""Virtual Xbox 360 controller via ViGEmBus (vgamepad).

Preferred over SendInput for Expedition 33 and UE5 titles generally: the game
reads XInput, so a ViGEm virtual pad is indistinguishable from real hardware at
the driver level -- no focus requirements, no raw-input quirks, and the game will
switch its on-screen prompts to controller glyphs when it sees the pad.

The tradeoff is an extra hop. press_button() only mutates a local report struct;
update() pushes it over the ViGEm bus to the driver, and XInput clients then poll
it on their own schedule. Budget ~0.5-2 ms for the round trip plus up to one
game poll interval. bench/loop.py measures the real number on your machine.

Requires the ViGEmBus driver installed system-wide:
  https://github.com/nefarius/ViGEmBus/releases
"""

from __future__ import annotations

from .base import InputBackend

_vg = None
try:
    import vgamepad as _vg
except Exception:  # noqa: BLE001 - missing driver raises non-ImportError too
    pass


def _button_map():
    b = _vg.XUSB_BUTTON
    return {
        "a": b.XUSB_GAMEPAD_A, "b": b.XUSB_GAMEPAD_B,
        "x": b.XUSB_GAMEPAD_X, "y": b.XUSB_GAMEPAD_Y,
        "lb": b.XUSB_GAMEPAD_LEFT_SHOULDER, "rb": b.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "ls": b.XUSB_GAMEPAD_LEFT_THUMB, "rs": b.XUSB_GAMEPAD_RIGHT_THUMB,
        "back": b.XUSB_GAMEPAD_BACK, "start": b.XUSB_GAMEPAD_START,
        "dup": b.XUSB_GAMEPAD_DPAD_UP, "ddown": b.XUSB_GAMEPAD_DPAD_DOWN,
        "dleft": b.XUSB_GAMEPAD_DPAD_LEFT, "dright": b.XUSB_GAMEPAD_DPAD_RIGHT,
    }


class _PadToken:
    __slots__ = ("kind", "value", "label")

    def __init__(self, kind, value, label):
        self.kind, self.value, self.label = kind, value, label


class GamepadBackend(InputBackend):
    """Bindings are bare names: 'rb', 'lb', 'a', 'rt', 'lt', 'dup', ..."""

    name = "gamepad"

    def __init__(self, bindings, hold_ms=40.0, async_release=True):
        if _vg is None:
            raise RuntimeError(
                "vgamepad unavailable. pip install vgamepad AND install the ViGEmBus "
                "driver from https://github.com/nefarius/ViGEmBus/releases")
        super().__init__(bindings, hold_ms, async_release)
        self.pad = _vg.VX360Gamepad()
        self._buttons = _button_map()

    @staticmethod
    def available() -> bool:
        return _vg is not None

    def bind(self, spec: str) -> _PadToken:
        what = spec.strip().lower().removeprefix("pad:")
        if what in ("lt", "rt"):
            return _PadToken("trigger", what, f"pad:{what}")
        if what in self._buttons:
            return _PadToken("button", self._buttons[what], f"pad:{what}")
        raise ValueError(
            f"unknown pad control {what!r}; known: {sorted(self._buttons) + ['lt', 'rt']}")

    def _press(self, token: _PadToken) -> None:
        if token.kind == "button":
            self.pad.press_button(button=token.value)
        elif token.value == "rt":
            self.pad.right_trigger(value=255)
        else:
            self.pad.left_trigger(value=255)
        self.pad.update()

    def _release(self, token: _PadToken) -> None:
        if token.kind == "button":
            self.pad.release_button(button=token.value)
        elif token.value == "rt":
            self.pad.right_trigger(value=0)
        else:
            self.pad.left_trigger(value=0)
        self.pad.update()

    def close(self) -> None:
        super().close()
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:  # noqa: BLE001 - teardown only
            pass
