"""No-op input backend.

Runs the full binding/scheduling machinery and records what would have fired,
without touching the OS. Use it to exercise the trigger loop safely while the
game is running, or on a machine with no ViGEmBus.
"""

from __future__ import annotations

from ..clock import now_ns
from .base import InputBackend


class NullBackend(InputBackend):
    name = "null"

    def __init__(self, bindings, hold_ms=40.0, async_release=True):
        super().__init__(bindings, hold_ms, async_release)
        self.events: list[tuple[int, str, str]] = []

    def bind(self, spec: str) -> str:
        return spec

    def _press(self, token: str) -> None:
        self.events.append((now_ns(), "down", token))

    def _release(self, token: str) -> None:
        self.events.append((now_ns(), "up", token))
