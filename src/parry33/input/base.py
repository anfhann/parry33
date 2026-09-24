"""Input backend interface + async release scheduling.

The hot-path rule: a detection loop must NEVER block for the hold duration.
At 144 Hz a 40 ms blocking tap() costs ~6 captured frames, which is enough to
miss the follow-up attack in a combo. press() returns immediately and the
release is handed to a scheduler thread.

Why hold at all, and why 40 ms: the game samples input once per tick. A press
released inside a single tick can be missed entirely -- the down and up both land
between two polls. Hold for >= 2 game frames. At 60 fps game logic that is 33 ms,
so 40 ms is a safe floor. Shorter holds trade reliability for nothing, since the
release timing has no effect on a parry.
"""

from __future__ import annotations

import abc
import heapq
import threading

from ..clock import PreciseTimer, now_ns


class InputBackend(abc.ABC):
    name = "base"

    def __init__(self, bindings: dict, hold_ms: float = 40.0,
                 async_release: bool = True) -> None:
        self.bindings = dict(bindings)
        self.hold_ms = hold_ms
        self._scheduler = ReleaseScheduler(self) if async_release else None

    @abc.abstractmethod
    def bind(self, spec: str) -> object:
        """Resolve a config binding string into a prebuilt, ready-to-fire payload.

        Called once at startup so the hot path never parses or allocates.
        """

    @abc.abstractmethod
    def _press(self, token: object) -> None: ...

    @abc.abstractmethod
    def _release(self, token: object) -> None: ...

    def prepare(self):
        """Resolve every binding up front. Returns self."""
        self._tokens = {a: self.bind(s) for a, s in self.bindings.items()}
        return self

    def token(self, action: str):
        try:
            return self._tokens[action]
        except (AttributeError, KeyError):
            raise KeyError(
                f"action {action!r} is not bound for backend {self.name!r}. "
                f"Add it under [input.bindings.{self.name}] in config, or call "
                f"prepare() if you constructed the backend directly. "
                f"Bound actions: {sorted(getattr(self, '_tokens', {}))}") from None

    def press(self, action: str) -> int:
        """Fire the press. Returns perf ns sampled immediately after the syscall."""
        self._press(self.token(action))
        return now_ns()

    def release(self, action: str) -> int:
        self._release(self.token(action))
        return now_ns()

    def tap(self, action: str, hold_ms: float | None = None) -> int:
        """Press now, release after hold_ms. Returns the press timestamp.

        Non-blocking when async_release is on; otherwise busy-waits the hold.
        """
        hold = self.hold_ms if hold_ms is None else hold_ms
        t = self.press(action)
        if self._scheduler is not None:
            self._scheduler.schedule(action, t + int(hold * 1e6))
        else:
            PreciseTimer().sleep_until_ns(t + int(hold * 1e6))
            self.release(action)
        return t

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.stop()


class ReleaseScheduler:
    """Timer thread that fires deferred releases without touching the hot path."""

    def __init__(self, backend: InputBackend) -> None:
        self._backend = backend
        self._heap: list[tuple[int, str]] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="release-sched", daemon=True)
        self._thread.start()

    def schedule(self, action: str, at_ns: int) -> None:
        with self._lock:
            heapq.heappush(self._heap, (at_ns, action))
        self._wake.set()

    def _run(self) -> None:
        timer = PreciseTimer(spin_margin_us=300)
        while self._running:
            with self._lock:
                nxt = self._heap[0][0] if self._heap else None
            if nxt is None:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            if now_ns() < nxt:
                timer.sleep_until_ns(nxt)
            with self._lock:
                if not self._heap or self._heap[0][0] > now_ns():
                    continue
                _, action = heapq.heappop(self._heap)
            try:
                self._backend.release(action)
            except Exception:  # noqa: BLE001 - a stuck key is worse than a lost log line
                pass
        timer.close()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self._thread.join(timeout=1.0)


def build(cfg) -> InputBackend:
    """Construct the backend named by an InputConfig."""
    bindings = cfg.bindings.get(cfg.backend, {})
    if not bindings and cfg.backend == "null":
        # null stands in for whichever backend is being dry-run; borrow its map.
        bindings = cfg.bindings.get("sendinput", {})
    if cfg.backend == "sendinput":
        from .sendinput import SendInputBackend
        be = SendInputBackend(bindings, cfg.hold_ms, cfg.async_release)
    elif cfg.backend == "gamepad":
        from .gamepad import GamepadBackend
        be = GamepadBackend(bindings, cfg.hold_ms, cfg.async_release)
    elif cfg.backend == "null":
        from .nullbackend import NullBackend
        be = NullBackend(bindings, cfg.hold_ms, cfg.async_release)
    else:
        raise ValueError(f"unknown input backend: {cfg.backend!r}")
    return be.prepare()
