"""Injection syscall benchmark.

This measures only how long the injection call itself takes to return. That is
a floor, not a latency: SendInput returning means the event is queued in the raw
input stack, and pad.update() returning means the HID report is on its way to
the ViGEm bus. Neither says anything about when the game observes it.

Use this to confirm the hot path is cheap and jitter-free. Use bench/loop.py for
the number that actually matters.
"""

from __future__ import annotations

from .. import config as cfgmod
from ..clock import PreciseTimer, now_ns
from ..input import base as inbase
from ..util import prio
from ..util.stats import Samples


def run(cfg=None, backend: str | None = None, action: str = "parry",
        trials: int | None = None, gap_ms: float = 8.0) -> dict:
    cfg = cfg or cfgmod.load()
    if backend:
        cfg.input.backend = backend
    trials = trials or cfg.bench.trials

    prio.apply(cfg.timing)
    be = inbase.build(cfg.input)
    timer = PreciseTimer(spin_margin_us=cfg.timing.spin_margin_us)

    press_s = Samples("press()", capacity=trials * 2 + 16)
    release_s = Samples("release()", capacity=trials * 2 + 16)
    tap_s = Samples("tap() return", capacity=trials * 2 + 16)

    token = be.token(action)
    label = getattr(token, "label", str(token))
    print(f"inject: backend={be.name} action={action} -> {label}")
    print(f"  trials {trials}, hold {cfg.input.hold_ms:.0f} ms, "
          f"async_release={cfg.input.async_release}")
    print("  NOTE: syscall cost only. This is NOT end-to-end latency; run "
          "`parry33 bench loop` for that.\n")

    for _ in range(cfg.bench.warmup):
        be.press(action)
        be.release(action)

    ctx = prio.frozen_gc() if cfg.timing.freeze_gc else _null_ctx()
    with ctx:
        for _ in range(trials):
            t0 = now_ns()
            be.press(action)
            t1 = now_ns()
            be.release(action)
            t2 = now_ns()
            press_s.add(t1 - t0)
            release_s.add(t2 - t1)
            timer.sleep(gap_ms / 1000.0)

        # tap() is the call the trigger loop will actually make; with async
        # release it must return in roughly press() time, not hold_ms.
        for _ in range(min(trials, 200)):
            t0 = now_ns()
            be.tap(action)
            tap_s.add(now_ns() - t0)
            timer.sleep((cfg.input.hold_ms + gap_ms) / 1000.0)

    print(press_s.report())
    print(release_s.report())
    print(tap_s.report())

    t = tap_s.summary()
    if t.get("n"):
        if cfg.input.async_release and t["p99"] > 1.0:
            print(f"\n  WARNING: tap() p99 is {t['p99']:.2f} ms. With async_release "
                  f"on it should be well under 1 ms -- the hot path is blocking.")
        else:
            print(f"\n  tap() returns in {t['p50']:.3f} ms p50 -- hot path is clear "
                  f"({cfg.frame_period_ms:.2f} ms frame budget).")

    timer.close()
    be.close()
    return {"press": press_s.summary(), "release": release_s.summary(),
            "tap": tap_s.summary()}


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="injection syscall benchmark")
    p.add_argument("--backend", choices=("sendinput", "gamepad", "null"), default=None)
    p.add_argument("--action", default="parry")
    p.add_argument("--trials", type=int, default=None)
    a = p.parse_args(argv)
    print("\n  Focus a scratch text field first if using sendinput -- this will "
          "type into whatever has focus.\n")
    run(cfgmod.load(), backend=a.backend, action=a.action, trials=a.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
