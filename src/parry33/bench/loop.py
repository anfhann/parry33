"""Closed-loop latency: injection -> observed pixel change.

This is the only Phase 1 number that means anything. It measures the full chain
we are responsible for:

  press() syscall
    -> Windows raw input / ViGEm bus
    -> responder app wakes and repaints
    -> DWM composes
    -> DXGI Desktop Duplication makes the frame available
    -> our grab() returns it
    -> our threshold check fires

Everything except the last two steps is also present when the target is the game
itself, with the game's input poll and render pipeline substituted for a GDI
paint. So treat the result as a floor and assume the game adds one game-frame of
input polling plus its own render+present latency on top.

Budget arithmetic at 144 Hz (6.94 ms/frame): the display quantises twice, once
when the responder presents and once when we sample, so ~7 ms of the result is
structural and cannot be optimised away in software. What is left over is ours.

Default target is the flash-window harness, spawned automatically. Point it at
anything else with --no-spawn --region L T R B.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time

import numpy as np

from .. import config as cfgmod
from ..capture import base as capbase
from ..clock import PreciseTimer, now_ns
from ..input import base as inbase
from ..util import prio
from ..util.stats import Samples, budget_table

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_u32.FindWindowW.restype = ctypes.c_void_p
_u32.SetForegroundWindow.argtypes = (ctypes.c_void_p,)

WINDOW_CLASS = "Parry33FlashWindow"


def _spawn_flash(x: int, y: int, size: int, mode: str, vk: int,
                 pad_button: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "parry33.harness.flash_window",
           "--x", str(x), "--y", str(y), "--size", str(size),
           "--mode", mode, "--vk", hex(vk), "--pad-button", pad_button]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    deadline = time.time() + 8.0
    hwnd = None
    while time.time() < deadline:
        hwnd = _u32.FindWindowW(WINDOW_CLASS, None)
        if hwnd:
            break
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"flash window exited early:\n{out}")
        time.sleep(0.05)
    if not hwnd:
        proc.terminate()
        raise RuntimeError("flash window never appeared")
    _u32.AllowSetForegroundWindow(proc.pid)
    _u32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    time.sleep(0.35)   # let focus settle and the first paint land
    return proc


def _roi_level(arr: np.ndarray, stride: int = 4) -> float:
    """Cheap brightness probe. Subsampled single channel -- a few microseconds."""
    return float(arr[::stride, ::stride, 1].mean()) if arr.ndim == 3 else \
        float(arr[::stride, ::stride].mean())


def _drain_until(cap, predicate, timeout_ms: float = 800.0):
    deadline = now_ns() + int(timeout_ms * 1e6)
    last = None
    while now_ns() < deadline:
        f = cap.grab(timeout_ms=50)
        if f is None:
            continue
        last = _roi_level(f.data)
        if predicate(last):
            return last
    return last


def run(cfg=None, trials: int | None = None, backend: str | None = None,
        action: str = "parry", spawn: bool = True, region=None,
        window_pos=(300, 300), window_size=420, probe=200,
        responder_mode=None) -> dict:
    cfg = cfg or cfgmod.load()
    if backend:
        cfg.input.backend = backend
    trials = trials or cfg.bench.trials

    prio.apply(cfg.timing)
    be = inbase.build(cfg.input)
    token = be.token(action)
    label = getattr(token, "label", str(token))

    proc = None
    if spawn:
        mode = "pad" if cfg.input.backend == "gamepad" else (responder_mode or "raw")
        vk = 0x20
        pad_button = "rb"
        if cfg.input.backend == "sendinput":
            from ..input.sendinput import VK
            spec = cfg.input.binding(action)
            vk = VK.get(spec.partition(":")[2].strip().lower(), 0x20)
        elif cfg.input.backend == "gamepad":
            pad_button = cfg.input.binding(action).removeprefix("pad:")
        wx, wy = window_pos
        proc = _spawn_flash(wx, wy, window_size, mode, vk, pad_button)
        # Probe a small box inside the window: less copy cost, no edge artefacts.
        cx, cy = wx + window_size // 2, wy + window_size // 2
        region = (cx - probe // 2, cy - probe // 2, cx + probe // 2, cy + probe // 2)
    if region is None:
        raise ValueError("--no-spawn requires --region L T R B")

    cfg.capture.region = tuple(region)
    cap = capbase.build(cfg.capture)

    lat = Samples("inject->pixel", capacity=trials * 2 + 16)
    syscall = Samples("press() syscall", capacity=trials * 2 + 16)
    detect = Samples("threshold calc", capacity=trials * 2 + 16)
    frames_waited = []

    timer = PreciseTimer(spin_margin_us=cfg.timing.spin_margin_us)
    misses = 0

    try:
        with cap:
            print(f"closed loop: input={be.name}:{label}  capture={cap!r}")
            print(f"  probe roi {cfg.capture.size}  trials {trials}  "
                  f"display {cfg.refresh_hz} Hz ({cfg.frame_period_ms:.2f} ms)")

            for _ in range(cfg.bench.warmup):
                cap.grab(timeout_ms=100)
            dark = _drain_until(cap, lambda v: v < 64) or 0.0
            print(f"  baseline level {dark:.1f}\n")

            hi = 128.0
            ctx = prio.frozen_gc() if cfg.timing.freeze_gc else _null_ctx()
            with ctx:
                for i in range(trials):
                    if _drain_until(cap, lambda v: v < 64, 500) is None:
                        misses += 1
                        continue

                    t_pre = now_ns()
                    be.tap(action)
                    syscall.add(now_ns() - t_pre)

                    seen = None
                    nframes = 0
                    deadline = now_ns() + int(400e6)
                    while now_ns() < deadline:
                        f = cap.grab(timeout_ms=50)
                        if f is None:
                            continue
                        nframes += 1
                        t_d = now_ns()
                        level = _roi_level(f.data)
                        detect.add(now_ns() - t_d)
                        if level > hi:
                            seen = f.t_ns
                            break
                    if seen is None:
                        misses += 1
                    else:
                        lat.add(seen - t_pre)
                        frames_waited.append(nframes)
                    timer.sleep(0.030)
    finally:
        timer.close()
        be.close()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(lat.report(18))
    print(syscall.report(18))
    print(detect.report(18))
    if misses:
        print(f"\n  {misses}/{trials} trials produced no observable change.")
        if be.name == "sendinput":
            print("  In `key` responder mode the window must hold keyboard focus. "
                  "Use --responder-mode raw (default) to remove that dependency.")

    s = lat.summary()
    if s.get("n"):
        fw = np.array(frames_waited)
        print(f"\n  frames from inject to detect: p50 {np.percentile(fw, 50):.0f}  "
              f"p99 {np.percentile(fw, 99):.0f}")
        sysc = syscall.summary()["p50"]
        det = detect.summary()["p50"]
        ours = sysc + det
        rest = max(0.0, s["p50"] - ours)
        period = cfg.frame_period_ms
        window = cfg.game.parry_window_ms
        spread = s["p99"] - s["min"]
        margin = window / 2 - spread / 2

        print()
        print(f"  where the {s['p50']:.2f} ms goes (p50):")
        print(budget_table([
            ("our code (measured)", ours),
            ("rest: OS+paint+DWM+DDA", rest),
        ], window))
        print(f"    our code = press() {sysc:.2f} + threshold {det:.2f} ms "
              f"-- {ours / s['p50'] * 100:.1f}% of the total")
        print(f"    the rest = {rest / period:.2f} display frames of quantisation, "
              f"not ours to optimise")

        print()
        print(f"  against the {window:.0f} ms parry window:")
        print(f"    p50     {s['p50']:6.2f} ms  = {s['p50'] / window * 100:5.1f}% of window")
        print(f"    p99     {s['p99']:6.2f} ms  = {s['p99'] / window * 100:5.1f}% of window")
        print(f"    spread  {spread:6.2f} ms  = {spread / window * 100:5.1f}% of window "
              f" <- the only part that hurts")
        print()
        print("  Constant latency is free: aim that far ahead and it cancels.")
        print(f"  Only jitter eats the window. Timing budget left for the model:")
        print(f"    +/- {margin:.1f} ms  (~{margin / cfg.capture_period_ms:.0f} captured "
              f"frames at {1000 / cfg.capture_period_ms:.0f} fps)")
        if margin > 40:
            print()
            print("  That is wide enough for a REACTIVE trigger on a near-impact cue;")
            print(f"  predicting from the wind-up is not required at this window size.")
        if cfg.game.fps < cfg.refresh_hz:
            print()
            print(f"  NOTE: measured against a {cfg.refresh_hz} Hz responder; the game "
                  f"renders at {cfg.game.fps} fps,")
            print(f"  so against the real game expect quantisation in "
                  f"{cfg.capture_period_ms:.1f} ms steps, not {period:.1f} ms.")
            print(f"  Re-run with --no-spawn --region <game roi> to get the true figure.")

    return {"latency": s, "syscall": syscall.summary(), "misses": misses}


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="closed-loop latency benchmark")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--backend", choices=("sendinput", "gamepad"), default=None)
    p.add_argument("--action", default="parry")
    p.add_argument("--no-spawn", action="store_true",
                   help="do not spawn the flash window; use --region instead")
    p.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    p.add_argument("--window-pos", type=int, nargs=2, default=(300, 300))
    p.add_argument("--window-size", type=int, default=420)
    p.add_argument("--responder-mode", choices=("raw", "key"), default="raw",
                   help="raw = RIDEV_INPUTSINK, no focus needed (default); "
                        "key = WM_KEYDOWN, needs focus but is closest to a "
                        "focused game window")
    a = p.parse_args(argv)
    run(cfgmod.load(), trials=a.trials, backend=a.backend, action=a.action,
        spawn=not a.no_spawn, region=a.region,
        window_pos=tuple(a.window_pos), window_size=a.window_size,
        responder_mode=a.responder_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
