"""Capture throughput and latency benchmark.

Reports what the capture stage actually costs us, and what it leaves behind.

  grab()        -- wall time from entering grab() to holding pixels. This is
                   dominated by *waiting* for the next desktop update, so its
                   p50 tracks the frame period and is not a measure of our cost.
  interarrival  -- delivery cadence. Should sit on the frame period exactly.
  drops         -- gaps longer than 1.5 frame periods, i.e. frames we missed.
  headroom      -- how much downstream work per frame we can afford before we
                   start dropping. This is the actual inference budget.

Two things to know before reading any number here:

  * Desktop Duplication only emits a frame when the desktop actually changes.
    On a still screen you get nothing at all, not duplicates. Point this at the
    game, a video, or `parry33 flash --mode animate`.
  * There is no backlog. A frame that arrives while we are busy is discarded,
    not queued, so overrunning a frame period loses that frame outright. Do not
    try to measure "copy cost" by idling and then timing a grab -- that measures
    the wait to the next vsync and converges on frame_period/2.
"""

from __future__ import annotations

import numpy as np

from .. import config as cfgmod
from ..capture import base as capbase
from ..clock import now_ns
from ..util import prio
from ..util.ringbuffer import FrameRing
from ..util.stats import Samples


def run(cfg=None, seconds: float | None = None, mode: str | None = None,
        region=None, ring: bool = False, delay: float = 0.0) -> dict:
    cfg = cfg or cfgmod.load()
    seconds = seconds if seconds is not None else cfg.bench.seconds
    if mode:
        cfg.capture.mode = mode
    if region:
        cfg.capture.region = tuple(region)

    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game and get into "
              f"combat now.")
        print("(menus and idle areas are not representative; measure a fight)")
        for remaining in range(int(delay), 0, -1):
            print(f"  {remaining}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 20, end="\r")

    applied = prio.apply(cfg.timing)
    cap = capbase.build(cfg.capture)

    grab_s = Samples("grab()", capacity=400_000)
    inter_s = Samples("interarrival", capacity=400_000)
    copy_s = Samples("ring write", capacity=400_000)

    ringbuf = None
    with cap:
        size = cap.size
        print(f"capture: {cap!r}")
        print(f"  display {cfg.refresh_hz} Hz -> {cfg.frame_period_ms:.2f} ms/frame")
        if size:
            w, h = size
            mb = w * h * (4 if cfg.capture.color in ("BGRA", "RGBA") else 3) / 1e6
            print(f"  roi {w}x{h} = {mb:.2f} MB/frame "
                  f"({mb * cfg.refresh_hz:.0f} MB/s at full rate)")
        print(f"  priority {applied}")

        # Warm up: first grabs pay duplicator/surface setup.
        for _ in range(cfg.bench.warmup):
            cap.grab(timeout_ms=200)

        if ring:
            f = cap.grab(timeout_ms=500)
            if f is None:
                raise RuntimeError("no frames during warmup; is anything on screen moving?")
            ringbuf = FrameRing(256, f.data.shape, f.data.dtype)
            print(f"  ring {ringbuf.capacity} frames = {ringbuf.nbytes / 1e6:.1f} MB")

        t_end = now_ns() + int(seconds * 1e9)
        prev_t = 0
        timeouts = 0

        ctx = prio.frozen_gc() if cfg.timing.freeze_gc else _null_ctx()
        with ctx:
            while now_ns() < t_end:
                t0 = now_ns()
                frame = cap.grab(timeout_ms=200)
                t1 = now_ns()
                if frame is None:
                    timeouts += 1
                    continue
                grab_s.add(t1 - t0)
                if prev_t:
                    inter_s.add(frame.t_ns - prev_t)
                prev_t = frame.t_ns
                if ringbuf is not None:
                    t2 = now_ns()
                    ringbuf.write(frame.data, frame.t_ns, frame.seq)
                    copy_s.add(now_ns() - t2)

    n = len(grab_s)
    fps = n / seconds if seconds else 0.0
    print(f"\n  frames {n} in {seconds:.1f}s -> {fps:.1f} fps "
          f"({fps / max(cfg.refresh_hz, 1) * 100:.0f}% of display rate), "
          f"{timeouts} timeouts")
    print(grab_s.report())
    print(inter_s.report())
    if ringbuf is not None:
        print(copy_s.report())

    g = grab_s.summary()
    i = inter_s.summary()
    period = cfg.frame_period_ms
    if i.get("n"):
        # Self-calibrating: measure gaps against the OBSERVED cadence, not the
        # display refresh. The source here is the game, which may present well
        # below the panel rate -- comparing against refresh would report almost
        # every interval as a "drop" and hide the real signal.
        typical = i["p50"]
        drops = int((inter_s.values_ns > typical * 1.5e6).sum())
        print(f"\n  gaps > 1.5x the typical {typical:.1f} ms interval: "
              f"{drops}/{i['n']} ({drops / i['n'] * 100:.0f}%)")
        print(f"  Desktop Duplication keeps no backlog: grab() waits for the NEXT")
        print(f"  update and anything arriving while we are busy is discarded.")
        if i["p90"] > typical * 1.4:
            print(f"  Delivery is UNEVEN (p90 {i['p90']:.1f} ms vs p50 {typical:.1f} ms).")
            print(f"  If the game's own fps counter reads well above "
                  f"{1000 / typical:.0f}, we are losing frames in the capture")
            print(f"  path, not the game. Run `bench capture --compare` to find where.")

        # Budget on the slow frames, not the typical ones: p90 interarrival is
        # what the latency chain actually has to survive.
        fps_typical = 1000.0 / i["p50"]
        fps_slow = 1000.0 / i["p90"]
        jitter = i["p90"] - i["p50"]
        print()
        print(f"  frame delivery: {fps_typical:.0f} fps typical, "
              f"{fps_slow:.0f} fps at p90 (jitter {jitter:.1f} ms)")
        win = cfg.game.parry_window_ms
        print(f"  2-frame latency at the slow end: {2000.0 / fps_slow:.1f} ms "
              f"= {2000.0 / fps_slow / win * 100:.0f}% of the {win:.0f} ms window")
        if abs(fps_slow - cfg.game.fps) > 5:
            print()
            print(f"  config says [game] fps = {cfg.game.fps}. Measured p90 is "
                  f"{fps_slow:.0f}. Update config/local.toml:")
            print(f"      [game]")
            print(f"      fps = {int(fps_slow)}")
    return {"grab": g, "interarrival": i, "fps": fps, "timeouts": timeouts}


def _measure(cfg, seconds: float) -> dict:
    """Interarrival distribution for the current cfg.capture. No reporting."""
    cap = capbase.build(cfg.capture)
    inter = Samples("i", capacity=200_000)
    prev = 0
    try:
        with cap:
            for _ in range(20):
                cap.grab(timeout_ms=200)
            t_end = now_ns() + int(seconds * 1e9)
            while now_ns() < t_end:
                f = cap.grab(timeout_ms=200)
                if f is None:
                    continue
                if prev:
                    inter.add(f.t_ns - prev)
                prev = f.t_ns
    except Exception as e:  # noqa: BLE001 - one bad config must not kill the sweep
        return {"n": 0, "error": str(e)[:60]}
    return inter.summary()


def compare(cfg=None, seconds: float = 8.0, delay: float = 0.0) -> None:
    """Diagnose a capture rate that falls short of the game's render rate.

    If the game's own counter reads far above what we capture, the loss is in
    our path. The two things that plausibly cause it are the backend mode and
    the ROI size under real GPU contention -- neither of which can be settled by
    reasoning, because a synthetic desktop test does not reproduce a game
    saturating the GPU. So measure both, back to back, under identical load.
    """
    cfg = cfg or cfgmod.load()
    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game and get into combat")
        for r in range(int(delay), 0, -1):
            print(f"  {r}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 20, end="\r")

    base = cfg.capture.region
    if base:
        cx = (base[0] + base[2]) // 2
        cy = (base[1] + base[3]) // 2
        small = (cx - 160, cy - 160, cx + 160, cy + 160)
    else:
        small = (800, 440, 1120, 760)

    cases = [
        ("thread", base, "thread"),
        ("poll", base, "poll"),
        ("poll, 320x320 roi", small, "poll"),
        ("thread, 320x320 roi", small, "thread"),
    ]
    print(f"capture mode comparison -- {seconds:.0f}s each, same scene\n")
    print(f"  {'config':<22} {'fps p50':>8} {'fps p90':>8} {'jitter':>8}  verdict")
    best = None
    for label, region, mode in cases:
        cfg.capture.region = region
        cfg.capture.mode = mode
        d = _measure(cfg, seconds)
        if not d.get("n"):
            print(f"  {label:<22} {'--':>8} {'--':>8} {'--':>8}  "
                  f"{d.get('error', 'no frames')}")
            continue
        fps50, fps90 = 1000.0 / d["p50"], 1000.0 / d["p90"]
        jit = d["p90"] - d["p50"]
        if best is None or fps50 > best[1]:
            best = (label, fps50, mode, region)
        print(f"  {label:<22} {fps50:8.1f} {fps90:8.1f} {jit:7.1f}ms")
    cfg.capture.region = base
    if best:
        print(f"\n  best: {best[0]} at {best[1]:.0f} fps")
        print(f"  Set [capture] mode = \"{best[2]}\" in config/local.toml.")
        print(f"  Compare that against the game's own fps counter. If it still "
              f"falls well short,")
        print(f"  the loss is upstream of the mode choice -- suspect exclusive "
              f"fullscreen or")
        print(f"  a compositor path that is not presenting every frame to the "
              f"desktop.")


def measure_headroom(cfg, trials: int = 400, work_ms: float = 0.0) -> dict:
    """Measure how much of each frame period is left for us after capture.

    Why not "copy cost": Desktop Duplication keeps no backlog. grab() always
    waits for the NEXT desktop update, and any frame that arrives while we are
    not inside grab() is discarded rather than queued. Idling before a grab and
    timing it therefore measures the wait to the next vsync -- it converges on
    frame_period/2 for a random arrival phase, which is exactly what it does
    here, and tells you nothing about the copy.

    The consequence is the real design constraint: there is no catching up. If
    downstream work overruns one frame period, that frame is simply gone. So the
    number worth tracking is headroom -- frame_period minus everything we do
    between two grab() calls -- and the drop count when we exceed it.

    Pass work_ms to simulate a downstream inference cost and find where it breaks.
    """
    from ..clock import PreciseTimer
    cap = capbase.build(cfg.capture)
    inter = Samples("interarrival", capacity=trials * 2 + 16)
    timer = PreciseTimer(spin_margin_us=200)
    period_ns = int(cfg.frame_period_ms * 1e6)
    drops = 0
    prev = 0
    n = 0
    try:
        with cap:
            for _ in range(20):
                cap.grab(timeout_ms=200)
            while n < trials:
                f = cap.grab(timeout_ms=200)
                if f is None:
                    continue
                n += 1
                if prev:
                    d = f.t_ns - prev
                    inter.add(d)
                    if d > period_ns * 1.5:
                        drops += round(d / period_ns) - 1
                prev = f.t_ns
                if work_ms:
                    timer.sleep(work_ms / 1000.0)
    finally:
        timer.close()
    out = inter.summary()
    out["drops"] = drops
    out["frames"] = n
    return out


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sweep(cfg=None, sizes=((320, 180), (640, 360), (960, 540), (1280, 720)),
          seconds: float = 3.0) -> None:
    """Find the sustainable work budget at each ROI size.

    Bigger ROI means more context for the model but more bytes per frame. This
    reports where that tradeoff actually starts to cost frames.
    """
    cfg = cfg or cfgmod.load()
    cx, cy = 1920, 1080
    period = cfg.frame_period_ms
    print(f"ROI sweep -- can we hold {1000 / period:.0f} fps, and with how much "
          f"work per frame?\n")
    print(f"  {'roi':>12} {'MB/frame':>9} {'fps':>7} {'drops':>6} {'p99 gap':>8}  "
          f"max sustainable work/frame")
    trials = max(120, int(seconds * 120))
    for w, h in sizes:
        cfg.capture.region = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        mb = w * h * 4 / 1e6
        base = measure_headroom(cfg, trials=trials)
        if not base.get("n"):
            print(f"  {f'{w}x{h}':>12} {mb:9.2f}  (no frames -- is anything moving?)")
            continue
        fps = 1000.0 / base["p50"]
        # Walk the injected work up until frames start dropping.
        budget = 0.0
        for work in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            r = measure_headroom(cfg, trials=max(80, trials // 3), work_ms=work)
            if r.get("drops", 0) > r.get("frames", 1) * 0.02:
                break
            budget = work
        print(f"  {f'{w}x{h}':>12} {mb:9.2f} {fps:7.1f} {base['drops']:6d} "
              f"{base['p99']:8.2f}  {budget:.0f}+ ms")
    print(f"\n  Frame period is {period:.2f} ms. 'max sustainable work' is how much")
    print("  downstream time per frame you can spend before dropping >2% of frames.")
    print("  That is the real inference budget -- size the ROI and the model to it.")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="capture benchmark")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--mode", choices=("poll", "thread"), default=None)
    p.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    p.add_argument("--ring", action="store_true", help="also time ring-buffer writes")
    p.add_argument("--delay", type=float, default=0.0,
                   help="countdown before measuring, so you can tab into the game")
    p.add_argument("--sweep", action="store_true", help="ROI size sweep instead")
    p.add_argument("--compare", action="store_true",
                   help="compare poll vs thread and roi sizes under real load")
    a = p.parse_args(argv)
    cfg = cfgmod.load()
    if a.compare:
        compare(cfg, seconds=a.seconds or 8.0, delay=a.delay)
    elif a.sweep:
        sweep(cfg, seconds=a.seconds or 3.0)
    else:
        run(cfg, seconds=a.seconds, mode=a.mode, region=a.region, ring=a.ring,
            delay=a.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
