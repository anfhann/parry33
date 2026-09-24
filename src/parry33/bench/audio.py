"""Audio path benchmarks.

Three things, in order of how much they matter:

  cadence -- do blocks arrive on the audio clock, evenly, without overruns?
             This is the audio equivalent of interarrival, and unlike video it
             should be immune to GPU load.

  loop    -- burst emitted -> onset detected. The audio analogue of
             `bench loop`, and directly comparable to the ~33 ms the video path
             costs on this rig.

  listen  -- live level meter and onset counter. Not a benchmark: it exists
             because on a machine with virtual audio routing (Sonar, Voicemeeter)
             it is entirely possible to capture a silent device and spend an hour
             debugging a detector that was never fed anything.
"""

from __future__ import annotations

import numpy as np

from .. import config as cfgmod
from ..audio import onset as onsetmod
from ..audio.capture import LoopbackCapture
from ..clock import PreciseTimer, now_ns
from ..util.stats import Samples


def devices() -> int:
    rows = LoopbackCapture.list_devices()
    if not rows:
        print("no WASAPI loopback devices found. pip install pyaudiowpatch")
        return 1
    print("WASAPI loopback devices (default output first):\n")
    for d in rows:
        mark = "*" if d["is_default_output"] else " "
        print(f" {mark} [{d['index']:>3}] {d['name']}")
        print(f"       {d['channels']}ch @ {d['rate']} Hz")
    print("\n  * = loopback of the current default output device")
    print("  If the game routes through a virtual device (SteelSeries Sonar,")
    print("  Voicemeeter), the default may carry no game audio. Verify with:")
    print("      parry33 audio listen --device <index>")
    return 0


def listen(device: int | None = None, blocksize: int = 512, seconds: float = 20.0,
           detector: str = "flux", sensitivity: float = 3.0) -> int:
    """Live meter + onset counter. Confirms we are capturing the right device."""
    cap = LoopbackCapture(device_index=device, blocksize=blocksize)
    with cap:
        det = onsetmod.build(detector, cap.samplerate, blocksize,
                             sensitivity=sensitivity)
        print(f"{cap!r}")
        print(f"detector={detector} sensitivity={sensitivity}")
        print("play some game audio. bar = level, ! = onset\n")
        t_end = now_ns() + int(seconds * 1e9)
        timer = PreciseTimer(spin_margin_us=300)
        peak = 1e-6
        onsets = 0
        blocks = 0
        next_draw = 0
        while now_ns() < t_end:
            item = cap.read()
            if item is None:
                timer.sleep(0.002)
                continue
            samples, t_blk = item
            blocks += 1
            rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
            peak = max(peak, rms)
            hit = det.push(samples, t_ns=t_blk)
            onsets += hit
            # Throttle the redraw on the WALL CLOCK, not on a block count. A
            # console redraw costs milliseconds; at one per 4 blocks that is
            # ~47/s, which stalled the drain badly enough to overrun the ring
            # and lose 55% of the audio. Dropped blocks are not merely missing
            # data -- spectral flux compares each block to the previous one, so
            # a gap reads as a huge spectral change and manufactures a false
            # onset. A meter that corrupts the thing it measures is worse than
            # no meter.
            now = now_ns()
            if now >= next_draw:
                next_draw = now + 50_000_000              # 20 Hz
                n = int(min(1.0, rms / peak) * 40)
                db = 20 * np.log10(max(rms, 1e-9))
                print(f"\r  [{'#' * n}{'.' * (40 - n)}] {db:6.1f} dBFS  "
                      f"onsets={onsets:<4d} {'!' if hit else ' '}", end="")
        timer.close()
        print()
        if blocks == 0:
            # WASAPI loopback on an IDLE endpoint delivers nothing at all --
            # not silent blocks, no blocks. "Nothing is playing" and "wrong
            # device" therefore look identical unless the block count is
            # reported, which cost one confusing debugging detour.
            print(f"  NO BLOCKS ARRIVED from {cap.device_name!r}.")
            print("  The endpoint is idle: WASAPI loopback yields nothing when")
            print("  nothing is playing. Start the game (or any sound) first.")
            print("  If audio IS playing, the device is wrong -- try")
            print("  `parry33 audio devices` and --device <n>.")
            return 1
        if peak < 1e-5:
            print(f"  {blocks} blocks arrived, but every one was SILENT.")
            print("  The device is live and carries no sound, which usually")
            print("  means the wrong endpoint. Try `parry33 audio devices`.")
            return 1
        print(f"\n  {blocks} blocks, {onsets} onsets, peak {20 * np.log10(peak):.1f} dBFS, "
              f"{cap.ring.overruns} overruns")
        if cap.ring.overruns:
            print(f"  WARNING: {cap.ring.overruns} blocks dropped -- onset count "
                  f"is not trustworthy (gaps read as onsets)")
    return 0


def cadence(device: int | None = None, blocksize: int = 512,
            seconds: float = 10.0, delay: float = 0.0) -> dict:
    """Block delivery timing. Should track the audio clock, not the GPU."""
    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game")
        for r in range(int(delay), 0, -1):
            print(f"  {r}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 20, end="\r")

    cap = LoopbackCapture(device_index=device, blocksize=blocksize)
    inter = Samples("block gap", capacity=200_000)
    with cap:
        print(f"{cap!r}")
        cap.drain()
        timer = PreciseTimer(spin_margin_us=300)
        t_end = now_ns() + int(seconds * 1e9)
        prev = 0
        n = 0
        while now_ns() < t_end:
            item = cap.read()
            if item is None:
                timer.sleep(0.001)
                continue
            _, t = item
            n += 1
            if prev:
                inter.add(t - prev)
            prev = t
        timer.close()
        over = cap.ring.overruns
    d = inter.summary()
    print(f"\n  blocks {n} in {seconds:.1f}s -> {n / seconds:.1f} blocks/s "
          f"({n * blocksize / seconds / 1000:.1f} kHz effective), {over} overruns")
    print(inter.report())
    if d.get("n"):
        print(f"\n  nominal block period {cap.block_ms:.2f} ms, "
              f"measured p50 {d['p50']:.2f} ms")
        print(f"  jitter p99-p50 {d['p99'] - d['p50']:.2f} ms")
    return d


def loop(device: int | None = None, blocksize: int = 512, trials: int = 60,
         detector: str = "flux", sensitivity: float = 3.0,
         cfg=None) -> dict:
    """Burst emitted -> onset detected. Comparable to `bench loop` for video."""
    from ..harness.beep import BeepEmitter
    cfg = cfg or cfgmod.load()
    window = cfg.game.parry_window_ms

    cap = LoopbackCapture(device_index=device, blocksize=blocksize)
    lat = Samples("emit->detect", capacity=trials * 2 + 16)
    misses = 0

    with cap:
        emitter = BeepEmitter(samplerate=cap.samplerate, blocksize=256).start()
        det = onsetmod.build(detector, cap.samplerate, blocksize,
                             sensitivity=sensitivity)
        timer = PreciseTimer(spin_margin_us=300)
        print(f"capture {cap!r}")
        print(f"emit    {emitter.device_name!r}")
        print(f"detector={detector} sensitivity={sensitivity} trials={trials}")
        print("  this will play short clicks through your speakers\n")
        try:
            # Let the detector learn the room/idle noise floor first.
            t_warm = now_ns() + int(1.5e9)
            while now_ns() < t_warm:
                item = cap.read()
                if item is None:
                    timer.sleep(0.002)
                    continue
                det.push(item[0], t_ns=item[1])

            for _ in range(trials):
                cap.drain()
                emitter.fire()
                while not emitter.fired:
                    timer.sleep(0.0005)
                t_emit = emitter.t_emit_ns

                seen = None
                deadline = now_ns() + int(500e6)
                while now_ns() < deadline:
                    item = cap.read()
                    if item is None:
                        timer.sleep(0.0005)
                        continue
                    samples, t_block = item
                    if t_block < t_emit:
                        det.push(samples, t_ns=t_block)   # feed baseline, ignore stale
                        continue
                    if det.push(samples, t_ns=t_block):
                        seen = t_block
                        break
                if seen is None:
                    misses += 1
                else:
                    lat.add(seen - t_emit)
                timer.sleep(0.25)              # let the refractory window clear
        finally:
            timer.close()
            emitter.stop()

    print(lat.report(18))
    s = lat.summary()
    if misses:
        print(f"\n  {misses}/{trials} bursts not detected "
              f"(try --sensitivity {max(1.5, sensitivity - 1):.1f})")
    if s.get("n"):
        print(f"\n  block period {cap.block_ms:.2f} ms -- detection cannot beat this")
        print(f"\n  against the {window:.0f} ms parry window:")
        print(f"    p50 {s['p50']:6.2f} ms = {s['p50'] / window * 100:4.1f}%")
        print(f"    p99 {s['p99']:6.2f} ms = {s['p99'] / window * 100:4.1f}%")
        video_ms = 2 * cfg.capture_period_ms
        print(f"\n  video path for comparison: {video_ms:.1f} ms "
              f"({video_ms / window * 100:.0f}% of window)")
        if s["p50"] < video_ms:
            print(f"  audio is {video_ms - s['p50']:.1f} ms faster than video, and")
            print(f"  unlike video it does not degrade when the GPU saturates.")
        else:
            print(f"  audio is NOT faster than video here. Its value would be")
            print(f"  robustness and simplicity, not latency -- reconsider priority.")
        print(f"\n  NOTE: this measures OUR emit->detect path. The game's own")
        print(f"  audio-generation latency is upstream and not included, exactly")
        print(f"  as the flash window excludes the game's render latency.")
    return s
