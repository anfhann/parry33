# parry33

Vision-based auto-parry harness for *Clair Obscur: Expedition 33*.

Capture the screen with DXGI Desktop Duplication, decide whether an attack
telegraph is on screen, and inject the parry input inside the reaction window.

**Phase 1 is complete and measured.** Capture, injection and a closed-loop
latency harness all work and have been benchmarked on real hardware.

---

## Phase 1 results

> These were measured against the **synthetic 144 Hz responder**, on a 4K
> desktop, before the display was reconfigured. They are the clean-room ceiling.
> The real-game numbers are lower and are the ones to plan against -- see
> "Measured against the real game" below. Kept because the ratios and the
> reasoning still hold.

```
closed loop: inject -> observed pixel change     p50 13.90 ms   p99 19.41 ms
  of which our own code                          0.38 ms  (2.7%)
  press() syscall                                0.32 ms
  threshold check                                0.06 ms
  detection lands 2 display frames after injection

against the 150 ms parry window (9 frames @ 60 fps game logic):
  p50                                             9.7% of the window
  p99                                            15.1% of the window
  spread (min -> p99)                             6.8% of the window
  timing budget left for the model               +/- 70 ms
```

```
capture: 320x320 ROI, poll mode
  sustained                    144.2 fps, 0 drops, 0 timeouts
  interarrival                 p50 6.94 ms   p99 7.98 ms
  ring-buffer write            p50 0.09 ms
```

```
ROI sweep -- can we hold 144 fps, and with how much work per frame?
           roi   MB/frame     fps  drops  max sustainable work/frame
       320x180       0.23   143.7      0   5+ ms
       640x360       0.92   143.8      0   6+ ms
       960x540       2.07   144.3      0   6+ ms
      1280x720       3.69   143.4      0   6+ ms
```

### What these numbers actually mean

**The end-to-end latency is quantisation-bound, not compute-bound.** 13.9 ms is
almost exactly two display frames (2 x 6.94 = 13.88 ms), and only 0.38 ms of it
is code we wrote. Every stage in the chain — the input stack, the target's
repaint, DWM composition, DDA delivery — snaps to the refresh interval, and two
of those snaps are unavoidable.

More importantly, **latency is not the binding constraint at all.** Against a
150 ms parry window we spend under 10% at p50 and 15% at p99.

And of that, only the *variance* actually costs anything. Constant latency is
free: if the delay were a rock-solid 13.9 ms you would aim 13.9 ms early and land
dead centre. Only jitter — about 10-12 ms of spread, under 7% of the window —
cannot be aimed away. That leaves roughly **+/- 70 ms of timing error** before
the model misses.

Three consequences that shape Phase 3:

1. **Optimising the Python is pointless.** We are 18x under the frame budget
   already, our code is 2.7% of the latency, and the latency is 10% of the
   window. Rewriting the hot path in C++ would be measuring noise.
2. **A reactive trigger is viable.** With +/- 70 ms of slack the classifier can
   fire on a near-impact cue and still land inside the window. It does not have
   to forecast from wind-up kinematics, which is a far harder problem. Prefer
   reactive until measurement proves it insufficient.
3. **Refresh rate is a weak lever.** 144 -> 240 Hz saves ~5.6 ms out of 150,
   i.e. 3.7%. Not worth buying hardware for.

**ROI size barely matters for capture,** at least up to 1280x720 — cost is flat
and we hold 144 fps with 6+ ms/frame to spare at every size tested. So choose the
ROI for what the *model* needs, and spend the saved budget on inference. Note
that DXGI copies the region on the GPU side, so a smaller ROI does not scale the
capture cost the way a naive `numpy` crop would.

**There is no frame backlog.** Desktop Duplication emits a frame only when the
desktop changes, and a frame that arrives while we are busy is discarded, not
queued. Overrun one frame period and that frame is gone permanently. The
detection loop must fit inside 6.94 ms — there is no catching up.

> Measuring caveat: do not try to isolate "copy cost" by idling and then timing a
> `grab()`. That measures the wait to the next vsync and converges on
> `frame_period / 2`. An earlier version of the capture bench did exactly this
> and reported a confident, meaningless number.

---

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[capture,dev]"
```

`vgamepad` is optional and needs the **ViGEmBus driver** installed system-wide
([releases](https://github.com/nefarius/ViGEmBus/releases)). Without it the
`sendinput` backend still works.

Then check the environment:

```bash
.venv/Scripts/python -m parry33 doctor
```

---

## Usage

```bash
python -m parry33 doctor
```

```bash
python -m parry33 bench capture --seconds 10 --ring
```

```bash
python -m parry33 bench capture --sweep
```

```bash
python -m parry33 bench loop --trials 200
```

```bash
python -m parry33 bench inject --backend null
```

```bash
python -m parry33 flash --mode animate --hz 144
```

`bench capture` needs something on screen to be moving — Desktop Duplication
produces nothing on a still desktop. Run the game, or use `flash --mode animate`
as a paced synthetic source.

---

## Layout

```
src/parry33/
  clock.py              high-res waitable timer + spin; the sleep primitive
  config.py             TOML config, default.toml <- local.toml overlay
  doctor.py             environment preflight
  cli.py                command line
  capture/
    base.py             CaptureBackend ABC, Frame, build()
    dxgi.py             Desktop Duplication via bettercam/dxcam
    synthetic.py        GPU-free frame source for dev and CI
  input/
    base.py             InputBackend ABC + async ReleaseScheduler
    sendinput.py        Win32 SendInput, scancode-based, prebuilt structs
    gamepad.py          virtual Xbox 360 pad via ViGEmBus
    nullbackend.py      records instead of injecting
  bench/
    capture.py          throughput, drops, sustainable work budget
    inject.py           injection syscall cost
    loop.py             closed-loop inject -> pixel latency  <- the real one
  harness/
    flash_window.py     Win32 responder window (raw/key/pad/animate modes)
  util/
    ringbuffer.py       preallocated SPSC frame ring
    stats.py            percentile reporting, ASCII histograms
    prio.py             priority, affinity, DPI awareness, GC freeze
```

---

## Design notes worth keeping

**Scancodes, not virtual keys.** Games reading DirectInput or Raw Input look at
the scancode and ignore `wVk`. VK-only injection silently does nothing while
working perfectly in Notepad. `sendinput.py` uses `KEYEVENTF_SCANCODE`
throughout and prebuilds every `INPUT` struct at bind time, so the hot path is
one syscall with zero Python allocation.

**Hold the key long enough to be sampled.** The game polls input once per tick.
A press released inside a single tick can be missed entirely — both edges land
between two polls. Default hold is 40 ms (~2 frames at 60 fps game logic).

**Never block the detection loop for the hold.** A 40 ms blocking `tap()` costs
~6 captured frames at 144 Hz. `press()` returns immediately and the release goes
to a scheduler thread; measured `tap()` return is 0.024 ms.

**`ctypes` argtypes are not cosmetic.** `GetCurrentProcess()` returns the
pseudo-handle `(HANDLE)-1`. Without a declared `restype` ctypes hands back a C
int, which is then passed as a 32-bit `-1` into a 64-bit handle slot; the call
fails with `ERROR_INVALID_HANDLE` and priority is never applied — silently. This
bug was live in `prio.py` until the benchmark reported `high_priority: False`.

**Raw Input for the test responder.** The harness defaults to
`RIDEV_INPUTSINK`, which receives injected keys without holding focus. It is
both more reliable for scripted runs and closer to how a UE5 game actually reads
the keyboard than `WM_KEYDOWN`. `--responder-mode key` is available when you
want the strictest focused-window comparison.

---

## Known constraints

- Run the game **borderless / windowed-fullscreen**. True exclusive fullscreen
  can bypass the compositor and starve Desktop Duplication.
- `capture.gpu` must be the adapter the game renders on, or you duplicate an
  output that never updates.
- If the game runs elevated, UIPI silently drops `SendInput` from a
  standard-rights process. Run elevated to match.
- Protected/DRM content returns black frames by design.
- Keybinds in `[input.bindings]` are unverified defaults. Check them in-game
  before trusting a run.

---

## Audio path (Phase 2, built)

Audio does not go through the display pipeline at all, which matters because the
video path is quantisation-bound and degrades when the GPU saturates. Measured
on this rig with `parry33 bench audio`:

```
emit -> onset detected     p50 17.71 ms   p99 20.11 ms   (256-frame blocks)
video path, same window    p50 38.50 ms
```

Roughly 2x faster than video, with a tighter tail, and immune to the framerate
collapse that cost 3x capture throughput. Two things worth knowing:

* **Block size tightens the tail, not the median.** 512 -> 256 frames moved p50
  by 0.35 ms but halved p99 (29.8 -> 20.1 ms). The ~15 ms floor is WASAPI's own
  shared-mode buffer chain, not our block period.
* **Verify the device before trusting anything.** With virtual audio routing
  (SteelSeries Sonar, Voicemeeter) the default output may carry no game audio at
  all. `parry33 audio listen` shows a live meter and fails loudly on silence.

```bash
parry33 audio devices
parry33 audio listen --device 50
parry33 bench audio --what loop
```

## Phase 3 feasibility: the tell is learnable

Held-out AUC **0.949** (grouped CV, real control clips, shuffled floor 0.52),
using a 500-1000 ms window *before* the parry — 600 ms of lead. Generalises to
an unseen session (0.746 vs 0.153 for controls). Not a state gate: a control
one second from a real parry scores 0.233 against the parry's 0.795.

Full protocol and caveats in `docs/phase3-feasibility.md`. The blocker is a
7.3% per-window false-positive rate; temporal smoothing is the next thing to try.

## Next: Phase 2 — data collection

The ring buffer and capture loop are the pieces Phase 2 needs. The plan:

- hotkey-triggered clip dump: on a hit, write the preceding N frames from the
  ring to disk with capture timestamps
- a labelling pass marking `pre-hit / wind-up / impact / post-hit` per frame
- a dataset writer producing temporal stacks sized to whatever ROI Phase 3 wants

The key question Phase 2 has to answer, given the latency findings above: **how
many frames before impact is the telegraph reliably distinguishable?** That
number sets whether a frame-difference trigger is sufficient or a temporal model
is actually required. With +/- 70 ms of timing slack, the bar for "sufficient"
is much lower than it first appeared.

## Measured against the real game

Final configuration: 2560x1440 desktop @ 144 Hz, **HDR off**, game capped at
**60 fps**, bordered windowed.

```
capture efficiency   59.4 fps captured of 60 rendered  (~99%)
interarrival         p50 16.7 ms   p90 19.1 ms   jitter 2.3 ms
end-to-end (video)   ~33 ms = 22% of the 150 ms parry window
end-to-end (audio)   ~18 ms = 12% of the window
model timing margin  +/- 60 ms
```

Getting there took four configuration fixes, three of which were display or OS
settings rather than code:

| fix | before | after |
|---|---|---|
| **HDR off** | 30.5 fps, 13.0 ms jitter | 59.4 fps, 2.3 ms jitter |
| **Cap the framerate** | 19.1 fps uncapped | 59.4 fps at a 60 cap |
| **Desktop to 1440p** | DWM composing 33 MB/frame | 14.7 MB/frame |
| **poll not thread mode** | (reverted a bad default) | no change, but proven |

The framerate one is counterintuitive enough to restate: **more game fps gives
fewer captured frames.** Uncapped, the game saturates the GPU and Desktop
Duplication starves behind it. See `docs/game-settings.md`.
