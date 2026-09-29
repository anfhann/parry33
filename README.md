# parry33

A real-time auto-parry bot for *Clair Obscur: Expedition 33*, built as a machine
learning problem rather than a game hack: capture the screen and the system
audio, decide whether an attack is about to land, and inject the parry key
inside a 150 ms window.

**Current state:** it detects essentially every attack and presses at the wrong
instant. On the target boss, 4 of 14 presses are confirmed parries. On a simpler
enemy, held-out evaluation puts it at ~97% detection with 0.6 false presses per
minute. The detection problem is solved; the timing is not.

That gap is the interesting part, and most of this repository is the measuring
apparatus built to characterise it.

---

## What this project is actually about

Getting a bot to parry is easy to *start* and unexpectedly hard to *measure*.
Every obvious success metric turns out to be measuring something else:

- **The model scores AUC 0.947 and parries 48% of attacks.** Both numbers are
  correct. AUC is computed on sampled negatives; serving slides a window every
  10 ms. The gap between them is the whole problem, and it is invisible unless
  you deliberately evaluate the way you serve.
- **A "successful parry" has no direct signal.** The player's health only reveals
  attacks that *got through*, so every timing estimate is fitted to presses that
  failed. Confirming a success needs the *boss's* health bar, because a parry
  triggers a counter.
- **Labels derived from the player's own keypresses are circular.** "Not an
  attack" meant "no landed press nearby", which made attacks nobody reacted to
  invisible to every metric, and caused hard-negative mining to train the model
  to suppress real detections.

So the repo contains an OCR pipeline for reading health values off the screen, a
punching-bag protocol for collecting unbiased ground truth, negative controls,
and an evaluation harness that replays the game's actual mechanics — all in
service of answering "did that work?" honestly.

---

## Things I believed, then disproved

The documentation records these deliberately. Each looked like a result and was
not; each was caught by a specific check, and several were caught by the player
noticing something the analysis had not.

| Claim | How it died |
|---|---|
| Clash sounds identify successful parries | A run scored 14/14 landed; the in-game counter said 1. Clashes mark an attack *making contact*, parried or not — 100% of attacks that landed on an unparried player produced one |
| Hard-negative mining improves precision | 49% → 46% parried, and worse at matched false-alarm rates. The mined "false positives" included real attacks the player had whiffed |
| Longer audio context helps (52% → 56%) | The same configuration re-scored at 50% on a second run. Variance across random seeds exceeded every difference in the sweep; the grid was read before the noise floor was measured |
| A 150 ms refractory gives 77% parry rate | The metric counted a second press near a real parry as free. Simulating the actual mechanic — where that press whiffs and costs a 1500 ms lockout — collapsed it to ~50% |
| Bottom-centre motion predicts attacks (AUC 0.76) | 0.533 against real control clips. The original number was measured against within-clip baselines |
| The audio cue sits 553 ms before impact | Measured from the model's *own* score peak, so it inherited the model's bias. Two independent measurements put the correct press near the moment of contact |

The full record, including the reasoning that produced each mistake, is in
[`docs/phase4-learned-audio.md`](docs/phase4-learned-audio.md).

---

## How it works

```
system audio ──► log-mel ──► gradient-boosted classifier ──► "press now"
  (WASAPI)       141 ms         trained on ground truth        │
                 trailing                                       ▼
screen ────────► DXGI Desktop Duplication              SendInput (scancode)
  capture        HP readout OCR, boss HP bar
```

Audio drives the trigger, not video. It bypasses the display pipeline entirely:
~18 ms end-to-end against ~33 ms for video, with a tighter tail, and it is immune
to the framerate collapse that cost 3× capture throughput (see below).

The model answers *"given the trailing 141 ms of audio, should we press now?"* —
not *"is an attack happening?"*. That distinction matters: an earlier vision
model knew **whether** with 88% recall but its signal stayed high for 300–1300 ms
against a 150 ms window, so it knew whether and never when.

---

## Ground truth

The project's evaluation rests on reading the game's own numbers off the screen.

**Player health** is a 4-digit readout. Reading it took several failed
approaches, each of which produced plausible but wrong output rather than an
error:

- The readout **zooms** as a UI emphasis animation. Naive change-detection
  reported 31 attacks where the arithmetic allowed exactly 10.
- The digits turn **red** at low health. Stored as the green channel they
  vanished entirely below ~50% HP — at every threshold — while the white "/ max"
  beside them survived, silently losing half of every run.
- Downsampling the capture 2×4 to save disk turned 20 px digits into 5 px and
  made them unsegmentable. A *smaller region at full resolution* reads 100% of
  frames and costs four times less.

It now reads ~90% of frames and self-validates: health only falls during a
punching-bag run, so a misread announces itself as an impossible increase.

**Boss health** confirms successes. A parry triggers a counter, so the boss
losing health is proof the parry worked — the one thing the player's own health
cannot show. Stored as red-excess *and* luminance, because the bar animates
red → white dissolve → black and red-excess sees only the first stage.

**The protocol:** the player skips every turn, so they deal no damage and every
drop in the boss bar is necessarily a counter.

---

## Results

Measured leave-one-session-out across 15 recording sessions and 399 ground-truth
attacks, replaying the real mechanic (a landed parry is free and instant; a whiff
costs a 1500 ms lockout that can also cost the following attack):

```
detection (true recall)        97-100%
false presses                  0.6 / min
attacks parried, simulated     ~49%
timing error                   p50 +7 ms, p10 -49 / p90 +61 ms
```

On the target boss, live and armed: **4 of 14 presses confirmed as parries**,
evidenced by counters landing 2704–2820 ms after the press — a 116 ms band across
four events.

Latency, measured on real hardware:

```
audio: emit -> onset detected    p50 17.7 ms   p99 20.1 ms
video: end-to-end                ~33 ms  = 22% of the 150 ms window
capture efficiency               59.4 of 60 fps rendered  (~99%)
```

Latency is not the binding constraint, and constant latency is free — only jitter
(~2.3 ms) cannot be aimed away. That leaves roughly ±60 ms of timing margin.

---

## Three counterintuitive findings

**More game FPS gives fewer captured frames.** Uncapped, the game saturates the
GPU and Desktop Duplication starves behind it: 19.1 fps captured uncapped versus
59.4 fps at a 60 cap.

**HDR halves capture throughput.** With HDR on, Windows tone-maps the entire
desktop to SDR inside `AcquireNextFrame`. Turning it off took capture from
30.5 fps with 13.0 ms jitter to 59.4 fps with 2.3 ms jitter.

**There is no frame backlog.** Desktop Duplication emits only on desktop
*change*, and a frame arriving while you are busy is discarded, not queued.
Overrun one frame period and that frame is gone permanently.

Details in [`docs/game-settings.md`](docs/game-settings.md).

---

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[capture,dev]"
.venv/Scripts/python -m parry33 doctor
```

Windows only — it depends on DXGI Desktop Duplication, WASAPI loopback and
`SendInput`. `vgamepad` is optional and needs the
[ViGEmBus driver](https://github.com/nefarius/ViGEmBus/releases); without it the
`sendinput` backend still works.

Copy `config/local.toml.example` to `config/local.toml` and set your resolution
and keybinds. **The bindings are unverified defaults — check them in-game before
trusting a run.**

## Usage

```bash
python -m parry33 doctor                  # environment preflight
python -m parry33 bench capture --sweep   # ROI cost curve
python -m parry33 bench loop --trials 200 # closed-loop inject -> pixel latency
python -m parry33 audio listen            # live level meter; fails loudly on silence
python -m parry33 record --seconds 600    # aligned video + audio + input capture
python -m parry33 train --grunt           # train the attack-sound model
python -m parry33 trigger --seconds 300   # dry run (add --arm to inject)
```

`bench capture` needs something on screen to be moving — Desktop Duplication
produces nothing on a still desktop.

---

## Layout

```
src/parry33/
  clock.py              high-res waitable timer + spin; the sleep primitive
  trigger.py            the live trigger, and a record of two failed designs
  audio/
    capture.py          WASAPI loopback, SPSC ring with overrun counting
    grunt.py            the learned attack-sound model and its serving policy
    train.py            training from recorded sessions
  capture/dxgi.py       Desktop Duplication via bettercam
  input/sendinput.py    Win32 SendInput, scancode-based, prebuilt structs
  input/keywatch.py     low-level keyboard hook, filtered to a watchlist
  bench/loop.py         closed-loop inject -> pixel latency
scripts/
  record_hits.py        punching-bag capture: HP readout, boss bar, audio, input
  find_hits.py          OCR the health readout -> ground-truth attack times
  analyze_hits.py       true recall, cue timing, honest negatives
  train_gt.py           train on ground truth rather than keypress labels
  grade_presses.py      score the bot's presses from health, not from audio
```

---

## Design notes

**Scancodes, not virtual keys.** Games reading DirectInput or Raw Input look at
the scancode and ignore `wVk`. VK-only injection silently does nothing while
working perfectly in Notepad.

**Hold the key long enough to be sampled.** The game polls input once per tick; a
press released inside a single tick can be missed entirely. Default hold is 40 ms.

**Never block the detection loop for the hold.** A 40 ms blocking `tap()` costs
~6 captured frames at 144 Hz, so `press()` returns immediately and the release
goes to a scheduler thread.

**`ctypes` argtypes are not cosmetic.** `GetCurrentProcess()` returns the
pseudo-handle `(HANDLE)-1`; without a declared `restype` ctypes truncates it to a
32-bit `-1` and the call fails silently. This bug was live until a benchmark
happened to print `high_priority: False`.

**Do not measure "copy cost" by idling and timing a `grab()`.** That measures the
wait to the next vsync and converges on `frame_period / 2`. An earlier version of
the capture bench did exactly this and reported a confident, meaningless number.

---

## Known constraints

- Run the game **borderless / windowed-fullscreen**; exclusive fullscreen can
  bypass the compositor and starve Desktop Duplication.
- **HDR off.** See above.
- If the game runs elevated, UIPI silently drops `SendInput` from a
  standard-rights process.
- Protected/DRM content returns black frames by design.
- WASAPI loopback on an *idle* endpoint delivers nothing at all — not silent
  blocks, no blocks. "Nothing is playing" and "wrong device" look identical
  unless the block count is reported.

---

## Status

Working, incomplete, and honestly measured. Detection is solved; press timing is
not. The open questions and the current handoff are in
[`STATUS.md`](STATUS.md) and [`docs/phase4-learned-audio.md`](docs/phase4-learned-audio.md).

No recorded game footage is included — the capture data (frame arrays, audio) is
gitignored and was never committed. Regenerating any analysis requires new
recordings.

MIT licensed.
