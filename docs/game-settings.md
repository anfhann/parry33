# Game settings for a vision bot

These are **not** normal performance settings. Several standard "free fps"
choices actively damage the signal we detect on. Lock these once and never
change them — every settings change invalidates collected training data.

## The rule

Cut anything that costs GPU time but carries no telegraph information.
Keep anything the attack tell might live in. Never blend frames together.

For any option not listed below, the test is:

> **Does it change pixels when the game state didn't?**

- **Static** distortions (chromatic aberration, vignette, colour grading,
  tonemapping) are learnable and low priority. Disable if free.
- **Dynamic global** effects (auto-exposure, damage vignette, lens flare, film
  grain) are actively dangerous. They manufacture brightness changes
  uncorrelated with attacks — the exact false-positive class that makes the bot
  parry at random, often just *after* a hit lands.

## Before anything else: turn HDR off

This is a **Windows display setting**, not a game setting, and it matters more
than every graphics option combined. `Win+Alt+B` toggles it.

Desktop Duplication's legacy API returns an 8-bit BGRA surface (bettercam
hardcodes `DXGI_FORMAT_B8G8R8A8_UNORM` for its staging texture). With HDR on,
Windows tone-maps the entire desktop down to SDR *inside* `AcquireNextFrame`,
every single frame.

Measured on this rig, 2560x1440, game rendering 60 fps:

| | HDR on | HDR off |
|---|---|---|
| capture rate | 30.5 fps | **59.4 fps** |
| jitter (p90-p50) | 13.0 ms | **2.3 ms** |
| p99 interarrival | 57.7 ms | **22.6 ms** |
| share of a 150 ms parry window | 58% | **25%** |

HDR halved throughput and inflated jitter 5.6x. `parry33 doctor` now fails
loudly if it is on.

HDR is genuinely better for *playing* games on a capable display — this is a
real tradeoff, not a bad setting. Turn it on when you are not running the bot.
But note that HDR-on capture goes through the tone-mapping curve and HDR-off
capture does not, so **the pixels differ**. Collect all training data in one
mode and stay there.

## Settings

| Setting | Value | Why |
|---|---|---|
| **Effects / particle quality** | **HIGH** | Telegraphs live here. Low presets cut particle counts and post-processing and can remove the cue entirely. Non-negotiable. |
| **Temporal upscaling** (DLSS / FSR2 / TSR) | **OFF if possible** — see below | Accumulates samples across frames. A change in frame N smears into N+1 and N+2, blurring the temporal edge a reactive trigger fires on. Adds effective latency. |
| **Anti-aliasing** | **off / lowest non-temporal** | Same reason. FXAA/SMAA are spatial and harmless; TAA is temporal and is not. |
| **Motion blur** | **OFF** | Deliberately smears fast motion. Fast motion is the telegraph. |
| **Depth of field** | OFF | Blurs off-focus regions, which may include the enemy winding up. |
| **Auto-exposure / eye adaptation** | **OFF** | The worst offender. Continuously drifts global brightness from scene content, so the same telegraph looks different depending on what preceded it and the detector baseline never settles. |
| **Vignette** | **OFF** | Static vignette merely attenuates edge brightness. But many games *pulse* one on damage or low health — a large global brightness change uncorrelated with any attack, which a threshold trigger reads as a telegraph, and which fires right after you were hit. |
| **Film grain** | OFF | Per-frame random noise straight into a frame-difference trigger. |
| **Lens flare** | OFF | Bright artifacts that move with the camera. Pure false-positive generator. |
| **Chromatic aberration** | OFF | Static, so it adds no per-frame noise and is learnable. Off because it is free and it smears edges hardest at the periphery, where the camera-zoom signal lives. |
| **Bloom** | default | The one helpful effect: it amplifies bright telegraph flashes and can spread a cue into the ROI from just outside it. Mildly noisy too. Not worth tuning. |
| **Resolution** | **native 720p-1080p** | Native low beats upscaled high. We downsample for the model anyway. |
| **Textures** | LOW | Telegraphs are not texture detail. |
| **Shadows / reflections / foliage / volumetrics** | LOW | Pure GPU cost, no cue value. |
| **VSync** | OFF | Removes a queued frame of latency. |
| **Frame cap** | **the knee -- measure it** | NOT the max. Uncapped saturates the GPU and starves Desktop Duplication: measured 59.4 fps captured at a 60 cap vs 19.1 fps uncapped. Find the highest cap that still captures ~99% of the cap. |
| **Display mode** | **borderless windowed** | True exclusive fullscreen can bypass the compositor and starve Desktop Duplication. See README. |

## Audio settings

The audio path detects attacks by spectral transients, so every non-attack
transient in the mix is a false positive.

| Slider | Set to | Why |
|---|---|---|
| **Music** | **0** | Combat music is dense with transients -- drum hits, stingers, swells. Spectral flux fires on all of them. Measured ~1.5 onsets/second in combat with music on; most of that is the soundtrack. |
| **SFX** | high | The attack whoosh and impact live here. This is the signal. |
| **Voice / dialogue** | high | Community guidance names *screams* and *grunts* as the tell. If the game buckets enemy vocalisations under Voice, muting it deletes the cue. Do not reflexively mute everything that is not SFX. |
| **Ambience** | low | Steady background noise raises the detector adaptive floor, shrinking the margin a real onset has to clear. |

**Lock the mix like the graphics settings.** The detector learns a baseline and
threshold against whatever the mix is. Changing music from 0 to 50 later
invalidates every recording made before it.

Check the effect directly -- onset rate should drop noticeably with music off:

```
parry33 audio listen --seconds 30
```

## When upscaling cannot be turned off

UE5 titles often refuse to disable upscaling/AA unless the game is running at
the desktop's native resolution. If the menu greys out "off", the fix is usually
not to accept it:

**Set the Windows desktop resolution to the resolution you want to play at.**
Then the game is native and the option unlocks. Monitor-side scaling afterwards
is irrelevant to us — Desktop Duplication captures the desktop framebuffer
*before* the signal reaches the panel, so a monitor stretching 1440p onto a 4K
panel costs us nothing. It looks soft to you; our pixels are clean and native.

If it still cannot be disabled, rank by how much each blends across frames:

| | Option | Note |
|---|---|---|
| 1 | **FSR1** | Purely spatial, zero temporal accumulation. Outright winner if offered. Poor image quality is irrelevant here. |
| 2 | **DLSS, highest preset** | DLAA if available, else Quality. Motion-vector driven with learned history rejection — least ghosting of the temporal methods. |
| 3 | TSR | |
| 4 | FSR2/3 | |
| 5 | TAA | Crudest accumulation, worst ghosting. |

**The preset matters more than the choice.** Quality renders ~67% internal,
Performance ~50%. Higher internal resolution converges in fewer frames after a
sudden change. Never use Performance or Ultra Performance.

Why this is a smaller handicap than it looks:

- Most of the cost is a **constant** offset, and constant latency is free — aim
  earlier and it cancels. Only the variable part of convergence hurts.
- Temporal accumulation is a low-pass filter over time: it smears the onset
  (bad for timing) but suppresses per-frame noise (good for detection
  reliability). With ±62 ms of margin, missed detections cost far more than a
  few ms of smear. The trade is near-neutral.
- It is **measurable**: `bench loop` reports "frames from inject to detect".
  Smear shows up there as a higher frame count. Quantify rather than guess.

**Whatever you pick, lock it.** A consistent handicap is far less damaging than
an inconsistent one — a model can learn around predictable ghosting, but a
settings change mid-dataset invalidates everything collected before it.

## Why not just maximise fps

Margin gained across the whole range, against a 150 ms parry window:

| game fps | period | 2-frame latency | % of window | model margin |
|---|---|---|---|---|
| 40 | 25.0 ms | 50.0 ms | 33.3% | ±62.5 ms |
| 60 | 16.7 ms | 33.3 ms | 22.2% | ±66.7 ms |
| 90 | 11.1 ms | 22.2 ms | 14.8% | ±71.7 ms |
| 120 | 8.3 ms | 16.7 ms | 11.1% | ±72.5 ms |
| 144 | 6.9 ms | 13.9 ms | 9.3% | ±72.9 ms |

40 → 144 fps buys only ~10 ms of timing margin. **Latency is not the reason to
want framerate.**

The real reason is **sample density**: a 300 ms wind-up is 12 frames at 40 fps
and 36 at 120 fps. Three times the temporal detail for the classifier, and the
resolution needed to answer Phase 2's central question — how many frames before
impact the telegraph becomes separable.

But that table assumes we actually CAPTURE those frames. We do not.

## Cap the framerate. This is the biggest lever, and it runs backwards.

**More game fps gives us FEWER captured frames.** Measured on this rig, 1440p,
HDR off, same scene:

| game cap | captured | jitter (p90-p50) | share of 150 ms window |
|---|---|---|---|
| **60** | **59.4 fps** (~99% of cap) | **2.3 ms** | 25% |
| uncapped | 19.1 fps | 23.1 ms | 96% -- unusable |

Uncapped, the game saturates the GPU and Desktop Duplication's per-frame
AcquireNextFrame + CopyResource starves behind it. Capped, the game leaves
headroom and DDA is serviced immediately.

So the target is **not** maximum framerate, and not "diminishing returns around
90-120". It is the **highest cap that still captures ~99% of what the game
renders**. Past that knee, throughput collapses and jitter explodes.

Find it by bisection: cap, measure, compare captured fps against the cap.
Ratios near 1.0 are healthy; below ~0.9 means you are past the knee, come down.

Re-measure once inference is added -- that steals GPU too, so expect the knee
to move down.


## After changing anything here

1. Re-measure: `parry33 bench capture` against the live game.
2. Update `[game] fps` in `config/local.toml` to the measured value.
3. Update `[display] refresh_hz` if the panel mode changed.
4. **Discard training data collected under different settings.** Effects quality
   and upscaling change pixels materially; a model trained across a settings
   change learns the settings, not the telegraph.

## ROI note

Screen resolution changes invalidate absolute ROI coordinates, and borderless
windowed means the window can move. ROI should be derived from the game window's
client rect at runtime rather than hardcoded — see the window-tracking item in
Phase 2.
