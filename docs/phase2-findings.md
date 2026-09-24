# Phase 2 findings — audio

Measured from one 600 s session: 8 boss attempts, 388 s in combat, 61 parry
presses, 20 confirmed landed. Run `11887513fbc44`.

All of it is offline analysis of `audio.wav` + `events.jsonl`, which is the
point of recording raw audio: every conclusion below was re-derived after the
fact without re-recording.

## The structure, in the player's words

> grunt, space, parry window

Confirmed and quantified. There are **two distinct audio events**, not one:

| event | offset from a landed press | spectrum | IQR |
|---|---|---|---|
| **grunt** (the cue) | **−107 ms** | vocal, 150–2500 Hz | 70 ms |
| **clash** (the result) | **+32 ms** | metallic, 6–24 kHz | 38 ms |

The original detector — spectral flux summed from 200 Hz to Nyquist — only ever
found the clash, because the clash is the brightest thing in the mix. The grunt
was in the recording the whole time; the filter discarded it.

## Consequence: the trigger is scheduled, not reactive

The grunt precedes the window by ~110 ms, and its spread (IQR 70 ms) is
*narrower than the 150 ms window*. So the design is:

    detect grunt  ->  wait a learned delay  ->  press

not "detect something and press as fast as possible". This inverts the whole
latency argument from Phase 1: with ~110 ms of scheduled slack against an 18 ms
audio path, **latency has stopped being the constraint**. Delay accuracy is.

Do not spend further effort shaving milliseconds off capture or injection.

## What works

* **Recall.** A vocal-band (150–2500 Hz) flux detector at a permissive threshold
  finds the grunt before **19 of 20** landed parries.
* **The clash is a precise auto-label.** IQR 38 ms — tighter than anything else
  measured. It marks "a parry resolved here" with no human input, which removes
  the manual labelling bottleneck: find clashes, search backwards.

## What does not work

* **Precision.** Catching 19/20 grunts costs 2.57 detections/sec, against a real
  attack rate near 0.05/sec. Roughly **2% precision**. A bot on this alone would
  parry continuously.
* **Onset magnitude as a discriminator.** Attack onsets: score p50 22.8.
  Background: 21.3. Best single threshold gives 100% recall at 4% precision.
  Useless.
* **Spectral shape alone.** Nearest-centroid on 24 log-bands: 80% recall,
  6% precision. Better than magnitude, still not usable.

## Correction: the landed-vs-missed contrast was inflated

The first session reported landed parries clustering at IQR 52 ms against
missed at IQR 1128 ms, and called that proof the grunt tracks good timing.

It was an artifact. `E` was bound to BOTH parry and skill-confirm, so the
"missed parries" were largely menu selections at arbitrary times -- a control
group of random noise, which any real signal beats.

A second session with `F` for attacks (so every `E` was a genuine parry) gives:

| | contaminated | clean |
|---|---|---|
| landed IQR | 52 ms | 130 ms |
| missed IQR | 1128 ms | **91 ms** |

The grunt is still real and still precedes presses. It does **not** separate good
timing from bad. Do not build a trigger on the assumption that it does.

Lesson: when a control group is defined by "the thing the player did that did
not work", verify the player was actually attempting the same action.

## Dead ends, recorded so they are not retried

* **Adaptive thresholds self-normalise the onset rate.** Turning game music off
  did not change onsets/sec at all (1.53 → 1.54). Removing the music lowered the
  noise floor, the adaptive threshold followed it down, and it simply fired on
  quieter things. Tuning sensitivity does not reduce false positives; it moves
  which sounds trip it.
* **A random-baseline spectro-temporal average proves nothing.** Comparing
  attack windows against random points across the session showed every band
  elevated at every offset — because most of the session is quiet menu time. Any
  baseline must be drawn from *in-combat, non-attack* windows.
* **"Faster than human reaction time" does not identify a sound.** An onset
  60 ms before a keypress cannot have *caused* that keypress, but the player may
  be anticipating from a visual cue with the finger already committed. Both can
  be driven by the same animation. This reasoning led to misattributing the
  grunt as the clash; the histogram of onsets around presses settled it instead.

## Open questions

0. **How many attack patterns are there?** The player initially reported 3-4,
   then revised to 7 after closer observation. At 32 landed parries that is ~4
   examples per pattern -- far too few. Sample size is now the binding
   constraint on everything else.
1. **What gates the trigger?** Precision is the blocker. Two candidates:
   classify the grunt spectrally (needs far more than 20 examples), or gate on
   vision — the player reports reading the enemy's shoulder well before the
   grunt, which would make vision the earliest available cue.
2. **Is the gap per-attack?** The 19 delays group loosely (50–51, 71–93,
   106–131, 155–161, 190–255) and the boss has four combos with different
   rhythms. If each has its own gap, the trigger needs to classify the attack
   before choosing a delay. Not resolvable at n=19.
3. **How early is the visual telegraph?** Never measured. 61 parry contact
   sheets exist in the run and none have been reviewed.

## RETRACTED: the bottom-centre motion feature

An earlier version of this document reported that bottom-centre motion share
rises at -400 ms in 88% of clips at 3.7 sigma, matching the player's reported
-371 ms tell. **That result does not survive a proper control.**

| negatives used | AUC |
|---|---|
| each clip's own -1000..-600 ms frames | 0.762 |
| **independent control clips at random times** | **0.533** |

0.533 is chance. The early frames of a parry clip are a *lull that precedes an
attack*, not representative combat, so the feature was measuring "combat lull
vs combat action", not "attack incoming vs not".

This was the third finding this session invalidated by a proper negative set,
after the audio landed-vs-missed contrast and the grunt/clash attribution. The
common failure was identical every time: **measuring the moments something
happened, without measuring the moments it did not.**

Control clips (recorded automatically at ~7 s intervals) are the structural fix.
Any future feature or model is scored against them, and nothing gets reported
without that comparison.

The original (retracted) numbers follow, kept so the mistake is legible:

## Retracted detail: the visual tell is measurable

From 54 landed parries with raw frame clips (run `11c904d5ed148`), computing the
share of inter-frame motion falling in the bottom-centre of the ROI:

| offset | bottom-centre motion share |
|---|---|
| -800..-500 ms | 0.72-0.86 (flat baseline) |
| **-400 ms** | **1.07  <- rise begins** |
| -300 ms | 1.15 |
| -200 ms | 1.28 |
| -100 ms | 1.44 |
| 0 ms | 1.41 |

Present in **45/51 clips (88%)**, mean rise **3.7 sigma**.

**None of this survived control clips. See the retraction above.**

The rise begins at -400 ms. The player independently reported reading the tell
at **-371 ms**. That is the same event, measured two ways.

Two caveats before trusting it:

* **Precision is unmeasured.** Clips were only ever saved at keypresses, so we
  have no idea what this feature does during ordinary combat. The audio grunt
  had 91% recall and was still useless at 2% precision, and that was only
  visible once there was something to compare against. Control clips at random
  intervals are now recorded for exactly this.
* **Framing is not consistent.** Averaged-frame edge energy is 0.21x that of
  individual frames, so the enemy occupies a different screen position in every
  clip. No fixed sub-ROI will work; a detector must be translation-tolerant.

## What global features cannot do

Whole-ROI inter-frame difference finds **nothing**. Averaged over 79 landed
parries, motion *decreases* before the press, bottoming at 0.67x baseline around
-200 ms, and only rises after impact. The tell is a localised limb movement on a
character that moves around the frame; no global statistic captures it.

The bottom-centre motion *share* works precisely because it is a ratio -- it
measures where motion is, not how much.

## The patterns are not the unit of work

The player's taxonomy: short left jab, paused left jab, short left jab into
combo, left jab curl-around, a right-hand attack, a jump attack -- around seven.

But **"short left jab" and "short left jab into combo" are the same jab**. They
diverge only after the first hit lands. That means the information distinguishing
them does not exist at the moment the parry decision must be made -- not for the
player, and not for any detector we could build.

So do not classify patterns up front. The first hit of both is identical, so its
parry timing is identical. What differs is whether a second attack follows, which
is a separate detection made *after* the first parry resolves.

    per-hit loop:  detect hit incoming -> parry -> did another follow? -> repeat

not

    classify pattern -> look up its full timing sequence -> execute

This degrades gracefully: mispredicting a continuation costs one hit, not the
whole exchange. It also collapses seven patterns into a much smaller set of
distinct *first-hit* timings.

## Do not label during play

Attempted and abandoned. Seven patterns is more mapping than anyone can recall
mid-fight, and the player correctly predicted the failure mode: "I'll muddy the
left jab and the left jab into combo because it's the exact same left jab and I
get excited and type in short jab."

Labels produced under time pressure are worse than no labels, because they get
trusted. Label offline from the contact sheets instead, where there is no clock.

Better still, avoid needing them: landed parries are already auto-labelled from
the clash, and the grunt-to-press delays cluster on their own.

## Dataset as of the end of Phase 2

| | count |
|---|---|
| landed parries with raw clips | 107 |
| control clips (random times) | 103 |
| parry success rate | 64% -> 71-74% as the player adapted to keyboard |

Clips are 90-ish frames of 240x135 uint8 grayscale, spanning -1000..+500 ms
around the event. Landed parries are auto-labelled from the clash; no manual
marking is involved.

This is a balanced, controlled dataset. Every hand-crafted feature tried against
it has failed:

| feature | AUC vs controls |
|---|---|
| audio grunt onset | 2% precision at 91% recall |
| global inter-frame motion | no signal (motion *falls* before a parry) |
| bottom-centre motion share | 0.533 |

Three independent failures is sufficient evidence that a learned model is
required, not a threshold on a hand-designed statistic.

## Method notes for the next session

* **20 landed parries is not enough.** Every analysis here ran at the edge of
  significance. Target 100+.
* **Drop the `+` key.** Auto-label from clash detection instead.
* **Fix the fight markers.** A single toggle key is phase-ambiguous: one missed
  press inverted an entire session, and the recorded "fights" were the menu
  gaps. Use two distinct keys, or infer phase from which spans contain the
  keypresses.
* **Log both bands separately.** Vocal and bright flux are different events and
  collapsing them into one detector is what hid the grunt for a whole session.
* **Store `audio_start_ns` in meta.** The WAV had to be re-aligned by scanning
  for the offset that put onset timestamps on actual energy.
