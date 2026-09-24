# Phase 3 feasibility: yes, the tell is learnable

Run overnight on the two clip-bearing sessions. **Answer: yes, and by a clear
margin.** More playtime is now worth collecting; it was not before this.

## Headline

Trained on run `11dc0951236fc` (52 landed parries, 96 control clips), grouped
5-fold CV by contiguous time block, scored only against real control clips:

| | AUC |
|---|---|
| **held-out (logistic, L2)** | **0.949** |
| random forest | 0.953 |
| shuffled-label floor | 0.52 |

At threshold 0.70: **81% recall, 7.3% false-positive rate.**

The features are a 6x8 spatial grid of inter-frame motion over 6 time bins,
stored three ways (absolute level, share of that bin's total, per-bin total) --
582 numbers per clip. No neural network involved.

## Why this one is believable when three earlier findings were not

Every claim in `phase2-findings.md` that later collapsed did so because the
negatives were wrong. This was built to remove each of those failure modes:

* **Real controls only.** Negatives are control clips recorded at random times,
  never a parry clip's own early frames. The retracted motion-share feature
  scored 0.762 against the latter and 0.533 against the former.
* **Combat-only controls.** Controls are recorded every 7 s regardless of game
  state, so many land in menus. Restricting negatives to controls within 10 s
  of a real parry changes nothing (0.956 vs 0.963) -- so the model is not just
  detecting "combat is happening".
* **Same session for positives and negatives.** Run 1 has 54 positives and zero
  controls; mixing it with run 2's controls would let a model separate them by
  session lighting alone.
* **Grouped by time, not shuffled.** Clips seconds apart are near-duplicates.
  Folds are contiguous time blocks so they cannot straddle a split.
* **Shuffled-label floor computed.** 0.52, so 0.949 is not small-sample drift.

## It is a per-event detector, not a state gate

The worry was that it detects "the enemy's turn is underway" rather than a
specific attack. Scoring control clips by their distance from a real parry:

| clip | mean score |
|---|---|
| **the parry itself** | **0.795** |
| control 0-2 s away | 0.233 |
| control 4-7 s away | 0.088 |
| control 11-16 s away | 0.058 |

A control sitting one second from a real parry still scores 0.233. The model
localises to the specific attack. There is a mild proximity effect
(corr -0.275), which is expected and harmless.

## It works early

Sweeping the feature window (controls <15 s, best model):

| window before the press | AUC |
|---|---|
| -600 .. 0 ms | 0.953 |
| -600 .. -300 ms | 0.963 |
| -1000 .. -500 ms | 0.939 |
| **-1000 .. -600 ms** | **0.938** |

It does not need late frames, so it is not reading the attack already landing.
600 ms of lead against a 150 ms window and an 18 ms injection path is enormous
margin -- and earlier than the -371 ms the player reports reading.

**Caveat:** clips only span -1000 ms, so we cannot test earlier than that. The
true onset of the signal is unknown and may be earlier still.

Excluding combo follow-ups (a parry within 1.5 s of another) *improves* results
to 0.958-0.975, so it is not cheating off the "PARRIED" text or damage numbers
left on screen by a previous hit.

## It generalises across sessions

Trained on run 2, scored run 1's 49 landed parries -- a different session,
different fights, never seen in training:

| | mean score |
|---|---|
| run 2 held-out positives | 0.795 |
| **run 1 positives (unseen session)** | **0.746** (median 0.865, 76% above 0.5) |
| run 2 controls | 0.153 |

Only mild degradation across sessions. This is the strongest single result here.

## What is not yet good enough

**False-positive rate.** 7.3% per window is far too high for a detector running
continuously -- at one window per 100 ms that is a spurious trigger every ~1.4 s.

The obvious fix is temporal smoothing: require N consecutive windows above
threshold before firing. Isolated false positives are uncorrelated in time while
a real tell persists across many windows, so this should cut FPR by an order of
magnitude at a small cost in latency. Untested.

**Sample size.** 52 positives from one session for training. The cross-session
result is reassuring but a second training session would be much better.

**Representation.** Hand-designed motion grids. A small CNN on the raw frames
would likely do better and is now clearly worth building.

## Recommended next steps

1. **Temporal smoothing.** Cheapest and highest-leverage. Turn the per-window
   score into a firing rule and measure end-to-end false-alarm rate per minute.
2. **Widen the ROI.** The current 960x540 box is 14% of a 1440p screen and the
   `roi_check.png` montage shows the enemy frequently small, partial or nearly
   out of frame. Free per the Phase 1 measurements. Do this before collecting
   more data, since it changes the input.
3. **Then collect more.** Now justified. With the wider ROI, target another
   100+ landed parries. Control clips are automatic.
4. **Then a CNN**, scored against exactly the same controls and the same
   grouped-CV protocol. The bar to beat is 0.949.

## Reproducing

    from parry33 import learn
    X, y, t = learn.build(Path("runs/11dc0951236fc"))
    g = learn.time_groups(t, 5)
    learn.evaluate(X, y, g)
    learn.shuffled_control(X, y, g)
