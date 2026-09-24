# Phase 4 — the learned attack-sound model

State at the end of this phase: **~49% of real attacks parried, ~3.5 whiffs per
minute**, measured leave-one-session-out and confirmed live at 4/8 on the first
armed run. That is a working detector and not a working bot. The target is 19
parries in 20.

## What was replaced, and why

The two-stage trigger (vision gates, audio fires) failed in a way worth
recording, because the failure was not "the model was inaccurate".

Its audio stage was a spectral-flux onset detector. Flux answers *did a
transient just happen*, which in a combat mix is true about 3.4 times a second:
footsteps, impacts, the player's own attacks and the enemy's grunt all look
identical to it. So "the first grunt after the gate opened" is not a fixed point
in an attack — the press time was set by **when the gate happened to open**.

Measured gate-open → fire delays on a real fight:

```
+  86 ms   + 1318 ms   +  459 ms   +   39 ms
+1438 ms   +    4 ms   +   54 ms   + 1412 ms
+1272 ms   +   51 ms   +   49 ms   +  563 ms
```

Bimodal at ~4-90 ms and ~1270-1440 ms. The player described exactly this
("quite early or quite late, consistent each way, but random") before the log
was analysed.

A sensitivity sweep confirmed no threshold fixes it: on 728 s of labelled audio,
coverage fell from 91% to 47% as the rate came down from 3.4/s to 0.74/s. There
is no operating point where flux isolates the attack.

## The model

Given the trailing 141 ms of log-mel audio, *should we press now?* Positives are
presses that landed (a clash within 200 ms). All features are strictly earlier
than the decision instant, so the clash — which arrives ~32 ms **after** the
press — cannot leak in.

```
AUC 0.947                    shuffled-label floor 0.454
served, sliding every 10 ms  49% of attacks parried, 3.5 whiffs/min
timing error                 p50 +7 ms
```

**The AUC is not the result.** AUC is computed on sampled negatives; serving
slides continuously. That gap — 0.947 against 49% — is the same train/serve
mismatch that broke the first vision model, so it is now measured deliberately
rather than discovered in a fight. Any future number quoted for this system
should be the served one.

## The mechanic, which drives the policy

```
landed parry  -> instant reframe, 0 ms. Parry again immediately.
whiffed parry -> 1500 ms lockout; real attacks inside it are lost.
```

Correct presses are free; wrong ones are expensive and can cost the *next*
attack. So the policy is a high threshold with a short refractory: reluctant to
fire, able to fire again at once for combos.

An earlier version had this backwards — a permissive threshold with a 1200 ms
lock, on the mistaken belief that every parry locked out. That discarded real
attacks in combos, and 8% of the player's own consecutive parries are closer
together than 1200 ms.

It also invalidated a result. A metric that let a second fire near a real parry
count as free showed 77% at a 150 ms refractory. Simulating the actual mechanic
— where that second press whiffs and costs 1500 ms — collapsed it to ~50%. **Any
metric that does not price the whiff is measuring the wrong game.**

## Things that were tried and did not work

| idea | result |
|---|---|
| Hard-negative mining | 49% → 46%; loses at matched whiff rates too |
| Time-to-press regression head | 47% vs 50% for the classifier |
| Logistic regression | 6% — a linear model cannot do this |
| Small MLP | 41% vs 49%, though 7× cheaper to evaluate |

**Mining deserves the detail**, because the idea is sound and the label was not.
A mined negative is a window where the model fired and no *landed* parry was
nearby — but the player misses ~27% of their parries, so many of those windows
are real attacks that were whiffed. Mining trained the model to suppress correct
detections. It appeared to work at first (6.6 → 3.2 false/min) because that
measurement scored against landed parries too, baking in the same mistake.

Mining becomes useful the moment we have ground truth for "an attack occurred"
that does not route through whether the player parried it. See below.

## What actually helped

- **Near-miss negatives** (±120-400 ms from a real press: same sound, wrong
  instant). Without them the score plateaus across the whole grunt, so a
  confident detection still presses at an arbitrary point inside it. 28% of
  attacks were being lost that way.
- **Peak confirmation.** Firing on the first threshold crossing catches the
  sound's rising edge and locks out its true peak. Waiting 30 ms and taking the
  local maximum moved timing error from −26 ms to +2 ms. 60 ms overshoots.
- **Pinning OpenMP to one thread.** One row through 200 small trees measured
  4.62 ms because sklearn launches a thread team per tree; single-threaded it is
  2.98 ms. At 100 scores/second that is 30% of a core instead of 46%.

## Where the press should land

From 226 landed and 85 missed presses, measured against the model's score peak:

```
                    p25    p50    p75
landed presses      +8     +18    +30 ms

offset band      landed  missed  land rate
  0- 50 ms          195      47      81%
 50-100 ms           14       8      64%
100-150 ms            4       3      57%
150-200 ms            2       5      29%
```

So the acoustic landmark is good — 81% of presses within 50 ms of it landed. The
remaining ceiling is about *which* peaks we fire on, not *when* within them.

`--lead-sweep` exists to settle the residual offset empirically: it varies the
lead per press and scores each press by whether a clash followed, printing a
land rate per lead value. The clash discriminates on recorded human play (71-82%
of presses have one, not ~100%), but see the caveat at the end before treating
its output as ground truth.

## The open problem: circular labels

Every label used in this phase derives from the player's keypresses. Positives
are presses that landed; "not an attack" means "no landed press nearby". This
is circular, and it has two consequences:

1. It broke hard-negative mining, as above.
2. **Attacks nobody reacted to are invisible to every metric.** The "17% never
   crossed threshold" figure is a lower bound on a quantity we cannot see.

The fix is `scripts/record_hits.py` + `scripts/find_hits.py`: record a fight
without parrying at all, and read the player's HP off the screen. HP steps down
when an attack connects whether or not anyone reacted, so it yields a *complete*
list of attacks.

Two lessons from building it, both from looking at pixels instead of reasoning
about them:

- **The HP readout zooms** as a UI emphasis animation. Every zoom in and out
  looks like a change, so naive change-detection reported 31 attacks where the
  arithmetic (380 damage at 38 per hit) allows exactly 10. Segmenting individual
  glyphs and normalising each one separately is immune to it; normalising the
  whole readout is not.
- **Resolution beat area.** The first recorder kept the whole bottom strip
  downsampled 2×4, turning 20 px digits into 5 px — unsegmentable, 35% of frames
  readable, 154 phantom "values". A small region at full resolution reads 100%
  of frames and is four times cheaper on disk.

## Next

1. A punching-bag run on a weak enemy for the ground-truth attack list.
2. With it: honest recall, correct negatives, working hard-negative mining, and
   the true grunt→impact delay.
3. The parry window's position relative to *impact* and the total pipeline lag
   are engine/system constants — measured once, they apply to every enemy. The
   sound→impact delay is per-enemy and is what per-enemy profiling would capture.

## Caveat on the self-scoring

`--lead-sweep` grades each press by whether a clash followed within 250 ms. The
clash detector is a spectral-flux onset with an *adaptive* threshold — the same
property that made the grunt detector fire on inaudible transients once the mix
got quiet. During a bot run with many whiffs it could drift down and register
false clashes, which would inflate the land rate uniformly.

Evidence it is not broken: on recorded human play, 71-82% of presses have a
clash within 250 ms, not ~100%. It discriminates.

Evidence it is trustworthy in *this* use: not yet collected. The independent
check is free — the victory screen reports **Successful Parries**. Compare it
against the run's self-scored landed count. If they disagree materially, the
grader is drifting and the lead curve is measuring the detector rather than the
timing.

Note also that the player's own parries produce clashes, so the player must not
press E during a scored run, and that a clash arriving between two pending
presses is credited to the earlier one.

## Context length — RETRACTED, probably noise

The feature configuration had never been revisited: 141 ms of context in 40 mel
bands from 50 Hz to 16 kHz were first guesses. Swept against the same honest
metric (whole sessions held out, mechanic simulated):

```
frames    ms   mels        band   parried  whiff/min
    13   130     40    50-16000      52%       4.1
    20   200     40    50-16000      52%       3.4
    26   260     40    50-16000      54%       3.9
    34   340     40    50-16000      56%       4.0
    20   200     64    50-16000      54%       4.4
    20   200     40     50- 8000      55%      5.2
    20   200     40   200-20000      51%       4.3
```

That table reads as a monotonic rise with context length, 52% → 56%, and it was
briefly written up here as the biggest lever found. **It does not survive a
second run.** Extending the sweep re-evaluated the identical configuration
(34 frames, 40 mels, 50-16000 Hz) and got **50% instead of 56%**:

```
34 frames, 40 mels, 50-16000 Hz   ->  56%   (evaluated 4th in the first grid)
34 frames, 40 mels, 50-16000 Hz   ->  50%   (evaluated 1st in the second grid)
```

Same features, same data, same metric. The only difference is the position in
the grid, and therefore the state of the shared random generator that draws the
negative samples. **Run-to-run variance is larger than every difference in the
table**, so none of those comparisons is interpretable, and the extended sweep
shows no trend at all (50 / 53 / 51 / 52% across 340-800 ms).

The mistake was procedural, not arithmetic: the grid was read before the noise
floor was measured. A 4-point difference across configurations means nothing
until the same configuration is shown to reproduce within less than 4 points.
`--repeat N` now measures that spread directly, and any future configuration
claim must clear it.

The underlying hypothesis is still reasonable — 141 ms can only show the model
the impact, never the wind-up the player reads by eye — but it is untested, not
supported.

**Longer context costs no latency.** The window is trailing: it ends at the
decision instant either way. It costs feature width, and therefore a little
compute per score.

Adopting a new configuration requires retraining, because the geometry is baked
into the model. `GruntStream` now takes its shape from the saved bundle and
raises if the model's `n_features_in_` disagrees, so a model and a serving
window can no longer silently desynchronise.

### The noise floor, measured

`sweep_features.py --repeat 5` re-scores one configuration across five seeds:

```
13 frames (520 features)   52 52 53 53 52   mean 52.2%, sd 0.5%
34 frames (1360 features)  50 54 53 55 54   mean 53.3%, sd 1.8%

paired differences         -2 +2  0 +2 +2   mean +0.8, t ~ 1.0
```

The 1.1-point gap is not significant, which confirms the retraction above.

The more useful number is the **variance**, which is 3.6x higher for the longer
window. That is an overfitting signature, not a coincidence: 34 frames is 1360
features estimated from 351 positives, against 520 for 13 frames. Longer context
is not failing because the wind-up carries no information — it is failing
because there is not enough labelled data to use it.

That inverts the priority. Tuning the feature geometry against 351 positives
cannot resolve differences of a few points, so **more labelled attacks are the
prerequisite for any further feature work**, not a nice-to-have. It also raises
the value of ground-truth runs specifically: a punching-bag run labels every
attack, including the ones the player never reacted to, so it grows the positive
set far faster than more human play does.

Note also that two implementations of "the same" evaluation (`eval_grunt.py` and
`sweep_features.py`) report 49% and 52% for the baseline. They differ only in how
negatives are drawn. Treat any single evaluation number as +/- a few points.

---

# Phase 5 — ground truth, and two graders that were wrong

## Where it stands

The bot parries Julien. **4 of 14 presses confirmed** on the last two armed runs
(29%), evidenced by counters landing 2704-2820 ms after the press -- a 116 ms
band across four events. Every earlier grading method reported zero, because
every earlier grading method was measuring the wrong thing.

## The two wrong graders, both caught by the player, not by analysis

**Clash scoring said 14/14 landed when the victory screen said 1.** A clash marks
an attack making CONTACT, parried or not: measured on punching-bag runs where the
player parried nothing, 100% of attacks that landed still produced a clash. This
also contaminated the original training labels, which defined a "landed parry" as
a press followed by a clash -- so the model was trained to fire when an attack
was happening, not at the instant that parries it. That is exactly a model that
lands close but early, which is what the player reported.

**The training anchor was circular.** It sat at 553 ms before impact, taken from
where the model's own score peaked -- so it inherited the bias above. Correcting
it to 180 ms was measured, but overshot in the other direction: 180 ms before the
HP drop is essentially the moment of contact, and a parry must intercept the blow
rather than coincide with it. The truth is between the two and is still unknown.

## The instrument that was missing

The player's own HP can only ever reveal FAILURES -- a parried attack leaves no
mark on it. So every measurement of "where should we press" was fitted to presses
that did not work.

The boss's health bar closes it (the player's idea). A successful parry triggers a
counter, so the boss losing health is proof the parry worked. Both the recorder
and the trigger now capture it, as red-excess AND luminance, because the bar
animates red -> white dissolve -> black and red-excess sees only the first stage.
The bar is segmented, so it confirms some successes rather than all: a confirmed
parry is certain, an unconfirmed one is unknown.

PROTOCOL THAT MAKES IT CLEAN: the player skips every turn. Then they deal no
damage, so every drop in the boss bar is a counter. Without that, the player's own
attacks are indistinguishable from counters.

## What is now measurable, per run

    player HP    attacks that got through        (failures)
    boss HP      counters                        (successes)
    audio        the cue both player and bot react to
    numpad 1-6   which of Julien's attacks each press was aimed at

## Julien

    1 Left Jab   2 Long Left Jab   3 Left Jab Combo (3 hits)
    4 Right Jab Combo (2)   5 Rotating Jab   6 Jump Attack   (+ an uppercut)

Every observed fight opens with Long Left Jab. Combo hits land 1128 ms apart
(p25 1050, p75 1324), which is why the 250 ms refractory is safe and why the old
1200 ms one would have blocked every second and third hit of every combo.

Per-attack press offsets were consistent (-88, -100, -106 ms relative to the HP
drop) across attack types, so one lead may serve all of them. The jump attack
does draw presses -- 4 of 6 -- contradicting the impression that it was being
ignored; they simply all failed.

## Open

* Two boss-HP drops had no press within 6 s. If a burn or damage-over-time effect
  is running, the success detector needs to filter it.
* 4 confirmed hits against 10 misses is too thin to fit a timing correction on.
  More runs at a FIXED --lead-ms, turns skipped, build that up at ~5 presses per
  30 s run.
* The counter lag measured 1961 ms on the player's fight and ~2790 ms on the
  bot's. Same bar. Recheck before porting the constant between runs.
