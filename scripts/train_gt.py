"""Train the attack-sound model on GROUND TRUTH instead of keypress labels.

Until now every label came from the player's own presses: positives were presses
that landed, and "not an attack" meant "no landed press nearby". That definition
is circular, and it cost us twice -- attacks nobody reacted to were invisible to
every metric, and hard-negative mining actively hurt (49% -> 46%) because it
trained the model to suppress real attacks the player had whiffed.

The punching-bag runs break the circle. The player never parried, so:

  POSITIVES  come from the HP readout: every attack that landed, including the
             ones nobody reacted to. Anchored at impact - ANCHOR_MS.
  NEGATIVES  are moments the model would have pressed with no attack incoming.
             Because the player never parried, "maybe it was an attack he
             whiffed" cannot excuse them. These are the honest hard negatives
             mining always needed.
  ALSO       the timestamps of our own injected turn presses, which are the
             sounds most easily confused with an enemy's.

THE ANCHOR WAS WRONG AND IS NOW MEASURED. It was 553 ms, taken from where the
model's own score peaked before impact -- circular in exactly the way keypress
labels were, and it taught the model to fire about 350 ms early. Two independent
measurements, neither involving the model, agree on the truth:

  * Align 399 attacks on the HP drop and look at raw audio consistency: it peaks
    at -170 ms. That is the blow itself; the readout lags it.
  * Align 51 human presses that parried 41 times: the cue they react to sits at
    -240 ms, with the press about 240 ms later -- one reaction time.

Both put the correct press near the moment of contact, so ANCHOR_MS is 180.

TWO KINDS OF RUN feed this. Punching-bag runs (player never parries) give
impact-anchored positives and proven negatives. Runs where the player parries
well give almost no HP drops -- three Julien fights yielded one each against 41
successful parries -- so there the PRESSES are the label.

    python scripts/train_gt.py --runs E:/parry33
    python scripts/train_gt.py --runs E:/parry33 --save
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod          # noqa: E402
from parry33.audio import grunt as G          # noqa: E402
from parry33.audio import train as gtrain     # noqa: E402

# Positives sit this far before the HP DROP. Not 550, which was inherited from
# the old model's own score peak and was therefore circular. Measured two ways
# and they agree: aligning 399 attacks on the HP drop, audio consistency peaks
# at -170 ms (the blow itself -- the readout lags it), and aligning 51 human
# presses that parried 41 times, the cue they react to sits at -240 ms with the
# press ~240 ms later. Both put the correct press at roughly the moment of
# contact, not half a second before it.
ANCHOR_MS = 180.0
JITTER_MS = 30.0         # spread positives slightly so timing is not memorised
POS_COPIES = 3
NEAR_MS = (150.0, 450.0)  # near-miss negatives: right sound, wrong instant
NEAR_PER_POS = 4
FAR_PER_POS = 6
FAR_MIN_MS = 700.0
HIT_MS = 120.0           # a press this close to the anchor counts as on-target


def load_press_runs(root: Path, max_drops=5):
    """Runs where the player parried well: the PRESSES are the ground truth.

    On a boss the player stops almost everything, HP drops are rare -- three
    Julien fights yielded one drop each against 41 successful parries -- so
    impact-anchored labels have almost nothing to learn from. The presses do:
    the victory screen scored 41 of 51 as successful, so a press is a
    demonstrably correct moment about 80% of the time, and it is not derived
    from any model.
    """
    out = []
    for d in sorted(root.glob("hits_*")):
        pp, hj = d / "presses.npy", d / "hits.json"
        if not (pp.exists() and (d / "audio.wav").exists()):
            continue
        press = np.load(pp)
        if len(press) < 5:
            continue
        drops = (len(json.loads(hj.read_text())["hits"]) if hj.exists() else 0)
        if drops > max_drops:
            continue                      # a punching-bag run, handled elsewhere
        meta = json.loads((d / "meta.json").read_text())
        t0 = meta.get("audio_start_ns")
        if t0 is None:
            continue
        with wave.open(str(d / "audio.wav")) as w:
            if w.getframerate() != G.SR:
                continue
            ch = w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw).astype(np.float32) / 32768.0
        # Two kinds of press are not parries and must not become positives.
        # The first press of a fight lands 7-9 s in and is the start confirm (E
        # doubles as menu confirm). And a press followed by an HP drop failed by
        # definition. Removing both brings the count to 10/16/18 against victory
        # screens of 9/15/17 -- so what remains is ~94% real parries, including
        # the 1 s triplets, which are multi-hit combos parried individually.
        press = np.sort(press.astype(np.int64))[1:]
        if hj.exists():
            drops = np.array([h["t_ns"] for h in
                              json.loads(hj.read_text())["hits"]], dtype=np.int64)
            if len(drops):
                keep = [t for t in press
                        if not np.any((drops >= t) & (drops - t <= int(1.5e9)))]
                press = np.array(keep, dtype=np.int64)
        if len(press) < 5:
            continue
        out.append(dict(name=d.name, mel=G.mel_frames(pcm), t0=t0,
                        hits=press.astype(np.int64),
                        spurious=np.zeros(0, np.int64),
                        own=np.zeros(0, np.int64), dur=len(pcm) / G.SR,
                        press_anchored=True))
    return out


def load_hits_runs(root: Path):
    out = []
    for d in sorted(root.glob("hits_*")):
        hp = d / "hits.json"
        if not (hp.exists() and (d / "audio.wav").exists()
                and (d / "meta.json").exists()):
            continue
        meta = json.loads((d / "meta.json").read_text())
        t0 = meta.get("audio_start_ns")
        if t0 is None:
            continue
        hits = np.array([h["t_ns"] for h in json.loads(hp.read_text())["hits"]],
                        dtype=np.int64)
        if len(hits) < 5:
            continue        # too few drops to anchor on; see load_press_runs
        with wave.open(str(d / "audio.wav")) as w:
            if w.getframerate() != G.SR:
                continue
            ch = w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw).astype(np.float32) / 32768.0
        spur = np.load(d / "spurious_t_ns.npy") if (d / "spurious_t_ns.npy").exists() \
            else np.zeros(0, np.int64)
        own = np.load(d / "attacks.npy") if (d / "attacks.npy").exists() \
            else np.zeros(0, np.int64)
        out.append(dict(name=d.name, mel=G.mel_frames(pcm), t0=t0, hits=hits,
                        spurious=spur, own=own, dur=len(pcm) / G.SR))
    return out


def rows(s, times):
    mel, out = s["mel"], []
    for t in np.asarray(times, dtype=np.int64):
        fi = int((t - s["t0"]) / 1e9 * G.SR) // G.HOP
        if fi - G.NFRAMES + 1 >= 0 and fi + 1 <= len(mel):
            out.append(mel[fi - G.NFRAMES + 1:fi + 1].T.ravel())
    return (np.array(out, np.float32) if out
            else np.zeros((0, G.NMEL * G.NFRAMES), np.float32))


def build(sessions, rng):
    X, y = [], []
    for s in sessions:
        # Press-anchored runs already give the correct instant; impact-anchored
        # ones need the offset from the HP drop back to the blow.
        anchor = (s["hits"] if s.get("press_anchored")
                  else s["hits"] - int(ANCHOR_MS * 1e6))
        pos_t = np.concatenate([
            anchor + (rng.uniform(-JITTER_MS, JITTER_MS, len(anchor)) * 1e6).astype(np.int64)
            for _ in range(POS_COPIES)])
        pos = rows(s, pos_t)
        if len(pos) < 5:
            continue
        neg = [rows(s, s["spurious"]), rows(s, s["own"])]
        base = np.repeat(anchor, NEAR_PER_POS)
        off = rng.uniform(*NEAR_MS, size=len(base)) * 1e6
        sign = rng.choice([-1.0, 1.0], size=len(base))
        neg.append(rows(s, (base + sign * off).astype(np.int64)))
        lo, hi = int(s["hits"].min()), int(s["hits"].max())
        cand = rng.integers(lo, hi, size=len(anchor) * FAR_PER_POS * 3)
        gap = np.abs(cand[:, None] - anchor[None, :]).min(axis=1) / 1e6
        neg.append(rows(s, cand[gap > FAR_MIN_MS][:len(anchor) * FAR_PER_POS]))
        neg = np.vstack([b for b in neg if len(b)])
        X.append(np.vstack([pos, neg]))
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    return np.vstack(X), np.concatenate(y)


def evaluate(model, s, threshold=0.9):
    """True recall and honest false-press rate against ground truth."""
    p, t = gtrain.slide(model, s)
    fires = gtrain.fire_times(p, t, threshold=threshold)
    anchor = s["hits"] - int(ANCHOR_MS * 1e6)
    if not len(fires):
        return 0.0, 0.0
    d = np.abs(fires[:, None] - anchor[None, :]) / 1e6
    hit = (d.min(axis=0) <= HIT_MS).sum()
    false = (d.min(axis=1) > HIT_MS).sum()
    return hit / len(anchor), false / (s["dur"] / 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.9)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB

    root = Path(a.runs) if a.runs else cfgmod.REPO_ROOT / "runs"
    gt = load_hits_runs(root)
    if not gt:
        print(f"  no analysed punching-bag runs under {root}")
        return 1
    press_runs = load_press_runs(root)
    old = list(gtrain._sessions(cfgmod.REPO_ROOT / "runs"))
    print(f"  {len(gt)} ground-truth runs ({sum(len(s['hits']) for s in gt)} attacks, "
          f"{sum(len(s['spurious']) for s in gt)} proven false positives)")
    if press_runs:
        print(f"  {len(press_runs)} press-anchored runs "
              f"({sum(len(s['hits']) for s in press_runs)} human presses on the "
              f"real target)")
    print(f"  {len(old)} keypress-labelled sessions "
          f"({sum(len(s['landed']) for s in old)} landed parries)\n")

    rng = np.random.default_rng(0)
    print(f"  {'held-out run':<22} {'recall':>8} {'false/min':>10}  {'(baseline)':>18}")
    R, F, R0, F0 = [], [], [], []
    for held in gt:
        tr = [s for s in gt if s["name"] != held["name"]] + press_runs
        Xg, yg = build(tr, rng)
        Xo, yo = gtrain.build(old, rng, near=True)
        m = HGB(max_iter=200, random_state=0).fit(
            np.vstack([Xg, Xo]), np.r_[yg, yo])
        r, f = evaluate(m, held, a.threshold)
        m0 = gtrain.load()["model"]
        r0, f0 = evaluate(m0, held, a.threshold)
        R.append(r); F.append(f); R0.append(r0); F0.append(f0)
        print(f"  {held['name']:<22} {r:7.0%} {f:10.1f}  {r0:9.0%} {f0:7.1f}")
    print(f"\n  {'MEAN':<22} {np.mean(R):7.0%} {np.mean(F):10.1f}  "
          f"{np.mean(R0):9.0%} {np.mean(F0):7.1f}")
    print("  baseline = the currently deployed model, trained on keypress labels")

    if a.save:
        Xg, yg = build(gt + press_runs, rng)
        Xo, yo = gtrain.build(old, rng, near=True)
        X, y = np.vstack([Xg, Xo]), np.r_[yg, yo]
        m = HGB(max_iter=200, random_state=0).fit(X, y)
        import pickle
        out = cfgmod.REPO_ROOT / "runs" / "grunt.pkl"
        with open(out, "wb") as fh:
            pickle.dump(dict(
                model=m, threshold=G.THRESHOLD, refractory_ms=G.REFRACTORY_MS,
                lookahead_ms=G.LOOKAHEAD_MS, nmel=G.NMEL, nframes=G.NFRAMES,
                hop=G.HOP, nfft=G.NFFT, samplerate=G.SR,
                fmin=G.FMIN, fmax=G.FMAX,
                sessions=[s["name"] for s in gt] + [s["name"] for s in old],
                n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), n_hard=0,
                near_miss=True, ground_truth=True, anchor_ms=ANCHOR_MS), fh)
        print(f"\n  saved {out} (trained on all runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
