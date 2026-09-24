"""Train the deployed attack-sound model from recorded sessions.

Labels come free from the recordings: a parry keypress followed by a clash
within 200 ms is a parry that landed, and its press time is a moment where
"press now" was demonstrably correct. No hand-labelling.

Three kinds of negative, and the second two matter more than the first:

  far      random combat instants at least 400 ms from any human press. Teaches
           "this is not an attack". Cheap and plentiful, and on its own not
           enough -- a model trained only on these scores highly across the
           whole grunt.

  near     deliberately drawn 120-400 ms either side of a landed press. Same
           sound, wrong instant. Without these, 28% of attacks were detected
           confidently but pressed off-target, because nothing ever taught the
           model that early is wrong.

  hard     the model's own false fires, mined by sliding round 1 over the
           training sessions. DEFAULT OFF -- measured, it makes things worse:
           49% -> 46% parried, and it loses at matched whiff rates too (43% vs
           39% at ~1.8 whiffs/min).

           The reason is a bad label, not a bad idea. A mined negative is a
           window where the model fired and no LANDED parry was nearby -- but
           the player misses about 27% of their parries, so a large share of
           those windows are real attacks the player whiffed. Mining trained the
           model to suppress correct detections. It looked like it worked when
           first measured (6.6 -> 3.2 false/min) because that measurement scored
           against landed parries too, which bakes in the same mistake.

           It would become useful given a ground truth for "an attack occurred"
           that does not route through whether the player parried it.

Evaluation is deliberately NOT done here -- training on every session and
reporting a score on those same sessions would be meaningless. See
scripts/eval_grunt.py, which holds out whole sessions and simulates the actual
game mechanic (a landed parry is free, a whiff costs a 1500 ms lockout).
"""

from __future__ import annotations

import json
import pathlib
import pickle
import wave

import numpy as np

from .. import config as cfgmod
from . import grunt as G

CLASH_MAX_MS = 200.0        # a clash this soon after a press means it landed
FAR_MIN_MS = 400.0          # keep "far" negatives clear of any press
NEAR_MS = (120.0, 400.0)    # near-miss offsets either side of a press
FAR_PER_POS = 10
NEAR_PER_POS = 4


def _sessions(root: pathlib.Path):
    """Yield (name, mel, t0, landed, presses, duration) per usable session."""
    for run in sorted(root.glob("*/")):
        meta_p, wav_p, ev_p = run / "meta.json", run / "audio.wav", run / "events.jsonl"
        if not (meta_p.exists() and wav_p.exists() and ev_p.exists()):
            continue
        t0 = json.loads(meta_p.read_text()).get("audio_start_ns")
        if t0 is None:
            continue
        ev = [json.loads(l) for l in open(ev_p, encoding="utf-8") if l.strip()]
        clash = np.array(sorted(e["t_ns"] for e in ev if e["kind"] == "clash"))
        press = np.array(sorted(e["t_ns"] for e in ev
                                if e["kind"] in ("parry", "dodge")))
        if not len(clash) or not len(press):
            continue
        d = (clash[None, :] - press[:, None]) / 1e6
        landed = press[np.any((d >= 0) & (d <= CLASH_MAX_MS), axis=1)]
        if len(landed) < 2:
            continue
        with wave.open(str(wav_p)) as w:
            ch, sr = w.getnchannels(), w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if sr != G.SR:
            continue
        pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw).astype(np.float32) / 32768.0
        yield dict(name=run.name, mel=G.mel_frames(pcm), t0=t0, landed=landed,
                   press=press, dur=len(pcm) / G.SR)


def rows_at(s, times) -> np.ndarray:
    """Feature rows for windows *ending* at each time. Skips out-of-range."""
    mel, out = s["mel"], []
    for t in np.asarray(times, dtype=np.int64):
        fi = int((t - s["t0"]) / 1e9 * G.SR) // G.HOP
        if fi - G.NFRAMES + 1 >= 0 and fi + 1 <= len(mel):
            out.append(mel[fi - G.NFRAMES + 1:fi + 1].T.ravel())
    return np.array(out, dtype=np.float32) if out else \
        np.zeros((0, G.NMEL * G.NFRAMES), np.float32)


def build(sessions, rng, near=True):
    """Positives + far/near negatives across the given sessions."""
    X, y = [], []
    for s in sessions:
        pos = rows_at(s, s["landed"])
        if len(pos) < 2:
            continue
        press, n = s["press"], len(s["landed"])
        cand = rng.integers(int(press.min()), int(press.max()), size=n * FAR_PER_POS * 2)
        gap = np.abs(cand[:, None] - press[None, :]).min(axis=1) / 1e6
        neg = [rows_at(s, cand[gap > FAR_MIN_MS][:n * FAR_PER_POS])]
        if near:
            base = np.repeat(s["landed"], NEAR_PER_POS)
            off = rng.uniform(*NEAR_MS, size=len(base)) * 1e6
            sign = rng.choice([-1.0, 1.0], size=len(base))
            neg.append(rows_at(s, (base + sign * off).astype(np.int64)))
        neg = np.vstack([b for b in neg if len(b)])
        if len(neg) < 2:
            continue
        X.append(np.vstack([pos, neg]))
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    return np.vstack(X), np.concatenate(y)


def slide(model, s, chunk=20000):
    """Score every 10 ms instant in a session. Returns (probs, times_ns)."""
    n = len(s["mel"]) - G.NFRAMES + 1
    p = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        p[a:b] = model.predict_proba(G.stack(s["mel"][a:b + G.NFRAMES - 1]))[:, 1]
    t = s["t0"] + ((np.arange(n) + G.NFRAMES - 1) * G.HOP / G.SR * 1e9)
    return p, t.astype(np.int64)


def fire_times(p, t, threshold=G.THRESHOLD, refractory_ms=G.REFRACTORY_MS,
               lookahead_ms=G.LOOKAHEAD_MS):
    """Apply the serving policy offline. Mirrors GruntStream exactly."""
    look = int(lookahead_ms // 10)
    out, lock = [], -1
    for i in np.flatnonzero(p >= threshold):
        if t[i] < lock:
            continue
        j = i + int(np.argmax(p[i:min(i + look + 1, len(p))]))
        out.append(t[j] + int(lookahead_ms * 1e6))
        lock = t[j] + int(refractory_ms * 1e6)
    return np.array(out, dtype=np.int64)


def mine_hard(model, sessions, rng, margin_ms=150.0):
    """Round-1 false fires, as negatives for round 2."""
    hard = []
    for s in sessions:
        p, t = slide(model, s)
        f = fire_times(p, t)
        if not len(f):
            continue
        off = np.abs(f[:, None] - s["landed"][None, :]).min(axis=1) / 1e6
        bad = f[off > margin_ms] - int(G.LOOKAHEAD_MS * 1e6)
        if len(bad):
            hard.append(rows_at(s, bad))
    if not hard:
        return np.zeros((0, G.NMEL * G.NFRAMES), np.float32)
    return np.vstack(hard)


def train(out: pathlib.Path | None = None, mine: bool = False, near: bool = True):
    from sklearn.ensemble import HistGradientBoostingClassifier

    root = cfgmod.REPO_ROOT / "runs"
    sessions = list(_sessions(root))
    if not sessions:
        raise RuntimeError(f"no usable recorded sessions under {root}")
    rng = np.random.default_rng(0)
    X, y = build(sessions, rng, near=near)
    print(f"  {len(sessions)} sessions, {int(y.sum())} landed parries, "
          f"{len(y) - int(y.sum())} negatives, {X.shape[1]} dims")

    model = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)
    n_hard = 0
    if mine:
        hard = mine_hard(model, sessions, rng)
        n_hard = len(hard)
        if n_hard:
            X = np.vstack([X, hard])
            y = np.r_[y, np.zeros(n_hard)]
            model = HistGradientBoostingClassifier(
                max_iter=200, random_state=0).fit(X, y)
        print(f"  mined {n_hard} hard negatives, retrained")

    out = out or (root / "grunt.pkl")
    with open(out, "wb") as fh:
        pickle.dump(dict(
            model=model, threshold=G.THRESHOLD, refractory_ms=G.REFRACTORY_MS,
            lookahead_ms=G.LOOKAHEAD_MS, nmel=G.NMEL, nframes=G.NFRAMES,
            hop=G.HOP, nfft=G.NFFT, samplerate=G.SR,
            fmin=G.FMIN, fmax=G.FMAX,
            sessions=[s["name"] for s in sessions],
            n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), n_hard=n_hard,
            near_miss=near), fh)
    print(f"  saved {out}")
    print("  NOTE: this model saw every session. For an honest number run "
          "scripts/eval_grunt.py, which holds sessions out.")
    return out


def load(path: pathlib.Path | None = None):
    path = path or (cfgmod.REPO_ROOT / "runs" / "grunt.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"no grunt model at {path} -- run: parry33 train --grunt")
    with open(path, "rb") as fh:
        return pickle.load(fh)
