"""Train, persist and serve the attack detector.

Detection and response are separate problems. This model answers only "is an
attack incoming?", which is the same question whether you intend to parry or
dodge -- so parry and dodge events are pooled as positives. What differs between
the two actions is the delay from detection to keypress, and that lives in the
trigger, not here.

Trained with the same protocol used throughout: real control clips as negatives,
grouped by contiguous time block so near-duplicate clips cannot straddle a fold,
and a shuffled-label floor computed alongside so a score can be believed.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import learn

# Window offsets the model is trained on, relative to the keypress. Training at
# several offsets (rather than one) keeps the score stable as the window slides
# at inference time -- without it, consecutive-window firing rules collapse.
TRAIN_OFFSETS = np.arange(-1000, -400, 100)
WINDOW_MS = 500.0
MODEL_PATH = cfgmod.REPO_ROOT / "runs" / "detector.pkl"


def full_screen_runs(runs_dir: Path | None = None):
    """Runs recorded on the full-screen ROI. Older cropped runs are not poolable."""
    runs_dir = runs_dir or (cfgmod.REPO_ROOT / "runs")
    out = []
    for d in sorted(runs_dir.glob("*/")):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        reg = m.get("video_region")
        if reg and (reg[2] - reg[0]) >= 2000 and len(list((d / "clips").glob("*.npz"))):
            out.append(d)
    return out


def build_dataset(runs, combat_only_s: float = 15.0):
    """Pooled positives (parry+dodge) and control negatives across runs."""
    X, y, t, src = [], [], [], []
    for run in runs:
        ev = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
        pos = [e for e in ev if e["kind"] in ("parry", "dodge") and e.get("clip")]
        neg = [e for e in ev if e["kind"] == "control" and e.get("clip")]
        if not pos:
            continue
        pt = np.array(sorted(e["t_ns"] for e in pos))
        neg = [e for e in neg
               if np.min(np.abs(pt - e["t_ns"])) / 1e9 <= combat_only_s]
        for lbl, grp in ((1, pos), (0, neg)):
            for e in grp:
                for off in TRAIN_OFFSETS:
                    v = learn.features(run, e, win=(float(off), float(off + WINDOW_MS)))
                    if v is None:
                        continue
                    X.append(v)
                    y.append(lbl)
                    t.append(e["t_ns"])
                    src.append(run.name)
    return (np.array(X, np.float32), np.array(y), np.array(t), np.array(src))


def train(runs=None, save: bool = True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    runs = runs or full_screen_runs()
    if not runs:
        raise RuntimeError("no full-screen runs with clips found")
    X, y, t, src = build_dataset(runs)
    print(f"  runs {len(runs)}  samples {len(X)}  "
          f"positives {int(y.sum())}  negatives {int((y == 0).sum())}")

    g = learn.time_groups(t, min(5, len(np.unique(t)) // 2 or 2))
    mdl = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=4000,
                                           class_weight="balanced"))
    sc = np.zeros(len(y))
    n_splits = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        mdl.fit(X[tr], y[tr])
        sc[te] = mdl.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, sc)
    floor, _ = learn.shuffled_control(X, y, g, n=10)
    print(f"  held-out AUC {auc:.3f}   shuffled floor {floor:.3f}   "
          f"delta {auc - floor:+.3f}")

    mdl.fit(X, y)                       # refit on everything for deployment
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": mdl, "auc": float(auc), "floor": float(floor),
                         "window_ms": WINDOW_MS, "offsets": TRAIN_OFFSETS.tolist(),
                         "runs": [r.name for r in runs],
                         "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}, f)
        print(f"  saved -> {MODEL_PATH}")
    return mdl, auc, floor


def load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"no trained detector at {MODEL_PATH}. Run: parry33 train")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# --- training on the serving distribution -----------------------------------
#
# The clip-based model scores 0.865 offline and 38% recall live. The gap is a
# train/serve mismatch: offline every sample is clip-shaped, extracted at a
# known offset from something that happened, while live the model scores a
# continuously sliding window that spends most of its time in states no clip
# ever covered. Training on the windows actually served removes the mismatch
# rather than compensating for it.

LIVE_LABEL_WINDOW_MS = (250.0, 900.0)   # a press this far after the window = positive


def load_live(runs_dir: Path | None = None):
    """Feature vectors from live runs, labelled by whether a press followed."""
    runs_dir = runs_dir or (cfgmod.REPO_ROOT / "runs")
    X, y, t = [], [], []
    for fp in sorted(runs_dir.glob("live_*_feats.npz")):
        jl = fp.with_name(fp.name.replace("_feats.npz", ".jsonl"))
        if not jl.exists():
            continue
        rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
        press = np.array(sorted(r["t_ns"] for r in rows
                                if r.get("kind") == "human_parry"))
        if not len(press):
            continue
        z = np.load(fp)
        Xi, ti = z["X"], z["t_ns"]
        lo, hi = LIVE_LABEL_WINDOW_MS
        for j, tw in enumerate(ti):
            d = (press - tw) / 1e6
            X.append(Xi[j])
            y.append(int(np.any((d >= lo) & (d <= hi))))
            t.append(int(tw))
    if not X:
        return None, None, None
    return np.array(X, np.float32), np.array(y), np.array(t)


def train_live(save: bool = True):
    """Train on served windows. Falls back with a clear message if none exist."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, t = load_live()
    if X is None:
        raise RuntimeError(
            "no live feature logs found. Run `parry33 live` (dry run) first -- "
            "it writes runs/live_<id>_feats.npz alongside the .jsonl")
    print(f"  served windows {len(X)}  positives {int(y.sum())} "
          f"({y.mean()*100:.1f}%)  negatives {int((y==0).sum())}")
    if y.sum() < 10:
        print("  WARNING: very few positives; more play time needed")

    g = learn.time_groups(t, 5)
    mdl = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=4000,
                                           class_weight="balanced"))
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        mdl.fit(X[tr], y[tr])
        sc[te] = mdl.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, sc) if len(np.unique(y)) > 1 else float("nan")
    floor, _ = learn.shuffled_control(X, y, g, n=8)
    print(f"  held-out AUC {auc:.3f}   shuffled floor {floor:.3f}   "
          f"delta {auc-floor:+.3f}")
    print(f"  (clip-trained model scored 0.865 offline but 38% recall live)")

    mdl.fit(X, y)
    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": mdl, "auc": float(auc), "floor": float(floor),
                         "window_ms": WINDOW_MS, "offsets": TRAIN_OFFSETS.tolist(),
                         "runs": ["live"], "trained_on": "served_windows",
                         "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}, f)
        print(f"  saved -> {MODEL_PATH}")
    return mdl, auc, floor
