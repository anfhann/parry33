"""Is the parry tell learnable from the clips we have?

This is a feasibility test, not a trigger. The question it answers is narrow:
does ANY model score meaningfully above chance against real control clips? If
yes, more playtime is worth collecting. If no, more of the same data will not
help and the representation has to change.

Three things this does that the earlier hand-crafted analyses did not, each
because an earlier finding died for want of them:

* **Real controls only.** Negatives are control clips recorded at random times,
  never a parry clip's own early frames. Using the latter is what produced a
  bogus AUC of 0.762 for a feature that scores 0.533 against real controls.

* **One session at a time.** Positives and negatives must come from the same
  session. Run 1 has 54 positives and zero controls; mixing its positives with
  run 2's controls would let a model separate them by session -- lighting,
  camera, anything -- and score beautifully while learning nothing.

* **Grouped by time, not shuffled.** Clips seconds apart are near-duplicates.
  A shuffled split puts them either side of the fold boundary and inflates the
  score. Folds are contiguous time blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfgmod


def load_run(run_dir: Path, landed_only: bool = True):
    """Return (events, clash_times) for a run."""
    ev = [json.loads(l) for l in open(run_dir / "events.jsonl", encoding="utf-8")]
    clash = np.array(sorted(e["t_ns"] for e in ev if e["kind"] == "clash"))
    pos, neg = [], []
    for e in ev:
        if not e.get("clip"):
            continue
        if e["kind"] == "control":
            neg.append(e)
        elif e["kind"] == "parry":
            d = (clash - e["t_ns"]) / 1e6
            if not landed_only or np.any((d >= 0) & (d <= 200)):
                pos.append(e)
    return pos, neg


def features_from_frames(f, rel, win=(-600.0, 0.0), gh=6, gw=8, tb=6):
    """Spatio-temporal motion descriptor from an in-memory frame stack.

    Training and live inference MUST call this same function. A mismatch in
    grid size, binning or normalisation silently produces garbage scores rather
    than an error, so there is exactly one implementation.

    Motion is binned into a gh x gw spatial grid over tb time bins and stored
    three ways: absolute level, share of that bin's total (where the motion is,
    independent of how much), and the per-bin total. The share matters because
    total motion FALLS before a defensive input -- what changes is its location.
    """
    if len(f) < 12:
        return None
    d = np.abs(np.diff(f.astype(np.float32), axis=0))
    t = rel[:-1]
    m = (t >= win[0]) & (t <= win[1])
    if m.sum() < 6:
        return None
    d, t = d[m], t[m]
    H, W = d.shape[1:]
    edges = np.linspace(win[0], win[1], tb + 1)
    cells, totals = [], []
    for i in range(tb):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if sel.sum() == 0:
            sel = np.zeros(len(t), bool)
            sel[min(i, len(t) - 1)] = True
        blk = d[sel].mean(0)
        grid = (blk.reshape(gh, H // gh, gw, W // gw).mean(axis=(1, 3))
                if H % gh == 0 and W % gw == 0 else _resize_mean(blk, gh, gw))
        cells.append(grid)
        totals.append(blk.mean())
    cells = np.array(cells)
    totals = np.array(totals)
    share = cells / np.maximum(totals[:, None, None], 1e-6)
    return np.concatenate([cells.ravel(), share.ravel(), totals])


def features(run_dir: Path, e: dict, win=(-600.0, 0.0), gh=6, gw=8, tb=6,
             use_intensity=False):
    """Feature vector for a recorded clip, relative to its event time."""
    z = np.load(run_dir / e["clip"])
    f = z["frames"].astype(np.float32)
    rel = (z["t_ns"] - e["t_ns"]) / 1e6
    return features_from_frames(f, rel, win=win, gh=gh, gw=gw, tb=tb)


def _resize_mean(a, gh, gw):
    H, W = a.shape
    ys = np.linspace(0, H, gh + 1).astype(int)
    xs = np.linspace(0, W, gw + 1).astype(int)
    return np.array([[a[ys[i]:max(ys[i + 1], ys[i] + 1),
                        xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
                      for j in range(gw)] for i in range(gh)])


def build(run_dir: Path, **kw):
    pos, neg = load_run(run_dir)
    X, y, t = [], [], []
    for lbl, group in ((1, pos), (0, neg)):
        for e in group:
            v = features(run_dir, e, **kw)
            if v is None:
                continue
            X.append(v)
            y.append(lbl)
            t.append(e["t_ns"])
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    t = np.array(t)
    order = np.argsort(t)
    return X[order], y[order], t[order]


def time_groups(t, n_groups=6):
    """Contiguous time blocks, so near-duplicate clips stay in the same fold."""
    edges = np.quantile(t, np.linspace(0, 1, n_groups + 1))
    g = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_groups - 1)
    return g


def evaluate(X, y, groups, seed=0):
    """Grouped CV AUC for a few small models. Returns {name: (mean, std)}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "logistic (L2, C=0.01)": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
        "logistic (L2, C=0.1)": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1),
    }
    n_splits = min(6, len(np.unique(groups)))
    out = {}
    for name, mdl in models.items():
        aucs = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        out[name] = (float(np.mean(aucs)), float(np.std(aucs)), len(aucs))
    return out


def shuffled_control(X, y, groups, n=20, seed=0):
    """AUC with labels shuffled. Anything above this is the real floor.

    With 156 samples and 300+ features a grouped CV can drift well off 0.50 by
    chance alone; comparing against the honest result is the only way to know
    whether a score means anything.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    n_splits = min(6, len(np.unique(groups)))
    res = []
    for _ in range(n):
        ys = rng.permutation(y)
        aucs = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, ys, groups):
            if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
                continue
            mdl.fit(X[tr], ys[tr])
            aucs.append(roc_auc_score(ys[te], mdl.predict_proba(X[te])[:, 1]))
        if aucs:
            res.append(np.mean(aucs))
    return float(np.mean(res)), float(np.std(res))
