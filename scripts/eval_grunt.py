"""Honest evaluation of the attack-sound model.

Two rules make this trustworthy, and both were learned the hard way here.

1. HOLD OUT WHOLE SESSIONS. A model validated on shuffled windows from the same
   fight can memorise that fight's mix and enemy and score beautifully while
   being useless on the next one. Sessions are the unit of independence.

2. SIMULATE THE GAME, NOT THE CLASSIFIER. Reporting AUC, or even "fired within
   75 ms of a real parry", flatters the system. AUC is measured on sampled
   negatives while serving slides continuously every 10 ms -- that gap turned
   0.947 AUC into 48% of attacks parried. And a naive recall metric counts a
   second fire near a real parry as free, when in the game the first press
   lands and the second whiffs and costs 1500 ms.

   So this replays the fight under the real mechanic:

       landed parry  -> instant reframe, 0 ms, free. Combos are possible.
       whiffed parry -> 1500 ms lockout. Real attacks arriving inside it are
                        lost, so one bad press can cost the *next* attack too.

   The headline number is the fraction of real attacks actually parried.

Usage:
    python scripts/eval_grunt.py                 # default model
    python scripts/eval_grunt.py --model logreg  # compare families
    python scripts/eval_grunt.py --sweep         # threshold/refractory grid
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod              # noqa: E402
from parry33.audio import grunt as G              # noqa: E402
from parry33.audio import train as gt             # noqa: E402

WHIFF_LOCKOUT_MS = 1500.0
HIT_WINDOW_MS = 75.0
MIN_LANDED = 30


def build_model(kind: str):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    if kind == "hgb50":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=50, random_state=0)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.1))
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64,),
                                           max_iter=400, random_state=0))
    raise SystemExit(f"unknown model {kind}")


def simulate(fires, landed, dur_s):
    """Play the fight out. Returns (parried_fraction, whiffs_per_min)."""
    unclaimed = set(range(len(landed)))
    ok = whiff = 0
    lock = -1
    for f in fires:
        if f < lock:
            continue                      # input eaten by an earlier whiff
        near = [i for i in unclaimed if abs(f - landed[i]) / 1e6 <= HIT_WINDOW_MS]
        if near:
            ok += 1
            unclaimed.discard(min(near, key=lambda i: abs(f - landed[i])))
        else:
            whiff += 1
            lock = f + int(WHIFF_LOCKOUT_MS * 1e6)
    return ok / max(len(landed), 1), whiff / (dur_s / 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb",
                    choices=("hgb", "hgb50", "logreg", "mlp"))
    ap.add_argument("--no-mine", action="store_true", help="skip hard negatives")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    sessions = list(gt._sessions(cfgmod.REPO_ROOT / "runs"))
    big = [s for s in sessions if len(s["landed"]) >= MIN_LANDED]
    if not big:
        raise SystemExit("no session has enough landed parries to hold out")
    print(f"  {len(sessions)} sessions, holding out {len(big)} with "
          f">={MIN_LANDED} landed parries")
    print(f"  model={a.model}  mine={'no' if a.no_mine else 'yes'}\n")

    cache = {}
    for held in big:
        train_on = [s for s in sessions if s["name"] != held["name"]]
        X, y = gt.build(train_on, rng, near=True)
        m = build_model(a.model).fit(X, y)
        if not a.no_mine:
            hard = gt.mine_hard(m, [s for s in train_on
                                    if len(s["landed"]) >= MIN_LANDED], rng)
            if len(hard):
                m = build_model(a.model).fit(
                    np.vstack([X, hard]), np.r_[y, np.zeros(len(hard))])
        cache[held["name"]] = gt.slide(m, held)
        print(f"    {held['name']} scored", flush=True)

    v = np.zeros((1, G.NMEL * G.NFRAMES), np.float32)
    for _ in range(20):
        m.predict_proba(v)
    t0 = time.perf_counter_ns()
    for _ in range(300):
        m.predict_proba(v)
    per_call = (time.perf_counter_ns() - t0) / 300 / 1e6

    grid = ([(th, rf) for th in (0.80, 0.90, 0.95, 0.99) for rf in (150, 250, 400)]
            if a.sweep else [(G.THRESHOLD, G.REFRACTORY_MS)])
    print(f"\n  {'thresh':>7} {'refr':>6} {'parried':>8} {'whiffs/min':>11} {'err p50':>8}")
    for th, rf in grid:
        P, W, E = [], [], []
        for held in big:
            p, t = cache[held["name"]]
            f = gt.fire_times(p, t, threshold=th, refractory_ms=rf)
            ld = list(held["landed"])
            pr, wh = simulate(f, ld, held["dur"])
            P.append(pr)
            W.append(wh)
            if len(f):
                d = np.abs(f[:, None] - np.array(ld)[None, :]) / 1e6
                hit = d.min(axis=1) <= HIT_WINDOW_MS
                E += list((f[hit] - np.array(ld)[d.argmin(axis=1)[hit]]) / 1e6)
        print(f"  {th:7.2f} {rf:6.0f} {np.mean(P):7.0%} {np.mean(W):10.1f} "
              f"{np.median(E) if E else float('nan'):+6.0f}ms")
    print(f"\n  predict cost {per_call:.3f} ms/call -> "
          f"{per_call * 100 / 10:.0f}% of a core at 100 scores/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
