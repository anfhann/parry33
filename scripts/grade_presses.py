"""Grade the bot's presses from the HP readout, not from clashes.

The trigger used to score itself by listening for a clash after each press. That
was wrong: a clash marks an attack making CONTACT, whether it was parried or not.
Measured on punching-bag runs where the player parried nothing, 100% of attacks
that landed still produced a clash. Scored that way, an armed run reported 14/14
presses landed when the victory screen said 1.

HP is unambiguous. If the parry worked, no damage.

    a press with an HP drop shortly after   -> it did not parry that attack
    an HP drop with no press near it        -> an attack we never fired on
    a press with no HP drop and no attack   -> a wasted press (1500 ms lockout)

Run the trigger with --grade-hp to capture the readout, then point this at it.

    python scripts/grade_presses.py trig_1a38e7d2ddf74
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import find_hits as F                          # noqa: E402
from parry33 import config as cfgmod           # noqa: E402

DROP_WINDOW_MS = 1500.0     # an attack we failed to stop lands within this
NEAR_MS = 1200.0            # a press this close to a drop was aimed at it
# A successful parry triggers a counter, and the boss loses health a very fixed
# time later: measured over 9 events, p25 1948 / p50 1961 / p75 1991 ms, a 43 ms
# spread. So a boss-HP drop at this lag after a press is PROOF that press
# parried. Nothing else in the game gives direct evidence of success -- the
# player's own HP can only ever show failures, because a parried attack leaves
# no mark on it.
COUNTER_MS = (1750.0, 2200.0)
BOSS_ROWS = (25, 34)        # the bar within the boss ROI
BOSS_TH = 70                # red-excess; scenery is reddish, the fill is redder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="trigger log stem, e.g. trig_1a38e7d2ddf74")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()

    root = Path(a.runs) if a.runs else cfgmod.REPO_ROOT / "runs"
    log = root / f"{a.run}.jsonl"
    hp_dir = root / a.run
    if not log.exists():
        print(f"  no trigger log at {log}")
        return 1
    if not (hp_dir / "hp.npy").exists():
        print(f"  no HP capture at {hp_dir}. Re-run the trigger with --grade-hp;")
        print("  without it there is no honest grader (clash scoring was retracted).")
        return 1

    ev = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
    fires = [(e["t_ns"], e.get("lead_ms")) for e in ev if e["kind"] == "fire"]
    if not fires:
        print("  no presses in this log")
        return 1

    hp = np.load(hp_dir / "hp.npy")
    ts = np.load(hp_dir / "t_ns.npy")
    tpl = F.load_templates(root)
    if not tpl:
        print("  no digit templates -- seed them once with find_hits.py")
        return 1
    gray = hp[:, F.ROWS[0]:F.ROWS[1], :]
    mask = gray > F.BRIGHT
    vals = [F.read_value(mask[i], gray[i], tpl) for i in range(len(hp))]
    got = sum(v is not None for v in vals)
    print(f"  read HP in {got}/{len(hp)} frames ({got / max(len(hp), 1):.0%})")

    stable, s = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[s]:
            if vals[s] is not None and i - s >= F.MIN_RUN:
                stable.append((s, int(vals[s])))
            s = i
    seq = stable[:1]
    for f, v in stable[1:]:
        if v != seq[-1][1]:
            seq.append((f, v))
    drops = [int(ts[f]) for (_, v0), (f, v1) in zip(seq, seq[1:]) if v1 < v0]
    print(f"  {len(drops)} attacks got through (HP dropped)")
    print(f"  {len(fires)} presses\n")

    # Successful parries, from the boss bar.
    counters = []
    bp = hp_dir / "boss.npy"
    if bp.exists():
        bs = np.asarray(np.load(bp, mmap_mode="r"))[:len(ts),
                                                    BOSS_ROWS[0]:BOSS_ROWS[1], :]
        colf = (bs > BOSS_TH).mean(axis=1) >= 0.5
        edge = np.array([np.max(np.nonzero(c)[0]) if c.any() else -1
                         for c in colf])
        runs_, st = [], 0
        for i in range(1, len(edge) + 1):
            if i == len(edge) or abs(int(edge[i]) - int(edge[st])) > 8:
                if edge[st] >= 0 and i - st >= 10:
                    runs_.append((st, int(edge[st])))
                st = i
        merged_e = runs_[:1]
        for a_, v in runs_[1:]:
            if abs(v - merged_e[-1][1]) > 8:
                merged_e.append((a_, v))
        counters = [int(ts[a_]) for k, (a_, v) in enumerate(merged_e)
                    if k and v < merged_e[k - 1][1]]
        parried = 0
        for t, _ in fires:
            if any(COUNTER_MS[0] <= (c - t) / 1e6 <= COUNTER_MS[1]
                   for c in counters):
                parried += 1
        print(f"  boss bar: {len(counters)} segment drops -> {parried}/{len(fires)} "
              f"presses CONFIRMED as parries")
        print("  (the bar is segmented, so it confirms some successes, not all;")
        print("   a confirmed one is certain, an unconfirmed one is unknown)")
        print()

    ft = np.array([t for t, _ in fires], dtype=np.int64)
    dt = np.array(drops, dtype=np.int64)
    rows = []
    for t, lead in fires:
        if len(dt):
            after = dt[(dt >= t) & (dt - t <= DROP_WINDOW_MS * 1e6)]
            failed = len(after) > 0
        else:
            failed = False
        near = bool(len(dt)) and np.abs(dt - t).min() / 1e6 <= NEAR_MS
        rows.append((lead, failed, near))

    unattacked = 0
    if len(dt):
        for d in dt:
            if not len(ft) or np.abs(ft - d).min() / 1e6 > NEAR_MS:
                unattacked += 1
    print(f"  {unattacked}/{len(drops)} attacks had NO press anywhere near them")
    print(f"  (those are attacks the detector missed entirely)\n")

    by = {}
    for lead, failed, near in rows:
        k = lead if lead is not None else 0.0
        b = by.setdefault(k, [0, 0, 0])
        b[0] += 1
        b[1] += failed
        b[2] += (not failed) and near
    print(f"  {'lead':>7} {'presses':>8} {'attack still hit':>17} {'stopped one':>12}")
    for k in sorted(by):
        n, f_, ok = by[k]
        print(f"  {k:+6.0f}ms {n:8d} {f_:16d} {ok:12d}")
    print()
    print("  'stopped one' = a press near an attack, and no damage followed.")
    print("  It is the best available evidence of a parry, but it is not proof:")
    print("  a press with no attack incoming also takes no damage. Cross-check")
    print("  the total against the victory screen's Successful Parries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
