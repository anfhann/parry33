"""Combine trigger runs into a land-rate-vs-lead curve.

The bot grades its own presses: a landed parry emits a metallic clash ~32 ms
later, so every press in a --lead-sweep run is recorded with the lead it used
and whether a clash followed. Pooling those across fights answers empirically
what "feels early" and "feels late" cannot: where the press should actually go.

Each run is normally one fight (F10 between them), so files are already clean
combat. Presses made outside a fight -- menus, walking around, the victory
screen -- whiff by definition, so any run with a suspiciously low rate across
ALL leads is flagged rather than silently pooled.

    python scripts/lead_curve.py              # every sweep run found
    python scripts/lead_curve.py --last 5     # the five most recent
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod          # noqa: E402


def load(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue                        # a run killed mid-write
        if e.get("kind") == "outcome":
            out.append((float(e["lead_ms"]), bool(e["landed"])))
    return out


def wilson(k, n, z=1.96):
    """95% interval. With 8 presses a row means little; show that honestly."""
    if not n:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - m), min(1.0, c + m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()

    root = Path(a.runs) if a.runs else (cfgmod.REPO_ROOT / "runs")
    files = sorted(root.glob("trig_*.jsonl"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if load(f)]
    if not files:
        print(f"  no runs with outcome records under {root}")
        print("  record some with: parry33 trigger --arm "
              '"--lead-sweep=-20,0,20,40,60"')
        return 1
    if a.last:
        files = files[-a.last:]

    print(f"  {len(files)} run(s) with self-scored presses\n")
    print(f"  {'run':<20} {'presses':>8} {'landed':>7} {'rate':>6}")
    pooled = defaultdict(lambda: [0, 0])
    for f in files:
        rows = load(f)
        k = sum(1 for _, ok in rows if ok)
        flag = "  <- low across the board; a non-combat run?" \
            if rows and k / len(rows) < 0.15 else ""
        print(f"  {f.stem:<20} {len(rows):8d} {k:7d} "
              f"{k / len(rows):5.0%}{flag}")
        for lead, ok in rows:
            pooled[lead][0] += 1
            pooled[lead][1] += int(ok)

    total = sum(n for n, _ in pooled.values())
    print(f"\n  pooled over {total} presses\n")
    print(f"  {'lead':>7} {'presses':>8} {'landed':>7} {'rate':>6}  "
          f"{'95% interval':>14}")
    best = None
    for lead in sorted(pooled):
        n, k = pooled[lead]
        lo, hi = wilson(k, n)
        bar = "#" * int(round(k / n * 24)) if n else ""
        print(f"  {lead:+6.0f}ms {n:8d} {k:7d} {k / n:5.0%}  "
              f"{lo:5.0%} - {hi:4.0%}  {bar}")
        if n >= 5 and (best is None or k / n > best[1]):
            best = (lead, k / n, n)

    if best is None:
        print("\n  no row has 5+ presses yet -- run more fights")
        return 0
    lead, rate, n = best
    print(f"\n  best measured: lead {lead:+.0f} ms at {rate:.0%} over {n} presses")
    tight = sorted(pooled)
    i = tight.index(lead)
    lo = tight[max(0, i - 1)]
    hi = tight[min(len(tight) - 1, i + 1)]
    print(f"  to sharpen it:  --lead-sweep={lo:.0f},{(lo + lead) / 2:.0f},"
          f"{lead:.0f},{(lead + hi) / 2:.0f},{hi:.0f}")
    print(f"  to just use it: --lead-ms {lead:.0f}")
    print("\n  Treat a row under ~10 presses as a hint, not a result -- the "
          "intervals above show how wide the uncertainty still is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
