"""Latency sample collection and reporting.

Everything is stored in nanoseconds as ints and only converted at print time.
Means are near-useless for latency work; the tail is what drops a parry, so the
default report leads with percentiles and max.
"""

from __future__ import annotations

import numpy as np

_BLOCKS = " .:-=+*#%"


class Samples:
    """Preallocated append-only sample buffer. No allocation in the hot path."""

    __slots__ = ("name", "unit", "_buf", "_n")

    def __init__(self, name: str, capacity: int = 100_000, unit: str = "ms") -> None:
        self.name = name
        self.unit = unit
        self._buf = np.empty(capacity, dtype=np.int64)
        self._n = 0

    def add(self, value_ns: int) -> None:
        if self._n < self._buf.size:
            self._buf[self._n] = value_ns
            self._n += 1

    def __len__(self) -> int:
        return self._n

    @property
    def values_ns(self) -> np.ndarray:
        return self._buf[: self._n]

    def summary(self) -> dict:
        v = self.values_ns
        if v.size == 0:
            return {"n": 0}
        f = v / 1e6
        p = np.percentile(f, [50, 90, 99, 99.9])
        return {
            "n": int(v.size),
            "min": float(f.min()),
            "p50": float(p[0]),
            "p90": float(p[1]),
            "p99": float(p[2]),
            "p999": float(p[3]),
            "max": float(f.max()),
            "mean": float(f.mean()),
            "std": float(f.std()),
        }

    def sparkline(self, bins: int = 24, lo=None, hi=None) -> str:
        v = self.values_ns
        if v.size == 0:
            return ""
        f = v / 1e6
        lo = float(f.min()) if lo is None else lo
        hi = float(np.percentile(f, 99.5)) if hi is None else hi
        if hi <= lo:
            hi = lo + 1e-6
        counts, _ = np.histogram(np.clip(f, lo, hi), bins=bins, range=(lo, hi))
        peak = counts.max() or 1
        bars = "".join(_BLOCKS[min(8, int(round(c / peak * 8)))] for c in counts)
        return f"{lo:6.2f} [{bars}] {hi:6.2f} ms"

    def line(self, width: int = 16) -> str:
        s = self.summary()
        if not s.get("n"):
            return f"  {self.name:<{width}} (no samples)"
        return (f"  {self.name:<{width}} n={s['n']:<6d} "
                f"p50 {s['p50']:6.2f}  p90 {s['p90']:6.2f}  p99 {s['p99']:6.2f}  "
                f"max {s['max']:7.2f} ms")

    def report(self, width: int = 16) -> str:
        head = self.line(width)
        if len(self) > 8:
            return head + "\n" + " " * (width + 4) + self.sparkline()
        return head


def budget_table(rows, total_budget_ms: float) -> str:
    """Render a stage-by-stage latency breakdown against the budget."""
    out = []
    total = sum(v for _, v in rows)
    for name, v in rows:
        frac = v / total_budget_ms if total_budget_ms else 0
        n = int(min(1.0, frac) * 30)
        bar = "#" * n + "." * (30 - n)
        out.append(f"  {name:<22} {v:7.2f} ms  {bar}")
    verdict = "WITHIN" if total <= total_budget_ms else "OVER"
    out.append(f"  {'TOTAL':<22} {total:7.2f} ms  [{verdict} budget {total_budget_ms:.1f} ms]")
    return "\n".join(out)
