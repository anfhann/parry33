"""Sweep the audio feature configuration against the honest metric.

The deployed model sees 141 ms of trailing audio in 40 mel bands from 50 Hz to
16 kHz. Those numbers were first guesses that were never revisited, and they are
load-bearing: context length decides whether the model can see an attack's
wind-up or only its impact, and 141 ms is shorter than most of the animations
the player says they read.

Every configuration is scored the same way as the deployed one: whole sessions
held out, slid continuously every 10 ms, and replayed under the real mechanic
(a landed parry is free and instant; a whiff costs a 1500 ms lockout that can
also cost the following attack). The headline is the fraction of real attacks
parried -- not AUC, which measures sampled negatives and reads ~0.95 for a model
that parries half.

    python scripts/sweep_features.py                 # the standard grid
    python scripts/sweep_features.py --quick         # context length only
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

SR, NFFT, HOP = 48000, 1024, 480
CLASH_MAX_MS, FAR_MIN_MS = 200.0, 400.0
NEAR_MS, FAR_PER_POS, NEAR_PER_POS = (120.0, 400.0), 10, 4
WHIFF_MS, HIT_MS, MIN_LANDED = 1500.0, 75.0, 30
HANN = np.hanning(NFFT).astype(np.float32)


def mel_fb(n_mels, fmin, fmax):
    to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    b = np.clip(np.floor((NFFT + 1) * pts / SR).astype(int), 0, NFFT // 2)
    fb = np.zeros((n_mels, NFFT // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo = b[i]
        mid = max(b[i + 1], lo + 1)
        hi = min(max(b[i + 2], mid + 1), NFFT // 2 + 1)
        mid = min(mid, hi - 1)
        fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


def load_sessions():
    out = []
    for run in sorted((cfgmod.REPO_ROOT / "runs").glob("*/")):
        mp, wp, ep = run / "meta.json", run / "audio.wav", run / "events.jsonl"
        if not (mp.exists() and wp.exists() and ep.exists()):
            continue
        t0 = json.loads(mp.read_text()).get("audio_start_ns")
        if t0 is None:
            continue
        ev = [json.loads(l) for l in open(ep, encoding="utf-8") if l.strip()]
        clash = np.array(sorted(e["t_ns"] for e in ev if e["kind"] == "clash"))
        press = np.array(sorted(e["t_ns"] for e in ev
                                if e["kind"] in ("parry", "dodge")))
        if not len(clash) or not len(press):
            continue
        d = (clash[None, :] - press[:, None]) / 1e6
        landed = press[np.any((d >= 0) & (d <= CLASH_MAX_MS), axis=1)]
        if len(landed) < 2:
            continue
        with wave.open(str(wp)) as w:
            if w.getframerate() != SR:
                continue
            ch = w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw)
        out.append(dict(name=run.name, pcm=pcm.astype(np.float32) / 32768.0,
                        t0=t0, landed=landed, press=press, dur=len(pcm) / SR))
    return out


def spectra(pcm):
    """|STFT| once per session; the filterbank is applied per configuration."""
    fr = np.lib.stride_tricks.sliding_window_view(pcm, NFFT)[::HOP]
    return np.abs(np.fft.rfft(fr * HANN, axis=1)).astype(np.float32)


def simulate(fires, landed, dur):
    unclaimed, ok, whiff, lock = set(range(len(landed))), 0, 0, -1
    for f in fires:
        if f < lock:
            continue
        near = [i for i in unclaimed if abs(f - landed[i]) / 1e6 <= HIT_MS]
        if near:
            ok += 1
            unclaimed.discard(min(near, key=lambda i: abs(f - landed[i])))
        else:
            whiff += 1
            lock = f + int(WHIFF_MS * 1e6)
    return ok / max(len(landed), 1), whiff / (dur / 60)


def evaluate(sessions, nframes, n_mels, fmin, fmax, thresholds, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    fb = mel_fb(n_mels, fmin, fmax)
    mel = {s["name"]: np.log(s["spec"] @ fb.T + 1e-6).astype(np.float32)
           for s in sessions}

    def rows(s, times):
        m = mel[s["name"]]
        out = []
        for t in np.asarray(times, dtype=np.int64):
            fi = int((t - s["t0"]) / 1e9 * SR) // HOP
            if fi - nframes + 1 >= 0 and fi + 1 <= len(m):
                out.append(m[fi - nframes + 1:fi + 1].T.ravel())
        return (np.array(out, np.float32) if out
                else np.zeros((0, n_mels * nframes), np.float32))

    def build(subset):
        X, y = [], []
        for s in subset:
            pos = rows(s, s["landed"])
            if len(pos) < 2:
                continue
            n = len(s["landed"])
            p = s["press"]
            cand = rng.integers(int(p.min()), int(p.max()), size=n * FAR_PER_POS * 2)
            gap = np.abs(cand[:, None] - p[None, :]).min(axis=1) / 1e6
            neg = [rows(s, cand[gap > FAR_MIN_MS][:n * FAR_PER_POS])]
            base = np.repeat(s["landed"], NEAR_PER_POS)
            off = rng.uniform(*NEAR_MS, size=len(base)) * 1e6
            neg.append(rows(s, (base + rng.choice([-1.0, 1.0], len(base)) * off
                                ).astype(np.int64)))
            neg = np.vstack([b for b in neg if len(b)])
            if len(neg) < 2:
                continue
            X.append(np.vstack([pos, neg]))
            y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        return np.vstack(X), np.concatenate(y)

    big = [s for s in sessions if len(s["landed"]) >= MIN_LANDED]
    scored = {}
    for held in big:
        X, y = build([s for s in sessions if s["name"] != held["name"]])
        m = HGB(max_iter=200, random_state=0).fit(X, y)
        M = mel[held["name"]]
        nrow = len(M) - nframes + 1
        p = np.empty(nrow, np.float32)
        for a0 in range(0, nrow, 20000):
            b0 = min(a0 + 20000, nrow)
            v = np.lib.stride_tricks.sliding_window_view(
                M[a0:b0 + nframes - 1], nframes, axis=0)
            p[a0:b0] = m.predict_proba(v.reshape(v.shape[0], -1))[:, 1]
        t = (held["t0"] + (np.arange(nrow) + nframes - 1) * HOP / SR * 1e9)
        scored[held["name"]] = (p, t.astype(np.int64))

    best = None
    for th in thresholds:
        P, W = [], []
        for held in big:
            p, t = scored[held["name"]]
            fires, lock = [], -1
            for i in np.flatnonzero(p >= th):
                if t[i] < lock:
                    continue
                j = i + int(np.argmax(p[i:min(i + 4, len(p))]))
                fires.append(t[j] + int(30e6))
                lock = t[j] + int(250e6)
            a, b = simulate(np.array(fires, np.int64), list(held["landed"]),
                            held["dur"])
            P.append(a)
            W.append(b)
        if best is None or np.mean(P) > best[0]:
            best = (float(np.mean(P)), float(np.mean(W)), th)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--long", action="store_true",
                    help="push context length past 340 ms")
    ap.add_argument("--repeat", type=int, default=0,
                    help="re-evaluate two configs across N seeds to "
                         "measure the noise floor before believing "
                         "any difference between configs")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sessions = load_sessions()
    for s in sessions:
        s["spec"] = spectra(s["pcm"])
        del s["pcm"]
    print(f"  {len(sessions)} sessions, "
          f"{sum(len(s['landed']) for s in sessions)} landed parries")
    print(f"  baseline is nframes=13 (141 ms), 40 mels, 50-16000 Hz -> 49%\n")

    if a.repeat:
        # THE NOISE FLOOR. The same config scored 56% and 50% in two different
        # sweeps -- identical features, different position in the grid, hence a
        # different draw of random negatives from the shared generator. If that
        # spread is typical, no difference in the grid above is interpretable
        # and the apparent monotonic trend in context length was an artifact.
        # Measure the spread before trusting any comparison.
        print(f"  {'frames':>7} {'seed':>5} {'parried':>8} {'whiff/m':>8}")
        for nf in (13, 34):
            got = []
            for seed in range(a.repeat):
                pr, wh, th = evaluate(sessions, nf, 40, 50, 16000,
                                      (0.80, 0.90, 0.95),
                                      np.random.default_rng(seed))
                got.append(pr)
                print(f"  {nf:7d} {seed:5d} {pr:7.0%} {wh:8.1f}", flush=True)
            g = np.array(got)
            print(f"  -> nframes={nf}: mean {g.mean():.1%}, "
                  f"sd {g.std(ddof=1):.1%}, range {g.min():.0%}-{g.max():.0%}")
            print()
        print("  A difference between configs is only real if it exceeds this")
        print("  spread. Anything smaller is the negative sampling talking.")
        return 0

    grid = [(13, 40, 50, 16000), (20, 40, 50, 16000), (26, 40, 50, 16000),
            (34, 40, 50, 16000)]
    if a.long:
        # Parried rose monotonically 52/52/54/56% across 130-340 ms and had not
        # plateaued, so the first sweep simply stopped too early. 141 ms was an
        # arbitrary first guess that could only ever show the model the impact,
        # never the wind-up the player says they read. Trailing context costs
        # no latency -- the window still ends at the decision instant.
        grid = [(34, 40, 50, 16000), (45, 40, 50, 16000), (60, 40, 50, 16000),
                (80, 40, 50, 16000), (34, 40, 50, 8000), (45, 40, 50, 8000),
                (60, 40, 50, 8000), (45, 64, 50, 16000)]
    elif not a.quick:
        grid += [(20, 64, 50, 16000), (26, 64, 50, 16000),
                 (20, 40, 50, 8000), (20, 40, 200, 20000),
                 (13, 64, 50, 16000)]
    ths = (0.80, 0.90, 0.95, 0.99)
    print(f"  {'frames':>7} {'ms':>6} {'mels':>5} {'band':>12} "
          f"{'parried':>8} {'whiff/m':>8} {'th':>5}")
    results = []
    for nf, nm, f0, f1 in grid:
        # A FRESH, IDENTICALLY SEEDED generator per configuration. Sharing one
        # across the grid meant each config was scored against a different draw
        # of random negatives, so its result depended on its position in the
        # grid -- the same config scored 56% evaluated 4th and 50% evaluated
        # 1st. That unpaired comparison had more variance than the effects being
        # measured, and produced a convincing but false trend in context length.
        # Seeding identically pairs the comparison: configs now differ only in
        # their features.
        pr, wh, th = evaluate(sessions, nf, nm, f0, f1, ths,
                              np.random.default_rng(0))
        results.append((pr, wh, nf, nm, f0, f1, th))
        print(f"  {nf:7d} {nf * 10:6d} {nm:5d} {f'{f0}-{f1}':>12} "
              f"{pr:7.0%} {wh:8.1f} {th:5.2f}", flush=True)
    results.sort(reverse=True)
    pr, wh, nf, nm, f0, f1, th = results[0]
    print(f"\n  best: nframes={nf} ({nf * 10} ms), {nm} mels, {f0}-{f1} Hz, "
          f"th={th} -> {pr:.0%} parried at {wh:.1f} whiffs/min")
    print("  (baseline 49%. A gain under ~3 points is inside the noise of "
          "5 held-out sessions and is not worth changing the deployed model for.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
