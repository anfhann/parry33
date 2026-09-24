"""What the ground-truth attack list tells us that keypress labels cannot.

Consumes a punching-bag run: `record_hits.py` captured it, `find_hits.py` read
the HP readout and wrote hits.json. Because the player never parried, every
attack landed, so hits.json is a COMPLETE list of attacks -- not a sample
filtered through what the player noticed and reacted to.

Three things become measurable for the first time:

TRUE RECALL. Every previous recall figure was "of the attacks the player
successfully parried, how many did we also detect". Attacks nobody reacted to
were invisible. Here the denominator is every attack that happened.

THE SOUND -> IMPACT DELAY. How long after the model's score peaks does the blow
actually land. This is the quantity the whole --lead-ms sweep has been groping
at indirectly, and it is per-enemy: it is exactly what per-enemy profiling would
capture.

HONEST NEGATIVES. A high-scoring window far from any real attack is now provably
a false positive, rather than possibly an attack the player whiffed. That is the
label that hard-negative mining needed and did not have -- mining measurably
hurt (49% -> 46%) because it was trained to suppress real detections.

CAVEAT ON THE TIMESTAMP. A hit is stamped when the HP digits CHANGE on screen,
which trails the actual impact by the UI's own reaction plus up to a frame of
capture latency. It is a fixed offset, not noise, so comparisons between attacks
are sound while the absolute delay is an upper bound. Pinning it down needs a
frame-by-frame look at one impact.

    python scripts/analyze_hits.py hits_16893a03aad5c
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

LOOKBACK_MS = 900.0      # how far before an impact the cue may plausibly sit
NEAR_MS = 1200.0         # a fire this close to an attack is "on" that attack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--runs", default=None,
                    help="directory holding runs "
                         "(default <repo>/runs)")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    root = (Path(a.runs) if a.runs
            else cfgmod.REPO_ROOT / "runs") / a.run
    meta = json.loads((root / "meta.json").read_text())
    hits_p = root / "hits.json"
    if not hits_p.exists():
        print(f"  no hits.json -- run: python scripts/find_hits.py {a.run}")
        return 1
    hits = np.array([h["t_ns"] for h in json.loads(hits_p.read_text())["hits"]])
    t0 = meta.get("audio_start_ns")
    if t0 is None or not (root / "audio.wav").exists():
        print("  this run has no aligned audio")
        return 1
    with wave.open(str(root / "audio.wav")) as w:
        ch, sr = w.getnchannels(), w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != G.SR:
        print(f"  audio is {sr} Hz, model expects {G.SR}")
        return 1
    pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw).astype(np.float32) / 32768.0

    bundle = gtrain.load()
    th = a.threshold if a.threshold is not None else bundle.get(
        "threshold", G.THRESHOLD)
    mel = G.mel_frames(pcm)
    n = len(mel) - G.NFRAMES + 1
    p = np.empty(n, np.float32)
    for i in range(0, n, 20000):
        j = min(i + 20000, n)
        p[i:j] = bundle["model"].predict_proba(
            G.stack(mel[i:j + G.NFRAMES - 1]))[:, 1]
    t = (t0 + (np.arange(n) + G.NFRAMES - 1) * G.HOP / G.SR * 1e9).astype(np.int64)

    dur = len(pcm) / G.SR
    print(f"  {len(hits)} ground-truth attacks over {dur:.0f}s "
          f"({len(hits) / (dur / 60):.1f}/min)")
    print(f"  model: threshold {th}, scored {n} windows\n")

    print(f"  {'thresh':>7} {'detected':>9} {'true recall':>12} "
          f"{'peak->impact p50':>18}")
    for probe in (0.5, 0.8, 0.9, 0.95, 0.99):
        found, delays = 0, []
        for h in hits:
            m = (t >= h - int(LOOKBACK_MS * 1e6)) & (t <= h)
            if not m.any():
                continue
            seg, segt = p[m], t[m]
            if seg.max() >= probe:
                found += 1
                delays.append((h - segt[int(np.argmax(seg))]) / 1e6)
        rec = found / max(len(hits), 1)
        d50 = np.median(delays) if delays else float("nan")
        print(f"  {probe:7.2f} {found:9d} {rec:11.0%} {d50:15.0f} ms")

    delays = []
    for h in hits:
        m = (t >= h - int(LOOKBACK_MS * 1e6)) & (t <= h)
        if m.any() and p[m].max() >= th:
            delays.append((h - t[m][int(np.argmax(p[m]))]) / 1e6)
    if delays:
        d = np.array(delays)
        print(f"\n  peak -> impact, at threshold {th} ({len(d)} attacks):")
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3d} {np.percentile(d, q):6.0f} ms")
        print(f"    spread p90-p10 = {np.percentile(d, 90) - np.percentile(d, 10):.0f} ms")
        print("\n  A TIGHT spread means the cue is a reliable landmark for this")
        print("  enemy and a fixed lead will work. A wide one means the enemy has")
        print("  several attacks with different wind-ups, and one offset cannot")
        print("  serve them all -- which is the case for per-enemy attack models.")

    fires = gtrain.fire_times(p, t, threshold=th)
    if len(fires) and len(hits):
        off = np.abs(fires[:, None] - hits[None, :]).min(axis=1) / 1e6
        spurious = int((off > NEAR_MS).sum())
        print()
        print("  SIMULATED presses. Nothing was injected during this recording;")
        print("  the model is replayed over the captured audio to ask where it")
        print("  WOULD have pressed had it been armed.")
        print(f"    {len(fires)} presses: {len(fires) - spurious} near a real "
              f"attack, {spurious} nowhere near one "
              f"({spurious / (dur / 60):.1f}/min)")
        print(f"  Those {spurious} are provably false positives: the player never")
        print("  parried in this run, so 'maybe it was an attack he whiffed'")
        print("  cannot excuse them. They are the honest hard negatives that")
        print("  mining needed and did not have.")
        np.save(root / "spurious_t_ns.npy", fires[off > NEAR_MS])
        print(f"  saved {root / 'spurious_t_ns.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
