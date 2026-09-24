"""Onset detector tests. No audio hardware required.

Both of the first two tests are regressions for bugs that cost real measurement
time: the detector was unfirable against digital silence, and its refractory
lockout leaked across trials because it counted blocks instead of wall time.
"""

from __future__ import annotations

import numpy as np
import pytest

from parry33.audio import onset as om
from parry33.audio.capture import AudioRing

SR, BS = 48000, 512
BLOCK_NS = int(BS / SR * 1e9)


def _silence():
    return np.zeros(BS, dtype=np.float32)


def _burst(rng, amp=0.25):
    return (rng.standard_normal(BS) * amp).astype(np.float32)


def _warm(det, t0, n=60):
    """Push enough silence to clear the warmup gate."""
    t = t0
    for _ in range(n):
        t += BLOCK_NS
        det.push(_silence(), t_ns=t)
    return t


@pytest.mark.parametrize("kind", ["flux", "rms"])
def test_fires_against_digital_silence(kind):
    """A silent baseline gives zero variance; the threshold must not become unfirable."""
    rng = np.random.default_rng(0)
    det = om.build(kind, SR, BS)
    t = _warm(det, 0)
    t += BLOCK_NS
    assert det.push(_burst(rng), t_ns=t) is True


@pytest.mark.parametrize("kind", ["flux", "rms"])
def test_refractory_is_wall_clock_not_block_count(kind):
    """A gap in feeding must not carry a lockout into the next event.

    The lockout used to decrement per pushed block, so a pause between trials
    left it armed and swallowed the start of the next burst.
    """
    rng = np.random.default_rng(1)
    det = om.build(kind, SR, BS, refractory_ms=80.0)
    t = _warm(det, 0)
    t += BLOCK_NS
    assert det.push(_burst(rng), t_ns=t) is True

    # Nothing pushed for 500 ms, then a fresh burst. Must fire.
    t += int(500e6)
    assert det.push(_burst(rng), t_ns=t) is True


@pytest.mark.parametrize("kind", ["flux", "rms"])
def test_refractory_suppresses_a_sustained_sound(kind):
    """One long sound is one onset, not five."""
    rng = np.random.default_rng(2)
    det = om.build(kind, SR, BS, refractory_ms=80.0)
    t = _warm(det, 0)
    fires = 0
    for _ in range(6):                      # 6 blocks ~= 64 ms, inside the lockout
        t += BLOCK_NS
        fires += det.push(_burst(rng), t_ns=t)
    assert fires == 1


def test_silence_produces_no_onsets():
    rng = np.random.default_rng(3)
    det = om.build("flux", SR, BS)
    t = _warm(det, 0)
    t += BLOCK_NS
    det.push(_burst(rng), t_ns=t)           # establish a peak
    fires = 0
    for _ in range(200):
        t += BLOCK_NS
        fires += det.push(_silence(), t_ns=t)
    assert fires == 0


def test_flux_ignores_steady_tone_but_catches_its_onset():
    """Spectral flux should fire when a tone starts, then go quiet while it holds."""
    det = om.build("flux", SR, BS)
    t = _warm(det, 0)
    tone = (0.3 * np.sin(2 * np.pi * 1000 *
                         np.arange(BS) / SR)).astype(np.float32)
    t += BLOCK_NS
    assert det.push(tone, t_ns=t) is True
    fires = 0
    for _ in range(40):                     # well past the 80 ms lockout
        t += BLOCK_NS
        fires += det.push(tone, t_ns=t)
    assert fires == 0


def test_ring_is_sequential_not_latest_value():
    r = AudioRing(4, 8)
    for i in range(3):
        r.write(np.full(8, i, dtype=np.float32), t_ns=i)
    for i in range(3):
        blk, t = r.read()
        assert blk[0] == i and t == i
    assert r.read() is None


def test_ring_counts_overruns_and_drops_oldest():
    r = AudioRing(4, 8)
    for i in range(10):
        r.write(np.full(8, i, dtype=np.float32), t_ns=i)
    assert r.overruns > 0
    blk, _ = r.read()
    assert blk[0] > 0                        # oldest was discarded, not the newest


def test_live_features_match_training_exactly():
    """The live path and the training path must compute identical features.

    A mismatch here produces garbage scores rather than an error, so it is worth
    a test. This caught a one-frame timestamp misalignment: training pairs
    np.diff(f) with rel[:-1] (the earlier frame), and the live stream was
    stamping each diff with the later frame's time.
    """
    import json
    from pathlib import Path
    from parry33 import config as cfgmod, learn
    from parry33.live import GridStream

    runs = [d for d in (cfgmod.REPO_ROOT / "runs").glob("*/")
            if (d / "events.jsonl").exists() and list((d / "clips").glob("*.npz"))]
    if not runs:
        pytest.skip("no recorded runs with clips available")
    run = runs[-1]
    ev = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    clips = [e for e in ev if e.get("clip")][:5]
    if not clips:
        pytest.skip("no clips in the latest run")

    checked = 0
    for e in clips:
        z = np.load(run / e["clip"])
        f, ts = z["frames"], z["t_ns"]
        ref = learn.features_from_frames(f.astype(np.float32),
                                         (ts - e["t_ns"]) / 1e6,
                                         win=(-1000.0, -500.0))
        if ref is None:
            continue
        gs = GridStream()
        for i in range(len(f)):
            gs.push(f[i], int(ts[i]))
        live = gs.features(int(e["t_ns"] - 500e6), window_ms=500.0)
        if live is None:
            continue
        assert np.allclose(ref, live, atol=1e-4), "live/training feature mismatch"
        checked += 1
    assert checked > 0


def test_grunt_stream_matches_offline_features():
    """Streaming and offline feature paths must produce identical rows.

    Training reads whole WAVs through mel_frames/stack; serving reads blocks
    through GruntStream. If those two disagree by even one frame of alignment,
    the model scores noise at serve time and degrades silently rather than
    raising -- which is precisely how the first vision model failed. Block
    sizes are varied because the audio callback size is not guaranteed to
    divide the hop.
    """
    import numpy as np
    from parry33.audio import grunt as G

    class Spy:
        def __init__(self):
            self.seen = []

        def predict_proba(self, v):
            self.seen.append(v.ravel().copy())
            return np.array([[1.0, 0.0]])

    pcm = (np.random.default_rng(0).standard_normal(24000) * 0.1).astype(np.float32)
    offline = G.stack(G.mel_frames(pcm))
    for blk in (256, 512, 480, 137):
        spy = Spy()
        stream = G.GruntStream(spy)
        for i in range(0, len(pcm) - blk, blk):
            stream.push(pcm[i:i + blk], i * 1000)
        live = np.array(spy.seen)
        n = min(len(offline), len(live))
        assert n > 10, f"blocksize {blk} produced too few rows"
        assert np.abs(offline[:n] - live[:n]).max() < 1e-3, \
            f"feature mismatch at blocksize {blk}"
