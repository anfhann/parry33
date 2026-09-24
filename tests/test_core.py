"""Hardware-light unit tests. No DXGI, no key injection."""

from __future__ import annotations

import numpy as np
import pytest

from parry33 import config as cfgmod
from parry33.clock import PreciseTimer, now_ns
from parry33.util.ringbuffer import FrameRing
from parry33.util.stats import Samples


def test_config_defaults_load():
    cfg = cfgmod.load()
    assert cfg.refresh_hz > 0
    assert cfg.frame_period_ms == pytest.approx(1000.0 / cfg.refresh_hz)
    assert cfg.input.binding("parry", "sendinput").startswith("key:")


def test_capture_region_size():
    c = cfgmod.CaptureConfig(region=(100, 200, 420, 560))
    assert c.size == (320, 360)
    assert cfgmod.CaptureConfig(region=None).size is None


def test_binding_missing_action_raises():
    ic = cfgmod.InputConfig(backend="sendinput", bindings={"sendinput": {"parry": "key:space"}})
    with pytest.raises(KeyError):
        ic.binding("nope")


def test_ring_wraps_and_preserves_order():
    r = FrameRing(4, (2, 2, 4))
    for i in range(6):
        r.write(np.full((2, 2, 4), i, dtype=np.uint8), t_ns=i * 1000)
    assert r.count == 4
    frames, ts = r.latest(4)
    # oldest-first, and the two earliest writes have been overwritten
    assert [int(f[0, 0, 0]) for f in frames] == [2, 3, 4, 5]
    assert list(ts) == [2000, 3000, 4000, 5000]


def test_ring_latest_clamps_to_available():
    r = FrameRing(8, (2, 2, 4))
    r.write(np.zeros((2, 2, 4), np.uint8), 1)
    frames, ts = r.latest(5)
    assert len(frames) == 1 and len(ts) == 1


def test_ring_slot_commit_avoids_a_copy():
    r = FrameRing(2, (2, 2))
    r.slot()[:] = 7
    r.commit(t_ns=42)
    frames, ts = r.latest(1)
    assert frames[0][0, 0] == 7 and ts[0] == 42


def test_samples_percentiles():
    s = Samples("t", capacity=100)
    for v in range(1, 101):
        s.add(v * 1_000_000)          # 1..100 ms
    d = s.summary()
    assert d["n"] == 100
    assert d["p50"] == pytest.approx(50.5, abs=0.6)
    assert d["max"] == pytest.approx(100.0)


def test_samples_respects_capacity():
    s = Samples("t", capacity=3)
    for i in range(10):
        s.add(i)
    assert len(s) == 3


def test_samples_empty_is_safe():
    s = Samples("t", capacity=4)
    assert s.summary() == {"n": 0}
    assert s.sparkline() == ""


@pytest.mark.parametrize("target_ms", [1.0, 5.0, 10.0])
def test_precise_timer_beats_sleep_granularity(target_ms):
    """The whole design leans on sub-ms sleeps; assert we actually have them."""
    t = PreciseTimer(spin_margin_us=300)
    errs = []
    for _ in range(20):
        t0 = now_ns()
        t.sleep(target_ms / 1000.0)
        errs.append(abs((now_ns() - t0) / 1e6 - target_ms))
    t.close()
    assert np.percentile(errs, 90) < 1.0, f"p90 sleep error {np.percentile(errs, 90):.3f} ms"


def test_sendinput_binds_known_scancodes():
    from parry33.input.sendinput import SendInputBackend
    be = SendInputBackend({"parry": "key:space", "m": "mouse:right"},
                          hold_ms=40, async_release=False).prepare()
    assert "sc=0x39" in be.token("parry").label      # space
    assert be.token("m").label == "mouse:right"
    be.close()


def test_sendinput_rejects_bad_specs():
    from parry33.input.sendinput import SendInputBackend
    be = SendInputBackend({}, hold_ms=40, async_release=False)
    for bad in ("key:notakey", "joystick:1", "space"):
        with pytest.raises(ValueError):
            be.bind(bad)


def test_null_backend_async_release_timing():
    from parry33.input import base as inbase
    import time
    cfg = cfgmod.InputConfig(backend="null", hold_ms=25.0,
                             bindings={"null": {"parry": "key:space"}})
    be = inbase.build(cfg)
    be.tap("parry")
    time.sleep(0.1)
    be.close()
    assert [e[1] for e in be.events] == ["down", "up"]
    held = (be.events[1][0] - be.events[0][0]) / 1e6
    assert held == pytest.approx(25.0, abs=2.0)


def test_synthetic_capture_paces_frames():
    from parry33.capture.synthetic import SyntheticCapture
    cap = SyntheticCapture(region=(0, 0, 64, 64), fps=200)
    with cap:
        ts = [cap.grab().t_ns for _ in range(30)]
    gaps = np.diff(ts) / 1e6
    assert np.percentile(gaps, 50) == pytest.approx(5.0, abs=0.7)
