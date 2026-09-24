"""Live parry trigger.

Runs the detector against the screen in real time and, when armed, injects the
parry key. Defaults to dry-run: it logs every decision but touches nothing.

The loop must fit inside one frame period (16.7 ms at 60 fps) or it drops
frames, and Desktop Duplication has no backlog -- a dropped frame is gone. The
naive approach, recomputing the full feature vector over a 500 ms window on
every score, costs ~60 ms and is hopeless.

The fix is exact rather than approximate. Feature aggregation is linear in the
frames: averaging a spatial grid over frames equals gridding the frame-average.
So each frame-pair's 6x8 motion grid is computed once as the frame arrives
(~1 ms) and the window feature vector is assembled from stored grids (~0.1 ms).
The numbers are identical to training, which matters -- a feature mismatch
between training and inference produces garbage scores, not an error.

Timing model, from the offline work:

    detector fires on a window ending at T
      -> the human's keypress falls ~500 ms after T
      -> so schedule the press at T + press_delay_ms

Firing requires N consecutive windows above threshold. Isolated false positives
do not persist; a real tell does. Offset-augmented training is what makes the
score stable enough across a sliding window for this to work.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import model as modelmod
from .capture import base as capbase
from .clock import PreciseTimer, now_ns
from .input import base as inbase
from .input.keywatch import KeyWatcher
from .util import prio

GH, GW, TB = 6, 8, 6          # must match learn.features_from_frames
WINDOW_MS = 500.0


# Derived from 594 frames of real combat across three sessions (p1..p99 with
# margin); passes 98.5% of them. A death screen fails all three at once.
COMBAT_MEAN = (17.0, 205.0)
COMBAT_STD_MIN = 14.0
COMBAT_DARK_MAX = 0.60
# Combat motion measured over 6501 frames: p5 0.31, p50 5.94, p95 19.74.
# Static screens (FAILED, victory, alt-tabbed desktop) sit at ~0.0. A floor of
# 1.0 is far from both and keeps ~92% of combat frames -- and most of what it
# drops is the skill menu, which is the player's turn and should not fire
# anyway, so the motion floor doubles as a turn gate.
MOTION_FLOOR = 1.0
# Never judge the firing rate before this much time has elapsed.
GUARD_WARMUP_S = 60.0
# Minimum gap between fires. A whiffed parry locks the character out for 1500 ms,
# so firing again inside that window is worse than useless -- the press is eaten
# by the lockout AND it cannot help. A sustained high score used to fire every
# 200 ms; 12 of 13 gaps in the first successful armed run were under 1.5 s.
# The offline tuner assumed an 800 ms refractory when it produced the 88% recall
# figure, so the live rule was never the rule that was measured.
FIRE_REFRACTORY_MS = 1200.0

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_u32.GetForegroundWindow.restype = ctypes.c_void_p
VK_F10 = 0x79


def foreground_title() -> str:
    """Title of the focused window. The surest alt-tab defence: if the game is
    not in front, nothing we see is the game."""
    try:
        h = _u32.GetForegroundWindow()
        if not h:
            return ""
        n = _u32.GetWindowTextLengthW(ctypes.c_void_p(h))
        buf = ctypes.create_unicode_buffer(n + 1)
        _u32.GetWindowTextW(ctypes.c_void_p(h), buf, n + 1)
        return buf.value
    except Exception:  # noqa: BLE001 - never let a gate check kill the loop
        return ""


HUD_ROWS = 26          # top strip of the downsampled frame holding the name plate
BAR_RED_MIN = 40       # red-fill pixels needed to call it a boss HP bar


def hp_bar_red(hud_bgr: np.ndarray) -> int:
    """Red-fill pixels in the boss HP bar, or 0 if no bar is present.

    The bar is a thin horizontal strip: dark red fill on the left, black on the
    right. Presence is a far easier question than fill level, and it is the one
    that separates "in a fight" from "walking into the arena" -- which no
    whole-frame statistic can, because arena entry is bright and full of motion
    and sails past a gate designed to reject static screens.

    Measured on 51 real combat sheets: every one has red fill, p5 = 98 px.
    Requiring the pixels to be dark AND red-dominant excludes the sandy orange
    arena background, which is red-dominant but bright.
    """
    if hud_bgr is None or hud_bgr.ndim != 3 or hud_bgr.shape[2] < 3:
        return 0
    t = hud_bgr[..., :3].astype(np.int16)
    b, g, r = t[..., 0], t[..., 1], t[..., 2]      # capture is BGRA
    lum = t.mean(2)
    red = (r - np.maximum(g, b) > 18) & (lum < 150) & (lum > 18)
    dark = lum < 60
    rows = (red | dark).sum(1)
    if rows.max() < 40:
        return 0
    return int(red[int(np.argmax(rows))].sum())


def in_combat(frame: np.ndarray, recent_motion: float,
              require_title: str = "", hud_bgr=None,
              require_bar: bool = True) -> tuple:
    """Is this frame a live combat scene? Returns (ok, reason).

    Without this gate the detector is fed victory screens, death screens, menus
    and the alt-tabbed desktop -- none of which it has ever seen, because every
    control clip was recorded within 15 s of a real attack. Out of distribution
    it does not fail gracefully, it saturates: a dry run produced 40 consecutive
    fires at p=1.00 after the fight ended.

    An earlier version of this gate keyed on "longest dark run near the top",
    intending to find the boss HP bar. That returns the FULL frame width on a
    black screen, so the death screen -- the single most important thing to
    reject -- scored maximally and sailed through. Statistics of the whole frame
    are both simpler and not invertible in that way.
    """
    if require_title:
        t = foreground_title()
        if require_title.lower() not in t.lower():
            return False, "unfocused"
    m = float(frame.mean())
    if not (COMBAT_MEAN[0] <= m <= COMBAT_MEAN[1]):
        return False, f"mean {m:.0f}"
    sd = float(frame.std())
    if sd < COMBAT_STD_MIN:
        return False, f"flat std {sd:.0f}"
    dk = float((frame < 30).mean())
    if dk > COMBAT_DARK_MAX:
        return False, f"dark {dk:.2f}"
    # Alt-tabbed or paused: the desktop can pass the statistical test, but it
    # does not move. The game always does.
    if recent_motion < MOTION_FLOOR:
        return False, f"static {recent_motion:.2f}"
    # The boss HP bar only exists during a fight. Arena entry, cutscenes and the
    # overworld are all bright and moving, so nothing above rejects them.
    if require_bar:
        red = hp_bar_red(hud_bgr)
        if red < BAR_RED_MIN:
            return False, f"no bar ({red})"
    return True, "ok"


def _finish(logf, featpath, feat_rows, feat_ts, log_features) -> None:
    """Flush the log and the served feature vectors.

    Called from a finally block so an interrupt cannot discard a session. The
    first attempt at feature logging wrote nothing at all because the save sat
    after the main loop and a patch to add it silently failed to apply -- a
    whole fight's worth of served windows was collected in memory and dropped.
    """
    try:
        logf.close()
    except Exception:  # noqa: BLE001 - teardown must not raise
        pass
    if log_features and feat_rows:
        try:
            np.savez_compressed(featpath,
                                X=np.array(feat_rows, np.float32),
                                t_ns=np.array(feat_ts, np.int64))
            mb = featpath.stat().st_size / 1e6
            print(f"\n  features: {featpath.name} "
                  f"({len(feat_rows)} windows, {mb:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"\n  WARNING: could not save features: {e}")


class GridStream:
    """Rolling per-frame motion grids, aggregated into training-identical features."""

    def __init__(self, capacity: int = 240) -> None:
        self.cap = capacity
        self.grids = np.zeros((capacity, GH, GW), np.float32)
        self.totals = np.zeros(capacity, np.float32)
        self.ts = np.zeros(capacity, np.int64)
        self.n = 0
        self._prev = None
        self._prev_t = 0

    def push(self, frame: np.ndarray, t_ns: int) -> None:
        f = frame.astype(np.float32)
        if self._prev is not None:
            d = np.abs(f - self._prev)
            H, W = d.shape
            if H % GH == 0 and W % GW == 0:
                g = d.reshape(GH, H // GH, GW, W // GW).mean(axis=(1, 3))
            else:
                from .learn import _resize_mean
                g = _resize_mean(d, GH, GW)
            i = self.n % self.cap
            self.grids[i] = g
            self.totals[i] = d.mean()
            # Stamp the diff with the EARLIER frame's time. Training pairs
            # np.diff(f) with rel[:-1], so a diff belongs to the frame it starts
            # from. Using the later frame here shifts every sample by one frame
            # and silently changes which time bin it lands in.
            self.ts[i] = self._prev_t
            self.n += 1
        self._prev = f
        self._prev_t = t_ns

    def recent_motion(self, k: int = 6) -> float:
        """Mean motion over the last k frame-pairs.

        Must index through the ring modulus. `totals` is a fixed `cap`-slot
        array while `n` counts up forever, so slicing totals[n-k:n] runs off the
        end once n exceeds cap and returns NaN -- and NaN < floor is False, so
        the motion gate silently stopped rejecting anything about four seconds
        into every run.
        """
        if self.n == 0:
            return 0.0
        k = min(k, self.n, self.cap)
        idx = (np.arange(self.n - k, self.n) % self.cap)
        return float(self.totals[idx].mean())

    def features(self, end_ns: int, window_ms: float = WINDOW_MS):
        """Feature vector for the window ending at end_ns, or None if too sparse."""
        if self.n < TB * 2:
            return None
        k = min(self.n, self.cap)
        idx = (np.arange(self.n - k, self.n) % self.cap)
        ts = self.ts[idx]
        rel = (ts - end_ns) / 1e6
        m = (rel >= -window_ms) & (rel <= 0)
        if m.sum() < 6:
            return None
        g, tot, r = self.grids[idx][m], self.totals[idx][m], rel[m]
        edges = np.linspace(-window_ms, 0.0, TB + 1)
        cells, totals = [], []
        for i in range(TB):
            sel = (r >= edges[i]) & (r <= edges[i + 1])
            if sel.sum() == 0:
                sel = np.zeros(len(r), bool)
                sel[min(i, len(r) - 1)] = True
            cells.append(g[sel].mean(0))
            totals.append(tot[sel].mean())
        cells = np.array(cells)
        totals = np.array(totals)
        share = cells / np.maximum(totals[:, None, None], 1e-6)
        return np.concatenate([cells.ravel(), share.ravel(), totals])


def run(cfg=None, seconds: float = 300.0, arm: bool = False,
        threshold: float = 0.5, consecutive: int = 2,
        press_delay_ms: float = 500.0, score_every_ms: float = 100.0,
        action: str = "parry", delay: float = 0.0, out_root: Path | None = None,
        require_title: str = "", max_fires_per_min: float = 30.0,
        require_bar: bool = True,
        log_features: bool = True):
    cfg = cfg or cfgmod.load()
    prio.apply(cfg.timing)
    bundle = modelmod.load()
    mdl = bundle["model"]

    backend = None
    if arm:
        backend = inbase.build(cfg.input)

    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game")
        for r in range(int(delay), 0, -1):
            print(f"  {r}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 24, end="\r")

    # Log the human's own presses next to the detector's decisions. Without
    # ground truth in the same file we can count fires but cannot tell a true
    # detection from a false alarm, which is the only number that matters.
    # Injected input is ignored by the hook, so an armed run does not log its
    # own keystrokes as if they were the player's.
    keys = KeyWatcher(watch=("q", "e"))
    video = capbase.build(cfg.capture)
    stream = GridStream()
    last_small = None
    last_hud = None
    timer = PreciseTimer(spin_margin_us=cfg.timing.spin_margin_us)
    # Log incrementally. The first dry run was interrupted mid-session and the
    # entire fire log was lost because it was only written at the end.
    out_root = out_root or (cfgmod.REPO_ROOT / "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    stamp_id = f"{now_ns():x}"
    logpath = out_root / f"live_{stamp_id}.jsonl"
    logf = open(logpath, "w", encoding="utf-8", buffering=1)
    # Feature vectors for every scored window. Offline training used clips
    # centred on events; live scores a continuously sliding window that spends
    # most of its time in states no clip ever covered. Logging the actual served
    # features lets us train on the serving distribution instead of a proxy for
    # it -- ~3.7 MB per 3-minute run.
    featpath = out_root / f"live_{stamp_id}_feats.npz"
    feat_rows = []
    feat_ts = []
    log = []
    pending = []          # scheduled presses: (fire_at_ns, detect_ns)
    gated = 0
    human = 0
    gate_reasons = {}
    timeouts = 0
    run_len = 0
    fires = 0
    frames = 0
    score_ns = int(score_every_ms * 1e6)
    scored = 0
    cost = []

    try:
      with video, keys:
          first = video.grab(timeout_ms=1000)
          if first is None:
              raise RuntimeError("no frames -- is the game running?")
          sy = max(1, first.data.shape[0] // 270)
          sx = max(1, first.data.shape[1] // 480)

          mode = "ARMED - will inject" if arm else "DRY RUN - logging only"
          print(f"live trigger [{mode}]")
          print(f"  model AUC {bundle['auc']:.3f} (floor {bundle['floor']:.3f}), "
                f"{bundle['n_pos']} pos / {bundle['n_neg']} neg")
          print(f"  rule: {consecutive} consecutive windows >= {threshold}")
          print(f"  press scheduled {press_delay_ms:.0f} ms after detection")
          print(f"  action: {action}\n")

          t_start = now_ns()
          t_end = t_start + int(seconds * 1e9)
          next_score = t_start
          warned = False
          fire_lock = 0
          while now_ns() < t_end:
              try:
                  f = video.grab(timeout_ms=50)
              except Exception:
                  # DXGI raises on AcquireNextFrame timeout when the desktop is
                  # static (game minimised, alt-tabbed). Not fatal.
                  timeouts += 1
                  f = None
              if f is not None:
                  small_bgr = f.data[::sy, ::sx]
                  small = small_bgr[..., 1]          # green for motion features
                  last_hud = small_bgr[:HUD_ROWS]    # colour strip for the bar gate
                  stream.push(small, f.t_ns)
                  frames += 1
                  last_small = small

              now = now_ns()
              if now >= next_score:
                  next_score = now + score_ns
                  t0 = now_ns()
                  recent = stream.recent_motion()
                  ok, why = ((False, "no frame") if last_small is None
                             else in_combat(last_small, recent, require_title,
                                            last_hud, require_bar))
                  v = stream.features(now) if ok else None
                  if not ok:
                      gated += 1
                      run_len = 0
                      gate_reasons[why] = gate_reasons.get(why, 0) + 1
                  if v is not None:
                      p = float(mdl.predict_proba(v.reshape(1, -1))[0, 1])
                      cost.append(now_ns() - t0)
                      scored += 1
                      # Log EVERY score, not just the ones that fire. Without the
                      # non-firing scores the threshold and consecutive-window
                      # rule cannot be tuned offline, and each retune costs a
                      # play session.
                      logf.write(json.dumps({"t_ns": now, "kind": "score",
                                             "p": round(p, 3),
                                             "motion": round(recent, 1)}) + chr(10))
                      if log_features:
                          feat_rows.append(v.astype(np.float32))
                          feat_ts.append(now)
                      run_len = run_len + 1 if p >= threshold else 0
                      if run_len >= consecutive and now >= fire_lock:
                          run_len = 0
                          fire_lock = now + int(FIRE_REFRACTORY_MS * 1e6)
                          fires += 1
                          fire_at = now + int(press_delay_ms * 1e6)
                          pending.append((fire_at, now))
                          rec = {"t_ns": now, "kind": "fire", "score": round(p, 3),
                                 "motion": round(recent, 2), "fire_at_ns": fire_at}
                          log.append(rec)
                          logf.write(json.dumps(rec) + chr(10))
                          # Judge the firing rate only after a warm-up. Rate is
                          # fires/elapsed, so in the first seconds any burst
                          # reads as hundreds per minute: this guard was killing
                          # every session within ~20 s of starting rather than
                          # catching real runaway, losing whole fights.
                          # And in dry run nothing is pressed, so there is
                          # nothing to protect -- warn, do not stop.
                          elapsed = max((now - t_start) / 1e9, 1e-6)
                          rate = fires / elapsed * 60
                          if (elapsed > GUARD_WARMUP_S
                                  and rate > max_fires_per_min and not warned):
                              warned = True
                              print()
                              print(f"  WARNING: {rate:.0f} fires/min exceeds "
                                    f"{max_fires_per_min:.0f}")
                              if arm:
                                  print("  ARMED -- stopping to avoid key spam.")
                                  t_end = 0
                              else:
                                  print("  dry run, nothing pressed -- continuing.")
                          print(f"\r  FIRE #{fires} p={p:.2f} "
                                f"(press in {press_delay_ms:.0f}ms)" + " " * 20)

              for t_key, name, down in keys.drain():
                  if down:
                      rec = {"t_ns": int(t_key), "kind":
                             "human_parry" if name == "e" else "human_dodge"}
                      logf.write(json.dumps(rec) + chr(10))
                      human += 1

              if _u32.GetAsyncKeyState(VK_F10) & 0x8000:
                  print()
                  print("  F10 - stopping")
                  break

              still = []
              for fire_at, det in pending:
                  if now < fire_at:
                      still.append((fire_at, det))
                      continue
                  if arm and backend is not None:
                      backend.tap(action)
              pending = still

              if frames % 60 == 0:
                  print(f"\r  {frames} frames, {scored} scores, {fires} fires", end="")

    finally:
        _finish(logf, featpath, feat_rows, feat_ts, log_features)
    timer.close()
    if backend is not None:
        backend.close()
    print()
    c = np.array(cost) / 1e6 if cost else np.array([0.0])
    dur = seconds
    print(f"\n  {frames} frames ({frames/dur:.0f}/s), {scored} scores, {fires} fires "
          f"({fires/dur*60:.1f}/min)")
    print(f"  score cost: p50 {np.percentile(c,50):.2f} ms  p99 {np.percentile(c,99):.2f} ms")
    out = (out_root or (cfgmod.REPO_ROOT / "runs")) / f"live_{now_ns():x}.json"
    out.write_text(json.dumps({"armed": arm, "threshold": threshold,
                               "consecutive": consecutive,
                               "press_delay_ms": press_delay_ms,
                               "frames": frames, "scores": scored, "fires": fires,
                               "fire_log": log}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return out
