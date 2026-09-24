"""Live parry trigger driven by the learned attack-sound model.

HISTORY, because the shape of this file is a reaction to two measured failures.

The single-stage *vision* trigger failed because its "an attack is happening"
signal stays high for 300-1300 ms while the parry window is 150 ms. It knows
whether, not when. Firing at the start of that span plus a fixed offset landed
early on some attacks and on the recovery frames of others.

The two-stage vision-gates-audio-fires trigger failed for a subtler reason. Its
audio stage was a spectral-flux onset detector, which fires on any transient --
about 3.4 times a second in a combat mix. "First transient after the gate
opened" is not a fixed point in an attack, so the press time was set by when the
gate happened to open rather than by the attack itself. Measured gate-open to
fire delays were bimodal at ~4-90 ms and ~1270-1440 ms: consistent within each
mode, unrelated to the enemy. The player described it that way before the log
confirmed it.

So the audio stage is now a learned model (see audio/grunt.py) that answers
"should we press *now*" rather than "did something just happen". It carries its
own timing, which is why there is no longer a fixed press delay to tune.

THE MECHANIC drives the firing policy:

    landed parry  -> instant reframe, 0 ms. Can parry again immediately.
    whiffed parry -> 1500 ms lockout, during which real attacks are simply lost.

Correct presses are free; wrong ones are expensive and can cost the *next*
attack too. Hence a high threshold with a short refractory: reluctant to fire,
but able to fire again at once for combos.

Honest performance, leave-one-session-out over 272 labelled attacks, simulating
that mechanic: ~48% of real attacks parried at ~4-5 whiffs/min. That is well
short of a comfortable win. The dominant remaining loss is press timing inside a
correctly detected attack, not failure to detect it.

THE VISION GATE is off by default. It existed to suppress a noisy audio stage,
and the learned model is not noisy in that way -- it fires single-digit times
per minute, not 200. More importantly the 48% above was measured audio-only, so
running with the gate on means running a configuration nobody has measured. Turn
it on with --gate to test it as a deliberate change, not as an unexamined
default.
"""

from __future__ import annotations

import ctypes
import json
import os
from contextlib import ExitStack

import numpy as np
from pathlib import Path

from . import config as cfgmod
from .audio import grunt as G
from .audio import onset as onsetmod
from .audio import train as gtrain
from .audio.capture import LoopbackCapture
from .capture import base as capbase
from .clock import PreciseTimer, now_ns
from .input import base as inbase
from .input.keywatch import KeyWatcher
from .util import prio

VK_F10 = 0x79

# Julien's rotation, as named by the player. Typed live on the numpad while the
# bot plays, so each press can be attributed to the attack it was aimed at.
ATTACK_NAMES = {1: "Left Jab", 2: "Long Left Jab", 3: "Left Jab Combo (3)",
                4: "Right Jab Combo (2)", 5: "Rotating Jab", 6: "Jump Attack"}


def run(cfg=None, seconds: float = 600.0, arm: bool = False,
        threshold: float | None = None, refractory_ms: float | None = None,
        lookahead_ms: float | None = None, lead_ms: float = 0.0,
        lead_sweep: str = "", grade_hp: bool = False,
        hp_roi: tuple = (2240, 1230, 2520, 1370),
        boss_roi: tuple = (780, 55, 1800, 90), action: str = "parry",
        delay: float = 0.0, gate: bool = False, vision_threshold: float = 0.5,
        consecutive: int = 2, gate_ms: float = 1500.0,
        device: int | None = None, out_root: Path | None = None):
    cfg = cfg or cfgmod.load()
    prio.apply(cfg.timing)

    # One row of 520 features through 200 tiny trees should be microseconds;
    # it measured 4.62 ms because sklearn launches an OpenMP team per tree.
    # Pinning to one thread halves it (4.62 -> 2.98 ms p50). Must be set before
    # sklearn's compiled extensions load, which happens inside gtrain.load().
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    bundle = gtrain.load()
    threshold = G.THRESHOLD if threshold is None else threshold
    refractory_ms = G.REFRACTORY_MS if refractory_ms is None else refractory_ms
    lookahead_ms = G.LOOKAHEAD_MS if lookahead_ms is None else lookahead_ms

    backend = inbase.build(cfg.input) if arm else None
    video = mdl = stream = None
    if gate:
        from . import model as modelmod
        from .live import GridStream
        mdl = modelmod.load()["model"]
        video = capbase.build(cfg.capture)
        stream = GridStream()
    # Grade presses by the HP readout rather than by clashes. A clash marks an
    # attack making CONTACT -- measured, 100% of attacks that landed on an
    # unparried player produced one -- so it cannot distinguish a successful
    # parry from a whiff. Scored by clash, an armed run reported 14/14 landed
    # when the victory screen said 1. HP is unambiguous: parried means no damage.
    if grade_hp and video is None:
        video = capbase.build(cfg.capture)

    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game")
        for r in range(int(delay), 0, -1):
            print(f"  {r}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 24, end="\r")

    out_root = out_root or (cfgmod.REPO_ROOT / "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    logpath = out_root / f"trig_{now_ns():x}.jsonl"
    logf = open(logpath, "w", encoding="utf-8", buffering=1)

    audio = LoopbackCapture(
        device_index=device if device is not None else cfg.audio.device_index,
        blocksize=cfg.audio.blocksize)
    # Numpad 1-6 label the attack type live: 1 Left Jab, 2 Long Left Jab,
    # 3 Left Jab Combo, 4 Right Jab Combo, 5 Rotating Jab, 6 Jump Attack.
    # The numpad is used rather than the top row because top-row digits act in
    # combat. A label costs nothing while the bot plays and is the only way to
    # tell which attack a press was aimed at -- the jump attack, for instance,
    # appears to draw no press at all, which a per-attack breakdown would show.
    # Both numpad and top-row digits are watched. The numpad is preferred
    # because top-row digits act in combat, but numpad codes only arrive when
    # NumLock is ON -- with it off the keys send End/Down/PageDown instead and
    # nothing is logged at all, which is silent and confusing.
    LABEL_KEYS = ("num1", "num2", "num3", "num4", "num5", "num6",
                  "1", "2", "3", "4", "5", "6")
    keys = KeyWatcher(watch=("q", "e") + LABEL_KEYS)
    timer = PreciseTimer(spin_margin_us=cfg.timing.spin_margin_us)
    u32 = ctypes.WinDLL("user32", use_last_error=True)

    fires = human = blocked = labels = 0
    pending: list[int] = []
    peak_p = 0.0
    det = None

    # RETRACTED: clash-based self-scoring.
    #
    # This used to grade each press by whether a clash followed within 250 ms,
    # on the belief that a clash meant a successful parry. It does not. Measured
    # on punching-bag runs where the player parried nothing, 100% of attacks
    # that LANDED still produced a clash within 400 ms -- a clash marks contact,
    # not a parry. Scored that way an armed run read 14/14 landed when the
    # victory screen said 1.
    #
    # The clash counts are still logged, because they do locate attacks, but
    # "landed" now comes from the HP readout: if the parry worked, no damage.
    sweep = [float(x) for x in lead_sweep.split(",") if x.strip()]
    rng = __import__("random").Random(0)
    outcomes: list[dict] = []          # {press_at, lead, landed}
    CLASH_WINDOW_MS = 250.0
    hp_frames: list = []
    hp_ts: list = []
    # The BOSS bar, as red-excess. A successful parry triggers a counter, and
    # measured across 9 events the boss loses health 1961 ms later with a 43 ms
    # spread -- so a boss-HP drop at +1961 ms after a press is proof that press
    # parried. The player's own HP can only ever show FAILURES, because a parried
    # attack leaves no mark on it. This is the only direct evidence of success.
    boss_frames: list = []
    boss_lum: list = []
    pcm_log: list = []
    audio_t0 = None

    try:
        # The video capture must be ENTERED, not merely constructed -- building
        # it does not start Desktop Duplication. Without this every grab raises
        # into the bare except below, which starved the audio drain badly enough
        # that a --gate run scored zero windows where audio-only scores ~490 in
        # the same five seconds.
        with ExitStack() as stack:
            stack.enter_context(audio)
            stack.enter_context(keys)
            if video is not None:
                stack.enter_context(video)
            # Geometry from the bundle, not from module constants, so a model
            # trained with different features cannot be served with the wrong
            # window shape.
            det = G.GruntStream(bundle["model"], threshold=threshold,
                                lookahead_ms=lookahead_ms,
                                refractory_ms=refractory_ms,
                                nframes=bundle.get("nframes"),
                                nmel=bundle.get("nmel"),
                                fmin=bundle.get("fmin"),
                                fmax=bundle.get("fmax"))
            clash = onsetmod.SpectralFluxOnset(
                audio.samplerate, cfg.audio.blocksize,
                sensitivity=cfg.audio.clash_sensitivity,
                refractory_ms=cfg.audio.refractory_ms,
                floor_frac=cfg.audio.floor_frac,
                fmin=cfg.audio.clash_band[0],
                fmax=min(cfg.audio.clash_band[1], audio.samplerate / 2))
            mode = "ARMED - will inject" if arm else "DRY RUN - logging only"
            print(f"learned trigger [{mode}]")
            print(f"  model     : {bundle['n_pos']} landed parries, "
                  f"{bundle['n_neg']} negatives, {bundle['n_hard']} hard")
            print(f"  policy    : fire at p>={threshold}, confirm peak over "
                  f"{lookahead_ms:.0f} ms, refractory {refractory_ms:.0f} ms")
            if sweep:
                print(f"  lead      : SWEEPING {sweep} ms"
                      + (", graded by HP" if grade_hp else
                         "  -- WITHOUT --grade-hp there is no honest grader; "
                         "clash scoring was retracted"))
            elif lead_ms:
                print(f"  lead      : pressing {lead_ms:.0f} ms earlier than "
                      f"the model asks")
            print(f"  audio in  : {audio.device_name}")
            print(f"  gate      : {'vision ON' if gate else 'off (audio only)'}")
            print("  F10 to stop\n")

            t_end = now_ns() + int(seconds * 1e9)
            gate_until = 0
            run_len = 0
            hl, ht, hr, hb = (int(x) for x in hp_roi)
            bl2, bt2, br2, bb2 = (int(x) for x in boss_roi)
            while now_ns() < t_end:
                if grade_hp and not gate:
                    try:
                        vf = video.grab(timeout_ms=2)
                    except Exception:
                        vf = None
                    if vf is not None:
                        # max over B,G,R: the HP number turns red at low health
                        # and vanishes from the green channel alone.
                        hp_frames.append(
                            vf.data[ht:hb, hl:hr, :3].max(axis=2).copy())
                        bb_ = vf.data[bt2:bb2, bl2:br2]
                        boss_frames.append(
                            np.clip(bb_[..., 2].astype(np.int16) - bb_[..., 1],
                                    0, 255).astype(np.uint8))
                        # Luminance too: the bar animates red -> white dissolve
                        # -> black, and red-excess sees only the red. The white
                        # flash is the instant damage lands; the settled edge
                        # trails it by the whole animation.
                        boss_lum.append(bb_[..., :3].max(axis=2).copy())
                        hp_ts.append(vf.t_ns)
                if gate:
                    try:
                        f = video.grab(timeout_ms=5)
                    except Exception:
                        f = None
                    if f is not None:
                        sy = max(1, f.data.shape[0] // 270)
                        sx = max(1, f.data.shape[1] // 480)
                        stream.push(f.data[::sy, ::sx][..., 1], f.t_ns)
                        v = stream.features(now_ns())
                        if v is not None:
                            pv = float(mdl.predict_proba(v.reshape(1, -1))[0, 1])
                            run_len = run_len + 1 if pv >= vision_threshold else 0
                            if run_len >= consecutive:
                                run_len = 0
                                gate_until = now_ns() + int(gate_ms * 1e6)

                while True:
                    item = audio.read()
                    if item is None:
                        break
                    samples, t_blk = item
                    if grade_hp:
                        if audio_t0 is None:
                            audio_t0 = t_blk
                        pcm_log.append(samples.copy())
                    if clash.push(samples, t_ns=t_blk):
                        # Logged, not used to grade. A clash marks an attack
                        # making contact, whether parried or not, so it says
                        # "an attack happened here" and nothing about success.
                        logf.write(json.dumps(
                            {"t_ns": t_blk, "kind": "clash"}) + "\n")
                    fire_at = det.push(samples, t_ns=t_blk)
                    peak_p = max(peak_p, det.last_p)
                    if fire_at is None:
                        continue
                    fired_p = det.last_fire_p
                    # Press earlier than the model asks by lead_ms. The labels
                    # are the player's own presses, which land inside the 150 ms
                    # window but not necessarily at its centre -- so the model
                    # inherits whatever lag the player had, and the input path
                    # adds more. This is the knob for that, and its right value
                    # is empirical: too early whiffs just as hard as too late.
                    this_lead = rng.choice(sweep) if sweep else lead_ms
                    fire_at -= int(this_lead * 1e6)
                    if gate and fire_at >= gate_until:
                        blocked += 1
                        logf.write(json.dumps({"t_ns": fire_at,
                                               "kind": "gate_blocked"}) + "\n")
                        continue
                    fires += 1
                    pending.append(fire_at)
                    # Only track an outcome for a press we will actually make.
                    # Recording one before the gate check would score a press
                    # that never happened as a whiff.
                    outcomes.append({"press_at": fire_at, "lead": this_lead,
                                     "landed": None})
                    logf.write(json.dumps({"t_ns": fire_at, "kind": "fire",
                                           "p": round(fired_p, 4),
                                           "lead_ms": this_lead}) + "\n")
                    print(f"\r  FIRE #{fires}  p={fired_p:.3f}"
                          + (f"  lead {this_lead:+.0f}ms" if sweep else "")
                          + " " * 16)

                now = now_ns()
                still = []
                for at in pending:
                    if now < at:
                        still.append(at)
                    elif arm and backend is not None:
                        backend.tap(action)
                pending = still

                # Outcomes are graded OFFLINE from the HP capture, not here. The
                # old in-loop rule ("no clash within 250 ms means it whiffed")
                # was the inverse of a rule that was itself wrong, and produced
                # a 100% land rate on a run with one real parry.

                for t_key, name, down in keys.drain():
                    if not down:
                        continue
                    if name in LABEL_KEYS:
                        labels += 1
                        logf.write(json.dumps(
                            {"t_ns": int(t_key), "kind": "attack_label",
                             "attack": int(name[-1])}) + "\n")
                        print(f"\r  label {name[-1]}: "
                              f"{ATTACK_NAMES[int(name[-1])]}" + " " * 24)
                        continue
                    human += 1
                    logf.write(json.dumps(
                        {"t_ns": int(t_key),
                         "kind": "human_parry" if name == "e"
                         else "human_dodge"}) + "\n")

                if u32.GetAsyncKeyState(VK_F10) & 0x8000:
                    print("\n  F10 - stopping")
                    break
                if not gate:
                    # Nothing else to do between audio blocks; without this the
                    # loop spins a core and starves the capture callback.
                    timer.sleep(0.001)
    finally:
        logf.close()
        timer.close()
        if backend is not None:
            backend.close()


    scored = det.scored if det is not None else 0
    print(f"\n  {scored} windows scored, {fires} fires, peak score {peak_p:.3f}")
    print(f"  human presses: {human}")
    if labels:
        print(f"  attack labels typed: {labels}")
    if gate:
        print(f"  gate blocked {blocked} fires")
    if audio.ring.overruns:
        print(f"  AUDIO DROPPED {audio.ring.overruns} blocks -- features span "
              f"the gap, so scores around it are unreliable")
    if hp_frames:
        import numpy as _np
        hp_dir = logpath.with_suffix("")
        hp_dir.mkdir(exist_ok=True)
        _np.save(hp_dir / "hp.npy", _np.array(hp_frames, dtype=_np.uint8))
        _np.save(hp_dir / "t_ns.npy", _np.array(hp_ts, dtype=_np.int64))
        if boss_frames:
            _np.save(hp_dir / "boss.npy",
                     _np.array(boss_frames, dtype=_np.uint8))
            _np.save(hp_dir / "boss_lum.npy",
                     _np.array(boss_lum, dtype=_np.uint8))
        # Save the audio too. Without it there is no common reference between
        # the bot's presses and the player's: the player parries this boss 16
        # times in 18, so their timing IS the target, but the only shared
        # landmark is the acoustic cue both are reacting to. HP drops only exist
        # for attacks that got through, which is precisely the cases where the
        # player's timing cannot be observed.
        if pcm_log:
            import wave as _wave
            data = _np.concatenate(pcm_log)
            with _wave.open(str(hp_dir / "audio.wav"), "wb") as _w:
                _w.setnchannels(1)
                _w.setsampwidth(2)
                _w.setframerate(audio.samplerate)
                _w.writeframes(_np.clip(data * 32767, -32768, 32767)
                               .astype(_np.int16).tobytes())
            (hp_dir / "meta.json").write_text(json.dumps(
                {"audio_start_ns": audio_t0, "samplerate": audio.samplerate,
                 "frames": len(hp_frames), "hp_roi": list(hp_roi)}), encoding="utf-8")
        print(f"  HP capture: {len(hp_frames)} frames -> {hp_dir}")
        print(f"  grade it:   python scripts/grade_presses.py {logpath.stem}")
    elif sweep:
        print()
        print("  NO GRADE. Presses were logged but nothing scored them:")
        print("  clash-based scoring was retracted (a clash marks an attack")
        print("  making contact, parried or not -- it read 14/14 landed on a")
        print("  run the victory screen scored as 1). Re-run with --grade-hp")
        print("  to grade honestly from the health readout.")
    print(f"  log: {logpath}")
    return logpath
