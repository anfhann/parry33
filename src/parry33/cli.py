"""parry33 command line.

  parry33 doctor                     preflight the environment
  parry33 bench capture [--sweep]    capture throughput / ROI cost curve
  parry33 bench inject               injection syscall cost
  parry33 bench loop                 closed-loop inject -> pixel latency
  parry33 flash                      run the responder window on its own
"""

from __future__ import annotations

import argparse
import sys

from . import config as cfgmod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="parry33", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to a TOML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment preflight")

    b = sub.add_parser("bench", help="latency benchmarks")
    bsub = b.add_subparsers(dest="bench", required=True)

    bc = bsub.add_parser("capture")
    bc.add_argument("--seconds", type=float, default=None)
    bc.add_argument("--mode", choices=("poll", "thread"), default=None)
    bc.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    bc.add_argument("--ring", action="store_true")
    bc.add_argument("--sweep", action="store_true", help="ROI size cost curve")
    bc.add_argument("--compare", action="store_true",
                    help="compare poll vs thread and roi sizes under real load")
    bc.add_argument("--delay", type=float, default=0.0,
                    help="countdown before measuring, to tab into the game")

    bi = bsub.add_parser("inject")
    bi.add_argument("--backend", choices=("sendinput", "gamepad", "null"), default=None)
    bi.add_argument("--action", default="parry")
    bi.add_argument("--trials", type=int, default=None)

    bl = bsub.add_parser("loop")
    bl.add_argument("--trials", type=int, default=None)
    bl.add_argument("--backend", choices=("sendinput", "gamepad"), default=None)
    bl.add_argument("--action", default="parry")
    bl.add_argument("--no-spawn", action="store_true")
    bl.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    bl.add_argument("--window-pos", type=int, nargs=2, default=(300, 300))
    bl.add_argument("--window-size", type=int, default=420)
    bl.add_argument("--responder-mode", choices=("raw", "key"), default="raw")

    au = sub.add_parser("audio", help="WASAPI loopback tools")
    ausub = au.add_subparsers(dest="audio", required=True)
    ausub.add_parser("devices", help="list loopback devices")
    al = ausub.add_parser("listen", help="live level meter + onset counter")
    al.add_argument("--device", type=int, default=None)
    al.add_argument("--blocksize", type=int, default=None)
    al.add_argument("--seconds", type=float, default=20.0)
    al.add_argument("--detector", choices=("flux", "rms"), default=None)
    al.add_argument("--sensitivity", type=float, default=None)

    ba = bsub.add_parser("audio")
    ba.add_argument("--what", choices=("cadence", "loop"), default="loop")
    ba.add_argument("--device", type=int, default=None)
    ba.add_argument("--blocksize", type=int, default=None)
    ba.add_argument("--trials", type=int, default=60)
    ba.add_argument("--seconds", type=float, default=10.0)
    ba.add_argument("--detector", choices=("flux", "rms"), default=None)
    ba.add_argument("--sensitivity", type=float, default=None)
    ba.add_argument("--delay", type=float, default=0.0)

    tr = sub.add_parser("train", help="train + save the attack detector")
    tr.add_argument("--grunt", action="store_true",
                    help="train the learned attack-sound model from recordings")
    tr.add_argument("--live", action="store_true",
                    help="train on served windows from live logs "
                         "instead of event-centred clips")

    lv = sub.add_parser("live", help="live trigger (dry-run unless --arm)")
    lv.add_argument("--seconds", type=float, default=300.0)
    lv.add_argument("--arm", action="store_true",
                    help="ACTUALLY inject the parry key (default is dry-run)")
    lv.add_argument("--threshold", type=float, default=0.5)
    lv.add_argument("--consecutive", type=int, default=2)
    lv.add_argument("--press-delay-ms", type=float, default=500.0)
    lv.add_argument("--score-every-ms", type=float, default=100.0)
    lv.add_argument("--action", default="parry")
    lv.add_argument("--delay", type=float, default=20.0)
    lv.add_argument("--require-title", default="",
                    help="only fire while a window whose title contains this "
                         "has focus, e.g. --require-title Expedition")
    lv.add_argument("--no-bar-gate", action="store_true",
                    help="disable the boss-HP-bar requirement (debug)")
    lv.add_argument("--max-fires-per-min", type=float, default=30.0,
                    help="stop if firing exceeds this -- runaway guard")

    tg = sub.add_parser("trigger",
                        help="live trigger driven by the learned attack-sound model")
    tg.add_argument("--seconds", type=float, default=600.0)
    tg.add_argument("--arm", action="store_true",
                    help="ACTUALLY inject (default is dry-run)")
    tg.add_argument("--threshold", type=float, default=None,
                    help="fire above this probability (default from the model)")
    tg.add_argument("--refractory-ms", type=float, default=None,
                    help="minimum gap between fires; only prevents double-firing "
                         "on one attack -- a landed parry has no game lockout")
    tg.add_argument("--lookahead-ms", type=float, default=None,
                    help="wait this long after crossing to take the peak")
    tg.add_argument("--lead-ms", type=float, default=0.0,
                    help="press this many ms EARLIER than the model asks; the "
                         "model inherits the player's own reaction lag")
    tg.add_argument("--lead-sweep", default="",
                    help="comma-separated lead values to try at random, e.g. "
                         "\"-20,0,20,40,60\"; each press is graded by whether a "
                         "clash follows, and a land rate per lead is printed")
    tg.add_argument("--grade-hp", action="store_true",
                    help="capture the HP readout so presses can be graded "
                         "honestly afterwards. Required for --lead-sweep to "
                         "mean anything: clash-based scoring was retracted "
                         "(a clash marks an attack CONNECTING, parried or not)")
    tg.add_argument("--hp-roi", type=int, nargs=4,
                    default=(2240, 1230, 2520, 1370),
                    metavar=("L", "T", "R", "B"))
    tg.add_argument("--gate", action="store_true",
                    help="also require the vision model to see an attack "
                         "(UNMEASURED -- the honest numbers are audio-only)")
    tg.add_argument("--vision-threshold", type=float, default=0.5)
    tg.add_argument("--consecutive", type=int, default=2)
    tg.add_argument("--gate-ms", type=float, default=1500.0)
    tg.add_argument("--action", default="parry")
    tg.add_argument("--delay", type=float, default=0.0,
                    help="countdown before starting; 0 = go now")
    tg.add_argument("--device", type=int, default=None)

    rec = sub.add_parser("record", help="record aligned video+audio+onsets+marks")
    rec.add_argument("--seconds", type=float, default=120.0)
    rec.add_argument("--device", type=int, default=None)
    rec.add_argument("--delay", type=float, default=15.0)
    rec.add_argument("--strip-onsets", action="store_true",
                     help="also export a sheet per onset (slow at 10min scale)")
    rec.add_argument("--no-clips", action="store_true",
                     help="skip raw frame-stack export (no training data)")
    rec.add_argument("--no-strips", action="store_true",
                     help="skip contact-sheet export (faster, less to review)")
    rec.add_argument("--pre-ms", type=float, default=400.0)
    rec.add_argument("--post-ms", type=float, default=400.0)

    f = sub.add_parser("flash", help="responder window (standalone)")
    f.add_argument("--x", type=int, default=300)
    f.add_argument("--y", type=int, default=300)
    f.add_argument("--size", type=int, default=420)
    f.add_argument("--mode", choices=("key", "pad", "animate"), default="key")
    f.add_argument("--hz", type=float, default=144.0)

    a = p.parse_args(argv)
    cfg = cfgmod.load(a.config)

    if a.cmd == "doctor":
        from .doctor import run
        return run(cfg)

    if a.cmd == "train":
        if a.grunt:
            from .audio.train import train as train_grunt
            train_grunt()
            return 0
        from .model import train, train_live
        if a.live:
            train_live()
        else:
            train()
        return 0

    if a.cmd == "live":
        from .live import run as live_run
        live_run(cfg, seconds=a.seconds, arm=a.arm, threshold=a.threshold,
                 consecutive=a.consecutive, press_delay_ms=a.press_delay_ms,
                 score_every_ms=a.score_every_ms, action=a.action, delay=a.delay,
                 require_title=a.require_title,
                 max_fires_per_min=a.max_fires_per_min,
                 require_bar=not a.no_bar_gate)
        return 0

    if a.cmd == "trigger":
        from .trigger import run as trig_run
        trig_run(cfg, seconds=a.seconds, arm=a.arm, threshold=a.threshold,
                 refractory_ms=a.refractory_ms, lookahead_ms=a.lookahead_ms,
                 lead_ms=a.lead_ms, lead_sweep=a.lead_sweep,
                 grade_hp=a.grade_hp, hp_roi=tuple(a.hp_roi),
                 action=a.action, delay=a.delay, gate=a.gate,
                 vision_threshold=a.vision_threshold, consecutive=a.consecutive,
                 gate_ms=a.gate_ms, device=a.device)
        return 0

    if a.cmd == "record":
        from .record import run as rec_run
        rec_run(cfg, seconds=a.seconds, device=a.device, delay=a.delay,
                strips=not a.no_strips, strip_onsets=a.strip_onsets,
                clips=not a.no_clips,
                pre_ms=a.pre_ms, post_ms=a.post_ms)
        return 0

    if a.cmd == "audio":
        from .bench import audio as A
        if a.audio == "devices":
            return A.devices()
        return A.listen(
            device=a.device if a.device is not None else cfg.audio.device_index,
            blocksize=a.blocksize or cfg.audio.blocksize,
            seconds=a.seconds,
            detector=a.detector or cfg.audio.detector,
            sensitivity=a.sensitivity if a.sensitivity is not None
            else cfg.audio.sensitivity)

    if a.cmd == "flash":
        from .harness.flash_window import main as flash_main
        return flash_main(["--x", str(a.x), "--y", str(a.y), "--size", str(a.size),
                           "--mode", a.mode, "--hz", str(a.hz)])

    if a.bench == "capture":
        from .bench.capture import compare, run, sweep
        if a.compare:
            compare(cfg, seconds=a.seconds or 8.0, delay=a.delay)
        elif a.sweep:
            sweep(cfg, seconds=a.seconds or 3.0)
        else:
            run(cfg, seconds=a.seconds, mode=a.mode, region=a.region, ring=a.ring,
                delay=a.delay)
        return 0

    if a.bench == "inject":
        from .bench.inject import run
        run(cfg, backend=a.backend, action=a.action, trials=a.trials)
        return 0

    if a.bench == "audio":
        from .bench import audio as A
        dev = a.device if a.device is not None else cfg.audio.device_index
        bs = a.blocksize or cfg.audio.blocksize
        det = a.detector or cfg.audio.detector
        sens = a.sensitivity if a.sensitivity is not None else cfg.audio.sensitivity
        if a.what == "cadence":
            A.cadence(device=dev, blocksize=bs, seconds=a.seconds, delay=a.delay)
        else:
            A.loop(device=dev, blocksize=bs, trials=a.trials,
                   detector=det, sensitivity=sens, cfg=cfg)
        return 0

    if a.bench == "loop":
        from .bench.loop import run
        run(cfg, trials=a.trials, backend=a.backend, action=a.action,
            spawn=not a.no_spawn, region=a.region,
            window_pos=tuple(a.window_pos), window_size=a.window_size,
            responder_mode=a.responder_mode)
        return 0

    p.error(f"unhandled command {a.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
