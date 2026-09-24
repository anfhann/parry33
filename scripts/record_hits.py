"""Record a fight continuously, to establish ground truth for WHEN ATTACKS HIT.

Everything the detector has been trained and scored against so far derives from
the player's own keypresses: positives are presses that landed, and "not an
attack" means "no landed press nearby". That definition is circular. It caused
one measured failure -- hard-negative mining suppressed real detections, because
windows the player whiffed were labelled as non-attacks -- and it means attacks
nobody reacted to are invisible to every metric we have.

The player's HP readout is not circular. It steps down the instant an attack
connects, whether or not anyone reacted, so reading it gives a COMPLETE list of
attacks rather than a sample filtered through what the player noticed.

USE IT LIKE THIS: pick a weak enemy, start this, and DO NOT PARRY OR DODGE. Take
every hit. On a boss that hits for a third of your health you get three samples
before dying; on a weak one, twenty or more.

--auto-attack advances the turn on a fixed cadence so the whole fight runs
unattended. The fight is scripted enough that a turn resolves in ~5.6 s, so a
burst every 8 s always finds the game waiting. This is what makes 15-20 runs a
day practical instead of one at a time.

It uses SKIP TURN (Tab, held ~1.5 s) rather than attacking. Attacking advances
the turn too, but it damages the boss, and every point of damage shortens the
fight this script exists to record. That bites hardest where it matters most: a
better parry detector counters harder and kills faster, so the configurations
most worth measuring would produce the least data. Skipping keeps the boss at
full health and the attacks coming for as long as the player survives.

It also produces a second kind of ground truth for free. The times we pressed
attack are known exactly, and the player's own attack is one of the sounds that
most resembles an enemy's -- it was a prime suspect for the false fires. Those
timestamps are saved to attacks.npy as honest negatives: sounds we can prove are
not incoming attacks, without routing through whether anyone parried.

Saves, per run:
    hp.npy       the HP readout at FULL resolution, every frame   (required)
    small.npy    whole frame at 1/scale, for eyeballing later
    hud.npy      the wider HUD strip, only with --keep-hud
    attacks.npy  timestamps of our own injected turn presses
    t_ns.npy     capture timestamp per frame
    audio.wav    system audio, aligned by meta["audio_start_ns"]
    meta.json

    python scripts/record_hits.py --seconds 300 --auto-attack
    python scripts/record_hits.py --auto-attack --out E:/parry33
    python scripts/record_hits.py --auto-attack --attack-action attack         --attack-hold-ms 40 --attack-count 2      # the old damaging mode
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod                    # noqa: E402
from parry33.audio.capture import LoopbackCapture       # noqa: E402
from parry33.capture import base as capbase             # noqa: E402
from parry33.clock import now_ns                        # noqa: E402
from parry33.input import base as inbase                # noqa: E402
from parry33.input.keywatch import KeyWatcher           # noqa: E402
from parry33.util import prio                           # noqa: E402

VK_F10 = 0x79


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=None,
                    help="directory for runs (e.g. an external drive). "
                         "Defaults to <repo>/runs")
    # The HP readout, at FULL resolution. An earlier version kept the whole
    # bottom strip downsampled 2x4 to save disk, which turned 20px-tall digits
    # into 5px and made them unsegmentable -- 35% of frames readable and 154
    # phantom HP "values". A small region at full res reads 100% of frames AND
    # is four times cheaper. Defaults measured at 2560x1440.
    ap.add_argument("--hp-roi", type=int, nargs=4, default=(2240, 1230, 2520, 1370),
                    metavar=("L", "T", "R", "B"))
    ap.add_argument("--scale", type=int, default=8, help="context downscale")
    # The BOSS health bar, top-centre. A successful parry triggers a counter, so
    # the boss losing health is proof the parry worked -- the one thing the
    # player's own HP cannot show, because a parried attack leaves no trace on
    # it. Stored as red-excess (R minus G) rather than luminance: the fill is
    # red, and red is nearly invisible in the green channel, which is precisely
    # how half of every early run was silently lost.
    # Measured from a real fight: the bar occupies only ~9 screen rows (65-73),
    # so the band is cropped tight. That buys room to store TWO views of it.
    ap.add_argument("--boss-roi", type=int, nargs=4,
                    default=(780, 55, 1800, 90),
                    metavar=("L", "T", "R", "B"))
    ap.add_argument("--no-boss", action="store_true")
    # The context frames are for eyeballing only -- no analysis reads them, and
    # at full rate they cost MORE than the HP readout that everything depends
    # on. A few per second is plenty to confirm which boss it was and that the
    # fight happened, so they are subsampled rather than dropped.
    ap.add_argument("--context-every", type=int, default=12,
                    help="store one context frame per N captured (0 = none)")
    ap.add_argument("--keep-hud", action="store_true",
                    help="also store the wide HUD strip. Only needed to RE-LOCATE "
                         "the readout after a resolution or UI change; find_hits "
                         "reads hp.npy, so this is redundant and doubles the size")
    ap.add_argument("--hud-top", type=float, default=0.78)
    ap.add_argument("--hud-bottom", type=float, default=1.0)
    ap.add_argument("--hud-xstride", type=int, default=2)
    ap.add_argument("--hud-ystride", type=int, default=4)
    ap.add_argument("--auto-attack", action="store_true",
                    help="inject the attack key on a cadence so the fight runs "
                         "unattended")
    ap.add_argument("--attack-period", type=float, default=8.0,
                    help="seconds between bursts (a turn resolves in ~5.6 s)")
    ap.add_argument("--attack-delay", type=float, default=5.0,
                    help="seconds before the first burst, to tab into the game")
    ap.add_argument("--attack-gap", type=float, default=0.25,
                    help="seconds between presses when --attack-count > 1")
    ap.add_argument("--attack-count", type=int, default=1,
                    help="presses per burst")
    # Skip Turn, held. Attacking also advances the turn, but it damages the
    # boss, and every point of damage shortens the fight this run exists to
    # record -- worse, a GOOD detector kills faster, so the configurations most
    # worth measuring would yield the least data. Skipping keeps the boss alive
    # and the attacks coming.
    ap.add_argument("--attack-action", default="skip",
                    help="action to advance the turn: 'skip' (Tab, held) does "
                         "no damage; 'attack' is faster but kills the boss")
    ap.add_argument("--attack-hold-ms", type=float, default=1500.0,
                    help="press duration; Skip Turn needs ~1500 ms, a normal "
                         "tap is ~40")
    ap.add_argument("--device", type=int, default=None)
    a = ap.parse_args()

    cfg = cfgmod.load()
    prio.apply(cfg.timing)
    video = capbase.build(cfg.capture)
    audio = LoopbackCapture(
        device_index=a.device if a.device is not None else cfg.audio.device_index,
        blocksize=cfg.audio.blocksize)
    u32 = ctypes.WinDLL("user32", use_last_error=True)

    backend = None
    if a.auto_attack:
        backend = inbase.build(cfg.input)
        try:
            tok = backend.token(a.attack_action)
        except KeyError:
            print(f"  no binding for {a.attack_action!r} -- check config")
            return 1

    root = Path(a.out) if a.out else (cfgmod.REPO_ROOT / "runs")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"hits_{now_ns():x}"
    out.mkdir(parents=True, exist_ok=True)

    # Log the player's own parry/dodge presses. On a boss you must parry to
    # survive, HP drops only reveal attacks that LANDED -- a biased sample that
    # omits exactly the attacks handled well. A press plus a clash shortly after
    # identifies a parried attack. The conjunction matters: clashes ALONE are
    # not evidence, as a punching-bag run proved by yielding 87 "parries" in a
    # fight with none.
    keys = KeyWatcher(watch=("q", "e"))
    presses: list[tuple[int, str]] = []

    try:
        with video, audio, keys:
            first = video.grab(timeout_ms=2000)
            if first is None:
                print("  no frames -- is the game rendering?")
                return 1
            H, W = first.data.shape[:2]
            top, bot = int(H * a.hud_top), int(H * a.hud_bottom)
            hud_h = len(range(top, bot, a.hud_ystride))
            hud_w = len(range(0, W, a.hud_xstride))
            sh, sw = H // a.scale, W // a.scale
            hl, ht, hr, hb = a.hp_roi
            hl, ht = max(0, hl), max(0, ht)
            hr, hb = min(W, hr), min(H, hb)
            # Size the buffer from the DISPLAY rate, not the game's frame cap.
            # Desktop Duplication emits on change and a smoke test measured 87
            # fps against a configured 52, so a 1.3x margin on game.fps would
            # have hit the cap at 234 s of a 300 s run -- silently truncating
            # the unattended fight this script exists to capture. Preallocating
            # for the refresh rate costs disk, which is cheap, instead of data.
            rate = max(cfg.game.fps, cfg.refresh_hz, 60)
            cap_frames = int(a.seconds * rate * 1.1) + 120
            every = max(0, a.context_every)
            cap_ctx = (cap_frames // every + 2) if every else 0
            bl, bt, br, bb = (int(x) for x in a.boss_roi)
            bl, bt = max(0, bl), max(0, bt)
            br, bb = min(W, br), min(H, bb)
            boss_on = not a.no_boss and br > bl and bb > bt
            per = ((hb - ht) * (hr - hl)
                   + (2 * (bb - bt) * (br - bl) if boss_on else 0)
                   + (sh * sw // every if every else 0)
                   + (hud_h * hud_w if a.keep_hud else 0))

            print(f"  frame {W}x{H}")
            print(f"  HP readout {hr - hl}x{hb - ht} at FULL resolution "
                  f"(rows {ht}-{hb}, cols {hl}-{hr})")
            print(f"  context {sw}x{sh} every {every} frames"
                  + (f", HUD strip {hud_w}x{hud_h}" if a.keep_hud else ""))
            print(f"  budget {cap_frames} frames = {cap_frames * per / 1e9:.2f} GB")
            print(f"  -> {out}")
            if backend is not None:
                print(f"  AUTO-TURN: {a.attack_action} "
                      f"({getattr(tok, 'label', tok)}) x{max(1, a.attack_count)}, "
                      f"held {a.attack_hold_ms:.0f} ms, every "
                      f"{a.attack_period:.0f}s, first in {a.attack_delay:.0f}s")
            print("  DO NOT PARRY OR DODGE. Take every hit. F10 to stop.\n")

            hp = np.lib.format.open_memmap(out / "hp.npy", mode="w+", dtype=np.uint8,
                                           shape=(cap_frames, hb - ht, hr - hl))
            boss = (np.lib.format.open_memmap(
                out / "boss.npy", mode="w+", dtype=np.uint8,
                shape=(cap_frames, bb - bt, br - bl)) if boss_on else None)
            # Luminance as well as red-excess. The bar animates red -> white
            # dissolve -> fade -> black, and red-excess sees only the red: white
            # has R about equal to G so it reads as zero, and so does black. The
            # WHITE FLASH is the instant damage is applied, while the settled
            # edge trails it by the length of the animation -- which is why the
            # press-to-edge lag measured 1961 ms rather than something short.
            bosslum = (np.lib.format.open_memmap(
                out / "boss_lum.npy", mode="w+", dtype=np.uint8,
                shape=(cap_frames, bb - bt, br - bl)) if boss_on else None)
            small = (np.lib.format.open_memmap(
                out / "small.npy", mode="w+", dtype=np.uint8,
                shape=(cap_ctx, sh, sw)) if every else None)
            small_idx = np.zeros(cap_ctx, dtype=np.int32)
            n_ctx = 0
            hud = (np.lib.format.open_memmap(
                out / "hud.npy", mode="w+", dtype=np.uint8,
                shape=(cap_frames, hud_h, hud_w)) if a.keep_hud else None)
            ts = np.zeros(cap_frames, dtype=np.int64)

            audio_t0 = None
            pcm = []
            n = bursts = 0
            attack_ts: list[int] = []
            queued: list[int] = []
            t_start = now_ns()
            t_end = t_start + int(a.seconds * 1e9)
            next_burst = (t_start + int(a.attack_delay * 1e9)
                          if backend is not None else None)

            while now_ns() < t_end and n < cap_frames:
                f = video.grab(timeout_ms=20)
                if f is not None:
                    # Green channel: the capture is BGRA and green carries most
                    # of the luminance, so this is a cheap grey with no convert.
                    g = f.data[..., 1]
                    # The HP readout specifically is stored as the per-pixel MAX
                    # over B,G,R, not the green channel. The number turns RED at
                    # low health, and red has almost no green -- measured, the
                    # digits vanished entirely below ~50% HP at every threshold
                    # while the white "/ max" beside them survived, so half of a
                    # 300 s run was silently unreadable. Green is fine for the
                    # context frames, which are only ever eyeballed.
                    hp[n] = f.data[ht:hb, hl:hr, :3].max(axis=2)
                    if boss is not None:
                        b = f.data[bt:bb, bl:br]
                        boss[n] = np.clip(b[..., 2].astype(np.int16)
                                          - b[..., 1], 0, 255).astype(np.uint8)
                        bosslum[n] = b[..., :3].max(axis=2)
                    if small is not None and n % every == 0 and n_ctx < cap_ctx:
                        small[n_ctx] = g[::a.scale, ::a.scale][:sh, :sw]
                        small_idx[n_ctx] = n
                        n_ctx += 1
                    if hud is not None:
                        hud[n] = g[top:bot:a.hud_ystride, ::a.hud_xstride]
                    ts[n] = f.t_ns
                    n += 1

                now = now_ns()
                if next_burst is not None and now >= next_burst:
                    queued += [now + i * int(a.attack_gap * 1e9)
                               for i in range(max(1, a.attack_count))]
                    next_burst += int(a.attack_period * 1e9)
                    if next_burst <= now:
                        # Fell a whole period behind -- a loading screen, an
                        # alt-tab, a stall. Advancing by one period each time
                        # would fire the backlog as a burst storm, so resync to
                        # now instead of trying to catch up.
                        next_burst = now + int(a.attack_period * 1e9)
                    bursts += 1
                # tap() is non-blocking (async_release), so injecting here does
                # not stall the capture loop.
                while queued and now >= queued[0]:
                    attack_ts.append(
                        backend.tap(a.attack_action, hold_ms=a.attack_hold_ms))
                    queued.pop(0)

                while True:
                    item = audio.read()
                    if item is None:
                        break
                    blk, t_blk = item
                    if audio_t0 is None:
                        audio_t0 = t_blk
                    pcm.append(blk.copy())

                for t_key, name, down in keys.drain():
                    if down:
                        presses.append((int(t_key), name))

                if u32.GetAsyncKeyState(VK_F10) & 0x8000:
                    print("\n  F10 - stopping")
                    break
                if n % 60 == 0 and n:
                    print(f"\r  {n} frames, {n / cfg.game.fps:.0f}s, "
                          f"{bursts} attack bursts  ", end="")

            hp.flush()
            if boss is not None:
                boss.flush()
                bosslum.flush()
            if small is not None:
                small.flush()
                np.save(out / "small_idx.npy", small_idx[:n_ctx])
            if hud is not None:
                hud.flush()
            np.save(out / "t_ns.npy", ts[:n])
            np.save(out / "attacks.npy", np.array(attack_ts, dtype=np.int64))
            np.save(out / "presses.npy",
                    np.array([t for t, _ in presses], dtype=np.int64))
            (out / "presses.json").write_text(json.dumps(
                [{"t_ns": t, "key": k} for t, k in presses]), encoding="utf-8")
            if pcm:
                data = np.concatenate(pcm)
                with wave.open(str(out / "audio.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(audio.samplerate)
                    w.writeframes(np.clip(data * 32767, -32768, 32767)
                                  .astype(np.int16).tobytes())
            (out / "meta.json").write_text(json.dumps({
                "frames": n, "width": W, "height": H, "scale": a.scale,
                "context_every": every, "context_frames": n_ctx,
                "boss_roi": [bl, bt, br, bb] if boss_on else None,
                "audio_start_ns": audio_t0, "hp_roi": [hl, ht, hr, hb],
                "hud_top": top, "hud_bottom": bot, "kept_hud": bool(a.keep_hud),
                "hud_xstride": a.hud_xstride, "hud_ystride": a.hud_ystride,
                "samplerate": audio.samplerate, "capacity": cap_frames,
                "auto_attack": bool(a.auto_attack),
                "attack_period_s": a.attack_period,
                "attack_gap_s": a.attack_gap,
                "attack_action": a.attack_action,
                "attack_hold_ms": a.attack_hold_ms,
                "attack_count": a.attack_count,
                "attack_presses": len(attack_ts),
                "player_presses": len(presses),
                "overruns": audio.ring.overruns}, indent=2), encoding="utf-8")
    finally:
        if backend is not None:
            backend.close()

    # Elapsed comes from the capture timestamps, not from n/game.fps: the
    # configured frame cap is not the rate we actually capture at (measured 139
    # fps against a configured 52), so dividing by it reported a 6 s run as 16 s.
    elapsed = (ts[n - 1] - ts[0]) / 1e9 if n > 1 else 0.0
    print(f"\n  {n} frames over {elapsed:.0f}s ({n / max(elapsed, 1e-9):.0f} fps), "
          f"{len(attack_ts)} attack presses in {bursts} bursts")
    if n >= cap_frames:
        print(f"  WARNING: hit the {cap_frames}-frame buffer cap; the recording "
              f"was truncated before --seconds elapsed.")
    if audio.ring.overruns:
        print(f"  audio dropped {audio.ring.overruns} blocks")
    print(f"  {out}")
    print(f"\n  Next: python scripts/find_hits.py {out.name}"
          + (f" --runs {root}" if a.out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
