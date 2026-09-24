"""Session recorder: aligned video, audio, onsets and manual marks.

This exists to answer the one question Phase 2 has to settle:

    Does an audio transient reliably precede an enemy attack landing, and by
    how many milliseconds?

`audio listen` showed ~1.5 onsets/second during combat, which is far more than
the enemy attack rate. So onsets alone are not the trigger -- some subset of
them is. Separating that subset needs ground truth, and ground truth needs
recorded evidence you can look at afterwards.

Design notes:

* **Single-threaded.** Video grab() blocks until the next desktop update
  (~16.7 ms at 60 fps); audio blocks arrive every 5.33 ms and pile up in their
  ring meanwhile. Draining them after each frame is fine because onset
  timestamps come from the audio callback, not from when we got round to
  processing -- accuracy is preserved without a second thread fighting for the
  GIL.
* **Frames are downsampled on ingest.** Full 960x540 BGRA is 2 MB/frame; at
  60 fps a 2-second ring would be 250 MB. Halving both axes costs nothing for
  review purposes and drops it to 62 MB.
* **Dumps are deferred.** An onset at T needs frames from T-400 ms to T+400 ms,
  and the second half has not happened yet. Dumps are queued and serviced once
  enough future has accumulated.

Ground truth comes from the keys you already press to survive: Q to dodge, E to
parry, and '+' right after one that LANDED. A landed parry is a hard timestamp
anchor -- the game's parry window provably contained that keypress -- which beats
tapping a marker key, because that carries 200-300 ms of human reaction lag.

Output per run, under runs/<timestamp>/:
    meta.json      config snapshot, device, geometry
    events.jsonl   onsets, parries, dodges, with scores and landed flags
    audio.wav      whole session, mono 16-bit
    strips/*.jpg   contact sheet per event, frames annotated with relative ms
"""

from __future__ import annotations

import ctypes
import json
import wave
from pathlib import Path

import numpy as np

from . import config as cfgmod
from .audio import onset as onsetmod
from .audio.capture import LoopbackCapture
from .input.keywatch import KeyWatcher
from .capture import base as capbase
from .clock import now_ns
from .util import prio
from .util.ringbuffer import FrameRing

_u32 = ctypes.WinDLL("user32", use_last_error=True)
VK_F10 = 0x79

# Generous: in turn-based combat you parry, finish the exchange, and only then
# tag what it was. A 4 s window silently dropped almost every tag.
TAG_WINDOW_NS = 20e9


def _feedback(msg: str) -> None:
    """Marker keys MUST confirm. A tag that fails silently is worse than no tag:
    the player believes the run is labelled and only finds out afterwards."""
    print("  >> " + msg + " " * 40)


def _key_down(vk: int) -> bool:
    return bool(_u32.GetAsyncKeyState(vk) & 0x8000)


def _strip(frames: np.ndarray, rel_ms, out_path: Path, cols: int = 8,
           tile_w: int = 240) -> None:
    """Contact sheet of a frame sequence, each tile labelled with relative ms.

    Reading a strip is how you answer "how early is the telegraph visible" --
    scrub back from the impact tile until the wind-up disappears.
    """
    from PIL import Image, ImageDraw
    n = len(frames)
    if n == 0:
        return
    h, w = frames[0].shape[:2]
    tile_h = max(1, int(tile_w * h / w))
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (16, 16, 16))
    for i in range(n):
        f = frames[i]
        rgb = f[..., 2::-1] if f.ndim == 3 and f.shape[2] >= 3 else f
        im = Image.fromarray(np.ascontiguousarray(rgb)).resize((tile_w, tile_h))
        d = ImageDraw.Draw(im)
        label = f"{rel_ms[i]:+.0f}ms"
        d.rectangle([0, 0, 58, 14], fill=(0, 0, 0))
        d.text((3, 2), label, fill=(0, 255, 120))
        sheet.paste(im, ((i % cols) * tile_w, (i // cols) * tile_h))
    sheet.save(out_path, quality=85)


def _write_wav(path: Path, blocks, samplerate: int) -> None:
    if not blocks:
        return
    a = np.concatenate(blocks)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())


def run(cfg=None, seconds: float = 120.0, device: int | None = None,
        strips: bool = True, strip_onsets: bool = False, clips: bool = True,
        pre_ms: float = 400.0, post_ms: float = 400.0,
        out_root: Path | None = None, delay: float = 0.0) -> Path:
    cfg = cfg or cfgmod.load()
    prio.apply(cfg.timing)

    if delay > 0:
        import time as _time
        print(f"starting in {delay:.0f}s -- switch to the game and get into combat")
        for r in range(int(delay), 0, -1):
            print(f"  {r}...", end="\r", flush=True)
            _time.sleep(1.0)
        print(" " * 24, end="\r")

    stamp = f"{now_ns():x}"
    out = (out_root or (cfgmod.REPO_ROOT / "runs")) / stamp
    (out / "strips").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    dev = device if device is not None else cfg.audio.device_index
    audio = LoopbackCapture(device_index=dev, blocksize=cfg.audio.blocksize)
    video = capbase.build(cfg.capture)

    events = []
    wav_blocks = []
    pending = []          # (dump_at_ns, event_index)
    marks = 0
    onsets = 0
    frames_seen = 0
    last_attempt = None   # index of the most recent q/e, for '+' to confirm
    variants = 0
    n_ctrl = 0
    ctrl_pending = []
    ctrl_every_s = 7.0     # ~1 control clip per 7 s of session
    fights = []           # [t_start, t_end] spans; excludes menus and reloads
    fight_open = None

    keys = KeyWatcher(watch=("q", "e", "plus", "numplus", "minus", "numminus",
                         "1", "2", "3", "4", "5", "6", "7", "8"))
    with audio, video, keys:
        # Two band-limited detectors, not one. The grunt is the cue and the
        # clash is the result; a single wideband detector only finds the clash.
        dets = {}
        for kind, band, sens in (
                ("grunt", cfg.audio.grunt_band, cfg.audio.grunt_sensitivity),
                ("clash", cfg.audio.clash_band, cfg.audio.clash_sensitivity)):
            dets[kind] = onsetmod.SpectralFluxOnset(
                audio.samplerate, cfg.audio.blocksize, sensitivity=sens,
                refractory_ms=cfg.audio.refractory_ms,
                floor_frac=cfg.audio.floor_frac,
                fmin=band[0], fmax=min(band[1], audio.samplerate / 2))

        first = video.grab(timeout_ms=1000)
        if first is None:
            raise RuntimeError("no video frames -- is the game rendering?")
        # Downsample to a fixed target, not a fixed stride. With a full-screen
        # ROI a ::2 stride would make each ring frame 1280x720x4 = 3.7 MB and a
        # 225-frame ring 830 MB. Targeting ~480x270 keeps the ring at ~116 MB
        # whatever the ROI is, so the ROI can be widened for free.
        sy = max(1, first.data.shape[0] // 270)
        sx = max(1, first.data.shape[1] // 480)
        small = first.data[::sy, ::sx]
        ring_len = int((pre_ms + post_ms) / 1000.0 * cfg.game.fps * 2.5) + 30
        ring = FrameRing(ring_len, small.shape, small.dtype)

        print(f"recording -> {out}")
        print(f"  video {video!r}")
        print(f"  audio {audio!r}")
        print(f"  ring  {ring.capacity} frames @ {small.shape[1]}x{small.shape[0]} "
              f"= {ring.nbytes / 1e6:.0f} MB  (from {first.data.shape[1]}x"
              f"{first.data.shape[0]} ROI)")
        print("\n  Q = dodge   E = parry   + = the last one LANDED")
        print("  1..8 = optional pattern tag (skip it -- just play)")
        print("  -  = fight start / fight end (toggle)   F10 = stop early\n")

        audio_start = now_ns()   # anchors audio.wav to the event timeline
        next_ctrl = audio_start + int(ctrl_every_s * 1e9)
        t_end = audio_start + int(seconds * 1e9)
        while now_ns() < t_end:
            f = video.grab(timeout_ms=50)
            if f is not None:
                ring.write(f.data[::sy, ::sx], f.t_ns)
                frames_seen += 1

            while True:
                item = audio.read()
                if item is None:
                    break
                samples, t_blk = item
                wav_blocks.append(samples.copy())
                for kind, det in dets.items():
                    if det.push(samples, t_ns=t_blk):
                        onsets += 1
                        events.append({"kind": kind, "t_ns": t_blk,
                                       "score": round(det.last_value, 4),
                                       "threshold": round(det.last_threshold, 4)})
                        pending.append((t_blk + int(post_ms * 1e6),
                                        len(events) - 1))

            for t_key, name, down in keys.drain():
                if not down:
                    continue
                if name in ("plus", "numplus"):
                    # Optional now: landed parries are auto-detected from the
                    # clash that follows them. Kept for manual override.
                    ok = False
                    if last_attempt is not None:
                        ev = events[last_attempt]
                        if t_key - ev["t_ns"] < int(TAG_WINDOW_NS) and not ev.get("landed"):
                            ev["landed"] = True
                            marks += 1
                            ok = True
                    _feedback("LANDED" if ok else "'+' ignored (no recent attempt)")
                    continue
                if name in ("1", "2", "3", "4", "5", "6", "7", "8"):
                    # Tag the variant of the attack just parried. The early
                    # visual tell says an attack is committed but not which of
                    # the branches it becomes, and each branch has its own gap
                    # to the window -- so the variant IS the timing label.
                    ok = False
                    if last_attempt is not None:
                        ev = events[last_attempt]
                        age = (t_key - ev["t_ns"]) / 1e9
                        if age < TAG_WINDOW_NS / 1e9:
                            ev["variant"] = int(name)
                            variants += 1
                            ok = True
                    _feedback(f"variant {name}" if ok
                              else f"variant {name} IGNORED - no attempt in the "
                                   f"last {TAG_WINDOW_NS / 1e9:.0f}s")
                    continue
                if name in ("minus", "numminus"):
                    # Toggle: odd press opens a fight, even press closes it.
                    if fight_open is None:
                        fight_open = t_key
                        print(f"\r  FIGHT {len(fights) + 1} start" + " " * 40)
                    else:
                        fights.append([fight_open, t_key])
                        print(f"\r  FIGHT {len(fights)} end  "
                              f"({(t_key - fight_open) / 1e9:.0f}s)" + " " * 30)
                        fight_open = None
                    continue
                kind = "parry" if name == "e" else "dodge"
                events.append({"kind": kind, "t_ns": t_key, "landed": False})
                last_attempt = len(events) - 1
                pending.append((t_key + int(post_ms * 1e6), last_attempt))

            now = now_ns()

            # Control clips at random times, for measuring FALSE POSITIVES.
            # Without them every feature looks predictive: we only ever see the
            # moments a parry happened. The audio grunt had 91% recall and was
            # still useless at 2% precision, and that only became visible once
            # there was something to compare against.
            if clips and now >= next_ctrl:
                next_ctrl = now + int(ctrl_every_s * 1e9)
                if ring.count > 20:
                    ctrl_pending.append((now + int(post_ms * 1e6), now))

            still_c = []
            for due, t_ev in ctrl_pending:
                if now < due:
                    still_c.append((due, t_ev)); continue
                fr, ts = _frames_between(ring, t_ev - int(pre_ms * 1e6),
                                         t_ev + int(post_ms * 1e6))
                if len(fr) > 20:
                    g = fr[..., 1].astype(np.uint8)
                    np.savez_compressed(out / "clips" / f"c{n_ctrl:04d}_control.npz",
                                        frames=g, t_ns=ts, event_t_ns=t_ev)
                    events.append({"kind": "control", "t_ns": int(t_ev),
                                   "clip": f"clips/c{n_ctrl:04d}_control.npz"})
                    n_ctrl += 1
            ctrl_pending = still_c

            if _key_down(VK_F10):
                print("\n  stopped early")
                break

            # Service deferred dumps once their future has arrived.
            now = now_ns()
            still = []
            for due, idx in pending:
                if now < due:
                    still.append((due, idx))
                    continue
                ev = events[idx]
                # Strip only the events we are labelling. At ~1.5 onsets/s a
                # 10-minute session is ~900 sheets, each a PIL encode of ~50
                # tiles inside the capture loop -- enough to stall it and drop
                # frames. Onsets are fully described by their timestamps in
                # events.jsonl; it is the keypresses we need pictures of.
                if clips and ev["kind"] in ("parry", "dodge"):
                    # Raw frame stack for model training. Contact sheets are
                    # JPEG-compressed 240x135 tiles -- fine for a human to
                    # eyeball, useless as training data. This keeps the pixels.
                    t0 = ev["t_ns"] - int(pre_ms * 1e6)
                    t1 = ev["t_ns"] + int(post_ms * 1e6)
                    fr, ts = _frames_between(ring, t0, t1)
                    if len(fr):
                        g = fr[..., 1].astype(np.uint8)   # green channel, already downsampled
                        np.savez_compressed(
                            out / "clips" / f"{idx:04d}_{ev['kind']}.npz",
                            frames=g, t_ns=ts, event_t_ns=ev["t_ns"])
                        ev["clip"] = f"clips/{idx:04d}_{ev['kind']}.npz"
                if strips and (strip_onsets or ev["kind"] in ("parry", "dodge")):
                    t0 = ev["t_ns"] - int(pre_ms * 1e6)
                    t1 = ev["t_ns"] + int(post_ms * 1e6)
                    fr, ts = _frames_between(ring, t0, t1)
                    if len(fr):
                        rel = (ts - ev["t_ns"]) / 1e6
                        name = f"{idx:04d}_{ev['kind']}.jpg"
                        _strip(fr, rel, out / "strips" / name)
                        ev["strip"] = f"strips/{name}"
                        ev["n_frames"] = int(len(fr))
            pending = still

            if frames_seen % 30 == 0:
                el = (now - (t_end - int(seconds * 1e9))) / 1e9
                print(f"\r  {el:6.1f}s  frames {frames_seen}  onsets {onsets}  "
                      f"marks {marks}", end="")

        if fight_open is not None:
            fights.append([fight_open, now_ns()])
        fights = _fix_fight_phase(fights, events)
        samplerate = audio.samplerate
        overruns = audio.ring.overruns

    print()
    _write_wav(out / "audio.wav", wav_blocks, samplerate)
    with open(out / "events.jsonl", "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    meta = {
        "stamp": stamp,
        "audio_start_ns": int(audio_start),
        "seconds": seconds,
        "video_region": list(cfg.capture.region) if cfg.capture.region else None,
        "game_fps": cfg.game.fps,
        "refresh_hz": cfg.refresh_hz,
        "parry_window_ms": cfg.game.parry_window_ms,
        "audio_device": audio.device_name,
        "audio_samplerate": samplerate,
        "blocksize": cfg.audio.blocksize,
        "detector": cfg.audio.detector,
        "sensitivity": cfg.audio.sensitivity,
        "frames": frames_seen,
        "onsets": onsets,
        "landed": marks,
        "variants_tagged": variants,
        "control_clips": n_ctrl,
        "parries": sum(1 for e in events if e["kind"] == "parry"),
        "dodges": sum(1 for e in events if e["kind"] == "dodge"),
        "ignored_injected_keys": keys.ignored_injected,
        "fights": [[int(a), int(b)] for a, b in fights],
        "fight_seconds": round(sum(b - a for a, b in fights) / 1e9, 1),
        "audio_overruns": overruns,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_parry = sum(1 for e in events if e["kind"] == "parry")
    n_dodge = sum(1 for e in events if e["kind"] == "dodge")
    print(f"\n  {frames_seen} frames, {onsets} onsets, {overruns} audio overruns")
    print(f"  {n_parry} parries ({marks} landed), {n_dodge} dodges")
    if fights:
        total = sum(b - a for a, b in fights) / 1e9
        print(f"  {len(fights)} fights, {total:.0f}s in combat "
              f"({total / max(seconds, 1) * 100:.0f}% of the session)")
    print(f"  wrote {out}")
    if marks:
        _report_lead(events, cfg.game.parry_window_ms, fights)
    else:
        print("\n  No manual '+' marks -- that is expected and fine.")
        print("  Landed parries are auto-detected from the clash during analysis,")
        print("  not at record time. Nothing was lost.")
    return out


def _fix_fight_phase(fights, events):
    """Repair an inverted fight/menu toggle.

    A single toggle key has no absolute reference: miss one press at the start
    and every span afterwards is off by one, so the recording marks the menu
    gaps as combat and the combat as menus. That happened on the first real
    session -- 10 of 61 parries landed inside the "fights", and zero of the
    successful ones.

    The player's own keypresses disambiguate it. Whichever pairing of the toggle
    edges contains more parry/dodge presses is the one that covers combat.
    """
    if len(fights) < 2:
        return fights
    edges = sorted({t for span in fights for t in span})
    presses = [e["t_ns"] for e in events if e["kind"] in ("parry", "dodge")]
    if not presses:
        return fights

    def pair(offset):
        e = edges[offset:]
        return [[e[i], e[i + 1]] for i in range(0, len(e) - 1, 2)]

    def covered(spans):
        return sum(any(a <= p <= b for a, b in spans) for p in presses)

    a, b = pair(0), pair(1)
    if covered(b) > covered(a):
        print(f"\n  NOTE: fight markers were inverted (a '-' press was missed).")
        print(f"  Auto-corrected: {covered(b)}/{len(presses)} presses now inside "
              f"combat, was {covered(a)}.")
        return b
    return a


def _frames_between(ring: FrameRing, t0: int, t1: int):
    frames, ts = ring.latest(ring.count)
    if len(ts) == 0:
        return frames[:0], ts[:0]
    sel = (ts >= t0) & (ts <= t1)
    return frames[sel], ts[sel]


def _lead_ms(onsets: np.ndarray, t: int):
    """Time from the nearest preceding onset to t, in ms. None if there is none."""
    before = onsets[onsets <= t]
    return (t - before[-1]) / 1e6 if len(before) else None


def _report_lead(events, window_ms: float, fights=None) -> None:
    """How long before each keypress did the nearest audio onset fire?

    A LANDED parry is a hard anchor. The game's parry window provably contained
    that keypress, so impact sits within roughly one window after it. Unlike a
    human "that was an attack" tap there is no reaction-time bias to subtract,
    which is why this protocol is worth more than F9 marking.

    The comparison that matters is landed vs missed. If landed parries have a
    consistent onset lead and missed ones do not, the onset is genuinely
    tracking the attack and the trigger is learnable. If both look the same,
    spectral flux is firing on things unrelated to attacks.
    """
    def in_fight(t):
        if not fights:
            return True
        return any(a <= t <= b for a, b in fights)

    events = [e for e in events if in_fight(e["t_ns"])]
    onsets = np.array(sorted(e["t_ns"] for e in events if e["kind"] == "grunt"))
    if not len(onsets):
        return
    if fights:
        print()
        print(f"  (restricted to {len(fights)} marked fight spans -- menu and "
              f"reload audio excluded)")
    groups = {
        "parry LANDED": [e for e in events
                         if e["kind"] == "parry" and e.get("landed")],
        "parry missed": [e for e in events
                         if e["kind"] == "parry" and not e.get("landed")],
        "dodge": [e for e in events if e["kind"] == "dodge"],
    }
    print(f"\n  time from nearest audio onset to your keypress:")
    print(f"    {'group':<14} {'n':>4} {'p50':>7} {'p25':>7} {'p75':>7} "
          f"{'IQR':>7}")
    stats = {}
    for label, evs in groups.items():
        leads = [x for x in (_lead_ms(onsets, e["t_ns"]) for e in evs)
                 if x is not None]
        if not leads:
            print(f"    {label:<14} {0:>4}")
            continue
        a = np.array(leads)
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        stats[label] = (len(a), q50, q75 - q25)
        print(f"    {label:<14} {len(a):>4} {q50:6.0f}ms {q25:6.0f}ms "
              f"{q75:6.0f}ms {q75 - q25:6.0f}ms")

    landed = stats.get("parry LANDED")
    if landed and landed[0] >= 5:
        n, med, iqr = landed
        print()
        if iqr < window_ms:
            print(f"  Landed parries cluster: IQR {iqr:.0f} ms, inside the "
                  f"{window_ms:.0f} ms window.")
            print(f"  An onset fires ~{med:.0f} ms before a parry that works, so the "
                  f"cue is")
            print(f"  real and an audio trigger is learnable. Next: separate THAT "
                  f"onset")
            print(f"  class from the ~1.5/s background, by score, spectrum or both.")
        else:
            print(f"  Landed parries scatter: IQR {iqr:.0f} ms against a "
                  f"{window_ms:.0f} ms window.")
            print(f"  Spectral flux alone is not finding the attack cue. Options:")
            print(f"  a sharper discriminator, or vision-first with audio as "
                  f"confirmation.")
        miss = stats.get("parry missed")
        if miss and miss[0] >= 5:
            print(f"\n  landed p50 {med:.0f} ms vs missed p50 {miss[1]:.0f} ms -- "
                  f"a large gap here")
            print(f"  means the onset predicts the window, not just 'a sound "
                  f"happened'.")
    else:
        print(f"\n  Need at least ~5 landed parries for this to mean anything.")
    print(f"\n  The strips in strips/ show the frames around each event; scrub "
          f"back from")
    print(f"  the impact tile to see how early the wind-up becomes visible.")
