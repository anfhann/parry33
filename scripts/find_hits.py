"""Read the player's HP per frame, and derive when attacks actually landed.

This is the ground truth the project was missing. Every other label derives from
the player's own keypresses: positives are presses that landed, and "not an
attack" means "no landed press nearby". That is circular. It made attacks nobody
reacted to invisible to every metric, and it broke hard-negative mining, which
learned to suppress real attacks the player had whiffed.

The HP readout does not care whether anyone reacted. It steps down when an attack
connects, full stop.

WHY THIS READS THE DIGITS INSTEAD OF DETECTING CHANGE. Detecting "the HP region
changed" looks obviously sufficient and is not, because the readout ZOOMS as a UI
emphasis animation. Measured: of twelve detected changes, three were real HP
drops and nine were zoom. Every cheaper approach was tried and failed:

  * whole-region pixel diff        -- ~3x too many events
  * bbox-normalised whole readout  -- 58 "values" where ~10 exist
  * per-glyph binary signatures    -- the same value across a zoom differs MORE
                                      than different values do; no separation
  * grayscale correlation          -- same overlapping distributions
  * bucketing frames by zoom state -- the zoom ANIMATES through intermediate
                                      sizes, so there are no discrete states

Reading the number sidesteps all of it and self-validates: HP only falls during a
punching-bag run, so a misread announces itself as an impossible increase. On a
300 s run, all 36 differences were negative.

TWO CAPTURE DETAILS, both learned by getting them wrong:

  * The readout turns RED at low health. Stored as the green channel the digits
    vanished entirely below ~50% HP -- at every threshold -- while the white
    "/ max" beside them survived, silently losing half of every run.
    record_hits.py now stores max(B,G,R).
  * That lifts the background too, so the threshold must be near-white (248). At
    185 the mask flickered and glyph counts ranged 0-5 for a number that always
    has 3 or 4 digits.

TEMPLATES. The font is fixed, so digit templates built once transfer to every
run. Seeding needs a few frames whose value a human has read; they are then
cached in digits.npz beside the runs and never needed again.

    python scripts/find_hits.py <run> --runs E:/parry33
    python scripts/find_hits.py <run> --seed 50=1211 1380=1167 --save-templates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod          # noqa: E402

BRIGHT = 248        # near-white; max(B,G,R) lifts the background too
ROWS = (28, 118)    # search window; the digit band is located inside it
BAND_COLS = (95, 172)
BAND_MIN_ROWS = 20  # digits span ~30 rows; the bar below spans 11-15
BAND_MIN_COUNT = 5
DIGIT = (14, 20)    # every glyph normalised to this before matching
MAX_W = 45
TALL_FRAC = 0.80    # current-HP glyphs ~31px vs ~22 for the "/ max" after them
GAP = 10            # px gap before the slash; digits sit 1-3px apart
CUR_MAX_X0 = 170    # the current value must begin left of the slash
MATCH_MIN = 0.55    # correlation below this is not a confident digit
MIN_RUN = 8         # frames a value must hold to be real, not a redraw artifact
TEMPLATES = "digits.npz"


def digit_band(mask):
    """Rows occupied by the HP digits, found per frame rather than fixed.

    The readout zooms and the segment bar beneath it moves with it: digits at
    rows 47-85 with the bar at 96-110 when zoomed, 49-80 with the bar at 86-99
    when not. Any fixed crop clearing the bar in one state includes it in the
    other, and a bar touching a digit merges with it -- which made the same "0"
    measure 40 px tall beside a 28 px "1", wrecking the height filter below.
    """
    prof = (mask[:, BAND_COLS[0]:BAND_COLS[1]].sum(axis=1) >= BAND_MIN_COUNT)
    lo = None
    for y, on in enumerate(np.r_[prof, False]):
        if on and lo is None:
            lo = y
        elif not on and lo is not None:
            if y - lo >= BAND_MIN_ROWS:
                return lo, y
            lo = None
    return None


def components(mask):
    cols = mask.any(axis=0)
    out, start = [], None
    for x, on in enumerate(np.r_[cols, False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 3:
                out.append((start, x, mask[:, start:x]))
            start = None
    return out


def glyphs(mask, gray):
    """Normalised grayscale patches for the current-HP digits, left to right.

    Height separates the current value from the "/ max" that follows it, and the
    ~10 px gap before the slash bounds it on the right. Grayscale rather than
    binary because at this size a one-pixel stroke change is a large fraction of
    a binary patch -- which is exactly why binary matching failed across zooms.
    """
    band = digit_band(mask)
    if band is None:
        return []
    m, g = mask[band[0]:band[1]], gray[band[0]:band[1]]
    comps = []
    for x0, x1, sub in components(m):
        ys = np.nonzero(sub.any(axis=1))[0]
        if not len(ys) or x1 - x0 > MAX_W:
            continue
        comps.append((x0, x1, int(ys.min()), int(ys.max())))
    if not comps:
        return []
    tall = max(c[3] - c[2] + 1 for c in comps)
    if tall < 12:
        return []
    kept = [c for c in comps if (c[3] - c[2] + 1) >= tall * TALL_FRAC]
    if not kept or kept[0][0] >= CUR_MAX_X0:
        return []
    grp = [kept[0]]
    for c in kept[1:]:
        if c[0] - grp[-1][1] > GAP:
            break
        grp.append(c)
    out = []
    for x0, x1, y0, y1 in grp:
        p = g[y0:y1 + 1, x0:x1].astype(np.float32)
        im = np.asarray(Image.fromarray(p).resize(DIGIT, Image.BILINEAR))
        im = im - im.mean()
        nn = float(np.linalg.norm(im))
        out.append(im.ravel() / nn if nn > 1e-6 else np.zeros(im.size, np.float32))
    return out


def read_value(mask, gray, tpl):
    g = glyphs(mask, gray)
    if not g or len(g) > 4:
        return None
    s = ""
    for v in g:
        best, score = None, -2.0
        for ch, lst in tpl.items():
            sc = max(float(np.dot(v, u)) for u in lst)
            if sc > score:
                best, score = ch, sc
        if score < MATCH_MIN:
            return None
        s += best
    return int(s) if s else None


def load_templates(runs_root):
    for p in (runs_root / TEMPLATES, cfgmod.REPO_ROOT / "runs" / TEMPLATES):
        if p.exists():
            z = np.load(p)
            return {k: [z[k][i] for i in range(len(z[k]))] for k in z.files}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--seed", nargs="*", default=[], metavar="FRAME=VALUE",
                    help="frames whose HP value you read by eye; builds the "
                         "digit templates (needed once, ever)")
    ap.add_argument("--save-templates", action="store_true")
    ap.add_argument("--with-clashes", action="store_true",
                    help="also count PARRIED attacks, detected as clashes in the "
                         "audio. HP drops alone only see attacks that landed, so "
                         "on a boss you must parry to survive they are a biased "
                         "sample. Together the two cover every attack.")
    ap.add_argument("--clash-merge-ms", type=float, default=700.0)
    a = ap.parse_args()

    runs_root = Path(a.runs) if a.runs else cfgmod.REPO_ROOT / "runs"
    root = runs_root / a.run
    meta = json.loads((root / "meta.json").read_text())
    n = meta["frames"]
    ts = np.load(root / "t_ns.npy")
    if not (root / "hp.npy").exists():
        print("  no hp.npy -- this run predates full-resolution HP capture")
        return 1
    hp = np.asarray(np.load(root / "hp.npy", mmap_mode="r")[:n])
    gray = hp[:, ROWS[0]:ROWS[1], :]
    mask = gray > BRIGHT

    tpl: dict[str, list] = {}
    for spec in a.seed:
        f, _, val = spec.partition("=")
        f = int(f)
        g = glyphs(mask[f], gray[f])
        if len(g) != len(val):
            print(f"  seed {f} ({val}): got {len(g)} glyphs, want {len(val)}"
                  f" -- skipped")
            continue
        for ch, v in zip(val, g):
            tpl.setdefault(ch, []).append(v)
    if not tpl:
        tpl = load_templates(runs_root) or {}
    if not tpl:
        print("  no digit templates. Seed them once from frames you have read:")
        print("    --seed 50=1211 1380=1167 ... --save-templates")
        return 1
    print(f"  templates for digits: {''.join(sorted(tpl))}")
    if a.save_templates:
        out = runs_root / TEMPLATES
        np.savez_compressed(out, **{k: np.array(v) for k, v in tpl.items()})
        print(f"  saved {out}")

    vals = [read_value(mask[i], gray[i], tpl) for i in range(n)]
    got = sum(v is not None for v in vals)
    print(f"  read HP in {got}/{n} frames ({got / max(n, 1):.0%})")

    stable, s = [], 0
    for i in range(1, n + 1):
        if i == n or vals[i] != vals[s]:
            if vals[s] is not None and i - s >= MIN_RUN:
                stable.append((s, int(vals[s])))
            s = i
    seq = stable[:1]
    for f, v in stable[1:]:
        if v != seq[-1][1]:
            seq.append((f, v))
    if len(seq) < 2:
        print("  not enough distinct values to find attacks")
        return 1

    hits, rises = [], 0
    for (_, v0), (f1, v1) in zip(seq, seq[1:]):
        if v1 < v0:
            hits.append((int(ts[f1]), v0 - v1))
        else:
            rises += 1

    parried = []
    if a.with_clashes:
        # HP drops only reveal attacks that LANDED. On a boss the player must
        # parry to survive, that is a biased sample -- exactly the attacks the
        # player handled well are missing. A parried attack instead emits a
        # metallic clash, so the two sources together cover every attack while
        # the player plays normally. Neither depends on the model, so the
        # labels stay non-circular.
        import wave as _wave
        from parry33.audio import onset as onsetmod
        wav = root / "audio.wav"
        t0 = meta.get("audio_start_ns")
        if not wav.exists() or t0 is None:
            print("  --with-clashes: no aligned audio in this run")
        else:
            cfg = cfgmod.load()
            with _wave.open(str(wav)) as w:
                sr, ch = w.getframerate(), w.getnchannels()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pcm = (raw.reshape(-1, ch).mean(1) if ch > 1 else raw)
            pcm = pcm.astype(np.float32) / 32768.0
            bs = cfg.audio.blocksize
            det = onsetmod.SpectralFluxOnset(
                sr, bs, sensitivity=cfg.audio.clash_sensitivity,
                refractory_ms=cfg.audio.refractory_ms,
                floor_frac=cfg.audio.floor_frac,
                fmin=cfg.audio.clash_band[0],
                fmax=min(cfg.audio.clash_band[1], sr / 2))
            clash = []
            for i in range(0, len(pcm) - bs, bs):
                tt = t0 + int(i / sr * 1e9)
                if det.push(pcm[i:i + bs], t_ns=tt):
                    clash.append(tt)
            # A clash ALONE is not evidence of a parry. Checked against a
            # punching-bag run in which the player parried zero times, the
            # detector still reported 271 clashes and 87 "parried attacks" --
            # all false. What identifies a parry is the CONJUNCTION: the player
            # pressed, and a clash followed within ~250 ms.
            pp = root / "presses.npy"
            if not pp.exists():
                print("  --with-clashes: this run has no presses.npy, so a "
                      "clash cannot be tied to a press. Clashes alone are not "
                      "evidence -- skipping. Re-record to capture presses.")
            else:
                press = np.load(pp)
                clash = np.array(clash, dtype=np.int64)
                hit_t = np.array([t for t, _ in hits], dtype=np.int64)
                for t_press in press:
                    d = (clash - t_press) / 1e6
                    if not np.any((d >= 0) & (d <= 250.0)):
                        continue                      # pressed, nothing parried
                    if len(hit_t) and np.abs(hit_t - t_press).min() / 1e6 <= a.clash_merge_ms:
                        continue                      # already counted as landed
                    if parried and (t_press - parried[-1]) / 1e6 <= a.clash_merge_ms:
                        continue
                    parried.append(int(t_press))
                print(f"  {len(press)} player presses, {len(clash)} clashes -> "
                      f"{len(parried)} parried attacks")

    dur = (ts[n - 1] - ts[0]) / 1e9
    dmg = sorted({d for _, d in hits})
    print(f"  {len(seq)} distinct values -> {len(hits)} attacks over {dur:.0f}s "
          f"({len(hits) / max(dur / 60, 1e-9):.1f}/min)")
    print(f"  damage per hit: {dmg}")
    if rises:
        # HP cannot rise during a punching-bag run, so this is the built-in
        # honesty check: an increase is a misread, never an event.
        print(f"  WARNING: {rises} apparent HP INCREASES -- these are misreads. "
              f"Seed more digits, or check the capture.")
    if len(hits) > 1:
        g = np.diff([t for t, _ in hits]) / 1e9
        print(f"  interval between attacks: p50 {np.median(g):.1f}s "
              f"min {g.min():.1f} max {g.max():.1f}")

    records = [{"t_ns": t, "damage": int(d), "outcome": "landed"} for t, d in hits]
    records += [{"t_ns": t, "outcome": "parried"} for t in parried]
    records.sort(key=lambda r: r["t_ns"])
    if parried:
        print(f"  {len(hits)} attacks landed + {len(parried)} parried "
              f"= {len(records)} total")
    out = root / "hits.json"
    out.write_text(json.dumps(
        {"hits": records,
         "source": ("player HP readout + clashes" if parried
                    else "player HP readout (template OCR)"),
         "run": a.run, "frames_read": got, "frames": n,
         "hp_increases": rises, "n_landed": len(hits), "n_parried": len(parried)},
        indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
