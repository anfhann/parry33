"""Compare parry-mode vs dodge-mode boss attempts.

Mode is inferred from the E/Q ratio -- no tagging needed. Enemy attack count is
estimated from clash onsets plus defensive inputs, which is independent of which
key was pressed.
"""
import json, pathlib, sys, numpy as np

def summarise(run):
    ev=[json.loads(l) for l in open(run/"events.jsonl",encoding="utf-8")]
    meta=json.load(open(run/"meta.json",encoding="utf-8"))
    par=[e for e in ev if e["kind"]=="parry"]
    dod=[e for e in ev if e["kind"]=="dodge"]
    clash=[e for e in ev if e["kind"]=="clash"]
    n=len(par)+len(dod)
    mode = "parry" if len(par) > 3*max(len(dod),1) else (
           "dodge" if len(dod) > 3*max(len(par),1) else "mixed")
    ts=[e["t_ns"] for e in ev]
    dur=(max(ts)-min(ts))/1e9 if ts else 0
    return dict(run=run.name, mode=mode, dur=dur, parry=len(par), dodge=len(dod),
                defensive=n, clash=len(clash),
                frames=meta.get("frames"), landed=None)

if __name__=="__main__":
    runs=[pathlib.Path(p) for p in sys.argv[1:]]
    rows=[summarise(r) for r in runs]
    print(f"{'run':>15} {'mode':>7} {'dur s':>7} {'E':>4} {'Q':>4} {'defend':>7} {'clash':>6}")
    for r in rows:
        print(f"{r['run'][:15]:>15} {r['mode']:>7} {r['dur']:7.0f} {r['parry']:4d} "
              f"{r['dodge']:4d} {r['defensive']:7d} {r['clash']:6d}")
    for m in ("parry","dodge"):
        s=[r for r in rows if r["mode"]==m]
        if s:
            print(f"\n  {m}-mode: {len(s)} attempts, mean {np.mean([r['dur'] for r in s]):.0f}s, "
                  f"mean {np.mean([r['defensive'] for r in s]):.0f} defensive inputs")
    p=[r for r in rows if r["mode"]=="parry"]; d=[r for r in rows if r["mode"]=="dodge"]
    if p and d:
        rt=np.mean([r['dur'] for r in d])/np.mean([r['dur'] for r in p])
        ra=np.mean([r['defensive'] for r in d])/np.mean([r['defensive'] for r in p])
        print(f"\n  RATIO dodge/parry -- duration {rt:.2f}x, defensive inputs {ra:.2f}x")
        print(f"  (predicted 2.00x; crossover at 1.5x -- above it parry wins on time-to-win)")
