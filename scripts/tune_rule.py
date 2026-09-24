"""Find the best (threshold, consecutive) firing rule from a live score log.

Requires a log containing every score, not just the fires. Scored against the
player's own presses: a fire is a true detection if a press follows within
[-200, 1500] ms, and recall counts distinct presses anticipated.
"""
import json, sys, pathlib
import numpy as np

def load(p):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    sc=[(r["t_ns"], r["p"]) for r in rows if r.get("kind")=="score"]
    hp=sorted(r["t_ns"] for r in rows if r.get("kind")=="human_parry")
    return sorted(sc), np.array(hp)

def simulate(sc, hp, th, n, refractory_ms=800):
    fires=[]; run=0; lock=0
    for t,p in sc:
        if t < lock: run=0; continue
        run = run+1 if p>=th else 0
        if run>=n:
            fires.append(t); run=0; lock=t+refractory_ms*1_000_000
    fires=np.array(fires)
    if not len(fires): return 0,0,0.0,0
    tp=sum(1 for f in fires if np.any(((hp-f)/1e6>=-200)&((hp-f)/1e6<=1500)))
    cov=sum(1 for h in hp if np.any(((h-fires)/1e6>=-200)&((h-fires)/1e6<=1500)))
    return len(fires), tp, cov/max(len(hp),1), len(fires)-tp

if __name__=="__main__":
    logs=[pathlib.Path(a) for a in sys.argv[1:]]
    if not logs:      # default to the newest live log; no placeholder to mistype
        runs=pathlib.Path(__file__).resolve().parents[1]/"runs"
        logs=sorted(runs.glob("live_*.jsonl"), key=lambda q:q.stat().st_mtime)[-1:]
        print("  (no log given, using " + (logs[0].name if logs else "nothing") + ")")
    sc=[]; hp=[]
    for L in logs:
        s,h=load(L); sc+=s; hp+=list(h)
    sc=sorted(sc); hp=np.array(sorted(hp))
    span=(sc[-1][0]-sc[0][0])/1e9 if len(sc)>1 else 1
    print(f"  {len(sc)} scores, {len(hp)} human parries, {span:.0f}s\n")
    print(f"  {'thr':>5} {'n':>2} {'fires':>6} {'recall':>7} {'FA/min':>7} {'prec':>6}")
    best=[]
    for th in (0.5,0.6,0.7,0.8,0.9,0.95):
        for n in (2,3,4,5):
            nf,tp,rec,fp=simulate(sc,hp,th,n)
            fa=fp/max(span,1)*60
            prec=tp/max(nf,1)
            best.append((rec,fa,th,n,prec))
            print(f"  {th:5.2f} {n:2d} {nf:6d} {rec:6.0%} {fa:7.1f} {prec:6.0%}")
    ok=[b for b in best if b[1]<=3.0]
    if ok:
        b=max(ok,key=lambda r:r[0])
        print(f"\n  best recall at <=3 FA/min: thr={b[2]} n={b[3]} -> "
              f"recall {b[0]:.0%}, {b[1]:.1f} FA/min")
    else:
        print("\n  no rule achieves <=3 FA/min on this log")
