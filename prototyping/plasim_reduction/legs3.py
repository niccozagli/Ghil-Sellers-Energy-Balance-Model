"""Robustness of the natural edge-on-M1 slope: moving-block bootstrap (100 yr) 90% interval, edge drift, and (check only)
the slope after removing a linear trend from both series."""
import numpy as np
src = open('prototyping/plasim_reduction/legs.py').read().split("print('NATURAL")[0]; exec(src)
rng = np.random.default_rng(1)
def boot(y, x, nb=100, n=400):
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + nb) for s in rng.integers(0, len(y) - nb, len(y) // nb)])
        out.append(slope(y[idx], x[idx]))
    return np.percentile(out, [5, 95])
def detr(v): t = np.arange(len(v)); return v - np.polyval(np.polyfit(t, v, 1), t)
print('edge on M1 (deg/K): slope [5-95% block bootstrap] | detrended slope | edge drift (sd)    forced slow-stage: 3.0-3.7 (zonal), 1.8-3.6 (pac)')
for edge in ('e', 'e_pac'):
    print('--', edge)
    for lab in LABELS:
        S = seg(lab, *load(lab)); y, x = S[edge], S['M1']; lo, hi = boot(y, x); t = np.arange(len(y))
        print(f'{lab:12s} {slope(y, x):+5.1f} [{lo:+5.1f}, {hi:+5.1f}] | {slope(detr(y), detr(x)):+5.1f} | {np.polyfit(t, y, 1)[0] * len(t) / y.std():+.2f}')
