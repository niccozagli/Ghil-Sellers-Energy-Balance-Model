"""(A2) Fast part of the edge response; (C) stationary early-warning statistics of the annual Southern edge.

A2: f = mean of r_e(t) over years 15-40 after the step (a block mean, used only to read off the fast plateau), with
its standard error from the annual scatter (n_eff from the lag-1 autocorrelation). Fast edge sensitivity = (1 - f) x
total edge change per W m-2 of mu... reported as fast change = (1 - f)(e_eq - e0) per unit dmu.
C: annual edge in the analysis segments: standard deviation, autocorrelation at 1, 5, 10 yr; the same after regressing
out the gyre-cycle eigenfunction (state A KDMD, lag 5, as in restoring.py).
"""
import numpy as np, json
from gsebm.plasim_global import read_global_series, ice_edge_latitude, RUNS, run_mu
from pathlib import Path
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
rel = json.load(open(TMP + 'relax.json'))
print('A2: fast fraction of the edge response (years 15–40 after the step)')
for r in rel:
    s = read_global_series(Path('data/Plasim'), r['run'], full=True)
    e = ice_edge_latitude(s.ocean_ice, s.lat, south=True); t = s.years - s.years[0] + r['missing_start']
    re = (e - r['eq']) / (r['e0'] - r['eq']); blk = re[(t >= 15) & (t <= 40)]
    if blk.size < 5: print(f"  {r['run']:12s} no early data"); continue
    a1 = np.corrcoef(blk[:-1], blk[1:])[0, 1]; neff = blk.size * (1 - a1) / (1 + a1)
    f = blk.mean(); se = blk.std() / np.sqrt(max(neff, 1))
    dmu = run_mu(r['run']) - run_mu(r['parent'])
    print(f"  {r['run']:12s} from {r['parent']} step {r['step']:+.2f}° | slow fraction r(15–40 yr) {f:.2f} ± {se:.2f} | fast change {(1-f)*r['step']:+.2f}° "
          f"(fast sensitivity {(1-f)*r['step']/dmu:+.3f} °/(W m⁻²); total {r['step']/dmu:+.3f})")
