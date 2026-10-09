"""Plain look: Southern ice edge by sector. Annual values over each run's analysis segment (as in earlier checks).

Forced change: window-mean edge 1245 -> 1232.5 per sector. Natural: standard deviation of the annual edge and its
autocorrelation at 10/25/50 yr (the gyre cycle gives a negative value near half its period, ~25-30 yr).
"""
import numpy as np
from gsebm.plasim_global import RUNS
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/full/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SKIP = {'1242p5': (500, 0), '1232p5': (0, 1000)}   # drop first 500 yr / last 1000 yr (stationarity check)
Q = ('zonal', 'atl', 'ind', 'pac')

def segment(lab):
    d = dict(np.load(TMP + f'{lab}.npz')); a, b = RUNS[lab].window; s0, s1 = SKIP.get(lab, (0, 0))
    keep = (d['years'] >= a + s0) & (d['years'] <= b - s1)
    return {q: d[f'edge_{q}'][keep] for q in Q}

def acf(x, k):
    x = x[np.isfinite(x)] - np.nanmean(x); return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))

seg = {lab: segment(lab) for lab in LABELS}
mean = {lab: {q: np.nanmean(seg[lab][q]) for q in Q} for lab in LABELS}
print('window-mean S edge (deg)   ' + '  '.join(f'{q:>6s}' for q in Q))
for lab in LABELS: print(f'{lab:12s}              ' + '  '.join(f'{mean[lab][q]:6.2f}' for q in Q))
print('forced 1245->1232.5        ' + '  '.join(f'{mean["1232p5"][q] - mean["1245"][q]:+6.2f}' for q in Q))
print('\nannual sd (deg) | acf 10 / 25 / 50 yr')
for lab in LABELS:
    print(f'{lab:12s} ' + ' | '.join(f'{q}: {np.nanstd(seg[lab][q]):.2f} ({acf(seg[lab][q],10):+.2f} {acf(seg[lab][q],25):+.2f} {acf(seg[lab][q],50):+.2f})' for q in Q))
print('\ncorrelation of sector edges with each other (annual)')
for lab in LABELS:
    e = np.array([seg[lab][q] for q in Q[1:]]); c = np.corrcoef(e)
    print(f'{lab:12s} atl-ind {c[0,1]:+.2f} atl-pac {c[0,2]:+.2f} ind-pac {c[1,2]:+.2f}')
