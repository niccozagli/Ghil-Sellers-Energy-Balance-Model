"""Test 2: the two legs of the edge-ocean loop, natural vs forced, in physical units.

Natural (each run's analysis segment, annual values, no filtering): least-squares slope of the edge on each ocean box
at lag 0 (edge in deg per K of box theta). Annual edge noise does not bias the slope; the box is slowly varying.
Also the trend of each box over the segment (drift check) against its standard deviation.
Forced (transients from 1240): slope of edge against box during the slow stage, between the 80-120 yr block and the
new equilibrium (block means of annual values): d e / d box = (e_eq - e_100) / (box_eq - box_100).
Edges: zonal and Pacific sector. Boxes as in fdt_boxes.py.
"""
import numpy as np
from gsebm.plasim_global import RUNS, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
BOX = ['U1', 'U2', 'M1', 'M2', 'D']

def load(lab):
    d = dict(np.load(TMP + f'full/{lab}.npz')); ll = d['lsg_lat']
    def lsg(field, vol, rows):
        v = vol * rows; return (np.nan_to_num(d[field]) * v).sum(1) / v.sum()
    b1, b2, sall = (ll <= -15) & (ll > -35), (ll <= -35) & (ll >= -70), ll < 0
    X = {'U1': lsg('up', d['vol_up'], b1), 'U2': lsg('up', d['vol_up'], b2), 'M1': lsg('mid', d['vol_mid'], b1),
         'M2': lsg('mid', d['vol_mid'], b2), 'D': lsg('deep', d['vol_deep'], sall), 'e': d['edge_zonal'], 'e_pac': d['edge_pac']}
    ok = np.isfinite(d['Ts'][:, 0])
    return d['years'], {k: np.where(ok, v, np.nan) for k, v in X.items()}

def seg(lab, yrs, X):
    a, b = RUNS[lab].window
    if lab == '1232p5': b -= 1000
    if lab == '1242p5': a += 500
    k = (yrs >= a) & (yrs <= b) & np.isfinite(X['e']); return {n: v[k] for n, v in X.items()}

def slope(y, x):
    x = x - x.mean(); return float(np.dot(x, y - y.mean()) / np.dot(x, x))

print('NATURAL: edge on box, lag 0 (deg per K); [halves]; r')
for edge in ('e', 'e_pac'):
    print(f'-- edge = {edge}')
    for lab in LABELS:
        S = seg(lab, *load(lab)); n = len(S['e']); h = n // 2
        out = []
        for b in BOX:
            s = slope(S[edge], S[b]); s1 = slope(S[edge][:h], S[b][:h]); s2 = slope(S[edge][h:], S[b][h:])
            out.append(f'{b} {s:+6.1f} [{s1:+6.1f} {s2:+6.1f}] r={np.corrcoef(S[edge], S[b])[0,1]:+.2f}')
        print(f'{lab:12s} ' + ' | '.join(out))
print('\nDRIFT: change of box over the segment from a linear fit, in units of its sd')
for lab in LABELS:
    S = seg(lab, *load(lab)); t = np.arange(len(S['e']))
    print(f'{lab:12s} ' + ' '.join(f'{b}: {np.polyfit(t, S[b], 1)[0] * len(t) / S[b].std():+.2f}' for b in BOX))
print('\nFORCED (transients from 1240): slow-stage slope (e_eq - e_100)/(box_eq - box_100), deg per K')
y0, X0 = load('1240'); S0 = seg('1240', y0, X0)
for lab in ['1242p5', '1237p5', '1235_new_IC', '1233p75', '1232p5']:
    yrs, X = load(lab); t = yrs - 14999; S = seg(lab, yrs, X)
    k = (t >= 80) & (t < 120)
    out = []
    for edge in ('e', 'e_pac'):
        for b in BOX:
            de = np.nanmean(S[edge]) - np.nanmean(X[edge][k]); dx = np.nanmean(S[b]) - np.nanmean(X[b][k])
            out.append(f'{edge}/{b} {de / dx:+6.1f}')
    print(f'{lab:12s} ' + ' '.join(out))
