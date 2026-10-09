"""Tests 2 and 3 on the cold state 1230 (window 12369-16368), with 1265 (far from the transition) as a control and the
warm runs repeated for comparison. Same boxes and statistics as legs.py / legs3.py / test3b.py:
* natural lag-0 slope of the edge (zonal, Pacific) on M1 (700-2000 m, 15-35 S), 5-95% block bootstrap, detrended check;
* corr(M1 10 and 20 yr after the Pacific edge); r(U1, M1);
* distributed-lag response of U1 and M1 to low-latitude shortwave, summed to 10 yr.
Forced ratios on the cold branch: (a) warm -> cold jump, equilibrium differences 1232.5 -> 1230;
(b) the 1228.5 run (from 1230; records start at 15000, ~500 yr after the step): (e_eq - e_first100)/(M1_eq - M1_first100).
In the cold state the edge (~25 S) lies inside the 15-35 S boxes; a second set of boxes is therefore given relative to
the edge: Ue/Me = 0-700 / 700-2000 m in the 10 deg equatorward of the window-mean edge.
"""
import numpy as np
src = open('prototyping/plasim_reduction/test3.py').read().split("KS = [")[0]; exec(src)
RUNS['1265'] = RUNS['1265']
rng = np.random.default_rng(5)

def edge_boxes(lab, X, d_edge):
    d = dict(np.load(TMP + f'full/{lab}.npz')); ll = d['lsg_lat']
    band = (ll <= -(d_edge - 10)) & (ll > -d_edge)
    for name, field, vol in (('Ue', 'up', 'vol_up'), ('Me', 'mid', 'vol_mid')):
        v = d[vol] * band; X[name] = (np.nan_to_num(d[field]) * v).sum(1) / v.sum()
    return X

def boot_slope(y, x, n=400, nb=100):
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + nb) for s in rng.integers(0, len(y) - nb, len(y) // nb)])
        out.append(slope(y[idx], x[idx]))
    return np.percentile(out, [5, 95])

def detr(v): t = np.arange(len(v)); return v - np.polyval(np.polyfit(t, v, 1), t)
def lagc(e, x, k): return np.corrcoef(x[k:], e[:len(e) - k])[0, 1]   # x k yr after e
K = 30
def fir10(x, F):
    n = len(F); A = np.column_stack([np.ones(n - K)] + [F[K - k:n - k] for k in range(K + 1)])
    h = np.linalg.lstsq(A, x[K:], rcond=None)[0][1:]; return h[:11].sum()

print(f"{'run':8s} {'edge':>5s} | e on M1 [5-95] detr | e_pac on M1 [5-95] | M1 after e_pac 10/20 | r(U1,M1) | e on Me [5-95] | r(Ue,Me) | U1, M1 resp. to F (10 yr)")
for lab in ['1265', '1245', '1240', '1237p5', '1235', '1233p75', '1232p5', '1230']:
    yrs, X = series(lab); S = seg(lab, yrs, X); emean = float(np.nanmean(S['e']))
    X = edge_boxes(lab, X, emean); S = seg(lab, yrs, X)
    lo, hi = boot_slope(S['e'], S['M1']); lp, hp = boot_slope(S['e_pac'], S['M1']); le, he = boot_slope(S['e'], S['Me'])
    print(f"{lab:8s} {emean:5.1f} | {slope(S['e'], S['M1']):+4.1f} [{lo:+4.1f},{hi:+4.1f}] {slope(detr(S['e']), detr(S['M1'])):+4.1f}"
          f" | {slope(S['e_pac'], S['M1']):+4.1f} [{lp:+4.1f},{hp:+4.1f}] | {lagc(S['e_pac'], S['M1'], 10):+.2f} {lagc(S['e_pac'], S['M1'], 20):+.2f}"
          f" | {np.corrcoef(S['U1'], S['M1'])[0,1]:+.2f} | {slope(S['e'], S['Me']):+4.1f} [{le:+4.1f},{he:+4.1f}] | {np.corrcoef(S['Ue'], S['Me'])[0,1]:+.2f}"
          f" | {fir10(S['U1'], S['F']):+.2f} {fir10(S['M1'], S['F']):+.3f}")

print('\nFORCED ratios (deg per K)')
y2, X2 = series('1232p5'); S2 = seg('1232p5', y2, X2); y3, X3 = series('1230'); S3 = seg('1230', y3, X3)
e2, e3 = np.nanmean(S2['e']), np.nanmean(S3['e'])
for b in ('U1', 'M1', 'M2'):
    print(f'  jump 1232.5 -> 1230: e/{b} {(e3 - e2) / (np.nanmean(S3[b]) - np.nanmean(S2[b])):+.1f}')
y4, X4 = series('1228p5'); a, b_ = RUNS['1228p5'].window; w4 = (y4 >= a) & (y4 <= b_); first = y4 < y4[0] + 100
for b in ('U1', 'M1', 'M2'):
    de = np.nanmean(X4['e'][w4]) - np.nanmean(X4['e'][first]); dx = np.nanmean(X4[b][w4]) - np.nanmean(X4[b][first])
    print(f'  1228.5 late slow stage: e/{b} {de / dx:+.1f}  (de {de:+.2f} deg, d{b} {dx:+.3f} K)')
print(f'  1228.5 vs 1230 equilibria: de {np.nanmean(X4["e"][w4]) - e3:+.2f} deg, dM1 {np.nanmean(X4["M1"][w4]) - np.nanmean(S3["M1"]):+.3f} K')
