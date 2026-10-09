"""(C) Stationary early-warning statistics, annual values, analysis segments (as in step 3), no smoothing/detrending.
Fast variable: Southern edge e (zonal). Slow variables: global LSG heat content H (from zonal_ocean_heat_content),
Southern upper-ocean theta 0-700 m (wet-area weighted, all longitudes). Statistics: standard deviation, autocorrelation
at 1, 5, 10, 50, 100 yr; for e also after regressing out the gyre-cycle eigenfunction (state A KDMD, lag 5 yr).
Uncertainty: the two halves of each segment.
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, run_mu
from gsebm.linear_modes import fit_kdmd
from gsebm.time import YEAR
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
def acf(x, k): x = x - x.mean(); return float(x[:-k] @ x[k:] / (len(x) - k) / x.var())
def stats(x, lags): return dict(sd=float(x.std()), **{f'a{k}': acf(x, k) for k in lags})
def stateA(d, seg):
    cols = []
    for X, w in ((d['sa_surface'][seg], d['sa_surface_w']), (d['sa_ocean'][seg], d['sa_ocean_w'])):
        Z = (X - X.mean(0)) * np.sqrt(w / w.sum()); cols.append(Z / np.sqrt(Z.var(0).sum()))
    return np.column_stack(cols)
out = {}
print(f"{'run':12s} | edge sd, a1 a5 a10 [halves a5] | edge cycle-removed a1 a5 a10 | S θ0–700 sd(mK) a10 a50 a100 | global OHC a50 a100 [halves a100]")
for lab in LABELS:
    d = dict(np.load(TMP + f'states/{lab}.npz')); n = len(d['years']); s0, s1 = SEGMENT[lab]; seg = slice(s0, s1 if s1 else n)
    e = ice_edge_latitude(np.nan_to_num(d['ice'][seg]), d['lat'], south=True)
    so = (d['lsg_lat'] < 0) & np.isfinite(d['theta_up_rows']).all(0) & (d['lsg_area'] > 0)
    th = d['theta_up_rows'][seg][:, so] @ (d['lsg_area'][so] / d['lsg_area'][so].sum())
    H = d['ohc_global'][seg].astype(float)
    h = len(e) // 2
    X = stateA(d, seg); kd = fit_kdmd(X, np.ones(X.shape[1]), 5, YEAR, len(e) - 5, seed=0, rel_threshold=1e-3)
    rates = kd.rates * YEAR; c = np.flatnonzero(rates.imag > 1e-9); ci = c[np.argmax(rates.real[c])]
    psi = kd.eigenfunctions[:, ci]; o = kd.origins; Xc = np.column_stack([np.ones(len(o)), psi.real, psi.imag])
    er = e[o] - Xc @ np.linalg.lstsq(Xc, e[o], rcond=None)[0]
    r = dict(mu=run_mu(lab), edge=stats(e, (1, 5, 10)), edge_h=[stats(e[:h], (5,)), stats(e[h:], (5,))], edge_nc=stats(er, (1, 5, 10)),
             theta=stats(th, (10, 50, 100)), ohc=stats(H, (50, 100)), ohc_h=[stats(H[:h], (100,)), stats(H[h:], (100,))])
    out[lab] = r
    E, N, T, O = r['edge'], r['edge_nc'], r['theta'], r['ohc']
    print(f"{lab:12s} | {E['sd']:.2f}, {E['a1']:.2f} {E['a5']:.2f} {E['a10']:.2f} [{r['edge_h'][0]['a5']:.2f}/{r['edge_h'][1]['a5']:.2f}] | {N['a1']:.2f} {N['a5']:.2f} {N['a10']:.2f} | "
          f"{T['sd']*1e3:5.1f} {T['a10']:.2f} {T['a50']:.2f} {T['a100']:.2f} | {O['a50']:.2f} {O['a100']:.2f} [{r['ohc_h'][0]['a100']:.2f}/{r['ohc_h'][1]['a100']:.2f}]")
json.dump(out, open(TMP + 'ews.json', 'w'))
