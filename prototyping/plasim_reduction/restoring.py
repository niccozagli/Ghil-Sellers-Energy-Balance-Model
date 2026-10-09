"""Restoring of the Southern surface anomaly vs mu: radiative slope, ocean uptake response, persistence.

Full Southern Hemisphere, annual anomalies, run segments as in step 3, no smoothing/detrending.
  T_S  = Gaussian-weighted mean surface temperature over the 16 Southern T21 rows
  ice  = Southern sea-ice AREA = sum_j zonal ocean-ice concentration_j * (ocean cells in row j) * cell area_j  [10^12 m^2]
  RAD  = OLR - albedo term, Southern means (W m-2 of S hemisphere); albedo term = -mean(I_j) (alpha_j - mean alpha_j)
  UPT  = ocean heat uptake summed over Southern LSG rows (W) / S hemisphere area  (W m-2 of S hemisphere)
1. radiative restoring  = lag-0 slope of RAD on T_S (W m-2 K-1; positive = damping)
2. ocean restoring      = distributed-lag fit UPT(t) = c + sum_{k=0..20} g_k T_S(t-k); G = sum g_k (fitted linear model)
3. persistence          = autocorrelation of T_S and of ice at lags 1, 5, 10 yr (read off the data)
3b. same after regressing out the cycle eigenfunction (Re, Im of the leading complex KDMD mode of the
    S Atlantic state A, lag 5 yr) -- labelled second version
Uncertainty: moving-block bootstrap (100-yr blocks, 400 resamples), 2.5-97.5%; halves; the 1235 pair.
"""
import numpy as np, h5py, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import run_archive
from gsebm.linear_modes import fit_kdmd
from gsebm.time import YEAR
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
MU = {'1245': 1245, '1242p5': 1242.5, '1240': 1240, '1237p5': 1237.5, '1235': 1235, '1235_new_IC': 1235, '1233p75': 1233.75, '1232p5': 1232.5}
R = 6.371e6; HEMI = 2 * np.pi * R**2; LAGS = 20; rng = np.random.default_rng(0)
with h5py.File(run_archive(Path('data/Plasim'), '1240'), 'r') as f:
    lsm = f['lsm'][:]; gw = f['t21_gaussian_weight'][:]; nlon = f['t21_lon'].shape[0]
ocean_cells = (lsm < 0.5).sum(1); cell_area = gw * 2 * np.pi * R**2 / nlon  # gw sums to 2
row_ocean_area = ocean_cells * cell_area

def acf(x, k): x = x - x.mean(); return float(x[:-k] @ x[k:] / (len(x) - k) / x.var())
def slope(y, x): x = x - x.mean(); return float((y - y.mean()) @ x / (x @ x))
def dlag(F, T):
    X = np.column_stack([np.ones(len(T) - LAGS)] + [T[LAGS - k:len(T) - k] for k in range(LAGS + 1)])
    coef = np.linalg.lstsq(X, F[LAGS:], rcond=None)[0][1:]
    return coef
def stats(T, ice, RAD, UPT):
    g = dlag(UPT, T)
    return dict(rad=slope(RAD, T), G=float(g.sum()), g0=float(g[0]), g_after=float(g[1:].sum()),
                **{f'T_acf{k}': acf(T, k) for k in (1, 5, 10)}, **{f'ice_acf{k}': acf(ice, k) for k in (1, 5, 10)})
def block_boot(series, fn, n=400, block=100):
    N = len(series[0]); nb = int(np.ceil(N / block)); out = []
    for _ in range(n):
        starts = rng.integers(0, N - block, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        out.append(fn(*[s[idx] for s in series]))
    return {k: np.percentile([o[k] for o in out], [2.5, 97.5]) for k in out[0]}

# cycle eigenfunction from state A (same blocks/scales as plasim_modes.py)
D = {l: dict(np.load(TMP + f'states/{l}.npz')) for l in LABELS}
def stateA(d, seg):
    cols = []
    for X, w in ((d['sa_surface'][seg], d['sa_surface_w']), (d['sa_ocean'][seg], d['sa_ocean_w'])):
        Z = (X - X.mean(0)) * np.sqrt(w / w.sum()); cols.append(Z / np.sqrt((Z.var(0)).sum()))
    return np.column_stack(cols)

results = {}
for lab in LABELS:
    d = D[lab]; n = len(d['years']); s0, s1 = SEGMENT[lab]; seg = slice(s0, s1 if s1 else n)
    lat, w = d['lat'], d['weight']; S = lat < 0; wS = w[S] / w[S].sum()
    T = d['Ts'][seg][:, S] @ wS
    ice = (np.nan_to_num(d['ice'][seg][:, S]) * row_ocean_area[S]).sum(1) / 1e12
    asr, ins = d['asr'][seg], d['ins'][seg]; alb = 1 - asr / ins
    albterm = -(ins.mean(0)[S] * (alb[:, S] - alb[:, S].mean(0))) @ wS
    RAD = d['olr'][seg][:, S] @ wS - albterm
    ll, area = d['lsg_lat'], d['lsg_area']; So = (ll < 0) & (area > 0)
    UPT = (d['uptake'][seg][:, So] @ area[So]) / HEMI
    full = stats(T, ice, RAD, UPT); h = len(T) // 2
    halves = [stats(T[:h], ice[:h], RAD[:h], UPT[:h]), stats(T[h:], ice[h:], RAD[h:], UPT[h:])]
    ci = block_boot([T, ice, RAD, UPT], stats)
    # second version: cycle regressed out
    lag = 5; kd = fit_kdmd(stateA(d, seg), np.ones(stateA(d, seg).shape[1]), lag, YEAR, len(T) - lag, seed=0, rel_threshold=1e-3)
    rates = kd.rates * YEAR; cyc = np.flatnonzero(rates.imag > 1e-9); ci_ = cyc[np.argmax(rates.real[cyc])]
    psi = kd.eigenfunctions[:, ci_]; o = kd.origins
    Xc = np.column_stack([np.ones(len(o)), psi.real, psi.imag])
    Tr = T[o] - Xc @ np.linalg.lstsq(Xc, T[o], rcond=None)[0]; Ir = ice[o] - Xc @ np.linalg.lstsq(Xc, ice[o], rcond=None)[0]
    noc = {f'T_acf{k}': acf(Tr, k) for k in (1, 5, 10)} | {f'ice_acf{k}': acf(Ir, k) for k in (1, 5, 10)}
    noc['var_removed_T'] = 1 - Tr.var() / T[o].var(); noc['var_removed_ice'] = 1 - Ir.var() / ice[o].var()
    noc['period'] = 2 * np.pi / rates[ci_].imag
    results[lab] = dict(full=full, halves=halves, ci={k: v.tolist() for k, v in ci.items()}, nocycle=noc, n=len(T),
                        ice_mean=float((np.nan_to_num(d['ice'][seg][:, S]) * row_ocean_area[S]).sum(1).mean() / 1e12))
    f_ = full; c = ci
    print(f"{lab:12s} μ={MU[lab]:7.2f} n={len(T)} ice mean {results[lab]['ice_mean']:.1f}e12 m2")
    print(f"   radiative restoring {f_['rad']:+.3f} [{c['rad'][0]:+.3f},{c['rad'][1]:+.3f}] halves {halves[0]['rad']:+.3f}/{halves[1]['rad']:+.3f}")
    print(f"   ocean G {f_['G']:+.3f} [{c['G'][0]:+.3f},{c['G'][1]:+.3f}] (g0 {f_['g0']:+.3f}, lags1-20 {f_['g_after']:+.3f}) halves {halves[0]['G']:+.3f}/{halves[1]['G']:+.3f}")
    print(f"   total rad+G {f_['rad'] + f_['G']:+.3f}")
    for v in ('T', 'ice'):
        print(f"   {v} ACF 1/5/10: " + ' '.join(f"{f_[f'{v}_acf{k}']:.2f}[{c[f'{v}_acf{k}'][0]:.2f},{c[f'{v}_acf{k}'][1]:.2f}]" for k in (1, 5, 10))
              + f" | halves 10yr {halves[0][f'{v}_acf10']:.2f}/{halves[1][f'{v}_acf10']:.2f} | cycle removed ({noc['var_removed_'+v]*100:.0f}% var, P={noc['period']:.0f}yr): "
              + ' '.join(f"{noc[f'{v}_acf{k}']:.2f}" for k in (1, 5, 10)), flush=True)
json.dump(results, open(TMP + 'restoring.json', 'w'), default=float)

# figure: each measure vs mu with bootstrap bars
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
items = [('rad', 'radiative restoring\nOLR − albedo per K (W m⁻² K⁻¹)'), ('G', 'ocean restoring G = Σg_k\n(W m⁻² K⁻¹)'),
         ('T_acf10', 'autocorrelation of T_S at 10 yr'), ('ice_acf10', 'autocorrelation of S ice area at 10 yr')]
for a, (k, title) in zip(ax, items):
    for lab in LABELS:
        r = results[lab]; m = MU[lab] + (0.25 if lab == '1235_new_IC' else 0)
        col = '#9a9a9a' if lab in ('1235_new_IC', '1245') else '#2a6fb0'
        a.errorbar(m, r['full'][k], yerr=[[r['full'][k] - r['ci'][k][0]], [r['ci'][k][1] - r['full'][k]]], fmt='o', color=col, capsize=3)
        if k.endswith('acf10'): a.plot(m, r['nocycle'][k], 'x', color='#c0612b')
    a.set_title(title, fontsize=9); a.set_xlabel('μ (W m⁻²)')
ax[2].plot([], [], 'x', color='#c0612b', label='cycle regressed out'); ax[2].legend(fontsize=8)
fig.suptitle('Restoring of the Southern Hemisphere surface anomaly (grey: 1245 flagged, 1235_new_IC check only; bars: block bootstrap 95%)', fontsize=10)
fig.tight_layout(); fig.savefig('figures/plasim_mechanism/southern_restoring.png', dpi=120)
