"""Search for the slow edge-ocean mode: LIM and KDMD at lags 5-40 yr on an edge + ocean state.

State (annual, run segments; anomalies from segment means; no detrending), each block x sqrt(normalized weights) /
fixed block scale (median over runs of the block's total standard deviation):
  ice  = Southern zonal ocean-ice concentration per T21 row (weights: ocean area of the row; rows with variance only)
  Ts   = Southern zonal surface temperature per T21 row (Gaussian weights)
  up   = Southern zonal theta 0-700 m per LSG row, all longitudes (wet volume)
  mid  = Southern zonal theta 700-2000 m per LSG row (wet volume)
  SA   = South Atlantic sector theta 0-700 m per row (the gyre-cycle carrier)
Forced response r: slope of window means (scaled features) on the step-2 amplitude across the 8 warm runs.
Selection: real eigenvalue, largest |cos| with r (KDMD: cos >= 0.9 and most variance along r, if any).
A true eigenvalue gives the same rate s = log(lambda)/tau at every lag; aliasing of the cycle (period 46-67 yr) shows up
as rates that change with lag (lags 30, 40 are near/above half the period).
"""
import json, numpy as np, h5py
from pathlib import Path
from gsebm.linear_modes import eof_basis, fit_lim, fit_kdmd, pattern_cosine, weighted_inner
from gsebm.plasim_global import ice_edge_latitude, run_archive
from gsebm.time import YEAR
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
LAGS = (5, 10, 15, 20, 30, 40); SPHERE = 4 * np.pi * 6.371e6**2
with h5py.File(run_archive(Path('data/Plasim'), '1240'), 'r') as f:
    lsm = f['lsm'][:]; gw = f['t21_gaussian_weight'][:]
ocean_area = (lsm < 0.5).sum(1) * gw
D = {}
for l in LABELS:
    d = dict(np.load(TMP + f'states/{l}.npz')); d2 = dict(np.load(TMP + f'states2/{l}.npz'))
    assert np.array_equal(d['years'], d2['years']); d.update({k: d2[k] for k in ('up', 'mid', 'deep', 'vol_up', 'vol_mid')}); D[l] = d
d0 = D['1240']; S = d0['lat'] < 0; So = (d0['lsg_lat'] < 0)
ice_rows = S & (np.nanstd(np.nan_to_num(D['1240']['ice']), 0) > 1e-3)
up_rows = So & np.isfinite(d0['up']).all(0) & (d0['vol_up'] > 0); mid_rows = So & np.isfinite(d0['mid']).all(0) & (d0['vol_mid'] > 0)
BLOCKS = {
    'ice': lambda d: (np.nan_to_num(d['ice'][:, ice_rows]), ocean_area[ice_rows]),
    'Ts': lambda d: (d['Ts'][:, S], d['weight'][S]),
    'up': lambda d: (d['up'][:, up_rows], d0['vol_up'][up_rows]),
    'mid': lambda d: (d['mid'][:, mid_rows], d0['vol_mid'][mid_rows]),
    'SA': lambda d: (d['sa_ocean'], d['sa_ocean_w']),
}
names = list(BLOCKS)
scale = {b: np.median([np.sqrt((BLOCKS[b](D[l])[0].var(0) * BLOCKS[b](D[l])[1] / BLOCKS[b](D[l])[1].sum()).sum()) for l in LABELS]) for b in names}
def build(d, seg=slice(None)):
    cols = []
    for b in names:
        X, w = BLOCKS[b](d); X = X[seg]; cols.append((X - X.mean(0)) * np.sqrt(w / w.sum()) / scale[b])
    return np.column_stack(cols)
def means(d):
    return np.concatenate([BLOCKS[b](d)[0].mean(0) * np.sqrt(BLOCKS[b](d)[1] / BLOCKS[b](d)[1].sum()) / scale[b] for b in names])
amp = json.load(open(TMP + 'forced_amp.json')); a = np.array([dict(zip(amp['labels'], amp['a']))[l] for l in LABELS])
M = np.array([means(D[l]) for l in LABELS]); A = np.column_stack([np.ones(len(a)), a - a.mean()])
r = np.linalg.lstsq(A, M, rcond=None)[0][1]; ones = np.ones(r.size)
print('features per block:', {b: BLOCKS[b](d0)[0].shape[1] for b in names}, '| forced-response share per block:',
      {b: round(float(np.sum(seg_**2) / np.sum(r**2)), 2) for b, seg_ in zip(names, np.split(r, np.cumsum([BLOCKS[b](d0)[0].shape[1] for b in names])[:-1]))})
def lim_sel(X, lag):
    b = eof_basis(X, ones, 20); res = fit_lim(b.pcs, lag, YEAR, b.patterns); rr = res.rates * YEAR
    cos = np.array([pattern_cosine(p, r, ones) for p in res.field_patterns]); real = np.abs(rr.imag) < 1e-9
    i = int(np.argmax(np.where(real, cos, -1))); return rr[i].real, cos[i]
def kdmd_sel(X, lag, need_phi=False):
    kd = fit_kdmd(X, ones, lag, YEAR, X.shape[0] - lag, seed=0, rel_threshold=1e-3); rr = kd.rates * YEAR
    cos = np.array([pattern_cosine(p, r, ones) for p in kd.field_patterns]); real = np.abs(rr.imag) < 1e-9
    nr = np.sqrt(weighted_inner(r, r, ones).real); along = np.array([abs(weighted_inner(r, p, ones)) / nr for p in kd.field_patterns])
    sc = np.where(real & (cos >= 0.9), along**2 * kd.eigenfunction_variance, -np.inf)
    i = int(np.argmax(sc)) if np.isfinite(sc).any() else int(np.argmax(np.where(real, cos, -1)))
    cyc = np.flatnonzero(rr.imag > 1e-9); ci = cyc[np.argmax(rr.real[cyc])] if cyc.size else None
    out = (rr[i].real, cos[i], bool(np.isfinite(sc).any()), (2 * np.pi / rr[ci].imag if ci is not None else np.nan))
    return out + ((kd.eigenfunctions[:, i].real, kd.origins) if need_phi else ())
results = {}
for l in LABELS:
    d = D[l]; n = len(d['years']); s0, s1 = SEGMENT[l]; seg = slice(s0, s1 if s1 else n); X = build(d, seg)
    row = {'lim': {}, 'kdmd': {}}
    for lag in LAGS:
        row['lim'][lag] = lim_sel(X, lag)
        row['kdmd'][lag] = kdmd_sel(X, lag)[:4]
    h = X.shape[0] // 2
    row['halves_lim10'] = [lim_sel(X[:h], 10)[0], lim_sel(X[h:], 10)[0]]
    # physical checks on the KDMD mode at lag 10
    s10, c10, ok10, P10, phi, o = kdmd_sel(X, 10, need_phi=True)
    idx = np.arange(n)[seg][o]; phi = phi - phi.mean()
    e = ice_edge_latitude(np.nan_to_num(d['ice'][idx]), d['lat'], south=True)
    up_mean = d['up'][idx][:, up_rows] @ (d0['vol_up'][up_rows] / d0['vol_up'][up_rows].sum())
    mid_mean = d['mid'][idx][:, mid_rows] @ (d0['vol_mid'][mid_rows] / d0['vol_mid'][mid_rows].sum())
    N = (d['asr'][idx] - d['olr'][idx]) @ d['weight']; H = d['ohc_global'][idx].astype(float) / SPHERE
    sl = lambda y: float((y - y.mean()) @ phi / (phi @ phi))
    row['phys'] = dict(r_edge=float(np.corrcoef(phi, e)[0, 1]), r_up=float(np.corrcoef(phi, up_mean)[0, 1]), r_mid=float(np.corrcoef(phi, mid_mean)[0, 1]),
                       r_H=float(np.corrcoef(phi, H)[0, 1]), s_budget=sl(N) / sl(H) * YEAR * 360 / 365.25 if sl(H) != 0 else np.nan)
    results[l] = row
    L_, K_ = row['lim'], row['kdmd']
    print(f"\n{l}: LIM real-aligned rate (1/yr) [cos] at lags {LAGS}: " + ' '.join(f"{L_[k][0]:+.4f}[{L_[k][1]:.2f}]" for k in LAGS))
    print(f"   KDMD: " + ' '.join(f"{K_[k][0]:+.4f}[{K_[k][1]:.2f}{'*' if K_[k][2] else ''}]" for k in LAGS) + f"  (* = cos≥0.9 selection) | cycle period at lag 10: {K_[10][3]:.0f} yr")
    p = row['phys']
    print(f"   halves LIM lag10: {row['halves_lim10'][0]:+.4f} / {row['halves_lim10'][1]:+.4f} | KDMD lag10 mode: corr with edge {p['r_edge']:+.2f}, S θ0–700 {p['r_up']:+.2f}, S θ700–2000 {p['r_mid']:+.2f}, global OHC {p['r_H']:+.2f}; budget rate {p['s_budget']:+.4f}")
json.dump(results, open(TMP + 'slow_mode.json', 'w'), default=float)
