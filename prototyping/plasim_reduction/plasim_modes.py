"""Step 3: LIM and KDMD on warm-branch PlaSim runs; select the mode aligned with the forced response.

States (annual, run segments below, anomalies from the segment mean, no detrending):
  A = S Atlantic sector zonal Ts + S Atlantic sector θ 0-700 m rows (the earlier Koopman state)
  B = zonal Ts all longitudes (32 Gaussian rows) + S Atlantic θ 0-700 m rows
  C = B + zonal θ 0-700 m rows of the Southern Hemisphere (all longitudes)
Each feature is multiplied by sqrt(normalized area weight) and each block divided by a fixed scale (the median over
runs of its total standard deviation), so the metric is the same at every μ.
Forced response r: slope of window means of each feature on the step-2 amplitude (K of global Ts), across the 8 runs.
"""
import json, numpy as np
from gsebm.linear_modes import eof_basis, fit_lim, fit_kdmd, pattern_cosine, weighted_inner
from gsebm.time import YEAR
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
FLAG = {'1245': 'deep variability', '1235_new_IC': 'check only (last 1500 yr)'}
SPHERE = 4 * np.pi * 6.371e6**2
D = {l: dict(np.load(TMP + f'states/{l}.npz')) for l in LABELS}
amp = json.load(open(TMP + 'forced_amp.json')); a_run = np.array([dict(zip(amp['labels'], amp['a']))[l] for l in LABELS])
mu = {l: m for l, m in zip(amp['labels'], amp['mu'])}

d0 = D['1240']; so = (d0['lsg_lat'] < 0) & np.isfinite(D['1240']['theta_up_rows']).all(0) & (d0['lsg_area'] > 0)
BLOCKS = {
    'saTs': lambda d: (d['sa_surface'], d['sa_surface_w']),
    'saO': lambda d: (d['sa_ocean'], d['sa_ocean_w']),
    'zTs': lambda d: (d['Ts'], d['weight']),
    'zSO': lambda d: (d['theta_up_rows'][:, so], d['lsg_area'][so] / d['lsg_area'][so].sum()),
}
STATES = {'A': ['saTs', 'saO'], 'B': ['zTs', 'saO'], 'C': ['zTs', 'saO', 'zSO']}
scale = {b: np.median([np.sqrt((BLOCKS[b](D[l])[0].var(0) * BLOCKS[b](D[l])[1]).sum()) for l in LABELS]) for b in BLOCKS}

def build(d, blocks, seg=None):
    cols = []
    for b in blocks:
        X, w = BLOCKS[b](d); X = X if seg is None else X[seg]
        cols.append((X - X.mean(0)) * np.sqrt(w / w.sum()) / scale[b])
    return np.column_stack(cols)

def forced(blocks):
    means = np.array([np.concatenate([BLOCKS[b](D[l])[0].mean(0) * np.sqrt(BLOCKS[b](D[l])[1] / BLOCKS[b](D[l])[1].sum()) / scale[b] for b in blocks]) for l in LABELS])
    A = np.column_stack([np.ones(len(LABELS)), a_run - a_run.mean()])
    return np.linalg.lstsq(A, means, rcond=None)[0][1]

def slope(y, x):
    x = x - x.mean(); return float((y - y.mean()) @ x / (x @ x))

def physics(d, seg, phi_t, origins):
    """Regress global observables on the eigenfunction (real part) at the given states."""
    idx = np.arange(len(d['years']))[seg][origins]
    phi = phi_t.real; phi = phi - phi.mean()
    w = d['weight']; Tg = d['Ts'][idx] @ w
    asr, ins, olr = d['asr'][idx], d['ins'][idx], d['olr'][idx]
    alb = 1 - asr / ins; alb_term = -(ins.mean(0) * (alb - alb.mean(0))) @ w
    N = (asr - olr) @ w; H = d['ohc_global'][idx] / SPHERE
    iceS = d['ice'][idx][:, d['lat'] < 0]; iceS = np.nan_to_num(iceS) @ w[d['lat'] < 0]
    sT = slope(Tg, phi)
    out = dict(s_budget=slope(N, phi) / slope(H, phi) * YEAR, olr=slope(olr @ w, phi) / sT, albedo=slope(alb_term, phi) / sT,
               C_m=slope(H, phi) / sT / 4.18e6, r_Tg=np.corrcoef(phi, Tg)[0, 1], r_ice=np.corrcoef(phi, iceS)[0, 1])
    # in-phase test: lag of max |corr| between phi and Tg, and phi and S ice (positive = observable lags phi)
    def lagmax(v):
        c = [np.corrcoef(phi[:len(phi) - k], v[k:])[0, 1] if k >= 0 else np.corrcoef(phi[-k:], v[:len(v) + k])[0, 1] for k in range(-8, 9)]
        return int(np.argmax(np.abs(c))) - 8
    out['lag_Tg'] = lagmax(Tg); out['lag_ice'] = lagmax(iceS)
    return out

results = []
for state, blocks in STATES.items():
    r = forced(blocks); ones = np.ones(r.size)
    print(f'\n===== state {state} ({"+".join(blocks)}), {r.size} features =====')
    for l in LABELS:
        d = D[l]; n = len(d['years']); s0, s1 = SEGMENT[l]; seg = slice(s0, s1 if s1 else n)
        X = build(d, blocks, seg)
        line = f'{l:12s} μ={mu[l]:7.2f} n={X.shape[0]:4d}' + (f' [{FLAG[l]}]' if l in FLAG else '')
        print(line)
        for lag in (1, 2, 5):
            b = eof_basis(X, ones, 20)
            lim = fit_lim(b.pcs, lag, YEAR, b.patterns)
            lr = lim.rates * YEAR
            lcos = np.array([pattern_cosine(p, r, ones) for p in lim.field_patterns])
            real = np.abs(lr.imag) < 1e-9
            li = int(np.argmax(np.where(real, lcos, -1)))
            kd = fit_kdmd(X, ones, lag, YEAR, X.shape[0] - lag, seed=0, rel_threshold=1e-3)
            kr = kd.rates * YEAR
            kcos = np.array([pattern_cosine(p, r, ones) for p in kd.field_patterns])
            kreal = np.abs(kr.imag) < 1e-9
            norm = np.sqrt(weighted_inner(r, r, ones).real)
            along = np.array([abs(weighted_inner(r, p, ones)) / norm for p in kd.field_patterns])
            score = np.where(kreal & (kcos >= 0.9), along**2 * kd.eigenfunction_variance, -np.inf)
            ki = int(np.argmax(score)) if np.isfinite(score).any() else int(np.argmax(np.where(kreal, kcos, -1)))
            cyc = np.flatnonzero((kr.imag > 1e-9))
            ci = cyc[np.argmax(kr.real[cyc])] if cyc.size else None
            phys = physics(d, seg, kd.eigenfunctions[:, ki], kd.origins)
            row = dict(state=state, run=l, mu=mu[l], lag=lag, lim_rate=lr[li].real, lim_cos=lcos[li], kdmd_rate=kr[ki].real,
                       kdmd_cos=kcos[ki], kdmd_ok=bool(np.isfinite(score).any()), rank=kd.rank,
                       cycle=(kr[ci].real, 2 * np.pi / kr[ci].imag) if ci is not None else None, **phys)
            results.append(row)
            cyc_s = f'cycle {kr[ci].real:+.3f}/yr P={2*np.pi/kr[ci].imag:5.1f}yr' if ci is not None else 'no cycle'
            print(f'   lag {lag}: LIM real-aligned {lr[li].real:+.4f} (cos {lcos[li]:.2f}) | KDMD {"sel" if row["kdmd_ok"] else "best<0.9"} {kr[ki].real:+.4f} (cos {kcos[ki]:.2f}, rank {kd.rank}) {cyc_s} '
                  f'| budget s {phys["s_budget"]:+.4f}, OLR {phys["olr"]:.2f}, albedo {phys["albedo"]:.2f}, C {phys["C_m"]:.0f} m, r(Tg) {phys["r_Tg"]:+.2f}, r(Sice) {phys["r_ice"]:+.2f}, lag Tg/ice {phys["lag_Tg"]}/{phys["lag_ice"]}', flush=True)
json.dump(results, open(TMP + 'plasim_modes.json', 'w'), default=float)
