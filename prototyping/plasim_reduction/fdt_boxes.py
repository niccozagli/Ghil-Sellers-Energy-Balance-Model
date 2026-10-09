"""Test 1: does the natural variability of 1240 predict its response to a mu step? (fluctuation-dissipation via a LIM)

State (annual, physical boxes, no filtering):
  e    Southern zonal ice edge (deg latitude, larger = more poleward)
  TsS, TsN  hemispheric-mean surface temperature (Gaussian weights)
  U1, U2    Southern ocean theta 0-700 m, rows 15-35 S and 35-70 S (wet-volume weighted)
  M1, M2    same for 700-2000 m
  D         Southern theta below 2000 m (all Southern rows)
LIM on the 1240 analysis window: A(tau) = C(tau) C(0)^-1, L = log(A)/tau. Step response to a forcing f (per W m^-2 of
solar constant): G(t) f with G(t) = L^-1 (exp(L t) - I). f is not known; it is fitted to the first T_FIT years of
the small steps from 1240 (1242.5, 1237.5, 1235_new_IC), with f = 0 on the ocean boxes below 700 m (mu heats the
surface; the deeper ocean can only respond). The test is the prediction for t > T_FIT and for the new equilibrium.
LIM uncertainty: moving-block bootstrap of the 1240 window (100-yr blocks, lag pairs within blocks).
"""
import sys, json, numpy as np
from scipy.linalg import expm, logm
from gsebm.plasim_global import RUNS, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
NAMES = ['e', 'TsS', 'TsN', 'U1', 'U2', 'M1', 'M2', 'D']
FORCED = np.array([1, 1, 1, 1, 1, 0, 0, 0], bool)
T_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TAU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FREE_F = len(sys.argv) > 3 and sys.argv[3] == 'free'
STEPS = ['1242p5', '1237p5', '1235_new_IC', '1233p75', '1232p5']
SMALL = ['1242p5', '1237p5', '1235_new_IC']
BRANCH = 14999

def boxes(lab):
    d = dict(np.load(TMP + f'full/{lab}.npz')); lat, ll = d['lat'], d['lsg_lat']
    w = np.cos(np.deg2rad(lat))  # T21 rows: Gaussian weights ~ cos(lat) spacing; use file weights when present
    try:
        w = dict(np.load(TMP + 'states/1240.npz'))['weight']
    except Exception:
        pass
    s, n = lat < 0, lat > 0
    def lsg(field, vol, rows):
        v = vol * rows; return (np.nan_to_num(d[field]) * v).sum(1) / v.sum()
    b1, b2, sall = (ll <= -15) & (ll > -35), (ll <= -35) & (ll >= -70), ll < 0
    X = np.column_stack([d['edge_zonal'], d['Ts'][:, s] @ w[s] / w[s].sum(), d['Ts'][:, n] @ w[n] / w[n].sum(),
                         lsg('up', d['vol_up'], b1), lsg('up', d['vol_up'], b2), lsg('mid', d['vol_mid'], b1),
                         lsg('mid', d['vol_mid'], b2), lsg('deep', d['vol_deep'], sall)])
    return d['years'], X

def window(lab, years, X):
    a, b = RUNS[lab].window
    if lab == '1232p5': b -= 1000
    if lab == '1242p5': a += 500
    k = (years >= a) & (years <= b) & np.isfinite(X).all(1); return X[k]

def lim(X, tau, scale, blocks=None):
    Z = (X - X.mean(0)) / scale
    if blocks is None: blocks = [np.arange(len(Z))]
    c0 = sum(Z[i[:-tau]].T @ Z[i[:-tau]] for i in blocks); ct = sum(Z[i[tau:]].T @ Z[i[:-tau]] for i in blocks)
    A = ct @ np.linalg.inv(c0); L = logm(A).real / tau
    return np.diag(scale) @ L @ np.diag(1 / scale)  # back to physical units

def G(L, t):
    return np.linalg.solve(L, expm(L * t) - np.eye(len(L)))

def fit_f(L, obs):
    """Least squares f on the forced components from (t, y per unit dmu, dmu); residuals are anomalies (y dmu) in
    units of the 1240 standard deviations, so that small steps do not dominate through their noise."""
    rows, rhs = [], []
    for t, y, dm in obs:
        g = G(L, t) * abs(dm) / SCALE[:, None]
        rows.append(g[:, FORCED] if not FREE_F else g); rhs.append(y * abs(dm) / SCALE)
    sol = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    f = np.zeros(len(NAMES)); f[FORCED if not FREE_F else slice(None)] = sol; return f

y40, X40 = boxes('1240'); W = window('1240', y40, X40); SCALE = W.std(0); x_eq = W.mean(0)
mu0 = run_mu('1240')
runs = {}
for lab in STEPS:
    yrs, X = boxes(lab); t = yrs - BRANCH - 0.5; dmu = run_mu(lab) - mu0
    runs[lab] = dict(t=t, Y=(X - x_eq) / dmu, eq=(window(lab, yrs, X).mean(0) - x_eq) / dmu, dmu=dmu)
obs = [(t, y, runs[lab]['dmu']) for lab in SMALL for t, y in zip(runs[lab]['t'], runs[lab]['Y']) if t < T_FIT and np.isfinite(y).all()]

def predict(L):
    f = fit_f(L, obs); return f, {T: G(L, T) @ f for T in (T_FIT, 100, 300, 1000, 3000)}, -np.linalg.solve(L, f)

L = lim(W, TAU, SCALE); ev = np.linalg.eigvals(L)
print(f'LIM 1240, tau = {TAU} yr, T_FIT = {T_FIT} yr, f {"free" if FREE_F else "on e, TsS, TsN, U1, U2 only"}')
print('eigenvalues (1/yr), slowest first:', ' '.join(f'{z.real:+.4f}{z.imag:+.3f}i' for z in sorted(ev, key=lambda z: -z.real)))
f, pred, sens = predict(L)
rng = np.random.default_rng(0); nb = 100; boot = []
for _ in range(50):
    starts = rng.integers(0, len(W) - nb, len(W) // nb); blocks = [np.arange(s, s + nb) for s in starts]
    try:
        Lb = lim(W, TAU, SCALE, blocks); boot.append(predict(Lb))
    except Exception:
        pass
def spread(fn):
    v = np.array([fn(b) for b in boot]); return np.percentile(v, [5, 95], axis=0)
print('fitted f (per W m^-2 per yr):', ' '.join(f'{n}={x:+.3g}' for n, x in zip(NAMES, f)))

def block(lab, a, b):
    r = runs[lab]; k = (r['t'] >= a) & (r['t'] < b); return np.nanmean(r['Y'][k], 0)
print('\nresponse per W m^-2 of solar constant (e in deg, T in K). observed = block means of annual values;')
print('prediction [5-95% over LIM bootstrap]')
for i, n in enumerate(NAMES):
    lo, hi = spread(lambda b: np.array([b[1][T][i] for T in (T_FIT, 100, 300, 1000, 3000)] + [b[2][i]]))
    print(f'\n{n}: pred t={T_FIT}: {pred[T_FIT][i]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]  100: {pred[100][i]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]'
          f'  300: {pred[300][i]:+.3f}  1000: {pred[1000][i]:+.3f} [{lo[3]:+.3f},{hi[3]:+.3f}]  3000: {pred[3000][i]:+.3f}'
          f'  equil: {sens[i]:+.3f} [{lo[5]:+.3f},{hi[5]:+.3f}]')
    for lab in STEPS:
        r = runs[lab]
        print(f'   obs {lab:12s} (dmu {r["dmu"]:+.2f}) {T_FIT-10}-{T_FIT}: {block(lab, T_FIT-10, T_FIT)[i]:+.3f}  80-120: {block(lab, 80, 120)[i]:+.3f}'
              f'  250-350: {block(lab, 250, 350)[i]:+.3f}  800-1200: {block(lab, 800, 1200)[i]:+.3f}  2500-3500: {block(lab, 2500, 3500)[i]:+.3f}  equil: {r["eq"][i]:+.3f}')
json.dump(dict(names=NAMES, f=f.tolist(), sens=sens.tolist(), eig=[[z.real, z.imag] for z in ev],
               eq={lab: runs[lab]['eq'].tolist() for lab in STEPS}),
          open(TMP + f'fdt_boxes_T{T_FIT}_tau{TAU}{"_free" if FREE_F else ""}.json', 'w'))
