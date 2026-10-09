import numpy as np, xarray as xr
from gsebm.ebm_stability import stability
from gsebm.linear_modes import fit_kdmd, fit_lim, eof_basis, pattern_cosine, weighted_inner
from gsebm.time import YEAR
PERIOD = {1.0: 20, 0.99: 22.2, 0.98: 25.3, 0.975: 27.6}
rng = np.random.default_rng(42)

def select(rates, Tpart, r, w, variance=None):
    cos = np.array([pattern_cosine(p, r, w) for p in Tpart])
    ok = cos >= 0.9
    if not ok.any(): return None, cos.max()
    if variance is None:
        i = int(np.argmax(np.where(ok, cos, -1)))
    else:
        along = np.array([abs(weighted_inner(r, p, w)) for p in Tpart]) / np.sqrt(weighted_inner(r, r, w).real)
        i = int(np.argmax(np.where(ok, along**2 * variance, -np.inf)))
    return i, cos[i]

fmt = lambda z: f'{z.real:.3f}{z.imag:+.3f}i'
guess = np.full(205, 300.0)
for tag, mu in (('1p0', 1.0), ('0p99', 0.99), ('0p98', 0.98), ('0p975', 0.975)):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    r = st.forced_response
    with xr.open_dataset(f'data/ebm_limit_cycle/limit_cycle_warm_mu{tag}_{{1}}.nc', engine='scipy') as d:
        t = d.time.values; T = d.temperature.values; a = d.cycle_amplitude.values; th = d.cycle_phase.values
    dt = float(np.median(np.diff(t)))
    c = a * np.cos(th)
    H = np.zeros_like(c); decay = np.exp(-dt / (10 * YEAR))
    for n in range(1, c.size): H[n] = H[n - 1] * decay + c[n - 1] * dt
    k = t >= 500 * YEAR; T, c, H = T[k], c[k], H[k]
    A = T - T.mean(0); b = eof_basis(T, w, 10); s0 = b.pcs[:, 0].std()
    z = lambda v: (v - v.mean()) / v.std()
    views = {
        'cos only': z(c),
        'cos + noise': z(z(c) + rng.standard_normal(c.size)),
        'reservoir H': z(H),
        'H + noise': z(z(H) + rng.standard_normal(H.size)),
    }
    target = -0.007 + 1j * 2 * np.pi / PERIOD[mu]
    print(f'mu={mu}: Jacobian s1={st.rates[0].real:.3f}; cycle ≈ {fmt(target)}; corr(H, cos) at lag 0 = {np.corrcoef(c, H)[0,1]:.2f}')
    for name, v in views.items():
        extra = (v * s0)[:, None]
        lim = fit_lim(np.column_stack([b.pcs, extra]), 6, dt); lr = lim.rates * YEAR
        i, cl = select(lr, lim.field_patterns[:, :10] @ b.patterns, r, w)
        kd = fit_kdmd(np.column_stack([A, extra]), np.concatenate([w, [1.0]]), 6, dt, 15000, seed=1, rel_threshold=1e-3)
        kr = kd.rates * YEAR
        j, ck = select(kr, kd.field_patterns[:, :205], r, w, kd.eigenfunction_variance)
        lsel = f'{fmt(lr[i])} (cos {cl:.2f})' if i is not None else f'none (max cos {cl:.2f})'
        ksel = f'{fmt(kr[j])} (cos {ck:.2f})' if j is not None else f'none (max cos {ck:.2f})'
        print(f'  {name:12s} LIM slow: {lsel:28s} cycle-like {fmt(lr[np.argmin(abs(lr - target))])} | '
              f'KDMD slow: {ksel:28s} cycle-like {fmt(kr[np.argmin(abs(kr - target))])}', flush=True)
