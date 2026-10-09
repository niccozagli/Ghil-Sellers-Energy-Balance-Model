import glob, numpy as np, xarray as xr
from gsebm.ebm_stability import stability
from gsebm.linear_modes import fit_kdmd, eof_basis, pattern_cosine, weighted_inner
from gsebm.time import YEAR
guess = np.full(205, 300.0)
for tag, mu in (('1p0', 1.0), ('0p99', 0.99), ('0p98', 0.98), ('0p975', 0.975)):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    r = st.forced_response
    print(f'mu={mu}: Jacobian s1={st.rates[0].real:.3f}')
    for f in sorted(glob.glob(f'data/ebm_limit_cycle/limit_cycle_warm_mu{tag}_*.nc')):
        with xr.open_dataset(f, engine='scipy') as d:
            t = d.time.values; T = d.temperature.values; a = d.cycle_amplitude.values; th = d.cycle_phase.values
        dt = float(np.median(np.diff(t))); k = t >= 500 * YEAR
        T, a, th = T[k], a[k], th[k]; A = T - T.mean(0)
        pc1_std = eof_basis(T, w, 1).pcs[:, 0].std()
        c, s = a * np.cos(th), a * np.sin(th)
        cyc = np.column_stack([c - c.mean(), s - s.mean()]) / c.std() * pc1_std
        S = np.column_stack([A, cyc]); wS = np.concatenate([w, [1.0, 1.0]])
        for thr in (1e-3, 1e-4):
            kd = fit_kdmd(S, wS, 6, dt, 15000, seed=1, rel_threshold=thr)
            Tpart = kd.field_patterns[:, :205]
            cos = np.array([pattern_cosine(p, r, w) for p in Tpart])
            along = np.array([abs(weighted_inner(r, p, w)) for p in Tpart]) / np.sqrt(weighted_inner(r, r, w).real)
            score = np.where(cos >= 0.9, along**2 * kd.eigenfunction_variance, -np.inf)
            sel = int(np.argmax(score)) if np.isfinite(score).any() else None
            osc = np.argmin(abs(kd.rates * YEAR - (-0.007 + 1j * 2 * np.pi / {1.0: 20, 0.99: 22.2, 0.98: 25.3, 0.975: 27.6}[mu])))
            slow = ', '.join(f'{z.real*YEAR:.3f}{z.imag*YEAR:+.3f}i' for z in kd.rates[:5])
            msg = f'{kd.rates[sel].real*YEAR:.3f}{kd.rates[sel].imag*YEAR:+.3f}i (cos {cos[sel]:.2f})' if sel is not None else 'none with cos>=0.9'
            print(f'  {f.split("_")[-1][:-3]} thr={thr:g} rank={kd.rank}: selected {msg} | cycle-like {kd.rates[osc]*YEAR:.3f} | slowest: {slow}', flush=True)
