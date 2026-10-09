import numpy as np, xarray as xr
from gsebm.ebm_stability import stability
from gsebm.linear_modes import eof_basis, fit_lim, best_aligned_mode, fit_kdmd, dominant_mode_along
from gsebm.time import YEAR
guess = np.full(205, 300.0)
for tag, mu, reals in (('1', 1.0, (1, 2, 3)), ('0p98', 0.98, (1, 2, 3)), ('0p97', 0.97, (1, 7, 8))):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    par = [np.sum(w * m.real * m.real[::-1]) / np.sum(w * m.real**2) for m in st.modes[:4]]
    print(f'mu={mu} Jacobian rates {np.round(st.rates[:4].real, 3)} parity (+1 sym, -1 antisym) {np.round(par, 2)}')
    for real in reals:
        with xr.open_dataset(f'data/new_stochastic_warm_mu{tag}_{{{real}}}.nc', engine='scipy') as d:
            t = d.time.values; T = d.temperature.values
        dt = float(np.median(np.diff(t))); T = T[t >= 500 * YEAR]
        out = []
        for name, sel in (('NH', x >= 0), ('globe', slice(None))):
            b = eof_basis(T[:, sel], w[sel], 10)
            for lag in (1, 12):
                res = fit_lim(b.pcs, lag, dt, b.patterns)
                i, c = best_aligned_mode(res, st.forced_response[sel], w[sel])
                out.append(f'{name} LIM lag{lag}: {res.rates[i].real*YEAR:.3f} (cos {c:.3f})')
        A = T - T.mean(0)
        k = fit_kdmd(A, w, 2, dt, 5000, seed=0, rel_threshold=5e-3)
        i, c = dominant_mode_along(k, st.forced_response, w)
        out.append(f'globe KDMD lag2: {k.rates[i].real*YEAR:.3f} (cos {c:.3f})')
        print(f'  realization {real}: ' + ' | '.join(out), flush=True)
