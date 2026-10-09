import glob, numpy as np, xarray as xr
from gsebm.ebm_stability import stability
from gsebm.linear_modes import eof_basis, fit_lim, pattern_cosine
from gsebm.time import YEAR
OMEGA = 2 * np.pi / 20; GAMMA = OMEGA / 10

def lim_modes(state, lag, dt):
    """LIM on an arbitrary state (n, d); returns rates (1/yr) and eigenvectors (columns), slowest first."""
    res = fit_lim(state, lag, dt)
    return res.rates * YEAR, res.field_patterns  # field_patterns rows = eigenvectors in state space

guess = np.full(205, 300.0)
for tag, mu in (('1p0', 1.0), ('0p98', 0.98), ('0p972', 0.972)):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    r = st.forced_response; s1 = st.rates[0].real
    print(f'\nmu={mu}: Jacobian s1={s1:.3f}; oscillator {-GAMMA:.3f}±{OMEGA:.3f}i')
    for f in sorted(glob.glob(f'data/ebm_oscillator/oscillator_warm_mu{tag}_*.nc')):
        with xr.open_dataset(f, engine='scipy') as d:
            t = d.time.values; T = d.temperature.values; q = d.oscillator_q.values
        dt = float(np.median(np.diff(t))); keep = t >= 500 * YEAR; T = T[keep]; q = q[keep]
        b = eof_basis(T, w, 10); P = b.pcs
        qs = (q - q.mean()) / q.std() * P[:, 0].std()
        line = []
        # (a) T only; (b) T + q and q lagged by 3 months (stand-in for the oscillator's (q, p)); (c) T EOFs with delays 0, 6, 12 months
        L = 12
        states = {
            'T only': (P[L:], b.patterns),
            'T + q,q(t-3mo)': (np.column_stack([P[L:], qs[L:], qs[L - 3:-3]]), None),
            'T delays 0,6,12mo': (np.column_stack([P[L:], P[L - 6:-6], P[:-L]]), None),
        }
        for name, (S, pat) in states.items():
            for lag in (6, 12):
                res = fit_lim(S, lag, dt)
                rates = res.rates * YEAR
                # field pattern of each mode = its T part at delay 0, mapped back through the EOFs
                fields = res.field_patterns[:, :10] @ b.patterns
                cos = np.array([pattern_cosine(fld, r, w) for fld in fields])
                real = np.abs(rates.imag) < 1e-6
                cand = np.where(cos >= 0.9)[0]
                sel = cand[np.argmax(cos[cand])] if cand.size else int(np.argmax(cos))
                osc = np.argmin(abs(rates - (-GAMMA + 1j * OMEGA)))
                line.append(f'{name} lag{lag}: best-aligned {rates[sel].real:.3f}{rates[sel].imag:+.3f}i (cos {cos[sel]:.2f}); oscillator-like {rates[osc]:.3f}')
        print(' ', f.split('/')[-1]); [print('    ' + l) for l in line]
