import numpy as np, xarray as xr, glob, re, json
from gsebm.ebm_stability import stability
from gsebm.linear_modes import eof_basis, fit_lim, best_aligned_mode, pattern_cosine
from gsebm.time import YEAR

MUS = {'1': 1.0, '0p99': 0.99, '0p98': 0.98, '0p975': 0.975, '0p972': 0.972, '0p97': 0.97}
KEEP_097 = {1, 7, 8}  # realizations that stay warm (plain look)
CUT = 500.0  # years dropped as transient
KS, LAGS = (5, 10, 20), (1, 3, 6, 12, 24)
rows = []; guess = np.full(205, 300.0); truth = {}
for tag, mu in MUS.items():
    st = stability(mu, guess); guess = st.temperature
    north = st.x >= 0
    w = np.diff(np.concatenate(([st.x[0]], 0.5 * (st.x[1:] + st.x[:-1]), [st.x[-1]])))[north]
    r = st.forced_response[north]; v1 = st.modes[0].real[north]
    truth[tag] = float(st.rates[0].real)
    for f in sorted(glob.glob(f'data/new_stochastic_warm_mu{tag}_{{*}}.nc')):
        k_real = int(re.search(r'\{(\d+)\}', f).group(1))
        if tag == '0p97' and k_real not in KEEP_097: continue
        with xr.open_dataset(f, engine='scipy') as d:
            t = d.time.values; T = d.temperature.values[:, north]
        keep = t >= CUT * YEAR; T = T[keep]; dt = float(np.median(np.diff(t)))
        n = T.shape[0]
        for part, sl in (('full', slice(None)), ('first', slice(0, n // 2)), ('second', slice(n // 2, None))):
            for k in KS:
                b = eof_basis(T[sl], w, k)
                for lag in LAGS:
                    res = fit_lim(b.pcs, lag, dt, b.patterns)
                    i, c = best_aligned_mode(res, r, w)
                    rows.append(dict(mu=mu, real=k_real, part=part, k=k, lag=lag,
                                     rate=float(res.rates[i].real * YEAR), imag=float(res.rates[i].imag * YEAR),
                                     rank=i, cos_r=c, cos_v1=pattern_cosine(res.field_patterns[i], v1, w),
                                     slowest=float(res.rates[0].real * YEAR), var1=float(b.variance_fraction[0])))
        print(tag, k_real, 'done', flush=True)
json.dump(dict(rows=rows, truth=truth), open('/Users/niccolo/.claude/jobs/96e03936/tmp/lim_ebm.json', 'w'))
