import sys, time, json, numpy as np, xarray as xr
from gsebm.ebm_stability import stability
from gsebm.linear_modes import fit_kdmd, dominant_mode_along, pattern_cosine
from gsebm.time import YEAR
count = int(sys.argv[1]); out = []
guess = np.full(205, 300.0)
for tag, mu, reals in (('1', 1.0, (1, 2)), ('0p98', 0.98, (1, 2)), ('0p972', 0.972, (1, 2)), ('0p97', 0.97, (1, 7))):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    for real in reals:
        with xr.open_dataset(f'data/new_stochastic_warm_mu{tag}_{{{real}}}.nc', engine='scipy') as d:
            t = d.time.values; T = d.temperature.values
        dt = float(np.median(np.diff(t))); T = T[t >= 500 * YEAR]; A = T - T.mean(0)
        for lag in (2, 6, 12):
            for thr in (5e-3, 1e-3):
                t0 = time.time()
                k = fit_kdmd(A, w, lag, dt, count, seed=real, rel_threshold=thr)
                i, c = dominant_mode_along(k, st.forced_response, w)
                row = dict(mu=mu, real=real, N=count, lag=lag, thr=thr, rank=k.rank, index=i,
                           rate=k.rates[i].real * YEAR, imag=k.rates[i].imag * YEAR, cos_r=c,
                           cos_v1=pattern_cosine(k.field_patterns[i], st.modes[0].real, w),
                           truth=st.rates[0].real, secs=time.time() - t0)
                out.append(row); print(row, flush=True)
json.dump(out, open(f'/Users/niccolo/.claude/jobs/96e03936/tmp/kdmd_globe_{count}.json', 'w'))
