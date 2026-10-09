import glob, numpy as np, xarray as xr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.ebm_stability import stability
from gsebm.linear_modes import eof_basis, fit_lim, pattern_cosine, fit_kdmd, dominant_mode_along
from gsebm.time import YEAR

def acf(v, lags):
    v = v - v.mean(); return np.array([v[:v.size - k] @ v[k:] / (v.size - k) for k in lags]) / v.var()

runs = (('1p0', 1.0), ('0p99', 0.99), ('0p98', 0.98), ('0p975', 0.975))
fig, ax = plt.subplots(len(runs), 2, figsize=(14, 2.4 * len(runs)), gridspec_kw={'width_ratios': [2, 1]})
guess = np.full(205, 300.0)
for row, (tag, mu) in enumerate(runs):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    w = np.diff(np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]])))
    r = st.forced_response
    print(f'\nmu={mu}: Jacobian s1={st.rates[0].real:.3f}/yr')
    for f in sorted(glob.glob(f'data/ebm_limit_cycle/limit_cycle_warm_mu{tag}_*.nc')):
        with xr.open_dataset(f, engine='scipy') as d:
            t = d.time.values; T = d.temperature.values; a = d.cycle_amplitude.values; th = d.cycle_phase.values
        dt = float(np.median(np.diff(t))); gm = T @ w / w.sum()
        ax[row, 0].plot(t / YEAR, gm, lw=0.3)
        keep = t >= 500 * YEAR
        if gm[keep].min() < 250:
            print(f'  {f.split("/")[-1]}: TIPPED (min global mean {gm.min():.1f} K), skipped'); continue
        T, a, th = T[keep], a[keep], th[keep]
        c, s = a * np.cos(th), a * np.sin(th)
        jumps = np.abs(np.diff(th)) > 2.0  # monthly phase advance is ~0.03 rad; slips are ±pi
        lags = np.arange(0, 1000 * 12 + 1, 12)
        ac = acf(c, lags); env = np.abs(acf(c, lags) + 1j * acf(s, lags) * 0)  # plain ACF of cos component
        ax[row, 1].plot(lags / 12, ac, lw=0.8)
        period = 2 * np.pi / np.median(np.diff(th)[~jumps] / (dt / YEAR))
        # ACF of the complex phase factor exp(i theta): its modulus gives the coherence
        z = np.exp(1j * th); coh = np.array([abs(np.mean(z[k:] * np.conj(z[:z.size - k]))) for k in lags])
        k300 = np.searchsorted(lags / 12, [100, 300, 600, 1000])
        print(f'  {f.split("/")[-1]}: global mean {gm[keep].mean():.2f}±{gm[keep].std():.2f} K; cycle period {period:.1f} yr; slips {jumps.sum()} in {(t[keep][-1]-t[keep][0])/YEAR:.0f} yr; phase coherence at 100/300/600/1000 yr {np.round(coh[k300], 2)}')
        b = eof_basis(T, w, 10); P = b.pcs
        scale = P[:, 0].std()
        cyc = np.column_stack([c, s]) / c.std() * scale
        for name, S in (('T only', P), ('T + cycle', np.column_stack([P, cyc]))):
            for lag in (6, 12):
                res = fit_lim(S, lag, dt); rates = res.rates * YEAR
                fields = res.field_patterns[:, :10] @ b.patterns
                cos = np.array([pattern_cosine(fl, r, w) for fl in fields])
                ok = np.where(cos >= 0.9)[0]
                sel = ok[np.argmax(cos[ok])] if ok.size else int(np.argmax(cos))
                slow = ', '.join(f'{z.real:.3f}{z.imag:+.3f}i' for z in rates[:4])
                print(f'    LIM {name:9s} lag{lag:2d}: aligned mode {rates[sel].real:.3f}{rates[sel].imag:+.3f}i (cos {cos[sel]:.2f}) | slowest: {slow}')
        A = T - T.mean(0)
        kd = fit_kdmd(A, w, 6, dt, 15000, seed=1, rel_threshold=1e-3)
        try:
            i, cs = dominant_mode_along(kd, r, w); msg = f'{kd.rates[i].real*YEAR:.3f}{kd.rates[i].imag*YEAR:+.3f}i (cos {cs:.2f})'
        except ValueError:
            msg = 'no mode with cos >= 0.9'
        slow = ', '.join(f'{z.real*YEAR:.3f}{z.imag*YEAR:+.3f}i' for z in kd.rates[:4])
        print(f'    KDMD T only lag 6: selected {msg} | slowest: {slow}', flush=True)
    ax[row, 0].set_ylabel(f'mu={mu}\nglobal mean T (K)'); ax[row, 1].set_ylabel('ACF of a cos(theta)')
ax[-1, 0].set_xlabel('year'); ax[-1, 1].set_xlabel('lag (yr)')
fig.suptitle('EBM + climate-coupled limit cycle with phase slips (monthly, no smoothing)'); fig.tight_layout()
fig.savefig('figures/ebm_benchmark/limit_cycle_overview.png', dpi=110)
