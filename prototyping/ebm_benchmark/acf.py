import numpy as np, xarray as xr, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.ebm_stability import stability, operator_at, tendency_jacobian
from gsebm.time import YEAR
D = json.load(open('/Users/niccolo/.claude/jobs/96e03936/tmp/lim_ebm.json'))
import pandas as pd; df = pd.DataFrame(D['rows'])
cases = [('1', 1.0), ('0p98', 0.98), ('0p972', 0.972), ('0p97', 0.97)]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8)); guess = np.full(205, 300.0)
for ax, (tag, mu) in zip(axes, cases):
    st = stability(mu, guess); guess = st.temperature
    J = tendency_jacobian(operator_at(mu), st.temperature)
    ev, L = np.linalg.eig(J.T); u1 = L[:, np.argmax(ev.real)].real   # adjoint (left) eigenvector of the slow mode
    lim = df[(df.mu == mu) & (df.part == 'full') & (df.k == 10) & (df.lag == 12)].rate.median()
    lags = np.arange(0, 12 * 30 + 1)
    for real in (1, 7, 8) if tag == '0p97' else (1, 2, 3):
        with xr.open_dataset(f'data/new_stochastic_warm_mu{tag}_{{{real}}}.nc', engine='scipy') as d:
            t = d.time.values; T = d.temperature.values
        dt = float(np.median(np.diff(t))); T = T[t >= 500 * YEAR]
        p = (T - T.mean(0)) @ u1; p -= p.mean()
        acf = np.array([p[:p.size - k] @ p[k:] / (p.size - k) for k in lags]) / p.var()
        ax.plot(lags * dt / YEAR, acf, lw=1, label=f'data, realization {real}')
    years = lags * dt / YEAR
    ax.plot(years, np.exp(st.rates[0].real * years), 'k--', label=f'exp(s1 t), Jacobian s1={st.rates[0].real:.3f}')
    ax.plot(years, np.exp(lim * years), 'r:', label=f'exp(s t), LIM s={lim:.3f}')
    ax.set_yscale('log'); ax.set_ylim(0.01, 1.1); ax.set_xlabel('lag (yr)'); ax.set_title(f'mu={mu}')
    ax.legend(fontsize=6)
    k5 = np.searchsorted(years, [5, 10, 20])
    print(mu, 'ACF at 5/10/20 yr:', np.round(acf[k5], 3), 'Jacobian:', np.round(np.exp(st.rates[0].real * years[k5]), 3), 'LIM:', np.round(np.exp(lim * years[k5]), 3))
axes[0].set_ylabel('autocorrelation of projection on adjoint slow mode')
fig.tight_layout(); fig.savefig('figures/ebm_benchmark/slow_mode_acf.png', dpi=120)
