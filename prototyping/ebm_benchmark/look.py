import numpy as np, xarray as xr, glob, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.ivp import build_ivp_operator
from gsebm.time import YEAR
op = build_ivp_operator(); w = op.control_widths; north = op.x >= 0
files = sorted(glob.glob('data/new_stochastic_warm_mu*_{*}.nc'))
mus = sorted({re.search(r'mu([0-9p]+)_', f).group(1) for f in files}, key=lambda s: -float(s.replace('p', '.')))
fig, axes = plt.subplots(len(mus), 1, figsize=(12, 2.2 * len(mus)), sharex=True)
store = {}
for ax, m in zip(axes, mus):
    for f in sorted(glob.glob(f'data/new_stochastic_warm_mu{m}_{{*}}.nc')):
        with xr.open_dataset(f, engine='scipy') as d:
            T = d.temperature.values[:, north]; t = d.time.values / YEAR
        Tm = T @ w[north] / w[north].sum()
        ax.plot(t, Tm, lw=0.3)
        k = int(re.search(r'\{(\d+)\}', f).group(1))
        print(m, k, f"len={t[-1]:.0f}yr n={t.size} min={Tm.min():.2f} first100={Tm[t<100].mean():.2f} 500-1000={Tm[(t>500)&(t<1000)].mean():.2f} last1000={Tm[t>4000].mean():.2f}")
    ax.set_ylabel(f'mu={m.replace("p",".")}\nNH mean T (K)')
axes[-1].set_xlabel('year'); fig.suptitle('Monthly NH area-mean temperature, all realizations (no smoothing)')
fig.tight_layout(); fig.savefig('figures/ebm_benchmark/stochastic_timeseries.png', dpi=110)
