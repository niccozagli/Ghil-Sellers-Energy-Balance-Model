import numpy as np, xarray as xr
from dataclasses import replace
from scipy.optimize import root
from gsebm import default_model_parameters
from gsebm.ivp import build_ivp_operator
from gsebm.time import YEAR

d = xr.open_dataset('data/warm_cold_mu_bifurcation.nc', engine='scipy')
op0 = build_ivp_operator()
assert np.allclose(op0.x, d.latitude.values)
x = op0.x; w = op0.control_widths

def equilibrium(mu, guess):
    op = build_ivp_operator(params=replace(default_model_parameters(), mu=mu))
    f = lambda T: op.rhs(0, T) * YEAR
    sol = root(f, guess, method='hybr', tol=1e-12)
    return op, sol.x, np.max(np.abs(f(sol.x)))

def jacobian(op, T, h=1e-4):
    n = T.size; J = np.empty((n, n)); f0 = op.rhs(0, T)
    for k in range(n):
        e = T.copy(); e[k] += h
        J[:, k] = (op.rhs(0, e) - f0) / h
    return J * YEAR  # per year

guess = d.warm_state_temperature.sel(mu=1.0, method='nearest').isel(time=-1).values
mus = [1.0, 0.99, 0.98, 0.975, 0.972, 0.97, 0.968, 0.967, 0.966, 0.9655, 0.965, 0.9645, 0.964, 0.9635, 0.963]
prev = None
for mu in mus:
    op, T, res = equilibrium(mu, guess)
    J = jacobian(op, T)
    ev, V = np.linalg.eig(J)
    order = np.argsort(-ev.real); ev = ev[order]; V = V[:, order]
    Tm = (T * w).sum() / w.sum()
    print(f"mu={mu:.4f} resid={res:.1e} Tmean={Tm:.2f} T(0)={T[x.size//2]:.1f} T(pole)={T[-1]:.1f} "
          f"rates[1/yr]={np.round(ev[:4].real,4)} max|imag|={np.abs(ev.imag).max():.1e} tau1={-1/ev[0].real:.2f} yr")
    guess = T

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
out = Path('figures/ebm_benchmark'); out.mkdir(parents=True, exist_ok=True)
guess = d.warm_state_temperature.sel(mu=1.0, method='nearest').isel(time=-1).values
mus = [1.0, 0.99, 0.98, 0.975, 0.972, 0.97, 0.968, 0.966, 0.9655]
inner = lambda a, b: (a * b * w).sum()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
rows = []
for mu in mus:
    op, T, _ = equilibrium(mu, guess); guess = T
    _, Tp, _ = equilibrium(mu + 5e-4, T); _, Tm_, _ = equilibrium(mu - 2e-4 if mu < 0.966 else mu - 5e-4, T)
    dm = 5e-4 + (2e-4 if mu < 0.966 else 5e-4)
    r = (Tp - Tm_) / dm
    J = jacobian(op, T); ev, V = np.linalg.eig(J); o = np.argsort(-ev.real); ev = ev[o].real; V = V[:, o].real
    v1 = V[:, 0] * np.sign(V[x.size//2:, 0].sum())
    cos = inner(r, v1) / np.sqrt(inner(r, r) * inner(v1, v1))
    c = np.linalg.solve(V, r)  # r = sum c_k v_k
    share = np.sqrt(inner(c[0] * V[:, 0], c[0] * V[:, 0]) / inner(r, r))
    rows.append((mu, ev[0], ev[1], cos, share))
    print(f"mu={mu:.4f} s1={ev[0]:.4f} s2={ev[1]:.4f} cos(r,v1)={cos:.4f} |c1 v1|/|r|={share:.3f}")
    north = x >= 0
    lat = np.degrees(np.arcsin(x[north]))
    ax[1].plot(lat, v1[north] / np.abs(v1[north]).max(), label=f"{mu}")
    ax[2].plot(lat, r[north] / np.abs(r[north]).max(), label=f"{mu}")
rows = np.array(rows)
ax[0].plot(rows[:, 0], -rows[:, 1], 'o-', label='slowest mode, -s1')
ax[0].plot(rows[:, 0], -rows[:, 2], 's-', label='second mode, -s2')
ax[0].set_xlabel('mu'); ax[0].set_ylabel('decay rate (1/yr)'); ax[0].legend()
ax[0].set_title('Jacobian eigenvalues at warm equilibrium')
ax[1].set_title('slowest eigenvector (normalized)'); ax[1].set_xlabel('latitude')
ax[2].set_title('forced response dT*/dmu (normalized)'); ax[2].set_xlabel('latitude'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / 'ground_truth.png', dpi=120)
