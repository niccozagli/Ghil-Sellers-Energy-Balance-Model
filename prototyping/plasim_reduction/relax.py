"""Sensitivity (A, B): relaxation of the Southern edge and of global ocean heat content after each run's mu step.

Annual values from the branch point. e(t): zonal Southern edge. H(t): global LSG heat content (sum of
zonal_ocean_heat_content, J). Parent equilibrium e0, H0 = parent window means; e_eq, H_eq = own window means.
r_e = (e - e_eq)/(e0 - e_eq), r_H likewise. Integral timescale tau = sum of r over the years before the window starts
(model-free; = e-folding time for an exponential). 0-D energy-balance prediction: tau_H = dH / dF, with dF the
global insolation term (dI (1 - mean alpha), map level) between parent and child equilibria (W).
"""
import numpy as np, h5py, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import run_archive, valid_record_range, RUNS, read_global_series, ice_edge_latitude, run_mu
ROOT = Path('data/Plasim'); TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'; R = 6.371e6; YS = 360 * 86400.0
CHILD = {'1265': ['1250', '1245', '1240', '1235'], '1240': ['1242p5', '1237p5', '1235_new_IC', '1233p75', '1232p5']}
def series(lab):
    s = read_global_series(ROOT, lab, full=True)
    e = ice_edge_latitude(s.ocean_ice, s.lat, south=True)
    with h5py.File(run_archive(ROOT, lab), 'r') as f:
        y = f['year'][:]; a, b = valid_record_range(y); i0 = int(np.flatnonzero(y == a)[0]); i1 = int(np.flatnonzero(y == b)[0]) + 1
        H = np.concatenate([np.nansum(np.asarray(f['zonal_ocean_heat_content'][k:min(k + 500, i1)], float), axis=1) for k in range(i0, i1, 500)])
    assert len(H) == len(e)
    return s.years, e, H
def eqmap(lab):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); w = m['t21_gaussian_weight']; nlon = m['t21_lon'].size
    area = (w / w.sum() * 4 * np.pi * R**2 / nlon)[:, None]
    I = m['rst__full'] - m['rsut__full']; return I, 1 - m['rst__full'] / I, area
cache = {}
def get(lab):
    if lab not in cache: cache[lab] = series(lab)
    return cache[lab]
rows = []
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for parent, kids in CHILD.items():
    yp, ep, Hp = get(parent); wa, wb = RUNS[parent].window; inw = (yp >= wa) & (yp <= wb)
    e0, H0 = np.nanmean(ep[inw]), np.nanmean(Hp[inw])
    Ip, ap, area = eqmap(parent)
    for lab in kids:
        y, e, H = get(lab); wa, wb = RUNS[lab].window; inw = (y >= wa) & (y <= wb); pre = y < wa
        eq, Hq = np.nanmean(e[inw]), np.nanmean(H[inw])
        re, rH = (e - eq) / (e0 - eq), (H - Hq) / (H0 - Hq)
        start_gap = int(y[0]) - (int(RUNS[lab].initial_condition.split(' at ')[1]) + 1)
        tau_e = float(np.nansum(re[pre])) + start_gap * 1.0   # missing first years counted as r = 1 (upper bound)
        tau_H = float(np.nansum(rH[pre])) + start_gap * 1.0
        Ic, ac, _ = eqmap(lab)
        dF = float(((Ic - Ip) * (1 - 0.5 * (ac + ap)) * area).sum())
        tau_pred = (Hq - H0) / dF / YS
        # tail check: r at the window start (should be ~0)
        rows.append(dict(run=lab, parent=parent, mu=run_mu(lab), e0=float(e0), eq=float(eq), step=float(eq - e0), tau_e=tau_e, tau_H=tau_H,
                         tau_pred=tau_pred, dH=float(Hq - H0), dF=dF / 1e15, missing_start=start_gap, pre_years=int(pre.sum()),
                         re_end=float(np.nanmean(re[pre][-100:])), rH_end=float(np.nanmean(rH[pre][-100:]))))
        t = y - y[0] + start_gap
        ax[0].plot(t[pre], re[pre], lw=0.4, label=f'{lab.replace("p", ".")} (from {parent})'); ax[1].plot(t[pre], rH[pre], lw=0.8)
for a, t_ in zip(ax, ('edge: (e − e_eq)/(e₀ − e_eq)', 'global ocean heat content: (H − H_eq)/(H₀ − H_eq)')):
    a.set(xscale='log', xlabel='years since the μ step', title=t_, ylim=(-1, 2)); a.axhline(0, color='k', lw=0.5); a.axhline(1, color='k', lw=0.3, ls=':')
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig('figures/plasim_mechanism/relaxation_after_step.png', dpi=110)
print(f"{'run':12s} {'from':5s} {'step°':>6s} | tau_edge  tau_H  tau_pred=dH/dF | dH (1e24 J) dF (PW) | r_e, r_H over last 100 pre-window yrs | missing first yrs")
for r in rows:
    print(f"{r['run']:12s} {r['parent']:5s} {r['step']:+6.2f} | {r['tau_e']:8.0f} {r['tau_H']:6.0f} {r['tau_pred']:8.0f} | {r['dH']/1e24:+8.2f} {r['dF']:+7.3f} | {r['re_end']:+.2f}, {r['rH_end']:+.2f} | {r['missing_start']}")
json.dump(rows, open(TMP + 'relax.json', 'w'))
