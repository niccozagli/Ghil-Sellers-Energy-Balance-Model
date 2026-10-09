"""Cap energy budget against the ice-edge position, with a fixed cap boundary (|lat| >= 20 deg) for all runs.

Per run (window means), integrated over the cap (PW): ASR split into an insolation part and an albedo part relative
to the reference run 1245 (dASR = dI (1 - a_mean) - I_mean da, row by row), OLR, ocean heat delivered into the cap
(poleward ocean transport at 20 deg), atmospheric import (total import - ocean). Plotted as changes from 1245
against the edge latitude.
"""
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location('eo', 'prototyping/plasim_reduction/edge_ocean_lib.py')
eo = importlib.util.module_from_spec(spec); spec.loader.exec_module(eo)
E, run_mu = eo.E, eo.run_mu
REF = '1245'; PHIC = 20.0
order = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5', '1230', '1228p5']
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
table = []
for h, south in enumerate((True, False)):
    sgn = -1 if south else 1
    ref = E[REF]; cr = eo.cap_terms(ref, PHIC, south)
    edge = lambda e: e['eq'].south_edge if south else e['eq'].north_edge
    pts = []
    for lab in order:
        e = E[lab]; c = eo.cap_terms(e, PHIC, south); r, ar = c['rows'], c['area']
        a_e, a_r = 1 - e['asr'] / e['ins'], 1 - ref['asr'] / ref['ins']
        Ib, ab = 0.5 * (e['ins'] + ref['ins']), 0.5 * (a_e + a_r)
        PW = lambda f: float((f[r] * ar[r]).sum() / 1e15)
        d = dict(run=lab, mu=run_mu(lab), edge=edge(e), insolation=PW((e['ins'] - ref['ins']) * (1 - ab)), albedo=PW(-Ib * (a_e - a_r)),
                 olr=-PW(e['olr'] - ref['olr']), ocean=c['ocean_in'] - cr['ocean_in'], atm=c['atm_in'] - cr['atm_in'],
                 ocean_abs=c['ocean_in'], atm_abs=c['atm_in'])
        pts.append(d); table.append(dict(hemi='S' if south else 'N', **d))
    x = np.array([p['edge'] for p in pts]); cold = np.array([p['run'] in ('1230', '1228p5') for p in pts])
    for k, col, lab in (('albedo', '#2a6fb0', 'albedo term (absorbed sunlight)'), ('olr', '#c0612b', '−OLR'), ('atm', '#4a9a5b', 'atmospheric import'),
                        ('ocean', '#7a4fa0', 'ocean heat delivered'), ('insolation', '#8a8a8a', 'insolation (μ forcing)')):
        y = np.array([p[k] for p in pts])
        ax[h, 0].plot(x[~cold], y[~cold], 'o-', color=col, ms=4, label=lab); ax[h, 0].plot(x[cold], y[cold], 's', color=col, ms=5, mfc='none')
    for p in pts:
        if p['run'] in ('1312', '1265', '1245', '1232p5', '1230'): ax[h, 0].annotate(p['run'].replace('p', '.'), (p['edge'], p['albedo']), fontsize=7, xytext=(2, 4), textcoords='offset points')
    ax[h, 0].axhline(0, color='k', lw=0.4); ax[h, 0].invert_xaxis()
    ax[h, 0].set(xlabel='ice-edge latitude (° from equator; right = further equatorward)', ylabel=f'change in heat gained by cap |lat|>{PHIC:.0f}° vs 1245 (PW)',
                 title=('Southern' if south else 'Northern') + ' cap budget vs edge (open squares: cold states)')
    ax[h, 0].legend(fontsize=7)
    mu = np.array([p['mu'] for p in pts])
    ax[h, 1].plot(mu[~cold], x[~cold], 'o-', color='#2a6fb0'); ax[h, 1].plot(mu[cold], x[cold], 's', color='k', mfc='none')
    ax[h, 1].set(xlabel='μ (W m⁻²)', ylabel='ice-edge latitude (°)', title=('Southern' if south else 'Northern') + ' edge vs μ')
fig.tight_layout(); fig.savefig('figures/plasim_mechanism/cap_budget_vs_edge.png', dpi=110)
print(f"{'h':1s} {'run':12s} {'mu':>7s} {'edge':>5s} | {'insol':>6s} {'albedo':>7s} {'-OLR':>6s} {'atm':>6s} {'ocean':>6s} | ocean in {'':>2s} atm in")
for t in table:
    print(f"{t['hemi']} {t['run']:12s} {t['mu']:7.2f} {t['edge']:5.1f} | {t['insolation']:+6.2f} {t['albedo']:+7.2f} {t['olr']:+6.2f} {t['atm']:+6.2f} {t['ocean']:+6.2f} | {t['ocean_abs']:5.2f} {t['atm_abs']:6.2f}")
json.dump(table, open('/Users/niccolo/.claude/jobs/96e03936/tmp/cap_fixed.json', 'w'), default=float)
