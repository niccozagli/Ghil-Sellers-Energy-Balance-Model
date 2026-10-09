import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('/Users/niccolo/.claude/jobs/96e03936/tmp/restoring.json'))
MU = {'1245': 1245, '1242p5': 1242.5, '1240': 1240, '1237p5': 1237.5, '1235': 1235, '1235_new_IC': 1235.25, '1233p75': 1233.75, '1232p5': 1232.5}
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
items = [('rad', 'radiative: OLR − albedo per K of T_S'), ('G', 'ocean: G = Σ g_k'), ('tot', 'total = radiative + ocean'),
         ('T_acf10', 'autocorrelation of T_S at 10 yr'), ('ice_acf10', 'autocorrelation of S ice area at 10 yr')]
for a, (k, title) in zip(ax, items):
    for lab, r in R.items():
        if k == 'tot':
            v = r['full']['rad'] + r['full']['G']
            e = np.hypot(*(np.diff(r['ci'][q])[0] / 2 for q in ('rad', 'G')))
        else:
            v = r['full'][k]; e = np.diff(r['ci'][k])[0] / 2
        col = '#9a9a9a' if lab in ('1235_new_IC', '1245') else '#2a6fb0'
        a.errorbar(MU[lab], v, yerr=e, fmt='o', color=col, capsize=3)
        for h in r['halves']:
            if k in h: a.plot(MU[lab], h[k], '_', color=col, ms=10)
        if k.endswith('acf10'): a.plot(MU[lab], r['nocycle'][k], 'x', color='#c0612b')
    a.set_title(title, fontsize=9); a.set_xlabel('μ (W m⁻²)'); a.axhline(0, color='k', lw=0.5)
ax[0].set_ylabel('W m⁻² K⁻¹ (positive = restoring)')
ax[3].plot([], [], 'x', color='#c0612b', label='cycle regressed out'); ax[3].plot([], [], '_', color='#2a6fb0', ms=10, label='halves'); ax[3].legend(fontsize=8)
fig.suptitle('Southern Hemisphere restoring vs μ (dots: full segment; bars: ± half the block-bootstrap 95% width; grey: 1245 flagged, 1235_new_IC check only)', fontsize=10)
fig.tight_layout(); fig.savefig('figures/plasim_mechanism/southern_restoring.png', dpi=120)
