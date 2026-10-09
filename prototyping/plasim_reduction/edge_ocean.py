"""Analyses 1 and 2: the ice edge against ocean heat convergence, and the edge's energy margin.

Window means (RUNS windows) of every warm run, plus the cold states 1230 and 1228.5.
Analysis 1: per hemisphere, profiles of ocean heat release R = -(surface heat flux into the ocean) on LSG rows
(W m-2 of ocean), and poleward ocean / atmosphere / total heat transport (PW); ice edge marked.
  ocean transport  = plasim_global.ocean_northward_transport (cumulative uptake, steady state)
  total transport  = plasim_global.implied_northward_transport (cumulative ASR - OLR)
  atmosphere       = total - ocean (ocean interpolated to the T21 faces)
Analysis 2: cap budget between neighbouring runs. Cap = poleward of phi_c = (pair-mean edge) - 10 deg, fixed
within the pair. Terms integrated over the cap (PW): dASR = insolation term dI(1-mean a) + albedo term -mean(I) da,
dOLR, ocean release in the cap (= poleward ocean transport at phi_c), atmospheric import (= total import - ocean).
Each change divided by the edge shift (deg; negative = edge moved equatorward).
"""
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import read_global_series, equilibrium_state, time_mean, run_mu, EARTH_RADIUS_M

ROOT = Path('data/Plasim')
WARM = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']
EXTRA = ['1235_new_IC', '1230', '1228p5']
A = 4 * np.pi * EARTH_RADIUS_M**2
E = {}
for lab in WARM + EXTRA:
    s = read_global_series(ROOT, lab); eq = equilibrium_state(s)
    o = np.argsort(s.lat); oo = np.argsort(s.lsg_lat)
    E[lab] = dict(eq=eq, lat=s.lat[o], w=s.weight[o], ins=time_mean(s.insolation)[o], asr=time_mean(s.absorbed_shortwave)[o],
                  olr=time_mean(s.outgoing_longwave)[o], lsg_lat=s.lsg_lat[oo], area=s.lsg_row_area[oo],
                  uptake=np.nan_to_num(time_mean(s.ocean_heat_uptake))[oo])
    print(lab, 'S edge %.1f N edge %.1f' % (eq.south_edge, eq.north_edge), flush=True)

def faces_t21(w):  # latitudes of the north faces of the T21 rows (south to north), w normalized to sum 1
    return np.degrees(np.arcsin(np.clip(-1 + 2 * np.cumsum(w), -1, 1)))
def faces_lsg(lat):
    d = np.diff(lat).mean(); return lat + d / 2

def transports(e):
    tot = e['eq'].total_transport; ft = faces_t21(e['w'])
    oce = e['eq'].ocean_transport; fo = faces_lsg(e['lsg_lat'])
    oce_t = np.interp(ft, fo, oce)
    return ft, tot, oce_t, tot - oce_t, fo, oce

# ---------- analysis 1 ----------
cmap = plt.get_cmap('viridis'); labs = WARM + ['1230', '1228p5']
mus = np.array([run_mu(l) for l in labs]); norm = plt.Normalize(mus.min(), mus.max())
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
rows = []
for lab in labs:
    e = E[lab]; col = 'k' if lab in ('1230', '1228p5') else cmap(norm(run_mu(lab))); ls = '--' if lab in ('1230', '1228p5') else '-'
    rel = -e['uptake']; ft, tot, oce_t, atm, fo, oce = transports(e)
    for h, (sgn, edge) in enumerate(((-1, e['eq'].south_edge), (1, e['eq'].north_edge))):
        m = (np.sign(e['lsg_lat']) == sgn) & (e['area'] > 0)
        ax[h, 0].plot(np.abs(e['lsg_lat'][m]), rel[m], ls, color=col, lw=1)
        ax[h, 0].axvline(edge, color=col, ls=ls, lw=0.6)
        mf = np.sign(fo) == sgn
        ax[h, 1].plot(np.abs(fo[mf]), sgn * oce[mf], ls, color=col, lw=1); ax[h, 1].axvline(edge, color=col, ls=ls, lw=0.6)
        mt = np.sign(ft) == sgn
        ax[h, 2].plot(np.abs(ft[mt]), sgn * atm[mt], ls, color=col, lw=1); ax[h, 2].axvline(edge, color=col, ls=ls, lw=0.6)
        # summary numbers
        lr = np.abs(e['lsg_lat'][m]); rr = rel[m]
        band = (lr > 15) & (lr < 75)
        i_rel = np.argmax(np.where(band, rr, -np.inf)); fo_h = np.abs(fo[mf]); oh = sgn * oce[mf]
        i_oht = np.argmax(oh)
        rows.append(dict(run=lab, hemi='S' if sgn < 0 else 'N', mu=run_mu(lab), edge=edge, lat_max_release=lr[i_rel], max_release=rr[i_rel],
                         lat_max_oht=fo_h[i_oht], max_oht=oh[i_oht], oht_at_edge=float(np.interp(edge, fo_h[np.argsort(fo_h)], oh[np.argsort(fo_h)])),
                         atm_at_edge=float(np.interp(edge, np.abs(ft[mt])[np.argsort(np.abs(ft[mt]))], (sgn * atm[mt])[np.argsort(np.abs(ft[mt]))])),
                         release_at_edge=float(np.interp(edge, lr[np.argsort(lr)], rr[np.argsort(lr)]))))
for h, name in enumerate(('Southern Hemisphere', 'Northern Hemisphere')):
    ax[h, 0].set(title=f'{name[:1]}H ocean heat release (W m⁻² of ocean)', xlabel='|latitude|', xlim=(0, 80))
    ax[h, 1].set(title=f'{name[:1]}H poleward ocean transport (PW)', xlabel='|latitude|', xlim=(0, 80))
    ax[h, 2].set(title=f'{name[:1]}H poleward atmospheric transport (PW)', xlabel='|latitude|', xlim=(0, 90))
    for a in ax[h]: a.axhline(0, color='k', lw=0.4)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); fig.colorbar(sm, ax=ax, label='μ (W m⁻²); dashed black = cold states 1230, 1228.5; vertical lines = ice edge', shrink=0.8)
fig.savefig('figures/plasim_mechanism/edge_vs_ocean.png', dpi=110)
print('\nAnalysis 1 (window means):')
print(f"{'run':8s} {'h':1s} {'edge':>5s} | {'max release':>11s} at {'lat':>5s} | {'release@edge':>12s} | {'max OHT':>7s} at {'lat':>5s} | {'OHT@edge':>8s} | {'atm@edge':>8s}")
for r in rows:
    print(f"{r['run']:8s} {r['hemi']} {r['edge']:5.1f} | {r['max_release']:11.1f} at {r['lat_max_release']:5.1f} | {r['release_at_edge']:12.1f} | {r['max_oht']:7.2f} at {r['lat_max_oht']:5.1f} | {r['oht_at_edge']:8.2f} | {r['atm_at_edge']:8.2f}")

# ---------- analysis 2 ----------
def cap_terms(e, phic, south):
    sgn = -1 if south else 1
    rowsT = (np.sign(e['lat']) == sgn) & (np.abs(e['lat']) >= phic)
    area_T = A * e['w'] / e['w'].sum()
    ft, tot, oce_t, atm, fo, oce = transports(e)
    oce_c = float(np.interp(phic, np.sort(np.abs(fo[np.sign(fo) == sgn])), (sgn * oce[np.sign(fo) == sgn])[np.argsort(np.abs(fo[np.sign(fo) == sgn]))]))
    tot_c = float(np.interp(phic, np.sort(np.abs(ft[np.sign(ft) == sgn])), (sgn * tot[np.sign(ft) == sgn])[np.argsort(np.abs(ft[np.sign(ft) == sgn]))]))
    return dict(rows=rowsT, area=area_T, ins=e['ins'], asr=e['asr'], olr=e['olr'], ocean_in=oce_c / 1, total_in=tot_c, atm_in=tot_c - oce_c)

def pair(a, b, south=True):
    ea, eb = E[a], E[b]
    edge = lambda e: e['eq'].south_edge if south else e['eq'].north_edge
    phic = 0.5 * (edge(ea) + edge(eb)) - 10
    ca, cb = cap_terms(ea, phic, south), cap_terms(eb, phic, south)
    r, ar = ca['rows'], ca['area']
    alb_a, alb_b = 1 - ea['asr'] / ea['ins'], 1 - eb['asr'] / eb['ins']
    Ibar, abar = 0.5 * (ea['ins'] + eb['ins']), 0.5 * (alb_a + alb_b)
    PW = lambda f: float((f[r] * ar[r]).sum() / 1e15)
    d = dict(insolation=PW((eb['ins'] - ea['ins']) * (1 - abar)), albedo=PW(-Ibar * (alb_b - alb_a)), olr=-PW(eb['olr'] - ea['olr']),
             ocean=cb['ocean_in'] - ca['ocean_in'], atm=cb['atm_in'] - ca['atm_in'])
    d['residual'] = d['insolation'] + d['albedo'] + d['olr'] + d['ocean'] + d['atm']
    dedge = edge(eb) - edge(ea)
    return phic, dedge, run_mu(b) - run_mu(a), d, ca, cb

print('\nAnalysis 2: cap budget changes between neighbouring runs (PW per degree of edge shift; edge shift < 0 = equatorward)')
print('  signs: each term is the change in heat gained by the cap; OLR term = -(change in OLR)')
out2 = []
for south in (True, False):
    print(' Southern cap' if south else ' Northern cap')
    seq = WARM + ['1230']
    for a, b in zip(seq[:-1], seq[1:]):
        phic, dedge, dmu, d, ca, cb = pair(a, b, south)
        per = {k: v / dedge for k, v in d.items()} if abs(dedge) > 1e-3 else {k: np.nan for k in d}
        out2.append(dict(hemi='S' if south else 'N', a=a, b=b, phic=phic, dedge=dedge, dmu=dmu, **{f'd_{k}': v for k, v in d.items()}, **{f'per_{k}': v for k, v in per.items()},
                         cap_ocean=cb['ocean_in'], cap_atm=cb['atm_in']))
        print(f"   {a:>7s}→{b:<7s} cap>{phic:4.1f}° Δμ {dmu:+6.2f} Δedge {dedge:+5.2f}° | per degree: insol {per['insolation']:+.3f} albedo {per['albedo']:+.3f} "
              f"OLR {per['olr']:+.3f} ocean {per['ocean']:+.3f} atm {per['atm']:+.3f} (resid {per['residual']:+.3f}) | edge sens {dedge/dmu:+.2f} °/(W m⁻²) | cap ocean in {cb['ocean_in']:.2f} atm in {cb['atm_in']:.2f} PW")
json.dump(dict(a1=rows, a2=out2), open('/Users/niccolo/.claude/jobs/96e03936/tmp/edge_ocean.json', 'w'), default=float)
