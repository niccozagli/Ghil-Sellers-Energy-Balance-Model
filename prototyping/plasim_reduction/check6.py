"""Check 6: energy closure at the edge. Window means.
(a) zonal_sst_mismatch (LSG - PlaSim SST, K) and zonal_ice_mismatch (LSG - PlaSim ice, fraction) at LSG rows within +-5 deg of
    the S edge, and their cap means.
(b) PlaSim net surface flux over ocean cells F_p = rss + rls + hfss + hfls (downward positive; W m-2 of ocean), versus
    LSG's received flux U (zonal_newtonian_coupling_heat_flux and zonal_ocean_heat_flux, W m-2 of wet area) on the same
    latitudes. In an annual-mean steady state with local thermodynamic ice, F_p + (heat from LSG) ~ 0, i.e. F_p ~ U.
    Cap (|lat| >= 20 S) integrals of F_p (T21 ocean cells) and U (LSG wet rows), and their difference, in PW.
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'; R = 6.371e6
RUNS_ = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5', '1230', '1228p5']
rows = []
print(f"{'run':12s} {'edge':>5s} | sst mism. edge±5 / cap (K) | ice mism. edge±5 / cap | cap PlaSim surface flux / LSG newtonian / LSG ocean_heat_flux (PW) | difference P-N (PW)")
for lab in RUNS_:
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: m[f'{k}__full']
    lat, lon, w = m['t21_lat'], m['t21_lon'], m['t21_gaussian_weight']; ocean = m['lsm'] < 0.5
    area = (w / w.sum() * 4 * np.pi * R**2 / lon.size)[:, None] * np.ones((1, lon.size))
    sic = np.nan_to_num(g('sea_ice_concentration'))
    ice = np.where(ocean.sum(1) > 0, (sic * ocean).sum(1) / np.maximum(ocean.sum(1), 1), np.nan)
    edge = float(ice_edge_latitude(ice[None], lat, south=True)[0])
    Fp = g('rss') + g('rls') + g('hfss') + g('hfls')
    capT = (lat <= -20)[:, None] & ocean
    P = float((Fp * area)[capT].sum() / 1e15)
    ll, wa = m['lsg_lat'], m['wet_surface_area']; capL = (ll <= -20) & (wa > 0)
    Nw = float((np.nan_to_num(g('zonal_newtonian_coupling_heat_flux')) * wa)[capL].sum() / 1e15)
    Oh = float((np.nan_to_num(g('zonal_ocean_heat_flux')) * wa)[capL].sum() / 1e15)
    near = (np.abs(np.abs(ll) - edge) <= 5) & (ll < 0) & (wa > 0)
    sm, im = np.nan_to_num(g('zonal_sst_mismatch')), np.nan_to_num(g('zonal_ice_mismatch'))
    wm = lambda f, msk: float((f * wa)[msk].sum() / wa[msk].sum())
    r = dict(run=lab, mu=run_mu(lab), edge=edge, sst_edge=wm(sm, near), sst_cap=wm(sm, capL), ice_edge=wm(im, near), ice_cap=wm(im, capL),
             plasim=P, newtonian=Nw, ocean_flux=Oh, diff=P - Nw, ocean_area_T=float(area[capT].sum()), wet_area_L=float(wa[capL].sum()))
    rows.append(r)
    print(f"{lab:12s} {edge:5.1f} | {r['sst_edge']:+.3f} / {r['sst_cap']:+.3f} | {r['ice_edge']:+.4f} / {r['ice_cap']:+.4f} | {P:+.3f} / {Nw:+.3f} / {Oh:+.3f} | {P - Nw:+.3f}")
by = {r['run']: r for r in rows}
for a, b in (('1265', '1245'), ('1245', '1232p5')):
    de = by[b]['edge'] - by[a]['edge']
    print(f"{a}→{b}: change of (PlaSim - LSG) cap difference per degree of edge: {(by[b]['diff'] - by[a]['diff'])/de:+.4f} PW/deg; sst mismatch at edge change {by[b]['sst_edge']-by[a]['sst_edge']:+.3f} K")
print('areas: T21 cap ocean %.3e m2, LSG cap wet %.3e m2' % (rows[0]['ocean_area_T'], rows[0]['wet_area_L']))
json.dump(rows, open(TMP + 'check6.json', 'w'))
