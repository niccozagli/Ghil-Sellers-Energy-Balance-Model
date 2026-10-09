"""Follow-up to check 6: cumulative (from the South Pole) PlaSim surface flux over ocean vs LSG received flux,
evaluated at the T21 row faces; gap(phi) = cumulative difference. Window means."""
import numpy as np
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'; R = 6.371e6
for lab in ('1265', '1245', '1232p5', '1230'):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: m[f'{k}__full']
    lat, lon, w = m['t21_lat'], m['t21_lon'], m['t21_gaussian_weight']; ocean = m['lsm'] < 0.5
    o = np.argsort(lat); lat, w = lat[o], w[o]; ocean = ocean[o]
    area = (w / w.sum() * 4 * np.pi * R**2 / lon.size)[:, None] * np.ones((1, lon.size))
    Fp = (((g('rss') + g('rls') + g('hfss') + g('hfls'))[o]) * area * ocean).sum(1) / 1e15
    sic = np.nan_to_num(g('sea_ice_concentration'))[o]; icef = (sic * ocean * area).sum(1) / np.maximum((ocean * area).sum(1), 1)
    faces = np.degrees(np.arcsin(-1 + np.cumsum(w / w.sum() * 2)))
    CP = np.cumsum(Fp)
    ll, wa = m['lsg_lat'], m['wet_surface_area']; ol = np.argsort(ll); ll, wa = ll[ol], wa[ol]
    U = (np.nan_to_num(g('zonal_newtonian_coupling_heat_flux'))[ol] * wa) / 1e15
    lf = ll + np.diff(ll).mean() / 2; CL = np.interp(faces, lf, np.cumsum(U))
    sel = (faces < -10) & (faces > -75)
    print(lab + ': face lat | cumulative PlaSim | cumulative LSG | gap | ice of row south of face')
    for f, a, b, ic in zip(faces[sel], CP[sel], CL[sel], icef[:-1][sel[:-1]] if False else icef[sel]):
        print(f'   {f:6.1f} | {a:+.3f} | {b:+.3f} | {a-b:+.3f} | {ic:.2f}')
