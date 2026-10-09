"""Check 1: ocean heat delivered into the Southern cap; independent LSG transport proxy; Atlantic / Indo-Pacific split."""
import numpy as np, json
from gsebm.plasim_global import ocean_northward_transport, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
RUNS_ = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5', '1230', '1228p5']
RHO_CP = 1030.0 * 4180.0
res = {}
print(f"{'run':12s} | surface-flux poleward OHT at 15/20/25/30°S (PW) | proxy (annual-mean flow) at 15/20/25/30°S | Atl / IndoPac proxy at 20°S, 30°S | net vol 20°S (Sv)")
for lab in RUNS_:
    m = dict(np.load(TMP + f'maps/{lab}.npz'))
    lat = m['lsg_lat']; o = np.argsort(lat); area = m['wet_surface_area']
    up = np.nan_to_num(m['zonal_newtonian_coupling_heat_flux__full'])
    oht = ocean_northward_transport(up[o], area[o]); faces = lat[o] + np.diff(lat[o]).mean() / 2
    pole = lambda F, x, phi: -float(np.interp(-phi, x[np.argsort(x)], F[np.argsort(x)]))  # poleward = -northward in SH
    vl = m['lsg_vector_lat']; ov = np.argsort(vl)
    prox = RHO_CP * np.nansum(m['zonal_meridional_temperature_transport_proxy__full'], axis=1) / 1e15
    atl = RHO_CP * np.nansum(m['atlantic_temperature_transport_proxy__full'], axis=1) / 1e15
    ind = RHO_CP * np.nansum(m['indo_pacific_temperature_transport_proxy__full'], axis=1) / 1e15
    vol = m['zonal_net_meridional_volume_transport__full'] / 1e6
    sf = [pole(oht, faces, p) for p in (15, 20, 25, 30)]
    px = [pole(prox[ov], vl[ov], p) for p in (15, 20, 25, 30)]
    sec = [pole(atl[ov], vl[ov], 20), pole(ind[ov], vl[ov], 20), pole(atl[ov], vl[ov], 30), pole(ind[ov], vl[ov], 30)]
    nv = float(np.interp(-20, vl[ov], vol[ov]))
    res[lab] = dict(mu=run_mu(lab), sf=sf, proxy=px, sector=sec, netvol20=nv)
    print(f"{lab:12s} | " + ' '.join(f'{v:5.2f}' for v in sf) + ' | ' + ' '.join(f'{v:5.2f}' for v in px) + ' | ' + f'{sec[0]:5.2f} {sec[1]:5.2f}, {sec[2]:5.2f} {sec[3]:5.2f}' + f' | {nv:+.2f}')
warm = ['1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']
for key, name in (('sf', 'surface-flux'), ('proxy', 'proxy')):
    arr = np.array([res[l][key] for l in warm])
    print(f'{name}: range over 1265–1232.5 at 15/20/25/30°S:', ' '.join(f'{r:.3f}' for r in arr.max(0) - arr.min(0)),
          '| change 1245→1232.5:', ' '.join(f'{v:+.3f}' for v in np.array(res['1232p5'][key]) - np.array(res['1245'][key])))
json.dump(res, open(TMP + 'check1.json', 'w'))
