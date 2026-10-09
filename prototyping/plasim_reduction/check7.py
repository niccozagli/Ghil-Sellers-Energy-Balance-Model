"""Check 7: transition runs (annual values, full valid records).
Southern edge: zonal (ocean ice, 0.5) and Atlantic sector (65W-20E). Ocean northward transport per year from LSG rows:
OHT(face_j) = sum_{rows south of face} (U_k area_k - dOHC_k/dt), U = heat flux into the ocean, OHC = zonal_ocean_heat_content,
dOHC/dt centred difference (360-day years). Latitude of the Southern poleward-transport maximum (|lat| < 40),
ocean heat delivered into the cap poleward of 20S = poleward transport at 20S.
"""
import numpy as np, h5py, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import run_archive, valid_record_range, RUNS, ice_edge_latitude
from gsebm.plasim_transitions import sector_mask
ROOT = Path('data/Plasim'); YS = 360 * 86400.0; TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
def rd(ds, a, b, step=200):
    out = []
    for k in range(a, b, step):
        try: out.append(np.asarray(ds[k:min(k + step, b)], dtype=float))
        except OSError: out.append(np.full((min(k + step, b) - k,) + ds.shape[1:], np.nan))
    return np.concatenate(out)
res = {}
fig, ax = plt.subplots(3, 3, figsize=(17, 10), sharex='col')
for c, lab in enumerate(('1230', '1228p5', '1225')):
    with h5py.File(run_archive(ROOT, lab), 'r') as f:
        years = f['year'][:]; y0, y1 = valid_record_range(years); a = int(np.flatnonzero(years == y0)[0]); b = int(np.flatnonzero(years == y1)[0]) + 1
        lat = f['t21_lat'][:]; lon = f['t21_lon'][:]; ocean = f['lsm'][:] < 0.5
        sic = rd(f['sea_ice_concentration'], a, b)
        U = np.nan_to_num(rd(f['zonal_newtonian_coupling_heat_flux'], a, b)); H = rd(f['zonal_ocean_heat_content'], a, b)
        ll = f['lsg_lat'][:]; wa = f['wet_surface_area'][:]
    yrs = years[a:b]
    def edge_of(mask):
        m = ocean & mask[None, :]
        z = np.where(m.sum(1) > 0, (np.nan_to_num(sic) * m[None]).sum(2) / np.maximum(m.sum(1), 1), np.nan)
        return ice_edge_latitude(z, lat, south=True)
    eZ = edge_of(np.ones(lon.size, bool)); eA = edge_of(sector_mask(lon, 'atlantic'))
    o = np.argsort(ll); ll_, wa_ = ll[o], wa[o]
    dH = np.full_like(H, np.nan); dH[1:-1] = (H[2:] - H[:-2]) / 2 / YS
    flux = U[:, o] * wa_ - dH[:, o]
    oht = np.cumsum(flux, axis=1) / 1e15; faces = ll_ + np.diff(ll_).mean() / 2
    sh = (faces < 0) & (faces > -40)
    pole = -oht[:, sh]; fl = np.abs(faces[sh])
    lat_max = fl[np.nanargmax(np.where(np.isfinite(pole), pole, -np.inf), axis=1)]
    deliv20 = -np.array([np.interp(-20, faces, row) for row in oht])
    res[lab] = dict(years=yrs.tolist(), edge_zonal=eZ.tolist(), edge_atl=eA.tolist(), oht_max_lat=lat_max.tolist(), deliv20=deliv20.tolist())
    ax[0, c].plot(yrs, eZ, lw=0.6, label='S edge, zonal'); ax[0, c].plot(yrs, eA, lw=0.6, label='S edge, Atlantic sector')
    ax[0, c].plot(yrs, lat_max, '.', ms=1, color='k', label='latitude of max poleward ocean transport')
    ax[1, c].plot(yrs, deliv20, lw=0.5); ax[2, c].plot(yrs, np.nanmax(pole, axis=1), lw=0.5)
    ax[0, c].set_title(f'{lab.replace("p", ".")} ({RUNS[lab].initial_condition})'); ax[0, c].set_ylabel('|latitude| (°)'); ax[0, c].legend(fontsize=7)
    ax[1, c].set_ylabel('ocean heat into cap >20°S (PW)'); ax[2, c].set_ylabel('max poleward ocean transport (PW)'); ax[2, c].set_xlabel('year')
    # summaries
    def at(y): i = np.argmin(np.abs(yrs - y)); return i
    print(f'\n{lab}: valid {yrs[0]}–{yrs[-1]}')
    for yy in list(range(int(yrs[0]) + 10, int(yrs[-1]), max(1, (int(yrs[-1]) - int(yrs[0])) // 12))):
        i = at(yy); s = slice(max(0, i - 5), i + 5)
        print(f'   year {yy}: edge zonal {np.nanmean(eZ[s]):5.1f}, Atlantic {np.nanmean(eA[s]):5.1f}, OHT-max lat {np.nanmedian(lat_max[s]):5.1f}, max OHT {np.nanmean(np.nanmax(pole[s], axis=1)):.2f} PW, into cap >20S {np.nanmean(deliv20[s]):.2f} PW   (10-yr means for this printout only)')
fig.suptitle('Transition runs: annual values (no smoothing)'); fig.tight_layout(); fig.savefig('figures/plasim_mechanism/transition_edge_ocean.png', dpi=110)
json.dump(res, open(TMP + 'check7.json', 'w'))
