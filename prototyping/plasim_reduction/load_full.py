"""Full-record annual series for the response-function tests (all valid years, including the adjustment after the mu step).

Per run, saved to scratch `full/<label>.npz`:
* Southern ice edge (latitude where ocean-only sea-ice concentration first reaches 0.5), zonal and per sector
  (Atlantic 65W-20E, Indian 20-115E, Pacific 115E-65W), on T21 rows, `ice_edge_latitude`.
* Zonal-mean T21 fields: surface temperature, ASR (rst), insolation (rst - rsut), OLR (rlut).
* LSG zonal potential temperature averaged (wet-volume weighted) over 0-700 m, 700-2000 m, > 2000 m per LSG row,
  and the row volumes, so that heat content of any latitude box is rho c_p sum(theta * vol).
"""
import sys, numpy as np, h5py
from pathlib import Path
from gsebm.plasim_global import run_archive, valid_record_range, ice_edge_latitude

ROOT = Path('data/Plasim'); OUT = Path('/Users/niccolo/.claude/jobs/96e03936/tmp/full'); OUT.mkdir(exist_ok=True)
SECTORS = {'atl': lambda lon: (lon >= 295) | (lon < 20), 'ind': lambda lon: (lon >= 20) & (lon < 115),
           'pac': lambda lon: (lon >= 115) & (lon < 295)}
for lab in sys.argv[1:]:
    out = OUT / f'{lab}.npz'
    if out.exists(): print(lab, 'exists'); continue
    with h5py.File(run_archive(ROOT, lab), 'r') as f:
        years = f['year'][:]; a, b = valid_record_range(years)
        i0 = int(np.flatnonzero(years == a)[0]); i1 = int(np.flatnonzero(years == b)[0]) + 1
        lat = f['t21_lat'][:]; lon = f['t21_lon'][:] % 360; ocean = f['lsm'][:] < 0.5
        bounds = f['depth_bounds'][:]; vol = f['wet_volume'][:]
        layers = {'up': bounds[:, 1] <= 700.0, 'mid': (bounds[:, 0] >= 700.0) & (bounds[:, 1] <= 2000.0), 'deep': bounds[:, 0] >= 2000.0}
        res = {k: [] for k in ('ice_zonal', 'ice_atl', 'ice_ind', 'ice_pac', 'Ts', 'asr', 'ins', 'olr', 'up', 'mid', 'deep')}
        bad = []
        for k in range(i0, i1, 100):
            s = slice(k, min(k + 100, i1)); n = s.stop - s.start
            blk = {}
            try:
                sic = np.asarray(f['sea_ice_concentration'][s], float)
                for name, m in [('zonal', np.ones(lon.size, bool))] + [(q, fn(lon)) for q, fn in SECTORS.items()]:
                    cells = ocean & m[None, :]
                    blk[f'ice_{name}'] = (np.where(cells.sum(1) > 0, (sic * cells).sum(2) / np.maximum(cells.sum(1), 1), np.nan))
                blk['Ts'] = (np.asarray(f['surface_temperature'][s], float).mean(2))
                rst = np.asarray(f['rst'][s], float).mean(2); rsut = np.asarray(f['rsut'][s], float).mean(2)
                blk['asr'] = rst; blk['ins'] = rst - rsut; blk['olr'] = -np.asarray(f['rlut'][s], float).mean(2)
                th = np.asarray(f['zonal_potential_temperature'][s], float); th = np.where(vol[None] > 0, np.nan_to_num(th), 0.0)
                for name, m in layers.items():
                    w = vol[:, m]; blk[name] = (np.where(w.sum(1) > 0, (th[:, :, m] * w).sum(2) / np.maximum(w.sum(1), 1e-30), np.nan))
            except OSError:
                bad.append(int(years[k]))
                for key in res:
                    shape = (n, 68) if key in ('up', 'mid', 'deep') else (n, lat.size)
                    blk[key] = np.full(shape, np.nan)
            for key in res:
                res[key].append(blk[key])
        lsg_lat = f['lsg_lat'][:]
    arr = {k: np.concatenate(v) for k, v in res.items()}
    edges = {f'edge_{q}': ice_edge_latitude(np.nan_to_num(arr[f'ice_{q}']), lat, south=True) for q in ('zonal', 'atl', 'ind', 'pac')}
    for q in edges:
        edges[q][~np.isfinite(arr['Ts'][:, 0])] = np.nan
    np.savez(out, years=years[i0:i1], lat=lat, lsg_lat=lsg_lat, bad=np.array(bad),
             vol_up=vol[:, layers['up']].sum(1), vol_mid=vol[:, layers['mid']].sum(1), vol_deep=vol[:, layers['deep']].sum(1),
             **edges, **{k: v.astype(np.float32) for k, v in arr.items()})
    print(lab, 'saved', len(years[i0:i1]), 'yr; unreadable blocks', bad, flush=True)
