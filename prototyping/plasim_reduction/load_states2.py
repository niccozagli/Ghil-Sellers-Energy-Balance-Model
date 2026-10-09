"""Annual zonal (all-longitude) potential temperature per LSG row for layers 0-700 m, 700-2000 m, >2000 m (wet-volume weighted)."""
import sys, numpy as np, h5py
from pathlib import Path
from gsebm.plasim_global import RUNS, run_archive
ROOT = Path('data/Plasim'); OUT = Path('/Users/niccolo/.claude/jobs/96e03936/tmp/states2')
for lab in sys.argv[1:]:
    out = OUT / f'{lab}.npz'
    if out.exists(): continue
    a, b = RUNS[lab].window
    with h5py.File(run_archive(ROOT, lab), 'r') as f:
        years = f['year'][:]; idx = np.flatnonzero((years >= a) & (years <= b)); i0, i1 = int(idx[0]), int(idx[-1]) + 1
        bounds = f['depth_bounds'][:]; vol = f['wet_volume'][:]; lat = f['lsg_lat'][:]
        layers = {'up': bounds[:, 1] <= 700.0, 'mid': (bounds[:, 0] >= 700.0) & (bounds[:, 1] <= 2000.0), 'deep': bounds[:, 0] >= 2000.0}
        res = {k: [] for k in layers}
        for k in range(i0, i1, 200):
            th = np.asarray(f['zonal_potential_temperature'][k:min(k + 200, i1)], float)
            th = np.where(vol[None] > 0, np.nan_to_num(th), 0.0)
            for name, m in layers.items():
                w = vol[:, m]; res[name].append(np.where(w.sum(1) > 0, (th[:, :, m] * w).sum(2) / np.maximum(w.sum(1), 1e-30), np.nan))
    np.savez(out, years=years[i0:i1], lat=lat, **{k: np.concatenate(v) for k, v in res.items()},
             vol_up=vol[:, layers['up']].sum(1), vol_mid=vol[:, layers['mid']].sum(1), vol_deep=vol[:, layers['deep']].sum(1))
    print(lab, 'saved', flush=True)
