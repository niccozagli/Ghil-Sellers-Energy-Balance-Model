"""Read annual state and physical observables of each warm-branch run (window only) into scratch npz."""
import sys, numpy as np, h5py
from pathlib import Path
from gsebm.plasim_global import RUNS, run_archive, read_global_series
from gsebm.plasim_koopman_single import load_fields

ROOT = Path('data/Plasim'); TMP = Path('/Users/niccolo/.claude/jobs/96e03936/tmp/states'); TMP.mkdir(exist_ok=True)
for label in sys.argv[1:]:
    out = TMP / f'{label}.npz'
    if out.exists(): print(label, 'exists'); continue
    a, b = RUNS[label].window
    s = read_global_series(ROOT, label)
    with h5py.File(run_archive(ROOT, label), 'r') as f:
        years = f['year'][:]; idx = np.flatnonzero((years >= a) & (years <= b)); i0, i1 = idx[0], idx[-1] + 1
        theta = np.concatenate([f['zonal_potential_temperature'][k:min(k + 200, i1)] for k in range(i0, i1, 200)])
        ohc = np.concatenate([f['zonal_ocean_heat_content'][k:min(k + 500, i1)] for k in range(i0, i1, 500)])
        bounds = f['depth_bounds'][:]; vol = f['wet_volume'][:]; area = f['wet_surface_area'][:]; lsg_lat = f['lsg_lat'][:]
    theta = np.where(vol[None] > 0, np.nan_to_num(theta), 0.0)  # dry cells are NaN and have zero volume
    up = np.flatnonzero(bounds[:, 1] <= 700.0); deep = np.flatnonzero(bounds[:, 0] >= 1000.0)
    wu = vol[:, up]; rows_ok = wu.sum(1) > 0
    theta_up_rows = np.where(rows_ok, (theta[:, :, up] * wu).sum(2) / np.maximum(wu.sum(1), 1e-30), np.nan)
    theta_up_global = (theta[:, :, up] * wu).sum((1, 2)) / wu.sum()
    wd = vol[:, deep]; theta_deep_global = (theta[:, :, deep] * wd).sum((1, 2)) / wd.sum()
    sa = load_fields(ROOT, a - 1, 700.0, label, end_year=b, detrend_ocean=False)
    assert np.array_equal(sa['years'], s.years)
    np.savez(out, years=s.years, lat=s.lat, weight=s.weight, Ts=s.surface_temperature, ice=s.ocean_ice,
             asr=s.absorbed_shortwave, ins=s.insolation, olr=s.outgoing_longwave, uptake=s.ocean_heat_uptake,
             lsg_lat=lsg_lat, lsg_area=area, theta_up_rows=theta_up_rows, theta_up_global=theta_up_global,
             theta_deep_global=theta_deep_global, ohc_global=ohc.sum(1),
             sa_ocean=sa['ocean_state'], sa_ocean_lat=sa['ocean_lat'], sa_ocean_w=sa['ocean_weights'],
             sa_surface=sa['surface_state'], sa_surface_lat=sa['surface_lat'], sa_surface_w=sa['surface_weights'],
             sa_ice=sa['ice_area'])
    print(label, 'saved', flush=True)
