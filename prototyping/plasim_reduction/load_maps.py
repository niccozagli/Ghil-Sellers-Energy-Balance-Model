"""Window means (full, first half, second half) of map and zonal fields for the picture tests."""
import sys, numpy as np, h5py
from pathlib import Path
from gsebm.plasim_global import RUNS, run_archive
ROOT = Path('data/Plasim'); OUT = Path('/Users/niccolo/.claude/jobs/96e03936/tmp/maps')
MAPS = ['rst', 'rsut', 'rss', 'as', 'rlut', 'rls', 'hfss', 'hfls', 'sea_ice_concentration', 'surface_temperature']
ZONAL = ['zonal_sst_mismatch', 'zonal_ice_mismatch', 'zonal_ocean_heat_flux', 'zonal_newtonian_coupling_heat_flux',
         'zonal_meridional_temperature_transport_proxy', 'atlantic_temperature_transport_proxy', 'indo_pacific_temperature_transport_proxy',
         'zonal_net_meridional_volume_transport', 'atlantic_volume_transport', 'indo_pacific_volume_transport',
         'zonal_meridional_volume_transport', 'zonal_lsg_ice_covered_fraction', 'zonal_toa_energy_imbalance']
STATIC = ['lsm', 't21_lat', 't21_lon', 't21_gaussian_weight', 'lsg_lat', 'lsg_vector_lat', 'wet_surface_area', 'wet_vector_cross_section_area', 'depth_bounds']
for label in sys.argv[1:]:
    out = OUT / f'{label}.npz'
    if out.exists(): continue
    a, b = RUNS[label].window
    with h5py.File(run_archive(ROOT, label), 'r') as f:
        years = f['year'][:]; idx = np.flatnonzero((years >= a) & (years <= b)); i0, i1 = int(idx[0]), int(idx[-1]) + 1
        mid = i0 + (i1 - i0) // 2
        res = {k: f[k][:] for k in STATIC}
        for name in MAPS + ZONAL:
            ds = f[name]; acc = {p: np.zeros(ds.shape[1:]) for p in ('full', 'h1', 'h2')}; cnt = {p: 0 for p in acc}
            for k in range(i0, i1, 50):
                stop = min(k + 50, i1)
                try:
                    blk = np.asarray(ds[k:stop], dtype=float)
                except OSError:
                    continue
                blk = np.where(np.isfinite(blk), blk, np.nan)
                # split the block at the midpoint
                for p, lo, hi in (('full', k, stop), ('h1', k, min(stop, mid)), ('h2', max(k, mid), stop)):
                    if hi > lo:
                        part = blk[lo - k:hi - k]
                        acc[p] += np.nansum(part, axis=0); cnt[p] += part.shape[0]
            for p in acc: res[f'{name}__{p}'] = acc[p] / max(cnt[p], 1)
    np.savez(out, **res); print(label, 'saved', flush=True)
