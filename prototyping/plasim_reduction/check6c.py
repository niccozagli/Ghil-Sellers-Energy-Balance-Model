"""Follow-up to check 6: zonal-mean PlaSim sea-ice thickness over Southern ocean cells (first 300 years of each window),
to see where the 9 m thickness limiter (icemod xmaxd = 9.0) is active."""
import h5py, numpy as np
from pathlib import Path
from gsebm.plasim_global import run_archive, RUNS
for lab in ('1265', '1245', '1232p5', '1230'):
    with h5py.File(run_archive(Path('data/Plasim'), lab), 'r') as f:
        y = f['year'][:]; a, b = RUNS[lab].window; i = np.flatnonzero((y >= a) & (y <= b))[:300]
        th = np.asarray(f['sea_ice_thickness'][i[0]:i[-1] + 1], float); lat = f['t21_lat'][:]; ocean = f['lsm'][:] < 0.5
    z = np.nanmean(np.where(ocean[None], th, np.nan), axis=(0, 2))
    print(lab, ' '.join(f'{la:.0f}:{v:.1f}' for la, v in zip(lat, z) if -70 < la < -25))
