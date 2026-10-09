"""Cache 10-yr block means (aligned to the first valid year) of LSG zonal fields by row and level, full record.

Usage: PYTHONPATH=src python load_ocean10.py <label> <out_dir>
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import run_archive, valid_record_range
from load_annual import ROOT

FIELDS = ("zonal_potential_temperature", "zonal_salinity", "zonal_convective_adjustment",
          "zonal_vertical_velocity", "zonal_meridional_volume_transport")


def main(label, out):
    with h5py.File(run_archive(ROOT, label), "r") as f:
        years = np.asarray(f["year"][:], dtype=int)
        y0, y1 = valid_record_range(years)
        idx = np.flatnonzero((years >= y0) & (years <= y1))
        a, b = int(idx[0]), int(idx[-1]) + 1
        starts = list(range(a, b - 9, 10))
        out_d = {}
        for name in FIELDS:
            ds = f[name]
            vals = np.full((len(starts),) + ds.shape[1:], np.nan, dtype=np.float32)
            for i, s in enumerate(starts):
                try:
                    vals[i] = np.nanmean(np.asarray(ds[s:s + 10], dtype=float), axis=0)
                except OSError:
                    pass
            out_d[name.replace("zonal_", "")] = vals
        np.savez_compressed(
            Path(out) / f"{label}.npz", block_start=years[starts], **out_d,
            lsg_lat=np.asarray(f["lsg_lat"][:], float), lsg_vector_lat=np.asarray(f["lsg_vector_lat"][:], float),
            depth_bounds=np.asarray(f["depth_bounds"][:], float), wet_volume=np.asarray(f["wet_volume"][:], float),
            wet_vector_area=np.asarray(f["wet_vector_cross_section_area"][:], float),
            wet_surface_area=np.asarray(f["wet_surface_area"][:], float))
    print(label, "blocks", len(starts))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
