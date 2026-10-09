"""Check 5 loader: annual capped area (ice thickness ≥ 8.9 m), its equivalent latitude, and the
coupling gap G south of 20°S (check 6's definition), for selected year ranges of one run.

Usage: PYTHONPATH=src python check5_load.py <label> <first_year> <last_year> <out_dir>
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import run_archive
from load_annual import ROOT, read_blocks

R = 6.371e6
CAP = 8.9


def main(label, y0, y1, out):
    with h5py.File(run_archive(ROOT, label), "r") as f:
        years = np.asarray(f["year"][:], dtype=int)
        idx = np.flatnonzero((years >= y0) & (years <= y1))
        a, b = int(idx[0]), int(idx[-1]) + 1
        lat = np.asarray(f["t21_lat"][:], float)
        gw = np.asarray(f["t21_gaussian_weight"][:], float)
        ocean = np.asarray(f["lsm"][:], float) < 0.5
        cell_area = (gw / gw.sum() * 4 * np.pi * R**2 / 64)[:, None] * np.ones((1, 64))
        south = (lat < 0)[:, None] & ocean
        cap20 = (lat <= -20)[:, None] & ocean
        thick, bad = read_blocks(f["sea_ice_thickness"], a, b, lambda x: x.astype(np.float32))
        capped = (thick >= CAP) & ocean[None]
        cap_global = (capped * cell_area).sum(axis=(1, 2)) / 1e12
        cap_south = (capped * (south * cell_area)).sum(axis=(1, 2)) / 1e12
        fp = 0.0
        for name in ("rss", "rls", "hfss", "hfls"):
            v, m = read_blocks(f[name], a, b, lambda x: (x * (cap20 * cell_area)).sum(axis=(1, 2)))
            fp = fp + v; bad += m
        lsg_lat = np.asarray(f["lsg_lat"][:], float)
        wa = np.asarray(f["wet_surface_area"][:], float)
        capL = (lsg_lat <= -20) & (wa > 0)
        u, m = read_blocks(f["zonal_newtonian_coupling_heat_flux"], a, b,
                           lambda x: (np.nan_to_num(x) * wa)[:, capL].sum(axis=1))
        bad += m
        # equivalent latitude of the Southern capped area (rows filled from the pole)
        rows = np.flatnonzero(lat < 0); rows = rows[np.argsort(lat[rows])]
        oarea = (ocean * cell_area).sum(axis=1)[rows] / 1e12
        absl = np.abs(lat[rows]); mids = 0.5 * (absl[:-1] + absl[1:])
        pole, eq = np.concatenate([[90.0], mids]), np.concatenate([mids, [0.0]])
        cum = np.concatenate([[0.0], np.cumsum(oarea)])
        k = np.clip(np.searchsorted(cum, cap_south, side="right") - 1, 0, len(oarea) - 1)
        frac = np.clip((cap_south - cum[k]) / oarea[k], 0, 1)
        cap_eqlat = pole[k] + frac * (eq[k] - pole[k])
        np.savez_compressed(Path(out) / f"{label}_{y0}_{y1}.npz", years=years[a:b], cap_global=cap_global,
                            cap_south=cap_south, cap_eqlat=cap_eqlat, Fp=fp / 1e15, U=u / 1e15,
                            G=(fp - u) / 1e15)
    print(label, y0, y1, "bad", sorted(set(bad)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
