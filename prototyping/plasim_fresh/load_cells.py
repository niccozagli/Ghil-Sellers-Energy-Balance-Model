"""Cache cell-level annual fields in the Southern edge rows (window years) and
sector-mean absorbed shortwave by row (full record) for one run.

Usage: PYTHONPATH=src python prototyping/plasim_fresh/load_cells.py <label> <out_dir>
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import RUNS, run_archive, valid_record_range
from load_annual import BLOCK, SECTORS, read_blocks

ROOT = Path(__file__).resolve().parents[2] / "data" / "Plasim"
ROW_LATS = (-19.38, -24.92, -30.46, -36.0, -41.53, -47.07)
FIELDS = ("sea_ice_concentration", "rst", "rss", "surface_temperature", "as", "sea_ice_thickness")


def main(label: str, out_dir: Path) -> None:
    with h5py.File(run_archive(ROOT, label), "r") as f:
        years = np.asarray(f["year"][:], dtype=int)
        y0, y1 = valid_record_range(years)
        idx = np.flatnonzero((years >= y0) & (years <= y1))
        a, b = int(idx[0]), int(idx[-1]) + 1
        lat = np.asarray(f["t21_lat"][:], float)
        lon = np.asarray(f["t21_lon"][:], float)
        rows = [int(np.argmin(np.abs(lat - r))) for r in ROW_LATS]
        names = list(SECTORS)
        smask = np.stack([SECTORS[k](lon) for k in names])  # (3, 64)

        def sector_rows(x):
            return np.einsum("tjl,sl->tjs", x, smask) / smask.sum(axis=1)

        rst_sector, bad = read_blocks(f["rst"], a, b, sector_rows)
        w0, w1 = RUNS[label].window
        win = np.flatnonzero((years >= w0) & (years <= w1))
        c, d = int(win[0]), int(win[-1]) + 1
        cells = {}
        for name in FIELDS:
            cells[name], m = read_blocks(f[name], c, d, lambda x: x[:, rows, :].astype(np.float32))
            bad += m
        np.savez_compressed(
            out_dir / f"{label}.npz", years=years[a:b], window_years=years[c:d], lat=lat, lon=lon,
            rows=np.array(rows), row_lat=lat[rows], ocean=np.asarray(f["lsm"][:], float)[rows] < 0.5,
            sectors=np.array(names), rst_sector=rst_sector, **{f"cell_{k}": v for k, v in cells.items()},
        )
    print(label, "bad blocks:", sorted(set(bad)))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
