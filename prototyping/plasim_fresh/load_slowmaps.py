"""Cache 100-yr block-mean maps over a run's stationary period (plain averaging, no other processing).

Usage: PYTHONPATH=src python load_slowmaps.py <label> <out_dir>
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import run_archive
from load_annual import ROOT

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json"))
FIELDS = {"sic": "sea_ice_concentration", "ts": "surface_temperature", "rst": "rst",
          "th0_100": "theta_layer_0_100m", "th300_600": "theta_layer_300_600m", "t1025_2000": "temperature_1025_2000m"}
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}


def main(label, out):
    a, b = WIN.get(label, (STAT[label]["0.95"], STAT[label]["end"]))
    with h5py.File(run_archive(ROOT, label), "r") as f:
        years = np.asarray(f["year"][:], dtype=int)
        idx = np.flatnonzero((years >= a) & (years <= b)); i0 = int(idx[0])
        L = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        nb = len(idx) // L
        res = {}
        for key, name in FIELDS.items():
            ds = f[name]; vals = []
            for k in range(nb):
                s = i0 + L * k
                try:
                    vals.append(np.nanmean(np.asarray(ds[s:s + L], dtype=float), axis=0).astype(np.float32))
                except OSError:
                    vals.append(np.full(ds.shape[1:], np.nan, dtype=np.float32))
            res[key] = np.stack(vals)
        np.savez_compressed(Path(out) / f"{label}.npz", block_start=years[i0 + L * np.arange(nb)], L=L, **res,
                            t21_lat=f["t21_lat"][:], t21_lon=f["t21_lon"][:], lsm=f["lsm"][:],
                            lsg_lat2d=f["lat"][:], lsg_lon2d=f["lon"][:])
    print(label, a, b, "blocks", nb)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
