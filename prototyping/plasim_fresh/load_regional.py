"""Annual regional ocean indices over a run's stationary period (area-weighted means of LSG layer maps).

Regions: EP = eastern subtropical Pacific (230–290°E, 10–35°S); EA = eastern South Atlantic (330–15°E, 10–35°S).
Usage: PYTHONPATH=src python load_regional.py <label> <out_dir>
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
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}


def main(label, out):
    a, b = WIN.get(label, (STAT[label]["0.95"], STAT[label]["end"]))
    with h5py.File(run_archive(ROOT, label), "r") as f:
        years = np.asarray(f["year"][:], dtype=int); idx = np.flatnonzero((years >= a) & (years <= b))
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        lat = f["lat"][:]; lon = np.mod(f["lon"][:], 360); area = f["lsg_horizontal_area"][:]
        regions = {"EP": (lat <= -10) & (lat >= -35) & (lon >= 230) & (lon <= 290),
                   "EA": (lat <= -10) & (lat >= -35) & ((lon >= 330) | (lon <= 15))}
        out_d = {}
        for fld, name in (("0_100", "theta_layer_0_100m"), ("300_600", "theta_layer_300_600m")):
            ds = f[name]
            for rk, m in regions.items():
                out_d[f"{rk}_{fld}"] = []
            ref = np.isfinite(np.asarray(ds[i0], dtype=float))
            for s in range(i0, i1, 10):
                n = min(s + 10, i1) - s
                try:
                    x = np.asarray(ds[s:s + n], dtype=float)
                except OSError:
                    x = np.full((n,) + ds.shape[1:], np.nan)
                for rk, m in regions.items():
                    w = area * m * ref
                    vals = np.nansum(np.nan_to_num(x) * w, axis=(1, 2)) / w.sum()
                    vals[~np.isfinite(x).any(axis=(1, 2))] = np.nan
                    out_d[f"{rk}_{fld}"].append(vals)
        np.savez(Path(out) / f"{label}.npz", years=years[i0:i1], **{k: np.concatenate(v) for k, v in out_d.items()})
    print(label, a, b)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
