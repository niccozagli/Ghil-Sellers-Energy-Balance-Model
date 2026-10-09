"""Cache plain annual zonal series of one PlaSim run (full valid record).

No smoothing, no detrending. Unreadable 10-year blocks become NaN.
Usage: PYTHONPATH=src python prototyping/plasim_fresh/load_annual.py <label> <out_dir>
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import run_archive, valid_record_range

ROOT = Path(__file__).resolve().parents[2] / "data" / "Plasim"
BLOCK = 10
# Sector longitudes on the T21 grid (degrees east): Atlantic 65°W–20°E,
# Indian 20–115°E, Pacific 115°E–65°W.
SECTORS = {
    "atlantic": lambda lon: (lon >= 295) | (lon < 20),
    "indian": lambda lon: (lon >= 20) & (lon < 115),
    "pacific": lambda lon: (lon >= 115) & (lon < 295),
}
LAYERS = ((0, 700), (700, 2025), (2025, 6000))


def read_blocks(dataset, a, b, reducer):
    out, bad = [], []
    for first in range(a, b, BLOCK):
        last = min(first + BLOCK, b)
        try:
            values = reducer(np.asarray(dataset[first:last], dtype=float))
        except OSError:
            values = None
            bad.append(first)
        out.append((first, last, values))
    shape = next(v.shape[1:] for _, _, v in out if v is not None)
    result = np.full((b - a,) + shape, np.nan)
    for first, last, values in out:
        if values is not None:
            result[first - a:last - a] = values
    return result, bad


def main(label: str, out_dir: Path) -> None:
    path = run_archive(ROOT, label)
    with h5py.File(path, "r") as f:
        years_all = np.asarray(f["year"][:], dtype=int)
        y0, y1 = valid_record_range(years_all)
        idx = np.flatnonzero((years_all >= y0) & (years_all <= y1))
        a, b = int(idx[0]), int(idx[-1]) + 1
        lat = np.asarray(f["t21_lat"][:], float)
        lon = np.asarray(f["t21_lon"][:], float)
        gw = np.asarray(f["t21_gaussian_weight"][:], float)
        ocean = np.asarray(f["lsm"][:], float) < 0.5
        masks = {"all": ocean}
        masks.update({k: ocean & fn(lon)[None, :] for k, fn in SECTORS.items()})
        names = list(masks)
        counts = np.stack([masks[k].sum(axis=1) for k in names], axis=-1)  # (32, 4)
        south = lat < 0

        def sector_mean(x):
            # x: (t, 32, 64) -> (t, 32, 4) mean over ocean cells of each sector
            return np.stack(
                [(x * masks[k]).sum(axis=2) / np.maximum(masks[k].sum(axis=1), 1) for k in names],
                axis=-1,
            )

        bad = []
        sic, m = read_blocks(f["sea_ice_concentration"], a, b, sector_mean); bad += m
        sic_map, m = read_blocks(f["sea_ice_concentration"], a, b,
                                 lambda x: x[:, south, :].astype(np.float16)); bad += m
        thick, m = read_blocks(f["sea_ice_thickness"], a, b, sector_mean); bad += m
        zonal = {}
        for name in ("surface_temperature", "rst", "rsut", "rlut"):
            zonal[name], m = read_blocks(f[name], a, b, lambda x: x.mean(axis=2)); bad += m
        wet_volume = np.asarray(f["wet_volume"][:], float)  # (68, 22)
        bounds = np.asarray(f["depth_bounds"][:], float)
        layer_w = []
        for top, bottom in LAYERS:
            sel = (bounds[:, 0] >= top) & (bounds[:, 1] <= bottom)
            w = wet_volume * sel[None, :]
            layer_w.append(w)
        layer_w = np.stack(layer_w, axis=-1)  # (68, 22, 3)

        def layers(x):
            # x: (t, 68, 22) -> (t, 68, 3) wet-volume-weighted layer means
            num = np.einsum("trk,rkl->trl", np.nan_to_num(x), layer_w)
            den = layer_w.sum(axis=1)
            return num / np.where(den > 0, den, np.nan)

        theta, m = read_blocks(f["zonal_potential_temperature"], a, b, layers); bad += m
        flux, m = read_blocks(f["zonal_newtonian_coupling_heat_flux"], a, b, lambda x: x); bad += m
        np.savez_compressed(
            out_dir / f"{label}.npz",
            years=years_all[a:b], lat=lat, lon=lon, gw=gw, ocean=ocean, sectors=np.array(names),
            ocean_counts=counts, sic=sic, sic_map_south=sic_map, thick=thick,
            ts=zonal["surface_temperature"], rst=zonal["rst"], rsut=zonal["rsut"],
            rlut=zonal["rlut"], theta_layers=theta, layer_bounds=np.array(LAYERS),
            layer_volume=layer_w.sum(axis=1), coupling_flux=flux,
            lsg_lat=np.asarray(f["lsg_lat"][:], float),
            lsg_area=np.asarray(f["wet_surface_area"][:], float),
            bad_blocks=np.array(sorted(set(int(years_all[k]) for k in bad))),
        )
    print(label, y0, y1, "bad blocks:", sorted(set(int(years_all[k]) for k in bad)))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
