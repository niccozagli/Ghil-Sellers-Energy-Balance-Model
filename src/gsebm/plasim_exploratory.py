"""In-memory annual diagnostics derived directly from PlaSim raw maps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr


ICE_THRESHOLD = 0.5
OCEAN_BOX_BANDS = (
    ("0_100m", 0.0, 100.0),
    ("100_312p5m", 100.0, 312.5),
    ("312p5_700m", 312.5, 700.0),
    ("700_1025m", 700.0, 1025.0),
    ("1025_2000m", 1025.0, 2000.0),
    ("2000_6000m", 2000.0, 6000.0),
)
EARTH_RADIUS_M = 6.371e6


def mask_path(archive: Path) -> Path:
    """Return the companion geographic mask for a raw-map archive."""
    if not archive.name.endswith("_raw_maps.nc"):
        raise ValueError(f"Not a raw-map archive: {archive}")
    return archive.with_name(archive.name.replace("_raw_maps.nc", "_basin_masks.nc"))


def compute_diagnostics(archive: Path) -> xr.Dataset:
    """Return annual diagnostics without creating another data product.

    Results are cached for the notebook process and recomputed if either input
    file changes. Only one 50-year block of maps is held during reduction.
    """
    archive = Path(archive).resolve()
    masks = mask_path(archive)
    if not masks.exists():
        raise FileNotFoundError(f"Missing basin masks: {masks}")
    archive_stat = archive.stat()
    mask_stat = masks.stat()
    return _compute_cached(
        str(archive), archive_stat.st_size, archive_stat.st_mtime_ns,
        mask_stat.st_size, mask_stat.st_mtime_ns,
    )


@lru_cache(maxsize=None)
def _compute_cached(
    archive_name: str,
    archive_size: int,
    archive_mtime_ns: int,
    mask_size: int,
    mask_mtime_ns: int,
) -> xr.Dataset:
    # File identity values are part of the cache key; source reads use the path.
    del archive_size, archive_mtime_ns, mask_size, mask_mtime_ns
    archive = Path(archive_name)
    masks = mask_path(archive)
    with h5netcdf.File(archive, "r") as source, h5netcdf.File(masks, "r") as basins:
        years = np.asarray(source.variables["year"][:], dtype=np.int32)
        if years.size == 0 or not np.all(np.diff(years) == 1):
            raise ValueError(f"Missing or nonconsecutive annual records in {archive}")
        lat = np.asarray(source.variables["t21_lat"][:], dtype=float)
        lsm = np.asarray(source.variables["lsm"][:], dtype=float)
        row_lat = np.asarray(source.variables["lat"][:], dtype=float).mean(axis=1)
        depths = np.asarray(source.variables["upper_depth"][:], dtype=float)
        depth_bounds = np.asarray(source.variables["depth_bounds"][:len(depths)], dtype=float)
        volume = np.asarray(source.variables["wet_cell_volume"][:len(depths)], dtype=float)
        sector = np.asarray(basins.variables["lsg_scalar_south_atlantic"][:], dtype=bool)
        sector_box = sector & (row_lat[:, None] >= -60) & (row_lat[:, None] < 0)
        surface_sector = (
            np.asarray(basins.variables["t21_south_atlantic"][:], dtype=bool)
            & (lat[:, None] >= -60)
        )
        wet_area = np.asarray(source.variables["wet_surface_area"][:], dtype=float)
        flux_lat = np.asarray(source.variables["lsg_lat"][:], dtype=float)

        nodes, latitude_weights = np.polynomial.legendre.leggauss(len(lat))
        if not np.allclose(lat, np.degrees(np.arcsin(nodes))[::-1], atol=1e-4):
            raise ValueError("T21 latitudes do not match the expected Gaussian grid.")
        area = latitude_weights[::-1, None] * np.ones_like(lsm)
        ocean = lsm < 0.5
        surface_sector &= ocean
        if not surface_sector.any():
            raise ValueError("South Atlantic T21 surface box has no ocean cells.")
        north = lat[:, None] > 0
        south = lat[:, None] < 0
        edge = (lat[:, None] >= -45) & (lat[:, None] <= -28)
        flux_rows = (flux_lat >= -60) & (flux_lat < 0)
        if not flux_rows.any():
            raise ValueError("No Southern Hemisphere coupling-flux rows.")

        box_masks = (np.ones_like(sector, dtype=bool), sector_box)
        upper_weights = []
        for _, lower, upper in OCEAN_BOX_BANDS[:4]:
            levels = (depth_bounds[:, 0] >= lower) & (depth_bounds[:, 1] <= upper)
            if not levels.any():
                raise ValueError(f"No native ocean level in {lower:g}–{upper:g} m")
            upper_weights.append(tuple(volume * levels[:, None, None] * mask[None] for mask in box_masks))
        deep_weights = [
            tuple(
                np.asarray(source.variables[f"deep_wet_volume_{label}"][:], dtype=float) * mask
                for mask in box_masks
            )
            for label, _, _ in OCEAN_BOX_BANDS[4:]
        ]
        if any(weight.sum() <= 0 for band in (*upper_weights, *deep_weights) for weight in band):
            raise ValueError("An ocean box has no wet volume.")

        scalars = {
            name: np.empty(years.size, dtype=np.float32)
            for name in (
                "global_temperature", "northern_temperature", "southern_temperature",
                "south_atlantic_surface_temperature",
                "southern_edge_temperature", "global_mean_sic", "southern_mean_sic",
                "southern_ice_area",
                "global_ice_covered_fraction", "southern_ice_covered_fraction",
                "global_toa_imbalance", "southern_toa_imbalance",
                "southern_coupling_flux",
            )
        }
        box_temperature = np.full(
            (years.size, 2, len(OCEAN_BOX_BANDS)), np.nan, dtype=np.float32
        )

        def weighted_mean(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
            return np.sum(values * weight[None], axis=(-2, -1)) / weight.sum()

        for start in range(0, years.size, 50):
            stop = min(start + 50, years.size)
            surface = np.asarray(source.variables["surface_temperature"][start:stop], dtype=float)
            sic = np.asarray(source.variables["sea_ice_concentration"][start:stop], dtype=float)
            toa = np.asarray(source.variables["zonal_toa_energy_imbalance"][start:stop], dtype=float)
            flux = np.asarray(source.variables["zonal_newtonian_coupling_heat_flux"][start:stop], dtype=float)
            theta = np.asarray(source.variables["temperature_upper"][start:stop], dtype=float)

            for band_index, band_weights in enumerate(upper_weights):
                for box_index, weight in enumerate(band_weights):
                    box_temperature[start:stop, box_index, band_index] = (
                        np.nansum(theta * weight[None], axis=(1, 2, 3)) / weight.sum()
                    )
            for deep_index, (label, _, _) in enumerate(OCEAN_BOX_BANDS[4:]):
                deep_map = np.asarray(source.variables[f"temperature_{label}"][start:stop], dtype=float)
                for box_index, weight in enumerate(deep_weights[deep_index]):
                    box_temperature[start:stop, box_index, deep_index + 4] = (
                        np.nansum(deep_map * weight[None], axis=(1, 2)) / weight.sum()
                    )

            for name, mask in (
                ("global_temperature", np.ones_like(lsm, dtype=bool)),
                ("northern_temperature", np.broadcast_to(north, lsm.shape)),
                ("southern_temperature", np.broadcast_to(south, lsm.shape)),
                ("south_atlantic_surface_temperature", surface_sector),
                ("southern_edge_temperature", np.broadcast_to(edge, lsm.shape)),
            ):
                scalars[name][start:stop] = weighted_mean(surface, area * mask)
            for name, mask in (("global", ocean), ("southern", ocean & south)):
                weighted = area * mask
                scalars[f"{name}_mean_sic"][start:stop] = (
                    np.nansum(sic * weighted[None], axis=(-2, -1)) / weighted.sum()
                )
                if name == "southern":
                    scalars["southern_ice_area"][start:stop] = (
                        scalars["southern_mean_sic"][start:stop]
                        * weighted.sum() * 2 * np.pi * EARTH_RADIUS_M**2 / lsm.shape[1]
                    )
                scalars[f"{name}_ice_covered_fraction"][start:stop] = (
                    np.sum((sic >= ICE_THRESHOLD) * weighted[None], axis=(-2, -1))
                    / weighted.sum()
                )
            zonal_weight = latitude_weights[::-1]
            scalars["global_toa_imbalance"][start:stop] = (
                toa @ zonal_weight / zonal_weight.sum()
            )
            scalars["southern_toa_imbalance"][start:stop] = (
                toa @ (zonal_weight * south[:, 0]) / np.sum(zonal_weight * south[:, 0])
            )
            scalars["southern_coupling_flux"][start:stop] = (
                flux[:, flux_rows] @ wet_area[flux_rows] / wet_area[flux_rows].sum()
            )

    return xr.Dataset(
        {
            **{name: (("year",), values) for name, values in scalars.items()},
            "ocean_box_temperature": (
                ("year", "box", "depth_band"), box_temperature,
                {"units": "K", "weighting": "native wet-cell volume within each depth band and box"},
            ),
        },
        coords={
            "year": years,
            "box": ["global_ocean", "south_atlantic_0_60s"],
            "depth_band": [label for label, _, _ in OCEAN_BOX_BANDS],
        },
        attrs={
            "source_archive": archive.name,
            "sea_ice_definition": "annual mean sic 0-1; covered area uses local annual sic >= 0.5",
            "southern_ice_area_definition": "sum of annual mean SIC times T21 Gaussian-grid ocean-cell area south of the equator; m2",
            "ocean_box_definition": "global wet ocean or South Atlantic wet ocean between 60 S and the equator",
            "surface_box_definition": "area-weighted T21 ts over ocean cells between 60 S and the equator, 65 W to 20 E",
            "temporal_processing": "annual raw values; no smoothing, detrending, or filtering",
        },
    )
