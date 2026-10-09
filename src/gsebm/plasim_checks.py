"""Checks of what to trust in the PlaSim–LSG runs: grid signature and energy closure.

Everything here works on annual values as stored in the raw-map archives;
nothing is smoothed. Window means are plain time means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_exploratory import EARTH_RADIUS_M
from gsebm.plasim_global import RUNS, ice_edge_latitude, run_archive, valid_record_range
from gsebm.plasim_mechanism import t21_cell_area
from gsebm.plasim_transitions import SECTORS, sector_mask


# LSG constants (lsgmod.f90): reference density and latent heat entmel = 80 cp.
LSG_DENSITY = 1030.0  # kg m⁻³
LSG_LATENT_HEAT = 80.0 * 4180.0  # J kg⁻¹
READ_BLOCK = 10
PARTIAL_COVER = (0.05, 0.95)


def _rows(dataset, first: int, last: int) -> np.ndarray:
    blocks = []
    for k in range(first, last, READ_BLOCK):
        stop = min(k + READ_BLOCK, last)
        try:
            blocks.append(np.asarray(dataset[k:stop], dtype=float))
        except OSError:
            blocks.append(np.full((stop - k,) + dataset.shape[1:], np.nan))
    return np.concatenate(blocks)


def _record_slice(years: np.ndarray, span: tuple[int, int]) -> tuple[int, int]:
    index = np.flatnonzero((years >= span[0]) & (years <= span[1]))
    if index.size == 0 or not np.all(np.diff(years[index]) == 1):
        raise ValueError(f"Years {span[0]}–{span[1]} are missing or not consecutive.")
    return int(index[0]), int(index[-1]) + 1


@dataclass(frozen=True)
class IceRows:
    """Annual sea-ice area per T21 row and sector (10¹² m²) and zonal concentration."""

    label: str
    years: np.ndarray
    lat: np.ndarray
    sector_area: dict[str, np.ndarray]
    zonal_concentration: np.ndarray


def read_ice_rows(root: Path, label: str, span: tuple[int, int] | None = None) -> IceRows:
    """Read annual ice area per row and sector, and zonal ocean ice concentration.

    Ice area of a cell is `sea_ice_concentration` times the T21 cell area,
    over ocean cells (`lsm < 0.5`). Zonal concentration is the mean over the
    ocean cells of each row (NaN for rows without ocean).
    """
    with h5py.File(run_archive(root, label), "r") as source:
        years = np.asarray(source["year"][:], dtype=int)
        a, b = _record_slice(years, span or valid_record_range(years))
        lat = np.asarray(source["t21_lat"][:], dtype=float)
        lon = np.asarray(source["t21_lon"][:], dtype=float)
        ocean = np.asarray(source["lsm"][:], dtype=float) < 0.5
        area = t21_cell_area(np.asarray(source["t21_gaussian_weight"][:]), lon.size)
        sic = _rows(source["sea_ice_concentration"], a, b)
    ice = sic * (area * ocean)[None] * 1e-12
    count = ocean.sum(axis=1)
    zonal = np.where(count > 0, (sic * ocean).sum(axis=2) / np.maximum(count, 1), np.nan)
    return IceRows(
        label=label,
        years=years[a:b],
        lat=lat,
        sector_area={
            name: (ice * sector_mask(lon, name)[None, None]).sum(axis=2) for name in SECTORS
        },
        zonal_concentration=zonal,
    )


def partial_rows(concentration: np.ndarray, lat: np.ndarray,
                 bounds: tuple[float, float] = PARTIAL_COVER) -> dict[str, np.ndarray]:
    """Latitudes of rows whose concentration lies strictly inside `bounds`, per hemisphere."""
    inside = (concentration > bounds[0]) & (concentration < bounds[1])
    return {"S": lat[inside & (lat < 0)], "N": lat[inside & (lat > 0)]}


def jump_contributions(rows: IceRows, before: tuple[int, int], after: tuple[int, int],
                       south: bool = True) -> list[tuple[str, float, float]]:
    """Split the change in hemispheric ice area into row × sector parts.

    Change = mean over `after` minus mean over `before` (plain time means).
    Returns (sector, row latitude, change in 10¹² m²), largest first.
    """
    pre = (rows.years >= before[0]) & (rows.years <= before[1])
    post = (rows.years >= after[0]) & (rows.years <= after[1])
    hemisphere = rows.lat < 0 if south else rows.lat > 0
    parts = []
    for name, area in rows.sector_area.items():
        change = area[post].mean(axis=0) - area[pre].mean(axis=0)
        parts += [(name, float(la), float(c))
                  for la, c in zip(rows.lat[hemisphere], change[hemisphere])]
    return sorted(parts, key=lambda part: -abs(part[2]))


def annual_edges(rows: IceRows) -> dict[str, np.ndarray]:
    """Annual zonal-mean ice-edge latitudes (see `plasim_global.ice_edge_latitude`)."""
    return {
        "S": ice_edge_latitude(rows.zonal_concentration, rows.lat, south=True),
        "N": ice_edge_latitude(rows.zonal_concentration, rows.lat, south=False),
    }


@dataclass(frozen=True)
class EnergySeries:
    """Annual global energy terms of one run, in W (10¹⁵ W = 1 PW)."""

    label: str
    years: np.ndarray
    toa_net: np.ndarray
    ocean_uptake: np.ndarray
    ocean_storage: np.ndarray
    ice_latent: np.ndarray


def read_energy_series(root: Path, label: str) -> EnergySeries:
    """Read the global energy terms over all valid years of a run.

    * `toa_net`: global integral of `rst + rlut` (net downward at the top).
    * `ocean_uptake`: `zonal_newtonian_coupling_heat_flux` times the wet
      surface area of each LSG row, summed (native sign: into the ocean).
    * `ocean_storage`: year-to-year difference of the summed
      `zonal_ocean_heat_content`, divided by one year; the value at year t
      is (OHC[t] − OHC[t−1]) / year, NaN for the first year.
    * `ice_latent`: latent heat stored by sea-ice growth, the year-to-year
      difference of `global_lsg_ice_volume` × LSG density × latent heat.
    """
    year_seconds = 360 * 86400.0  # PlaSim–LSG years have 360 days
    with h5py.File(run_archive(root, label), "r") as source:
        years = np.asarray(source["year"][:], dtype=int)
        a, b = _record_slice(years, valid_record_range(years))
        weight = np.asarray(source["t21_gaussian_weight"][:], dtype=float)
        weight = weight / weight.sum()
        sphere = 4 * np.pi * EARTH_RADIUS_M**2
        net = _rows(source["rst"], a, b).mean(axis=2) + _rows(source["rlut"], a, b).mean(axis=2)
        row_area = np.asarray(source["wet_surface_area"][:], dtype=float)
        uptake = _rows(source["zonal_newtonian_coupling_heat_flux"], a, b) @ row_area
        heat = _rows(source["zonal_ocean_heat_content"], a, b).sum(axis=1)
        volume = _rows(source["global_lsg_ice_volume"], a, b)
    storage = np.r_[np.nan, np.diff(heat)] / year_seconds
    latent = np.r_[np.nan, np.diff(volume)] * LSG_DENSITY * LSG_LATENT_HEAT / year_seconds
    return EnergySeries(
        label=label,
        years=years[a:b],
        toa_net=(net @ weight) * sphere,
        ocean_uptake=uptake,
        ocean_storage=storage,
        ice_latent=latent,
    )


def window_energy(series: EnergySeries) -> dict[str, float]:
    """Plain time means of the energy terms over the run's analysis window (W m⁻²)."""
    start, end = RUNS[series.label].window
    inside = (series.years >= start) & (series.years <= end)
    sphere = 4 * np.pi * EARTH_RADIUS_M**2
    return {
        name: float(np.nanmean(getattr(series, name)[inside]) / sphere)
        for name in ("toa_net", "ocean_uptake", "ocean_storage", "ice_latent")
    }


def coverage_table(rows: dict[str, IceRows], lat_limit: tuple[float, float] = (10.0, 66.0)
                   ) -> list[dict[str, object]]:
    """Window-mean zonal ice concentration per row (one table row per run)."""
    table = []
    for label, data in rows.items():
        start, end = RUNS[label].window
        inside = (data.years >= start) & (data.years <= end)
        finite = np.isfinite(data.zonal_concentration[inside])
        mean = np.where(finite.any(axis=0),
                        np.nansum(data.zonal_concentration[inside], axis=0)
                        / np.maximum(finite.sum(axis=0), 1), np.nan)
        partial = partial_rows(mean, data.lat)
        entry: dict[str, object] = {"run": label.replace("p", ".")}
        for la, value in zip(data.lat, mean):
            if lat_limit[0] < abs(la) < lat_limit[1]:
                entry[f"{la:+.1f}"] = round(float(value), 2)
        entry["partial S"] = ", ".join(f"{la:.1f}" for la in partial["S"])
        entry["partial N"] = ", ".join(f"{la:.1f}" for la in partial["N"])
        table.append(entry)
    return table


def energy_table(series: dict[str, EnergySeries]) -> list[dict[str, object]]:
    """Window means of the energy terms and the unexplained residual (W m⁻²)."""
    table = []
    for label, data in series.items():
        terms = window_energy(data)
        table.append({
            "run": label.replace("p", "."),
            "TOA net": round(terms["toa_net"], 3),
            "into ocean": round(terms["ocean_uptake"], 3),
            "ocean storage": round(terms["ocean_storage"], 3),
            "ice latent": round(terms["ice_latent"], 4),
            "TOA − storage − ice": round(
                terms["toa_net"] - terms["ocean_storage"] - terms["ice_latent"], 3),
            "into ocean − storage": round(terms["ocean_uptake"] - terms["ocean_storage"], 3),
        })
    return table


def period_difference(data: EnergySeries, event: tuple[int, int], quiet: tuple[int, int]
                      ) -> dict[str, float]:
    """Mean of each energy term over `event` minus over `quiet` (W m⁻² of globe)."""
    sphere = 4 * np.pi * EARTH_RADIUS_M**2
    result = {}
    for name in ("toa_net", "ocean_uptake", "ocean_storage", "ice_latent"):
        values = getattr(data, name)
        means = [np.nanmean(values[(data.years >= a) & (data.years <= b)]) / sphere
                 for a, b in (event, quiet)]
        result[name] = float(means[0] - means[1])
    return result
