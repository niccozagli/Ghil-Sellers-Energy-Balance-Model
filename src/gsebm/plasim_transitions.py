"""Anatomy of the warm → cold and cold → snowball transitions in the μ runs.

Annual series per ocean sector and hemisphere: ice edges, land snow,
volume-mean ocean temperatures, and the zonal ocean heat transport. Event
timing is measured as the first persistent departure from the
pre-transition linear trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_global import (
    RUNS, ice_edge_latitude, ocean_northward_transport, read_global_series, run_archive,
    valid_record_range,
)
from gsebm.plasim_mechanism import wrapped_longitude


# Longitude sectors (°E in [-180, 180)); the same split is used in both
# hemispheres, so the northern "Indian" sector is mostly Asia and its few
# ocean cells give an unreliable edge.
SECTORS = {
    "atlantic": (-65.0, 20.0),
    "indian": (20.0, 115.0),
    "pacific": (115.0, -65.0),
}
SNOW_DEPTH_THRESHOLD = 0.05  # m; land counts as snow-covered above this depth
READ_BLOCK = 10


def sector_mask(lon: np.ndarray, sector: str) -> np.ndarray:
    """Return a boolean longitude mask for one sector (wrapping at the date line)."""
    west, east = SECTORS[sector]
    lon = wrapped_longitude(lon)
    if west < east:
        return (lon >= west) & (lon < east)
    return (lon >= west) | (lon < east)


@dataclass(frozen=True)
class TransitionSeries:
    """Annual series of one run (T21 rows for surface, LSG rows for ocean)."""

    label: str
    years: np.ndarray
    lat: np.ndarray
    weight: np.ndarray
    surface_temperature: np.ndarray
    sector_ice: dict[str, np.ndarray]
    ocean_ice: np.ndarray
    land_snow: np.ndarray
    lsg_lat: np.ndarray
    depth: np.ndarray
    wet_volume: np.ndarray
    lsg_row_area: np.ndarray
    potential_temperature: np.ndarray
    ocean_heat_uptake: np.ndarray


def _block(dataset, first: int, last: int) -> np.ndarray:
    try:
        return np.asarray(dataset[first:last], dtype=float)
    except OSError:
        return np.full((last - first,) + dataset.shape[1:], np.nan)


def read_transition_series(root: Path, label: str,
                           years: tuple[int, int] | None = None) -> TransitionSeries:
    """Read one run's sector ice, land snow, and zonal ocean fields.

    Blocks that cannot be read become NaN. Only zonal reductions are kept.
    """
    archive = run_archive(root, label)
    with h5py.File(archive, "r") as source:
        all_years = np.asarray(source["year"][:], dtype=int)
        keep = np.ones(all_years.size, dtype=bool) if years is None else (
            (all_years >= years[0]) & (all_years <= years[1])
        )
        index = np.flatnonzero(keep)
        a, b = int(index[0]), int(index[-1]) + 1
        lat = np.asarray(source["t21_lat"][:], dtype=float)
        lon = np.asarray(source["t21_lon"][:], dtype=float)
        ocean = np.asarray(source["lsm"][:], dtype=float) < 0.5
        masks = {name: ocean & sector_mask(lon, name)[None] for name in SECTORS}
        counts = {name: np.where(mask.sum(axis=1) > 0, mask.sum(axis=1), np.nan)
                  for name, mask in masks.items()}
        land = ~ocean
        land_count = np.where(land.sum(axis=1) > 0, land.sum(axis=1), np.nan)
        ocean_count = np.where(ocean.sum(axis=1) > 0, ocean.sum(axis=1), np.nan)
        surface, snow, total = [], [], []
        ice = {name: [] for name in SECTORS}
        for first in range(a, b, READ_BLOCK):
            last = min(first + READ_BLOCK, b)
            sic = _block(source["sea_ice_concentration"], first, last)
            surface.append(_block(source["surface_temperature"], first, last).mean(axis=2))
            depth = _block(source["snd"], first, last)
            snow.append(((depth > SNOW_DEPTH_THRESHOLD) & land).sum(axis=2) / land_count)
            total.append((sic * ocean).sum(axis=2) / ocean_count)
            for name, mask in masks.items():
                ice[name].append((sic * mask).sum(axis=2) / counts[name])
        theta = np.concatenate([
            _block(source["zonal_potential_temperature"], first, min(first + READ_BLOCK, b))
            for first in range(a, b, READ_BLOCK)
        ])
        uptake = np.concatenate([
            _block(source["zonal_newtonian_coupling_heat_flux"], first, min(first + READ_BLOCK, b))
            for first in range(a, b, READ_BLOCK)
        ])
        weight = np.asarray(source["t21_gaussian_weight"][:], dtype=float)
        return TransitionSeries(
            label=label,
            years=all_years[a:b],
            lat=lat,
            weight=weight / weight.sum(),
            surface_temperature=np.concatenate(surface),
            sector_ice={name: np.concatenate(values) for name, values in ice.items()},
            ocean_ice=np.concatenate(total),
            land_snow=np.concatenate(snow),
            lsg_lat=np.asarray(source["lsg_lat"][:], dtype=float),
            depth=np.asarray(source["lsg_depth"][:], dtype=float),
            wet_volume=np.asarray(source["wet_volume"][:], dtype=float),
            lsg_row_area=np.asarray(source["wet_surface_area"][:], dtype=float),
            potential_temperature=theta,
            ocean_heat_uptake=uptake,
        )


def ocean_layer_mean(theta: np.ndarray, wet_volume: np.ndarray, lsg_lat: np.ndarray,
                     depth: np.ndarray, depth_range: tuple[float, float],
                     lat_range: tuple[float, float] = (-90.0, 90.0)) -> np.ndarray:
    """Volume-mean potential temperature over LSG rows and levels in the ranges.

    `theta` has shape (time, row, level); `wet_volume` (row, level).
    """
    rows = (lsg_lat >= lat_range[0]) & (lsg_lat <= lat_range[1])
    levels = (depth >= depth_range[0]) & (depth < depth_range[1])
    weight = wet_volume * rows[:, None] * levels[None, :]
    return np.nansum(theta * weight[None], axis=(1, 2)) / weight.sum()


def ocean_transport_series(series: TransitionSeries) -> tuple[np.ndarray, np.ndarray]:
    """Return (face latitudes, northward ocean transport in PW per year)."""
    order = np.argsort(series.lsg_lat)
    uptake = np.nan_to_num(series.ocean_heat_uptake[:, order])
    area = series.lsg_row_area[order]
    transport = np.stack([ocean_northward_transport(row, area) for row in uptake])
    # Rows are 2.5° apart; the north face sits half a row poleward.
    return series.lsg_lat[order] + 1.25, transport


@dataclass(frozen=True)
class TransitionIndices:
    """Scalar annual indices of one run."""

    years: np.ndarray
    indices: dict[str, np.ndarray]


def transition_indices(series: TransitionSeries) -> TransitionIndices:
    """Return the indices used to time the transitions."""
    indices = {"global Ts": series.surface_temperature @ series.weight}
    for name, ice in series.sector_ice.items():
        indices[f"S edge {name}"] = ice_edge_latitude(ice, series.lat, south=True)
        if name != "indian":
            indices[f"N edge {name}"] = ice_edge_latitude(ice, series.lat, south=False)
    snow = np.nan_to_num(series.land_snow)
    indices["S land snow edge"] = ice_edge_latitude(snow, series.lat, south=True)
    indices["N land snow edge"] = ice_edge_latitude(snow, series.lat, south=False)
    args = (series.potential_temperature, series.wet_volume, series.lsg_lat, series.depth)
    indices["tropical θ 0–700 m"] = ocean_layer_mean(*args, (0, 700), (-20, 20))
    indices["deep θ >1000 m"] = ocean_layer_mean(*args, (1000, 1e4))
    faces, transport = ocean_transport_series(series)
    for latitude in (-35.0, -20.0, 35.0):
        name = f"ocean transport {abs(latitude):g}°{'S' if latitude < 0 else 'N'}"
        indices[name] = np.array([np.interp(latitude, faces, row) for row in transport])
    return TransitionIndices(years=series.years, indices=indices)


def running_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centred running mean that averages only over existing, finite values.

    Near the ends of the series the window is truncated rather than padded,
    so the mean is not biased towards zero there.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    kernel = np.ones(width)
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def onset_year(years: np.ndarray, values: np.ndarray, reference: tuple[int, int],
               smooth: int = 11, sigmas: float = 4.0, persist: int = 30) -> int | None:
    """First year after the reference period with a persistent trend departure.

    A linear trend is fitted to the smoothed series over `reference`. The
    onset is the first later year from which the departure exceeds `sigmas`
    reference standard deviations for `persist` consecutive years.
    """
    years = np.asarray(years)
    smoothed = running_mean(values, smooth)
    ref = (years >= reference[0]) & (years <= reference[1])
    slope, intercept = np.polyfit(years[ref], smoothed[ref], 1)
    departure = smoothed - (slope * years + intercept)
    threshold = max(sigmas * np.std(departure[ref]), 1e-12)
    outside = np.abs(departure) > threshold
    for k in np.flatnonzero(years > reference[1]):
        if k + persist <= years.size and outside[k:k + persist].all():
            return int(years[k])
    return None


def variability(values: np.ndarray) -> tuple[float, float]:
    """Standard deviation and lag-1 autocorrelation after removing a linear trend."""
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    t = np.arange(values.size)
    residual = values - np.polyval(np.polyfit(t, values, 1), t)
    return float(residual.std()), float(np.corrcoef(residual[:-1], residual[1:])[0, 1])


@dataclass(frozen=True)
class WindowProfile:
    """Mean ocean transport, ocean heat uptake, and ice edges over a year range."""

    name: str
    faces: np.ndarray
    transport: np.ndarray
    lsg_lat: np.ndarray
    uptake: np.ndarray
    south_edge: float
    north_edge: float


def window_profile(series: TransitionSeries, years: tuple[int, int], name: str) -> WindowProfile:
    """Average transport, uptake, and zonal-mean ice edges over `years`."""
    select = (series.years >= years[0]) & (series.years <= years[1])
    faces, transport = ocean_transport_series(series)
    order = np.argsort(series.lsg_lat)
    ice = series.ocean_ice
    return WindowProfile(
        name=name,
        faces=faces,
        transport=transport[select].mean(axis=0),
        lsg_lat=series.lsg_lat[order],
        uptake=np.nanmean(series.ocean_heat_uptake[select][:, order], axis=0),
        south_edge=float(np.nanmean(ice_edge_latitude(ice[select], series.lat, True))),
        north_edge=float(np.nanmean(ice_edge_latitude(ice[select], series.lat, False))),
    )


@dataclass(frozen=True)
class RunTimeseries:
    """Full-length annual global indices of one run, with its analysis window."""

    label: str
    years: np.ndarray
    window: tuple[int, int]
    indices: dict[str, np.ndarray]


def run_timeseries(root: Path, label: str, series=None) -> RunTimeseries:
    """Return global indices for every valid year of a run.

    Indices: global Ts, planetary albedo, TOA imbalance, zonal-mean ice
    edges, and the volume-mean deep (> 1000 m) ocean temperature. Pass the
    full-length `GlobalSeries` if it has already been read.
    """
    if series is None:
        series = read_global_series(root, label, full=True)
    w = series.weight
    with h5py.File(run_archive(root, label), "r") as source:
        all_years = np.asarray(source["year"][:], dtype=int)
        first, last = valid_record_range(all_years)
        a = int(np.flatnonzero(all_years == first)[0])
        b = int(np.flatnonzero(all_years == last)[0]) + 1
        theta = np.concatenate([
            _block(source["zonal_potential_temperature"], k, min(k + READ_BLOCK, b))
            for k in range(a, b, READ_BLOCK)
        ])
        depth = np.asarray(source["lsg_depth"][:], dtype=float)
        volume = np.asarray(source["wet_volume"][:], dtype=float)
        lsg_lat = np.asarray(source["lsg_lat"][:], dtype=float)
    absorbed = series.absorbed_shortwave @ w
    return RunTimeseries(
        label=label,
        years=series.years,
        window=RUNS[label].window,
        indices={
            "global Ts (K)": series.surface_temperature @ w,
            "planetary albedo": 1 - absorbed / (series.insolation @ w),
            "TOA imbalance (W m⁻²)": absorbed - series.outgoing_longwave @ w,
            "S ice edge (°)": ice_edge_latitude(series.ocean_ice, series.lat, south=True),
            "N ice edge (°)": ice_edge_latitude(series.ocean_ice, series.lat, south=False),
            "deep θ >1000 m (K)": ocean_layer_mean(theta, volume, lsg_lat, depth, (1000, 1e4)),
        },
    )
