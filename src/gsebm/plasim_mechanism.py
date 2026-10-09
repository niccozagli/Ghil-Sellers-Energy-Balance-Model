"""Ice-cycle phase, composites, and budgets for the South Atlantic mechanism.

The cycle phase comes from the data alone: annual South Atlantic sea-ice area
is smoothed, its maxima are found with `plasim_composites.prepare_cycle_preview`,
and the phase runs linearly from 0 at one maximum to 1 at the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_composites import CyclePreview, prepare_cycle_preview
from gsebm.plasim_exploratory import EARTH_RADIUS_M, mask_path
from gsebm.plasim_koopman_irregularity import ANALYSIS_WINDOWS


# Two runs before and two after the change near 1235, plus 1235 itself.
MECHANISM_MUS = ("1245", "1240", "1235", "1233p75", "1232p5")
EXPERIMENT_PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
# The South Atlantic T21 mask spans 65°W–20°E; this splits it at 30°W.
WEST_EAST_SPLIT_LON = -30.0
ICE_ROW_RANGE = (-50.0, -20.0)


def mu_value(mu: str) -> float:
    """Return the solar constant for an archive label such as '1233p75'."""
    return float(mu.replace("p", "."))


def archive_path(root: Path, mu: str) -> Path:
    """Return the raw-map archive for one μ under an explicit archive root."""
    name = f"{EXPERIMENT_PREFIX}{mu}"
    return Path(root) / name / f"{name}_spinup_raw_maps.nc"


def wrapped_longitude(lon: np.ndarray) -> np.ndarray:
    """Return longitudes in [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def t21_cell_area(lat_weights: np.ndarray, nlon: int) -> np.ndarray:
    """Return T21 Gaussian-grid cell areas in m², shape (lat, lon)."""
    weights = np.asarray(lat_weights, dtype=float)
    return np.repeat(
        (2 * np.pi * EARTH_RADIUS_M**2 * weights / nlon)[:, None], nlon, axis=1,
    )


@dataclass(frozen=True)
class IceSeries:
    """Annual sea-ice area series for one μ window (areas in 10¹² m²)."""

    mu: str
    years: np.ndarray
    south_atlantic: np.ndarray
    southern_hemisphere: np.ndarray
    west: np.ndarray
    east: np.ndarray
    row_lat: np.ndarray
    rows: np.ndarray
    mean_concentration: np.ndarray
    t21_lat: np.ndarray
    t21_lon: np.ndarray
    south_atlantic_mask: np.ndarray


def read_ice_series(archive: Path, mu: str, window: tuple[int, int] | None = None) -> IceSeries:
    """Read annual sea-ice areas over the South Atlantic and its T21 rows.

    `rows` holds the South Atlantic ice area of each T21 row in
    `ICE_ROW_RANGE`, shape (year, row). `mean_concentration` is the
    window-mean concentration map used to show where the ice edge sits.
    """
    archive = Path(archive)
    start, end = window or ANALYSIS_WINDOWS[mu]
    with h5py.File(archive, "r") as source, h5py.File(mask_path(archive), "r") as masks:
        years = np.asarray(source["year"][:], dtype=int)
        index = np.flatnonzero((years >= start) & (years <= end))
        if index.size == 0 or not np.all(np.diff(years[index]) == 1):
            raise ValueError(f"Window {start}–{end} is missing or not consecutive in {archive}")
        lat = np.asarray(source["t21_lat"][:], dtype=float)
        lon = np.asarray(source["t21_lon"][:], dtype=float)
        ocean = np.asarray(source["lsm"][:], dtype=float) < 0.5
        area = t21_cell_area(source["t21_gaussian_weight"][:], lon.size) * ocean
        sector = np.asarray(masks["t21_south_atlantic"][:], dtype=bool) & ocean
        sic = np.asarray(source["sea_ice_concentration"][index[0]:index[-1] + 1], dtype=float)
    ice = sic * area
    west_mask = sector & (wrapped_longitude(lon) < WEST_EAST_SPLIT_LON)[None]
    row_index = np.flatnonzero((lat >= ICE_ROW_RANGE[0]) & (lat <= ICE_ROW_RANGE[1]))
    scale = 1e-12
    return IceSeries(
        mu=mu,
        years=years[index],
        south_atlantic=(ice * sector).sum(axis=(1, 2)) * scale,
        southern_hemisphere=(ice * (lat < 0)[:, None]).sum(axis=(1, 2)) * scale,
        west=(ice * west_mask).sum(axis=(1, 2)) * scale,
        east=(ice * (sector & ~west_mask)).sum(axis=(1, 2)) * scale,
        row_lat=lat[row_index],
        rows=(ice[:, row_index] * sector[row_index]).sum(axis=2) * scale,
        mean_concentration=sic.mean(axis=0),
        t21_lat=lat,
        t21_lon=lon,
        south_atlantic_mask=sector,
    )


def ice_cycle(series: IceSeries, smooth_years: int = 11) -> CyclePreview:
    """Find South Atlantic ice-area maxima with the shared composite tools."""
    return prepare_cycle_preview(
        series.years, series.south_atlantic * 1e12,
        int(series.years[0]), int(series.years[-1]), smooth_years=smooth_years,
    )


def event_phase(year_count: int, peak_indices: np.ndarray) -> np.ndarray:
    """Return a phase in [0, 1) that runs linearly between consecutive maxima.

    Years before the first or after the last maximum get NaN.
    """
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size < 2 or np.any(np.diff(peaks) <= 0):
        raise ValueError("Need at least two increasing maxima.")
    phase = np.full(year_count, np.nan)
    for a, b in zip(peaks[:-1], peaks[1:]):
        phase[a:b] = np.arange(b - a) / (b - a)
    return phase


def phase_composite(values: np.ndarray, phase: np.ndarray, bins: int = 20) -> np.ndarray:
    """Return the mean of `values` (time first) in equal phase bins.

    Years with a NaN phase are skipped. The result has shape (bins, ...).
    """
    values = np.asarray(values, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if values.shape[0] != phase.size:
        raise ValueError("Values and phase must share the time axis.")
    valid = np.isfinite(phase)
    index = np.minimum((phase[valid] * bins).astype(int), bins - 1)
    flat = values[valid].reshape(int(valid.sum()), -1)
    sums = np.zeros((bins, flat.shape[1]))
    np.add.at(sums, index, flat)
    counts = np.bincount(index, minlength=bins).astype(float)
    with np.errstate(invalid="ignore"):
        means = sums / counts[:, None]
    return means.reshape((bins,) + values.shape[1:])


def stage_durations(smoothed: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """Return (retreat, advance) years for each cycle between two maxima.

    Retreat runs from a maximum to the lowest smoothed value before the next
    maximum; advance runs from that minimum to the next maximum.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int)
    durations = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        low = a + int(np.nanargmin(smoothed[a:b + 1]))
        durations.append((low - a, b - low))
    return np.asarray(durations, dtype=int).reshape(-1, 2)


@dataclass(frozen=True)
class CycleSummary:
    """Ice series, detected maxima, and event phase for one μ."""

    series: IceSeries
    cycle: CyclePreview
    phase: np.ndarray
    stages: np.ndarray


def summarize_cycle(series: IceSeries, smooth_years: int = 11) -> CycleSummary:
    """Return maxima, event phase, and retreat/advance durations for one μ."""
    cycle = ice_cycle(series, smooth_years)
    return CycleSummary(
        series=series,
        cycle=cycle,
        phase=event_phase(series.years.size, cycle.peak_indices),
        stages=stage_durations(cycle.smoothed_area, cycle.peak_indices),
    )


def cycle_table(summaries: dict[str, CycleSummary]) -> list[dict[str, float]]:
    """Return one row of cycle statistics per μ."""
    rows = []
    for mu, summary in summaries.items():
        lengths = summary.cycle.cycle_lengths
        rows.append({
            "μ": mu_value(mu),
            "spectral period (yr)": round(summary.cycle.spectral_period, 1),
            "maxima": int(summary.cycle.peak_indices.size),
            "median cycle (yr)": float(np.median(lengths)),
            "cycle IQR (yr)": f"{np.percentile(lengths, 25):.0f}–{np.percentile(lengths, 75):.0f}",
            "median retreat (yr)": float(np.median(summary.stages[:, 0])),
            "median advance (yr)": float(np.median(summary.stages[:, 1])),
            "mean SA ice (10¹² m²)": round(float(summary.series.south_atlantic.mean()), 2),
        })
    return rows
