"""Phase-aligned vertical temperature timing for two South Atlantic regions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5netcdf
import numpy as np
from scipy.ndimage import uniform_filter1d

from gsebm.plasim_composites import CyclePreview, _phase_positions_for_events


PHASES = np.linspace(-0.6, 0.6, 61)
LAYER_NAMES = ("0–100 m", "150–300 m", "300–600 m", "Surface T")
REGIONS = (
    ("Ice edge", (-45.0, -30.0), (-42.0, -28.0, -55.0, 10.0)),
    ("Tropics", (-20.0, -5.0), (-20.0, -5.0, -40.0, 10.0)),
)
LAYER_FIELDS = (
    "theta_layer_0_100m", "theta_layer_150_300m", "theta_layer_300_600m",
)


@dataclass(frozen=True)
class VerticalTiming:
    """Composite anomalies and first-harmonic warm peaks by region and depth."""

    phases: np.ndarray
    depths: np.ndarray
    profiles_mk: np.ndarray  # region, depth, phase
    boxes_mk: np.ndarray  # region, field, phase
    profile_warm_phase: np.ndarray  # region, depth; fraction of cycle
    profile_amplitude_mk: np.ndarray
    box_warm_phase: np.ndarray  # region, field; fraction of cycle
    box_amplitude_mk: np.ndarray
    event_count: int
    median_cycle_years: float


def compute_vertical_timing(archive: Path, preview: CyclePreview) -> VerticalTiming:
    """Reduce annual fields, then composite around the selected SH ice maxima."""
    archive = Path(archive).resolve()
    stat = archive.stat()
    return _compute_cached(
        str(archive), stat.st_size, stat.st_mtime_ns,
        int(preview.years[0]), int(preview.years[-1]),
        tuple(int(index) for index in preview.peak_indices),
        preview.detrend_method, preview.smooth_years,
    )


@lru_cache(maxsize=4)
def _compute_cached(
    archive_name: str,
    archive_size: int,
    archive_mtime_ns: int,
    first_year: int,
    last_year: int,
    peak_indices: tuple[int, ...],
    detrend_method: str,
    smooth_years: int,
) -> VerticalTiming:
    del archive_size, archive_mtime_ns
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size < 5:
        raise ValueError("At least five ice maxima are needed for vertical timing.")
    events = peaks[1:-1]
    positions = _phase_positions_for_events(peaks, events, PHASES)
    half = smooth_years // 2
    length = last_year - first_year + 1
    valid = (positions.min(axis=1) >= half) & (
        positions.max(axis=1) <= length - 1 - half
    )
    positions = positions[valid]
    if len(positions) < 3:
        raise ValueError("Fewer than three complete cycles remain in the selected years.")

    with h5netcdf.File(archive_name, "r") as source:
        all_years = np.asarray(source.variables["year"][:], dtype=int)
        start = int(np.searchsorted(all_years, first_year))
        stop = int(np.searchsorted(all_years, last_year, side="right"))
        if stop - start != length or all_years[start] != first_year or all_years[stop - 1] != last_year:
            raise ValueError("Selected years are missing from the raw-map archive.")

        depths = np.asarray(source.variables["lsg_depth"][:], dtype=float)
        depth_indices = np.flatnonzero(depths < 1000)
        row_lat = np.asarray(source.variables["lsg_lat"][:], dtype=float)
        row_volume = np.asarray(source.variables["wet_volume"][:], dtype=float)
        ocean_lat = np.asarray(source.variables["lat"][:], dtype=float)
        ocean_lon = _wrap(np.asarray(source.variables["lon"][:], dtype=float))
        t21_lat = np.asarray(source.variables["t21_lat"][:], dtype=float)
        t21_lon = _wrap(np.asarray(source.variables["t21_lon"][:], dtype=float))
        lsm = np.asarray(source.variables["lsm"][:], dtype=float)
        t21_lat_grid, t21_lon_grid = np.meshgrid(t21_lat, t21_lon, indexing="ij")

        row_masks = []
        ocean_masks = []
        surface_masks = []
        for _, band, box in REGIONS:
            row_mask = (row_lat > band[0]) & (row_lat < band[1])
            y0, y1, x0, x1 = box
            ocean_mask = (
                (ocean_lat >= y0) & (ocean_lat <= y1)
                & (ocean_lon >= x0) & (ocean_lon <= x1)
            )
            surface_mask = (
                (t21_lat_grid >= y0) & (t21_lat_grid <= y1)
                & (t21_lon_grid >= x0) & (t21_lon_grid <= x1)
                & (lsm < 0.5)
            )
            if not row_mask.any() or not ocean_mask.any() or not surface_mask.any():
                raise ValueError(f"No cells in the {band} latitude band or {box} box.")
            row_masks.append(row_mask)
            ocean_masks.append(ocean_mask)
            surface_masks.append(surface_mask)

        profiles = np.empty((length, len(REGIONS), len(depth_indices)), dtype=float)
        boxes = np.empty((length, len(REGIONS), len(LAYER_NAMES)), dtype=float)
        for offset in range(0, length, 128):
            end = min(offset + 128, length)
            slab = slice(start + offset, start + end)
            zonal = np.asarray(source.variables["zonal_potential_temperature"][slab, :, :len(depth_indices)], dtype=float)
            for region, row_mask in enumerate(row_masks):
                weights = row_volume[row_mask, :len(depth_indices)]
                valid_zonal = np.isfinite(zonal[:, row_mask]) & (weights[None] > 0)
                numerator = np.sum(np.where(valid_zonal, zonal[:, row_mask], 0) * weights[None], axis=1)
                denominator = np.sum(valid_zonal * weights[None], axis=1)
                profiles[offset:end, region] = np.divide(
                    numerator, denominator,
                    out=np.full_like(numerator, np.nan), where=denominator > 0,
                )
            for field_index, field_name in enumerate(LAYER_FIELDS):
                layer = np.asarray(source.variables[field_name][slab], dtype=float)
                for region, mask in enumerate(ocean_masks):
                    boxes[offset:end, region, field_index] = np.nanmean(
                        layer[:, mask], axis=1
                    )
            surface = np.asarray(source.variables["surface_temperature"][slab], dtype=float)
            for region, mask in enumerate(surface_masks):
                boxes[offset:end, region, -1] = np.nanmean(surface[:, mask], axis=1)

    profiles = _preprocess_series(profiles, detrend_method, smooth_years)
    boxes = _preprocess_series(boxes, detrend_method, smooth_years)
    profile_composite = _sample_series(profiles, positions).transpose(1, 2, 0) * 1000
    box_composite = _sample_series(boxes, positions).transpose(1, 2, 0) * 1000
    profile_phase, profile_amplitude = _harmonic_timing(profile_composite)
    box_phase, box_amplitude = _harmonic_timing(box_composite)
    return VerticalTiming(
        phases=PHASES.copy(), depths=depths[depth_indices],
        profiles_mk=profile_composite, boxes_mk=box_composite,
        profile_warm_phase=profile_phase,
        profile_amplitude_mk=profile_amplitude,
        box_warm_phase=box_phase,
        box_amplitude_mk=box_amplitude,
        event_count=len(positions),
        median_cycle_years=float(np.median(np.diff(peaks))),
    )


def _wrap(longitude: np.ndarray) -> np.ndarray:
    return (longitude + 180) % 360 - 180


def _preprocess_series(values: np.ndarray, method: str, smooth_years: int) -> np.ndarray:
    """Apply the notebook's mean-centering or linear detrending and smoothing."""
    if method not in {"linear", "none"}:
        raise ValueError("Detrending must be 'linear' or 'none'.")
    if not np.isfinite(values).all():
        raise ValueError("A selected depth or box has missing annual temperatures.")
    center = (len(values) - 1) / 2
    mean = values.mean(axis=0)
    result = values - mean
    if method == "linear":
        time = np.arange(len(values), dtype=float) - center
        slope = np.einsum("t,tij->ij", time, result) / np.sum(time**2)
        result -= time[:, None, None] * slope[None]
    if smooth_years > 1:
        result = uniform_filter1d(result, size=smooth_years, axis=0, mode="nearest")
    return result


def _sample_series(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate each event at each phase and average across events."""
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    fraction = positions - lower
    sampled = values[lower] * (1 - fraction[..., None, None]) + values[upper] * fraction[..., None, None]
    return np.mean(sampled, axis=0)


def _harmonic_timing(composite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warm-peak phase and amplitude of the one-cycle harmonic."""
    within_cycle = (PHASES >= -0.5) & (PHASES < 0.5)
    coefficient = np.mean(
        composite[..., within_cycle] * np.exp(-2j * np.pi * PHASES[within_cycle]),
        axis=-1,
    )
    warm_phase = (-np.angle(coefficient) / (2 * np.pi) + 0.5) % 1 - 0.5
    return warm_phase, 2 * np.abs(coefficient)
