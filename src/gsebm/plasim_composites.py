"""Period-aware, event-aligned composites of annual PlaSim raw maps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend as signal_detrend
from scipy.signal import find_peaks, welch

from gsebm.plasim_exploratory import EARTH_RADIUS_M


DEFAULT_MAX_PHASE = (-1 / 51, 1 / 51)
DEFAULT_MIN_BEFORE_PHASE = (-26 / 51, -24 / 51)
DEFAULT_MIN_AFTER_PHASE = (24 / 51, 26 / 51)
PATHWAY_PHASES = (-0.4, -0.2, 0.0, 0.2, 0.4)


@dataclass(frozen=True)
class CyclePreview:
    """Ice-area processing and event timing for one consecutive year window."""

    years: np.ndarray
    raw_area: np.ndarray
    detrended_area: np.ndarray
    smoothed_area: np.ndarray
    spectral_period: float
    search_period: float
    spectral_periods: np.ndarray
    spectral_power: np.ndarray
    peak_indices: np.ndarray
    peak_radius_years: int
    detrend_method: str
    smooth_years: int

    @property
    def cycle_lengths(self) -> np.ndarray:
        return np.diff(self.years[self.peak_indices])


@dataclass(frozen=True)
class CompositeMaps:
    """Global phase differences, pathway snapshots, and geographic context."""

    t21_lat: np.ndarray
    t21_lon: np.ndarray
    lsm: np.ndarray
    surface_temperature_difference: np.ndarray
    sea_ice_concentration_difference: np.ndarray
    mean_sea_ice_concentration: np.ndarray
    lsg_lat: np.ndarray
    lsg_lon: np.ndarray
    wet_150_300m: np.ndarray
    theta_150_300m_difference: np.ndarray
    pathway_phases: tuple[float, ...]
    theta_150_300m_snapshots: np.ndarray
    lsg_vector_lat: np.ndarray
    lsg_vector_lon: np.ndarray
    u_mean_150_300m: np.ndarray
    v_mean_150_300m: np.ndarray
    u_150_300m_snapshots: np.ndarray
    v_150_300m_snapshots: np.ndarray
    edge_lat: np.ndarray
    edge_lon: np.ndarray
    detected_peaks: int
    composited_events: int
    median_cycle_years: float


def prepare_cycle_preview(
    years: np.ndarray,
    ice_area: np.ndarray,
    first_year: int,
    last_year: int,
    detrend_method: str = "linear",
    smooth_years: int = 11,
    radius_fraction: float = 0.4,
    manual_period: float = 0.0,
) -> CyclePreview:
    """Estimate the period and find SH ice-area maxima before reading maps."""
    years = np.asarray(years, dtype=int)
    ice_area = np.asarray(ice_area, dtype=float)
    if years.shape != ice_area.shape or not np.all(np.diff(years) == 1):
        raise ValueError("Ice-area years must be consecutive and match the series.")
    if detrend_method not in {"linear", "none"}:
        raise ValueError("Detrending must be 'linear' or 'none'.")
    if smooth_years < 1 or smooth_years % 2 != 1:
        raise ValueError("Running-mean width must be a positive odd number.")
    if not 0.1 <= radius_fraction < 0.5:
        raise ValueError("Peak-search radius must be between 0.1 and 0.5 cycle.")
    selected = (years >= first_year) & (years <= last_year)
    yy = years[selected]
    raw = ice_area[selected] / 1e12
    if yy.size < 200 or not np.all(np.isfinite(raw)):
        raise ValueError("Select at least 200 consecutive years with finite SH ice area.")
    if detrend_method == "linear":
        processed = signal_detrend(raw, type="linear")
    else:
        processed = raw - raw.mean()
    if smooth_years == 1:
        smoothed = processed.copy()
    else:
        smoothed = np.convolve(
            processed, np.ones(smooth_years) / smooth_years, mode="same"
        )
        half = smooth_years // 2
        smoothed[:half] = np.nan
        smoothed[-half:] = np.nan

    spectrum_series = smoothed[np.isfinite(smoothed)]
    segment = min(2048, spectrum_series.size)
    frequency, power = welch(
        spectrum_series, fs=1.0, window="hann", nperseg=segment,
        noverlap=segment // 2, detrend="constant", scaling="density",
    )
    in_band = (frequency >= 1 / 150) & (frequency <= 1 / 20)
    if not in_band.any():
        raise ValueError("The selected window cannot resolve a 20–150-year cycle.")
    band_frequency = frequency[in_band]
    band_power = power[in_band]
    spectral_period = float(1 / band_frequency[np.argmax(band_power)])
    search_period = float(manual_period) if manual_period > 0 else spectral_period
    radius = max(2, round(search_period * radius_fraction))
    finite = np.nan_to_num(smoothed, nan=-np.inf)
    candidates = find_peaks(finite, height=0)[0]
    peaks = np.asarray([
        k for k in candidates
        if radius <= k < yy.size - radius
        and finite[k] >= np.max(finite[k - radius:k + radius + 1])
    ], dtype=int)
    return CyclePreview(
        years=yy, raw_area=raw, detrended_area=processed,
        smoothed_area=smoothed, spectral_period=spectral_period,
        search_period=search_period, spectral_periods=1 / band_frequency,
        spectral_power=band_power, peak_indices=peaks,
        peak_radius_years=radius, detrend_method=detrend_method,
        smooth_years=smooth_years,
    )


def phase_sampling_schedule(
    peak_indices: np.ndarray,
    year_count: int,
    smooth_years: int,
    max_phase: tuple[float, float],
    min_before_phase: tuple[float, float],
    min_after_phase: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible peak indices and annual-index positions at nine phases."""
    windows = (max_phase, min_before_phase, min_after_phase)
    if any(a >= b or a < -1 or b > 1 for a, b in windows):
        raise ValueError("Phase ranges must be increasing and within one cycle.")
    if not (min_before_phase[1] < max_phase[0] < max_phase[1] < min_after_phase[0]):
        raise ValueError("The minimum and maximum phase windows must not overlap.")
    phases = np.concatenate([np.linspace(a, b, 3) for a, b in windows])
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size < 5:
        raise ValueError("At least five ice maxima are needed for three complete cycles.")
    event_peaks = peaks[1:-1]
    previous = event_peaks - peaks[:-2]
    following = peaks[2:] - event_peaks
    positions = event_peaks[:, None] + np.where(
        phases[None] < 0,
        phases[None] * previous[:, None],
        phases[None] * following[:, None],
    )
    half = smooth_years // 2
    valid = (positions.min(axis=1) >= half) & (
        positions.max(axis=1) <= year_count - 1 - half
    )
    event_peaks = event_peaks[valid]
    positions = positions[valid]
    if event_peaks.size < 3:
        raise ValueError("Fewer than three complete cycles remain in the selected years.")
    return event_peaks, positions


def compute_map_composite(
    archive: Path,
    preview: CyclePreview,
    max_phase: tuple[float, float] = DEFAULT_MAX_PHASE,
    min_before_phase: tuple[float, float] = DEFAULT_MIN_BEFORE_PHASE,
    min_after_phase: tuple[float, float] = DEFAULT_MIN_AFTER_PHASE,
    show_edge_cells: bool = True,
) -> CompositeMaps:
    """Read annual maps once for cached S1 differences and S3 snapshots."""
    archive = Path(archive).resolve()
    stat = archive.stat()
    return _compute_cached(
        str(archive), stat.st_size, stat.st_mtime_ns,
        int(preview.years[0]), int(preview.years[-1]),
        tuple(int(k) for k in preview.peak_indices),
        preview.detrend_method, preview.smooth_years,
        tuple(float(x) for x in max_phase),
        tuple(float(x) for x in min_before_phase),
        tuple(float(x) for x in min_after_phase),
        bool(show_edge_cells),
    )


@lru_cache(maxsize=4)
def _compute_cached(
    archive_name: str,
    archive_size: int,
    archive_mtime_ns: int,
    first_year: int,
    last_year: int,
    peaks: tuple[int, ...],
    detrend_method: str,
    smooth_years: int,
    max_phase: tuple[float, float],
    min_before_phase: tuple[float, float],
    min_after_phase: tuple[float, float],
    show_edge_cells: bool,
) -> CompositeMaps:
    del archive_size, archive_mtime_ns
    event_peaks, positions = phase_sampling_schedule(
        np.asarray(peaks), last_year - first_year + 1, smooth_years,
        max_phase, min_before_phase, min_after_phase,
    )
    with h5py.File(archive_name, "r") as source:
        all_years = np.asarray(source["year"][:], dtype=int)
        start = int(np.searchsorted(all_years, first_year))
        stop = int(np.searchsorted(all_years, last_year, side="right"))
        if stop - start != last_year - first_year + 1 or (
            all_years[start] != first_year or all_years[stop - 1] != last_year
        ):
            raise ValueError("Selected years are missing from the raw-map archive.")
        lat = np.asarray(source["t21_lat"][:], dtype=float)
        lon = np.asarray(source["t21_lon"][:], dtype=float)
        lsm = np.asarray(source["lsm"][:], dtype=float)
        lsg_lat = np.asarray(source["lat"][:], dtype=float)
        lsg_lon = np.asarray(source["lon"][:], dtype=float)
        vector_lat = np.asarray(source["lat_2"][:], dtype=float)
        vector_lon = np.asarray(source["lon_2"][:], dtype=float)
        u_mean = np.asarray(source["u_mean_150_300m"][:], dtype=float)
        v_mean = np.asarray(source["v_mean_150_300m"][:], dtype=float)
        wet = np.asarray(source["wet"][3:6], dtype=bool).any(axis=0)
        vector_wet = np.asarray(source["wetvec"][3:6], dtype=bool).any(axis=0)

        surface = np.asarray(
            source["surface_temperature"][start:stop], dtype=np.float32
        )
        surface_difference = _sample_difference(
            _preprocess_map(surface, detrend_method, smooth_years), positions
        )
        del surface

        sic = np.asarray(
            source["sea_ice_concentration"][start:stop], dtype=np.float32
        )
        mean_sic = np.mean(sic, axis=0, dtype=np.float64)
        if show_edge_cells:
            edge_lat, edge_lon = _edge_contributors(
                sic, lat, lon, lsm,
                float(np.median(np.diff(np.asarray(peaks)))),
            )
        else:
            edge_lat, edge_lon = np.array([]), np.array([])
        sic_difference = _sample_difference(
            _preprocess_map(sic, detrend_method, smooth_years), positions
        )
        sic_difference[lsm >= 0.5] = np.nan
        mean_sic[lsm >= 0.5] = np.nan
        del sic

        theta = np.asarray(
            source["theta_layer_150_300m"][start:stop], dtype=np.float32
        )
        processed_theta = _preprocess_map(theta, detrend_method, smooth_years)
        theta_difference = _sample_difference(processed_theta, positions)
        pathway_positions = _phase_positions_for_events(
            np.asarray(peaks), event_peaks, np.asarray(PATHWAY_PHASES)
        )
        snapshots = _sample_phase_maps(processed_theta, pathway_positions)
        theta_difference[~wet] = np.nan
        snapshots[:, ~wet] = np.nan
        current_snapshots = []
        for name in ("u_layer_150_300m", "v_layer_150_300m"):
            current = np.asarray(source[name][start:stop], dtype=np.float32)
            processed_current = _preprocess_map(current, detrend_method, smooth_years)
            phase_current = _sample_phase_maps(processed_current, pathway_positions)
            phase_current[:, ~vector_wet] = np.nan
            current_snapshots.append(phase_current)
    return CompositeMaps(
        t21_lat=lat, t21_lon=lon, lsm=lsm,
        surface_temperature_difference=surface_difference,
        sea_ice_concentration_difference=sic_difference,
        mean_sea_ice_concentration=mean_sic,
        lsg_lat=lsg_lat, lsg_lon=lsg_lon, wet_150_300m=wet,
        theta_150_300m_difference=theta_difference,
        pathway_phases=PATHWAY_PHASES,
        theta_150_300m_snapshots=snapshots,
        lsg_vector_lat=vector_lat, lsg_vector_lon=vector_lon,
        u_mean_150_300m=u_mean, v_mean_150_300m=v_mean,
        u_150_300m_snapshots=current_snapshots[0],
        v_150_300m_snapshots=current_snapshots[1],
        edge_lat=edge_lat, edge_lon=edge_lon,
        detected_peaks=len(peaks), composited_events=len(event_peaks),
        median_cycle_years=float(np.median(np.diff(np.asarray(peaks)))),
    )


def _preprocess_map(
    values: np.ndarray, detrend_method: str, smooth_years: int
) -> np.ndarray:
    """Process a map field in place, retaining only one full field in memory."""
    n = len(values)
    valid_cells = np.isfinite(values).all(axis=0)
    values[:, ~valid_cells] = 0
    if detrend_method == "linear":
        center = (n - 1) / 2
        mean = np.mean(values, axis=0, dtype=np.float64)
        sum_tx = np.zeros(values.shape[1:], dtype=np.float64)
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            time = np.arange(start, stop, dtype=np.float64) - center
            sum_tx += np.einsum(
                "t,tij->ij", time, values[start:stop], optimize=True
            )
        variance_t = n * (n**2 - 1) / 12
        slope = sum_tx / variance_t
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            time = np.arange(start, stop, dtype=np.float64) - center
            values[start:stop] -= (mean[None] + time[:, None, None] * slope[None]).astype(
                np.float32
            )
    elif detrend_method == "none":
        mean = np.mean(values, axis=0, dtype=np.float64)
        values -= mean.astype(np.float32)[None]
    else:
        raise ValueError("Detrending must be 'linear' or 'none'.")
    if smooth_years > 1:
        values = uniform_filter1d(values, size=smooth_years, axis=0, mode="nearest")
    values[:, ~valid_cells] = np.nan
    return values


def _sample_difference(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    maximum = np.zeros(values.shape[1:], dtype=np.float64)
    minimum = np.zeros(values.shape[1:], dtype=np.float64)
    for j in range(positions.shape[1]):
        lower = np.floor(positions[:, j]).astype(int)
        upper = np.minimum(lower + 1, len(values) - 1)
        fraction = (positions[:, j] - lower).astype(np.float32)
        sampled = (
            values[lower] * (1 - fraction)[:, None, None]
            + values[upper] * fraction[:, None, None]
        )
        count = np.isfinite(sampled).sum(axis=0)
        mean = np.divide(
            np.nansum(sampled, axis=0, dtype=np.float64), count,
            out=np.full(values.shape[1:], np.nan, dtype=np.float64), where=count > 0,
        )
        if j < 3:
            maximum += mean / 3
        else:
            minimum += mean / 6
    return (maximum - minimum).astype(np.float32)


def _phase_positions_for_events(
    peaks: np.ndarray,
    event_peaks: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Locate each requested cycle phase around the same S1 event maxima."""
    center_indices = np.searchsorted(peaks, event_peaks)
    previous = event_peaks - peaks[center_indices - 1]
    following = peaks[center_indices + 1] - event_peaks
    return event_peaks[:, None] + np.where(
        phases[None] < 0,
        phases[None] * previous[:, None],
        phases[None] * following[:, None],
    )


def _sample_phase_maps(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate annual maps and average events at each cycle phase."""
    output = np.empty((positions.shape[1], *values.shape[1:]), dtype=np.float32)
    for phase_index in range(positions.shape[1]):
        lower = np.floor(positions[:, phase_index]).astype(int)
        upper = np.minimum(lower + 1, len(values) - 1)
        fraction = (positions[:, phase_index] - lower).astype(np.float32)
        sampled = (
            values[lower] * (1 - fraction)[:, None, None]
            + values[upper] * fraction[:, None, None]
        )
        count = np.isfinite(sampled).sum(axis=0)
        output[phase_index] = np.divide(
            np.nansum(sampled, axis=0, dtype=np.float64), count,
            out=np.full(values.shape[1:], np.nan, dtype=np.float64), where=count > 0,
        )
    return output


def _edge_contributors(
    sic: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lsm: np.ndarray,
    cycle_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank SH ocean cells by covariance with the period-band ice-area cycle."""
    nodes, gaussian = np.polynomial.legendre.leggauss(len(lat))
    if not np.allclose(lat, np.degrees(np.arcsin(nodes))[::-1], atol=1e-4):
        raise ValueError("T21 latitude does not match Gaussian-grid area weights.")
    area = gaussian[::-1, None] * np.ones_like(lsm)
    area *= 2 * np.pi * EARTH_RADIUS_M**2 / lsm.shape[1]
    mask = (lat[:, None] < 0) & (lsm < 0.5) & np.isfinite(sic).all(axis=0)
    cell_indices = np.flatnonzero(mask.ravel())
    weights = area.ravel()[cell_indices]
    cells = sic.reshape(len(sic), -1)[:, cell_indices]
    total = np.asarray(cells @ weights, dtype=np.float64)
    low_period = max(10.0, 0.4 * cycle_years)
    high_period = 3.0 * cycle_years
    frequency = np.fft.rfftfreq(len(sic))
    band = (frequency >= 1 / high_period) & (frequency <= 1 / low_period)
    if not band.any():
        return np.array([]), np.array([])

    def bandpass(series: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(series - np.mean(series, axis=0), axis=0)
        spectrum[~band] = 0
        return np.fft.irfft(spectrum, n=len(sic), axis=0)

    total_cycle = bandpass(total)
    variance = np.var(total_cycle)
    if variance <= 0:
        return np.array([]), np.array([])
    contribution = np.empty(len(weights), dtype=float)
    for start in range(0, len(weights), 64):
        stop = min(start + 64, len(weights))
        cell_cycle = bandpass(cells[:, start:stop] * weights[None, start:stop])
        contribution[start:stop] = np.mean(
            cell_cycle * total_cycle[:, None], axis=0
        ) / variance
    order = np.argsort(-contribution)
    count = min(len(order), np.searchsorted(np.cumsum(contribution[order]), 0.5) + 1)
    top = cell_indices[order[:count]]
    latitude, longitude = np.meshgrid(lat, lon, indexing="ij")
    return latitude.ravel()[top], longitude.ravel()[top]
