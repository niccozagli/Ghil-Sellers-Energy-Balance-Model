"""Single-lag Koopman analysis of South Atlantic PlaSim fields.

The Marimo app is ``analysis/Koopman_analysis.py``. This module can
also be run directly with ``python -m gsebm.plasim_koopman_single`` to save
the figure. Its spectrum panel shows rates at the selected lag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist

from gsebm.plasim_raw_maps import RAW_MAP_ROOTS
from koopman_response import KoopmanSpectrumKDMD
from koopman_response.algorithms import KernelDMD, WeightedGaussianKernel
from koopman_response.algorithms.regularization import TSVDRegularizer


MU = "1240"
EXPERIMENT_PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
PHASE_MAP = LinearSegmentedColormap.from_list(
    "koopman_phase",
    ["#1f3b8c", "#2a9d8f", "#e9a23b", "#c2185b", "#1f3b8c"],
    N=256,
)


def available_mu_values(root: Path) -> list[str]:
    """List μ values under ``root`` with both a raw-map archive and its basin masks."""
    values = []
    for path in root.glob(f"{EXPERIMENT_PREFIX}*/*_spinup_raw_maps.nc"):
        experiment = path.parent.name
        if path.name != f"{experiment}_spinup_raw_maps.nc":
            continue
        if path.with_name(f"{experiment}_spinup_basin_masks.nc").is_file():
            values.append(experiment.removeprefix(EXPERIMENT_PREFIX))
    # Variants such as "1235_new_IC" sort after the plain run at the same μ.
    return sorted(
        values,
        key=lambda value: (float(value.split("_", 1)[0].replace("p", ".")), value),
    )


def source_file(root: Path, mu: str = MU) -> Path:
    """Return the raw-map archive for the selected PlaSim μ under ``root``."""
    experiment = f"{EXPERIMENT_PREFIX}{mu}"
    path = root / experiment / f"{experiment}_spinup_raw_maps.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mu={mu} raw-map archive: {path}")
    return path


def basin_mask_file(root: Path, mu: str = MU) -> Path:
    """Return the basin masks extracted alongside the selected raw maps."""
    experiment = f"{EXPERIMENT_PREFIX}{mu}"
    path = source_file(root, mu).with_name(f"{experiment}_spinup_basin_masks.nc")
    if not path.is_file():
        raise FileNotFoundError(f"Missing mu={mu} basin masks: {path}")
    return path


def load_fields(
    root: Path, start_year: int, maximum_depth: float, mu: str = MU,
    end_year: int | None = None, detrend_ocean: bool = True,
) -> dict[str, np.ndarray]:
    """Build South Atlantic zonal temperature states from native annual maps.

    Years after ``start_year`` are used, through ``end_year`` when given.
    Ocean rows are linearly detrended unless ``detrend_ocean`` is False; the
    undetrended row temperatures are always returned as ``ocean_state``.
    """
    with xr.open_dataset(source_file(root, mu)) as dataset, xr.open_dataset(
        basin_mask_file(root, mu)
    ) as masks:
        required = {
            "depth_bounds", "lat", "lsg_horizontal_area",
            "sea_ice_concentration", "surface_temperature", "t21_lat",
            "t21_lon", "temperature_upper", "wet_cell_volume", "year",
        }
        missing = required - set(dataset.variables)
        if missing:
            raise ValueError(f"The mu={mu} raw-map archive lacks {sorted(missing)}")
        mask_names = {"lsg_scalar_south_atlantic", "t21_south_atlantic"}
        missing_masks = mask_names - set(masks.variables)
        if missing_masks:
            raise ValueError(f"The mu={mu} basin masks lack {sorted(missing_masks)}")

        all_years = np.asarray(dataset.year.values, dtype=int)
        # Select by label rather than searchsorted: corrupt blocks elsewhere in
        # the archive can carry wrong (unsorted) year labels.
        selected = (all_years > start_year) & (
            True if end_year is None else all_years <= end_year
        )
        if not selected.any():
            raise ValueError("No annual records in the requested window")
        first = int(np.flatnonzero(selected)[0])
        last = int(np.flatnonzero(selected)[-1]) + 1
        years = all_years[first:last]
        if years.size < 100 or not np.all(np.diff(years) == 1):
            raise ValueError("The stationary annual record is too short or discontinuous")

        all_surface_lat = np.asarray(dataset.t21_lat.values, dtype=float)
        nodes, gaussian_weights = np.polynomial.legendre.leggauss(all_surface_lat.size)
        if not np.allclose(all_surface_lat, np.degrees(np.arcsin(nodes))[::-1]):
            raise ValueError("Unexpected PlaSim Gaussian latitude grid")
        if not (
            np.array_equal(masks.t21_lat.values, dataset.t21_lat.values)
            and np.array_equal(masks.t21_lon.values, dataset.t21_lon.values)
        ):
            raise ValueError("South Atlantic T21 mask coordinates do not match raw maps")
        ice_mask = np.asarray(masks.t21_south_atlantic.values, dtype=bool)
        native_mask = np.asarray(masks.lsg_scalar_south_atlantic.values, dtype=bool)
        if ice_mask.shape != (
            all_surface_lat.size, dataset.sizes["t21_lon"]
        ) or native_mask.shape != (
            dataset.sizes["south_north"], dataset.sizes["west_east"]
        ):
            raise ValueError("South Atlantic basin masks do not match the raw grids")
        surface_rows = ice_mask.any(axis=1)
        native_latitude = np.asarray(dataset.lat.isel(west_east=0).values, dtype=float)
        native_rows = np.flatnonzero(native_mask.any(axis=1))
        if not surface_rows.any() or native_rows.size == 0 or not ice_mask.any():
            raise ValueError("South Atlantic basin masks contain no wet rows")
        if np.any(all_surface_lat[surface_rows] >= 0.0) or np.any(
            native_latitude[native_rows] >= 0.0
        ):
            raise ValueError("South Atlantic basin masks include northern rows")

        depth_bounds = np.asarray(dataset.depth_bounds.values, dtype=float)
        levels = np.flatnonzero(depth_bounds[:, 1] <= maximum_depth)
        thickness = depth_bounds[levels, 1] - depth_bounds[levels, 0]
        if (
            not levels.size
            or levels[-1] >= dataset.sizes["upper_depth"]
            or not np.isclose(thickness.sum(), maximum_depth)
        ):
            raise ValueError("Selected native ocean layers do not span the requested depth")

        sector_surface = ice_mask[surface_rows]
        surface_count = sector_surface.sum(axis=1)
        surface_maps = np.asarray(
            dataset.surface_temperature.isel(
                year=slice(first, last), t21_lat=surface_rows
            ).values,
            dtype=float,
        )
        if not np.isfinite(surface_maps[:, sector_surface]).all():
            raise ValueError("Non-finite South Atlantic surface temperatures")
        surface = np.sum(
            np.where(sector_surface[None], surface_maps, 0.0), axis=2
        ) / surface_count[None]
        surface_lat = all_surface_lat[surface_rows]
        surface_weights = gaussian_weights[::-1][surface_rows] * surface_count
        surface_weights = surface_weights / surface_weights.sum()

        ice_concentration = np.asarray(
            dataset.sea_ice_concentration.isel(
                year=slice(first, last), t21_lat=surface_rows
            ).values,
            dtype=float,
        )
        if not np.isfinite(ice_concentration[:, sector_surface]).all():
            raise ValueError("Non-finite South Atlantic sea-ice concentration")
        ice_cell_area = (
            gaussian_weights[::-1][surface_rows, None]
            * sector_surface
            * (2.0 * np.pi * 6.371e6**2 / dataset.sizes["t21_lon"])
        )
        ice_area = np.nansum(
            ice_concentration * ice_cell_area[None], axis=(1, 2)
        ) / 1.0e12  # m² to 10^6 km²
        ice_rows = np.nansum(
            ice_concentration * ice_cell_area[None], axis=2
        ) / 1.0e12  # basin ice area per T21 row, 10^6 km²
        ice_row_area = ice_cell_area.sum(axis=1)

        sector_native = native_mask[native_rows]
        wet_volume = np.asarray(
            dataset.wet_cell_volume.isel(
                lsg_depth=levels, south_north=native_rows
            ).values,
            dtype=float,
        )
        if not np.isfinite(wet_volume).all() or np.any(wet_volume < 0.0):
            raise ValueError("Invalid South Atlantic wet-cell volumes")
        wet = (wet_volume > 0.0) & sector_native[None]
        volume_weights = np.where(wet, wet_volume, 0.0)
        zonal_volume = volume_weights.sum(axis=2)
        depth_weights = np.where(zonal_volume > 0.0, thickness[:, None], 0.0)
        wet_depth = depth_weights.sum(axis=0)
        surface_wet = np.asarray(
            dataset.wet_cell_volume.isel(
                lsg_depth=0, south_north=native_rows
            ).values
        ) > 0.0
        ocean_area = np.sum(
            np.asarray(
                dataset.lsg_horizontal_area.isel(south_north=native_rows).values,
                dtype=float,
            ) * sector_native * surface_wet,
            axis=1,
        )
        keep_rows = (wet_depth > 0.0) & (ocean_area > 0.0)
        if not keep_rows.any():
            raise ValueError("No wet South Atlantic upper-ocean rows were found")
        native_rows = native_rows[keep_rows]
        wet = wet[:, keep_rows]
        volume_weights = volume_weights[:, keep_rows]
        zonal_volume = zonal_volume[:, keep_rows]
        depth_weights = depth_weights[:, keep_rows]
        wet_depth = wet_depth[keep_rows]
        ocean_area = ocean_area[keep_rows]
        ocean_lat = native_latitude[native_rows]
        ocean_state = np.empty((years.size, native_rows.size), dtype=float)

        # The native temperature is a four-dimensional map, so reduce it to
        # basin-specific latitude profiles in bounded annual batches.
        for begin in range(0, years.size, 64):
            stop = min(begin + 64, years.size)
            temperature = np.asarray(
                dataset.temperature_upper.isel(
                    year=slice(first + begin, first + stop),
                    upper_depth=levels, south_north=native_rows,
                ).values,
                dtype=float,
            )
            if not np.isfinite(temperature[:, wet]).all():
                raise ValueError("Non-finite wet South Atlantic ocean temperatures")
            zonal_sum = np.einsum(
                "tdrl,drl->tdr",
                np.where(wet[None], temperature, 0.0),
                volume_weights,
                optimize=True,
            )
            zonal_mean = np.divide(
                zonal_sum, zonal_volume[None],
                out=np.zeros_like(zonal_sum),
                where=zonal_volume[None] > 0.0,
            )
            ocean_state[begin:stop] = (
                np.sum(zonal_mean * depth_weights[None], axis=1)
                / wet_depth[None]
            )

    order = np.argsort(ocean_lat)
    ocean_lat = ocean_lat[order]
    ocean_state = ocean_state[:, order]
    ocean_area = ocean_area[order]
    surface_anomaly = surface - surface.mean(axis=0)
    ocean_weights = ocean_area / ocean_area.sum()
    centred_year = years.astype(float) - years.mean()
    ocean_mean = ocean_state.mean(axis=0)
    slope = centred_year @ (ocean_state - ocean_mean) / (centred_year @ centred_year)
    ocean_anomaly = ocean_state - ocean_mean - (centred_year[:, None] * slope if detrend_ocean else 0.0)
    state = np.column_stack((surface_anomaly, ocean_anomaly))
    if not np.isfinite(state).all() or not np.isfinite(ice_area).all():
        raise ValueError("Non-finite Koopman states or South Atlantic ice area")
    return {
        "mu": mu,
        "maximum_depth": maximum_depth,
        "years": years,
        "state": state,
        "surface_anomaly": surface_anomaly,
        "surface_state": surface,
        "surface_lat": surface_lat,
        "surface_weights": surface_weights,
        "ocean_anomaly": ocean_anomaly,
        "ocean_state": ocean_state,
        "ocean_lat": ocean_lat,
        "ocean_weights": ocean_weights,
        "ice_area": ice_area,
        "ice_anomaly": ice_area - ice_area.mean(),
        "ice_rows_anomaly": ice_rows - ice_rows.mean(axis=0),
        "ice_row_lat": surface_lat,
        "ice_row_weights": ice_row_area / ice_row_area.sum(),
    }


STATE_BLOCKS = {
    # name: (values key, latitude key, weights key)
    "surface": ("surface_anomaly", "surface_lat", "surface_weights"),
    "ocean": ("ocean_anomaly", "ocean_lat", "ocean_weights"),
    "ice": ("ice_rows_anomaly", "ice_row_lat", "ice_row_weights"),
}


def combined_state(
    fields: dict[str, np.ndarray], blocks: tuple[str, ...] = ("surface", "ocean"),
) -> dict[str, np.ndarray]:
    """Copy of ``fields`` whose Koopman state is the given blocks side by side.

    ``"ice"`` is the basin sea-ice area per T21 row; rows whose ice never
    varies are dropped. Each block keeps its own weights, and ``fit_koopman``
    scales each block by its own median pairwise distance.
    """
    columns, layout, start = [], [], 0
    for name in blocks:
        values_key, lat_key, weights_key = STATE_BLOCKS[name]
        values = np.asarray(fields[values_key], dtype=float)
        weights = np.asarray(fields[weights_key], dtype=float)
        keep = values.std(axis=0) > 1e-9
        if not keep.any():
            raise ValueError(f"The {name} block has no varying rows")
        values, weights = values[:, keep], weights[keep]
        columns.append(values)
        layout.append((name, start, start + values.shape[1], weights / weights.sum(),
                       np.asarray(fields[lat_key])[keep]))
        start += values.shape[1]
    return {**fields, "state": np.column_stack(columns), "state_blocks": layout}


def kdmd_training_indices(
    state_count: int, lag: int, maximum_training_snapshots: int, seed: int,
) -> np.ndarray:
    """Reproduce the snapshot origins used by the single KDMD fit."""
    count = state_count - lag
    if count < 2 or maximum_training_snapshots < 2:
        raise ValueError("Too few lagged snapshot pairs")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(
        count, min(count, maximum_training_snapshots), replace=False,
    ))


def extract_eigenmode(
    spectrum: KoopmanSpectrumKDMD,
    rates: np.ndarray,
    fields: dict[str, np.ndarray],
    index: int,
    *, lag: int, maximum_training_snapshots: int, seed: int,
    batch_size: int = 256,
) -> dict:
    """Return one KDMD eigenfunction and its direct observable modes."""
    if not 0 <= index < rates.size:
        raise IndexError(f"Koopman mode index {index} is outside 0..{rates.size - 1}")
    indices = kdmd_training_indices(
        fields["state"].shape[0], lag, maximum_training_snapshots, seed,
    )
    state = fields["state"]
    # Evaluate only the chosen eigenfunction instead of the full spectrum.
    coefficient = spectrum.U_r @ (
        spectrum.right_eigvecs[:, index] / np.sqrt(spectrum.S_r)
    )
    raw_psi = np.concatenate([
        spectrum.kernel(
            state[begin:begin + batch_size], spectrum.reference_data,
        ) @ coefficient
        for begin in range(0, state.shape[0], batch_size)
    ])
    rms = float(np.sqrt(np.mean(np.abs(raw_psi)**2)))
    if rms == 0 or not np.isfinite(rms):
        raise ValueError("The selected eigenfunction has zero or invalid RMS")
    is_real_mode = bool(abs(rates[index].imag) < 1e-10)
    rotation = (
        np.exp(-0.5j * np.angle(np.mean(raw_psi**2)))
        if is_real_mode else 1.0 + 0.0j
    )
    scale = rotation / rms
    psi = raw_psi * scale

    observables = np.column_stack((fields["ice_anomaly"], state))
    mode = observable_modes(spectrum, state, observables, indices, [index], [scale])[0]
    n_surface = fields["surface_lat"].size
    return {
        "index": index,
        "eigenvalue": rates[index],
        "eigenfunction": psi,
        "ice_mode": mode[0],
        "surface_mode": mode[1:1 + n_surface],
        "ocean_mode": mode[1 + n_surface:],
        "is_real_mode": is_real_mode,
        "mode_source": "direct KDMD modes",
    }


def fit_koopman(
    fields: dict[str, np.ndarray], lag: int, rank_threshold: float,
    maximum_training_snapshots: int, seed: int,
    factorization_rel_threshold: float = 1e-5,
) -> tuple[KoopmanSpectrumKDMD, np.ndarray, int, tuple[float, float]]:
    """Fit one joint Gaussian-kernel KDMD operator at the requested lag.

    The state is ``fields["state"]``; its blocks come from
    ``fields["state_blocks"]`` (see ``combined_state``) or default to the
    zonal Ts and ocean rows. Each block is scaled by its median pairwise
    distance.
    """
    state = fields["state"]
    indices = kdmd_training_indices(
        state.shape[0], lag, maximum_training_snapshots, seed,
    )
    origin, target = state[indices], state[indices + lag]
    layout = fields.get("state_blocks")
    if layout is None:
        surface_count = fields["surface_anomaly"].shape[1]
        layout = [
            ("surface", 0, surface_count, fields["surface_weights"], None),
            ("ocean", surface_count, state.shape[1], fields["ocean_weights"], None),
        ]
    bandwidths, kernel_weights = [], []
    for _, begin, end, weights, _ in layout:
        bandwidth = float(np.median(pdist(origin[:, begin:end] * np.sqrt(weights))))
        if bandwidth <= 0:
            raise ValueError("A kernel block has zero median pairwise distance")
        bandwidths.append(bandwidth)
        kernel_weights.append(weights / bandwidth**2)
    kernel = WeightedGaussianKernel(sigma=1.0, weights=np.concatenate(kernel_weights))
    kdmd = KernelDMD(kernel=kernel)
    kdmd.fit_snapshots(X=origin, Y=target)
    factorization = TSVDRegularizer()
    factorization.factorize(kdmd.G, method="eigh", symmetrize=False,
                            rel_threshold=factorization_rel_threshold)
    matrix, basis, singular_values = factorization.solve_from_factorization(
        kdmd.A, rel_threshold=rank_threshold,
    )
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        matrix, kernel=kdmd.kernel, reference_data=kdmd.reference_data,
        U_r=basis, S_r=singular_values,
    )
    rates = spectrum.continuous_time_eigenvalues(lag)
    return spectrum, rates, singular_values.size, tuple(bandwidths)


def leading_eigenfunction(
    spectrum: KoopmanSpectrumKDMD, rates: np.ndarray,
    fields: dict[str, np.ndarray], batch_size: int = 256,
) -> tuple[int, np.ndarray]:
    """Choose the leading stable oscillation and put peak ice at phase zero."""
    candidates = np.flatnonzero((rates.imag > 1e-10) & (rates.real < 0))
    if not candidates.size:
        raise ValueError("No stable complex Koopman eigenvalue found")
    index = int(candidates[np.argmax(rates.real[candidates])])
    psi = np.array(spectrum.evaluate_eigenfunctions(
        fields["state"], batch_size=batch_size,
    )[:, index], dtype=complex, copy=True)
    psi /= np.sqrt(np.mean(np.abs(psi)**2))
    return index, align_ice_peak(psi, fields["ice_anomaly"])


def leading_and_harmonic_eigenfunctions(
    spectrum: KoopmanSpectrumKDMD, rates: np.ndarray,
    fields: dict[str, np.ndarray], batch_size: int = 256,
) -> tuple[int, np.ndarray, int, np.ndarray, np.ndarray]:
    """Evaluate the leading mode and the mode nearest twice its complex rate."""
    candidates = np.flatnonzero((rates.imag > 1e-10) & (rates.real < 0))
    if candidates.size < 2:
        raise ValueError("Two stable complex Koopman eigenvalues are required")
    leading_index = int(candidates[np.argmax(rates.real[candidates])])
    target_rate = 2 * rates[leading_index]
    harmonic_candidates = candidates[
        (candidates != leading_index)
        & (np.abs(rates[candidates].imag - target_rate.imag)
           <= 0.1 * target_rate.imag)
    ]
    if not harmonic_candidates.size:
        raise ValueError("No stable mode lies within 10% of twice the leading frequency")
    harmonic_index = int(harmonic_candidates[
        np.argmin(np.abs(rates[harmonic_candidates] - target_rate))
    ])
    evaluated = spectrum.evaluate_eigenfunctions(
        fields["state"], batch_size=batch_size,
    )
    raw_leading = np.array(evaluated[:, leading_index], dtype=complex, copy=True)
    raw_harmonic = np.array(evaluated[:, harmonic_index], dtype=complex, copy=True)
    del evaluated
    leading_psi = raw_leading / np.sqrt(np.mean(np.abs(raw_leading)**2))
    harmonic_psi = raw_harmonic / np.sqrt(np.mean(np.abs(raw_harmonic)**2))
    leading_psi = align_ice_peak(leading_psi, fields["ice_anomaly"])
    alignment = np.mean(leading_psi**2 * np.conj(harmonic_psi))
    if abs(alignment) == 0:
        raise ValueError("The selected harmonic cannot be phase-aligned")
    harmonic_psi *= np.exp(1j * np.angle(alignment))
    mode_scales = np.array([
        np.vdot(raw_leading, leading_psi) / np.vdot(raw_leading, raw_leading),
        np.vdot(raw_harmonic, harmonic_psi) / np.vdot(raw_harmonic, raw_harmonic),
    ])
    return leading_index, leading_psi, harmonic_index, harmonic_psi, mode_scales


def align_ice_peak(psi: np.ndarray, ice_anomaly: np.ndarray) -> np.ndarray:
    """Rotate psi so the peak ice composite is in the phase-zero bin."""
    phase = np.mod(np.angle(psi), 2 * np.pi)
    centres, ice_mean, _ = phase_composite(ice_anomaly, phase)
    peak_bin = int(np.nanargmax(ice_mean))
    if peak_bin in (0, ice_mean.size - 1):
        return psi
    return psi * np.exp(-1j * centres[peak_bin])


def mode_coefficients(psi: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """Project each centered zonal observable onto unit-RMS psi."""
    return np.mean(np.conj(psi)[:, None] * anomaly, axis=0)


def phase_composite(values: np.ndarray, phase: np.ndarray, bins: int = 36):
    """Return a circular phase composite and its within-bin spread."""
    index = np.minimum((phase / (2 * np.pi) * bins).astype(int), bins - 1)
    mean = np.asarray([values[index == k].mean(axis=0) for k in range(bins)])
    spread = np.asarray([values[index == k].std(axis=0) for k in range(bins)])
    centres = (np.arange(bins) + 0.5) * 2 * np.pi / bins
    return centres, mean, spread


def composite_fourier(composite: np.ndarray, centres: np.ndarray, harmonic: int):
    """Return B_l = <C(phi) exp(-i l phi)> over the phase bins (axis 0)."""
    weights = np.exp(-1j * harmonic * centres)
    composite = np.asarray(composite)
    return np.mean(composite * weights.reshape((-1,) + (1,) * (composite.ndim - 1)), axis=0)


def observable_modes(
    spectrum: KoopmanSpectrumKDMD,
    state: np.ndarray,
    observables: np.ndarray,
    indices: np.ndarray,
    mode_indices: list[int],
    mode_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Return direct KDMD modes of observables for the selected eigenfunctions.

    This is ``spectrum.koopman_modes`` for every column at once,
    ``a = W^* S_r^{-1/2} U_r^* G``, with ``G`` sampled at the KDMD training
    states ``state[indices]``. ``observables`` has one row per year and any
    trailing shape. Dividing by ``mode_scales`` makes each mode multiply the
    normalized eigenfunction, so the observable part is ``2 Re(a psi)``.
    """
    if not np.allclose(
        spectrum.reference_data, state[indices], rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("KDMD reference states do not match the training indices")
    values = np.asarray(observables)
    training = values.reshape(values.shape[0], -1)[indices]
    projection = spectrum.U_r.conj().T @ training / np.sqrt(spectrum.S_r)[:, None]
    modes = spectrum.left_eigvecs[:, mode_indices].conj().T @ projection
    if mode_scales is not None:
        modes = modes / np.asarray(mode_scales)[:, None]
    return modes.reshape((len(mode_indices),) + values.shape[1:])


def regression_modes(
    leading_psi: np.ndarray, harmonic_psi: np.ndarray, observables: np.ndarray,
) -> np.ndarray:
    """Fit observables on [1, psi1, psi2, conj(psi1), conj(psi2)] by least squares.

    Returns the psi1 and psi2 coefficients, shape ``(2, ...)``. Fitting both
    eigenfunctions and their conjugates jointly avoids leakage between them.
    """
    values = np.asarray(observables)
    design = np.column_stack((
        np.ones(leading_psi.size), leading_psi, harmonic_psi,
        np.conj(leading_psi), np.conj(harmonic_psi),
    ))
    coefficients = np.linalg.lstsq(
        design, values.reshape(values.shape[0], -1), rcond=None,
    )[0][1:3]
    return coefficients.reshape((2,) + values.shape[1:])


def time_to_peak(modes: np.ndarray, omega: float) -> np.ndarray:
    """Years after phase zero at which 2 Re(a psi) peaks, wrapped to [-P/2, P/2)."""
    period = 2 * np.pi / omega
    return np.mod(-np.angle(modes) / omega + period / 2, period) - period / 2


def pattern_correlation(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    """Weighted complex pattern correlation |<a, b>| / (|a| |b|)."""
    numerator = abs(np.sum(weights * first * np.conj(second)))
    return float(numerator / np.sqrt(
        np.sum(weights * np.abs(first)**2) * np.sum(weights * np.abs(second)**2)
    ))


def state_mode_comparison(
    spectrum: KoopmanSpectrumKDMD,
    fields: dict[str, np.ndarray],
    rates: np.ndarray,
    leading_index: int,
    leading_psi: np.ndarray,
    harmonic_index: int,
    harmonic_psi: np.ndarray,
    mode_scales: np.ndarray,
    *, lag: int, maximum_training_snapshots: int, seed: int,
) -> dict:
    """Compare three mode estimates for the zonal Ts and ocean state rows.

    ``direct`` are KDMD modes, ``regression`` the joint least-squares fit on
    both eigenfunctions, and ``composite`` the first Fourier coefficient of the
    phase composite divided by mean |psi1|, which estimates a1 without a linear
    model. Ocean values are in mK.
    """
    indices = kdmd_training_indices(
        fields["state"].shape[0], lag, maximum_training_snapshots, seed,
    )
    omega = float(rates[leading_index].imag)
    phase = np.mod(np.angle(leading_psi), 2 * np.pi)
    mean_amplitude = float(np.mean(np.abs(leading_psi)))
    blocks = {}
    for name, values, lat, weights, scale in (
        ("surface", fields["surface_anomaly"], fields["surface_lat"],
         fields["surface_weights"], 1.0),
        ("ocean", fields["ocean_anomaly"], fields["ocean_lat"],
         fields["ocean_weights"], 1e3),
    ):
        direct = scale * observable_modes(
            spectrum, fields["state"], values, indices,
            [leading_index, harmonic_index], mode_scales,
        )
        regression = scale * regression_modes(leading_psi, harmonic_psi, values)
        centres, composite, _ = phase_composite(scale * values, phase)
        composite_mode = composite_fourier(composite, centres, 1) / mean_amplitude
        blocks[name] = {
            "lat": lat, "weights": weights,
            "direct": direct, "regression": regression, "composite": composite_mode,
            "centres": centres, "observed": composite,
            "correlation": pattern_correlation(direct[0], regression[0], weights),
            "composite_correlation": pattern_correlation(
                direct[0], composite_mode, weights,
            ),
        }
    return {
        "mu": fields["mu"], "omega": omega, "period": 2 * np.pi / omega,
        "harmonic_omega": float(rates[harmonic_index].imag), **blocks,
    }


# Map observables outside the Koopman state: archive name, label, unit,
# display scale, grid, and whether to remove a linear trend (the ocean rows of
# the state are detrended, the surface rows are not).
MAP_OBSERVABLES = (
    ("surface_temperature", "Surface T", "K", 1.0, "t21", False),
    ("sea_ice_concentration", "Sea-ice concentration", "%", 100.0, "t21", False),
    ("theta_layer_0_100m", "θ 0–100 m", "mK", 1e3, "lsg", True),
    ("theta_layer_150_300m", "θ 150–300 m", "mK", 1e3, "lsg", True),
    ("theta_layer_300_600m", "θ 300–600 m", "mK", 1e3, "lsg", True),
    # Barotropic streamfunction, native LSG sign (not the eigenfunctions ψ).
    ("barotropic_streamfunction", "Ψ_BT", "Sv", 1.0, "lsg", True),
)


def load_map_fields(
    root: Path, mu: str, years: np.ndarray,
    names: tuple[str, ...] | None = None,
) -> dict:
    """Read annual lat-lon maps on the Koopman years as anomalies.

    Each field has its time mean removed, and the ocean layers also a linear
    trend, matching the state convention. Dry LSG cells are NaN in the archive;
    they are stored as zero anomalies with ``wet`` False so that mode
    projections stay finite.
    """
    selected = [item for item in MAP_OBSERVABLES if names is None or item[0] in names]
    years = np.asarray(years, dtype=int)
    with h5py.File(source_file(root, mu), "r") as source:
        all_years = np.asarray(source["year"][:], dtype=int)
        first = int(np.searchsorted(all_years, years[0]))
        stop = first + years.size
        if not np.array_equal(all_years[first:stop], years):
            raise ValueError("Map years do not match the Koopman state years")
        maps = {
            "t21_lat": np.asarray(source["t21_lat"][:], dtype=float),
            "t21_lon": np.asarray(source["t21_lon"][:], dtype=float),
            "lsm": np.asarray(source["lsm"][:], dtype=float),
            "lsg_lat": np.asarray(source["lat"][:], dtype=float),
            "lsg_lon": np.asarray(source["lon"][:], dtype=float),
            "fields": {},
        }
        with h5py.File(basin_mask_file(root, mu), "r") as basins:
            maps["t21_basin"] = np.asarray(basins["t21_south_atlantic"][:], dtype=bool)
            maps["lsg_basin"] = np.asarray(basins["lsg_scalar_south_atlantic"][:], dtype=bool)
        centred_year = years.astype(float) - years.mean()
        for name, label, unit, scale, grid, detrend in selected:
            values = np.asarray(source[name][first:stop], dtype=float)
            wet = np.isfinite(values).all(axis=0)
            values = np.where(wet[None], values, 0.0)
            values -= values.mean(axis=0)
            if detrend:
                slope = np.tensordot(centred_year, values, axes=1) / (centred_year @ centred_year)
                values -= centred_year[:, None, None] * slope[None]
            maps["fields"][name] = {
                "label": label, "unit": unit, "scale": scale, "grid": grid,
                "wet": wet, "anomaly": values.astype(np.float32),
            }
    return maps


def map_modes(
    spectrum: KoopmanSpectrumKDMD,
    fields: dict[str, np.ndarray],
    maps: dict,
    leading_index: int,
    harmonic_index: int,
    mode_scales: np.ndarray,
    *, lag: int, maximum_training_snapshots: int, seed: int,
) -> dict[str, np.ndarray]:
    """Return direct KDMD modes (2, lat, lon) of each map, in display units."""
    indices = kdmd_training_indices(
        fields["state"].shape[0], lag, maximum_training_snapshots, seed,
    )
    modes = {}
    for name, field in maps["fields"].items():
        mode = field["scale"] * observable_modes(
            spectrum, fields["state"], field["anomaly"], indices,
            [leading_index, harmonic_index], mode_scales,
        )
        modes[name] = np.where(field["wet"][None], mode, np.nan)
    return modes


def harmonic_contributions(
    spectrum: KoopmanSpectrumKDMD,
    fields: dict[str, np.ndarray],
    leading_index: int,
    leading_psi: np.ndarray,
    harmonic_index: int,
    harmonic_psi: np.ndarray,
    mode_scales: np.ndarray,
    *, lag: int, maximum_training_snapshots: int, seed: int,
) -> dict:
    """Build direct KDMD mode contributions on the leading phase clock."""
    indices = kdmd_training_indices(
        fields["state"].shape[0], lag, maximum_training_snapshots, seed,
    )
    observables = np.column_stack((fields["ice_anomaly"], fields["state"]))
    modes = observable_modes(
        spectrum, fields["state"], observables, indices,
        [leading_index, harmonic_index], mode_scales,
    )
    mode_source = "direct KDMD modes"

    n_surface = fields["surface_lat"].size
    v_ice = modes[:, 0]
    v_surface = modes[:, 1:1 + n_surface]
    v_ocean = modes[:, 1 + n_surface:]
    eigenfunctions = (leading_psi, harmonic_psi)
    ice_parts = [2 * np.real(v_ice[k] * eigenfunctions[k]) for k in range(2)]
    surface_parts = [
        2 * np.real(np.outer(eigenfunctions[k], v_surface[k])) for k in range(2)
    ]
    ocean_parts = [
        2e3 * np.real(np.outer(eigenfunctions[k], v_ocean[k])) for k in range(2)
    ]
    surface = fields["surface_anomaly"]
    ocean = fields["ocean_anomaly"] * 1e3
    surface_band = (fields["surface_lat"] >= -45) & (fields["surface_lat"] <= -28)
    ocean_band = (fields["ocean_lat"] >= -27) & (fields["ocean_lat"] <= -22)
    if not surface_band.any() or not ocean_band.any():
        raise ValueError("The South Atlantic diagnostic bands contain no rows")
    def band_mean(values, weights, selected):
        return np.average(values[:, selected], axis=1, weights=weights[selected])

    edge = band_mean(surface, fields["surface_weights"], surface_band)
    theta = band_mean(ocean, fields["ocean_weights"], ocean_band)
    edge_parts = [band_mean(part, fields["surface_weights"], surface_band)
                  for part in surface_parts]
    theta_parts = [band_mean(part, fields["ocean_weights"], ocean_band)
                   for part in ocean_parts]
    phase = np.mod(np.angle(leading_psi), 2 * np.pi)
    centres, ice_observed, ice_spread = phase_composite(fields["ice_area"], phase)
    ice_composites = [phase_composite(part, phase)[1] for part in ice_parts]
    edge_observed = phase_composite(edge, phase)[1]
    theta_observed = phase_composite(theta, phase)[1]
    edge_composites = [phase_composite(part, phase)[1] for part in edge_parts]
    theta_composites = [phase_composite(part, phase)[1] for part in theta_parts]
    surface_composites = [phase_composite(part, phase)[1] for part in surface_parts]
    ocean_composites = [phase_composite(part, phase)[1] for part in ocean_parts]

    def fourier(composite, harmonic):
        return complex(composite_fourier(composite, centres, harmonic))

    def summary(parts):
        b1, b2 = fourier(parts[0], 1), fourier(parts[1], 2)
        return {
            "B1": b1, "B2": b2,
            "amplitude_ratio": float(abs(b2) / abs(b1)) if b1 else float("nan"),
            "delta2": float(np.angle(b2 * np.conj(b1)**2)),
        }

    def rms_ratio(composites):
        rms1 = np.sqrt(np.mean(composites[0]**2, axis=0))
        rms2 = np.sqrt(np.mean(composites[1]**2, axis=0))
        return np.divide(
            rms2, rms1, out=np.full_like(rms1, np.nan), where=rms1 > 0,
        ), rms1

    ice_mean = float(fields["ice_area"].mean())
    ice_first = ice_mean + ice_composites[0]
    ice_both = ice_mean + ice_composites[0] + ice_composites[1]
    ice_total_variance = np.sum((ice_observed - ice_observed.mean())**2)
    if ice_total_variance <= 0:
        raise ValueError("The ice-area composite has zero phase variance")
    ice_variance_fractions = (
        float(1 - np.sum((ice_observed - ice_first)**2) / ice_total_variance),
        float(1 - np.sum((ice_observed - ice_both)**2) / ice_total_variance),
    )
    surface_ratio, _ = rms_ratio(surface_composites)
    ocean_ratio, ocean_first_rms = rms_ratio(ocean_composites)
    node_candidates = np.flatnonzero(
        (fields["ocean_lat"] >= -30) & (fields["ocean_lat"] <= -18)
    )
    if not node_candidates.size:
        raise ValueError("No native ocean row is available near the fundamental node")
    node_index = node_candidates[np.argmin(ocean_first_rms[node_candidates])]
    phase_difference = np.angle(harmonic_psi) - 2 * np.angle(leading_psi)
    locking_mean = np.mean(np.exp(1j * phase_difference))
    locking_offset = float(np.angle(locking_mean))
    locking = float(abs(np.mean(np.exp(1j * (phase_difference - locking_offset)))))
    observed_b1 = fourier(ice_observed, 1)
    observed_b2 = fourier(ice_observed, 2)
    return {
        "mu": fields["mu"],
        "maximum_depth": fields["maximum_depth"],
        "phase": phase,
        "centres": centres,
        "mode_source": mode_source,
        "ice": {
            "observed": ice_observed, "spread": ice_spread,
            "fundamental": ice_first, "combined": ice_both,
            "variance_fractions": ice_variance_fractions,
        },
        "loop": {
            "edge_annual": edge, "theta_annual": theta,
            "edge_observed": edge_observed, "theta_observed": theta_observed,
            "edge_first": edge_composites[0], "theta_first": theta_composites[0],
            "edge_combined": edge_composites[0] + edge_composites[1],
            "theta_combined": theta_composites[0] + theta_composites[1],
        },
        "surface": {
            "lat": fields["surface_lat"], "second": surface_composites[1],
            "ratio": surface_ratio,
        },
        "ocean": {
            "lat": fields["ocean_lat"], "second": ocean_composites[1],
            "ratio": ocean_ratio, "node_lat": float(fields["ocean_lat"][node_index]),
        },
        "summary": {
            "I": summary(ice_composites),
            "edge Ts": summary(edge_composites),
            "theta 22-27S": summary(theta_composites),
            "ice_variance_fractions": ice_variance_fractions,
            "ice_observed_delta2": float(np.angle(
                observed_b2 * np.conj(observed_b1)**2
            )),
            "phase_locking_R2": locking,
            "phase_locking_offset": locking_offset,
        },
    }


def make_figure(
    fields: dict[str, np.ndarray], rates: np.ndarray, index: int,
    psi: np.ndarray, *, lag: int, threshold: float, rank: int,
) -> plt.Figure:
    """Draw the leading eigenfunction's spectrum and physical panels."""
    phase = np.mod(np.angle(psi), 2 * np.pi)
    surface = fields["surface_anomaly"]
    ocean = fields["ocean_anomaly"] * 1e3
    surface_lat = fields["surface_lat"]
    ocean_lat = fields["ocean_lat"]
    surface_band = (surface_lat >= -45) & (surface_lat <= -28)
    ocean_band = (ocean_lat >= -27) & (ocean_lat <= -22)
    if not (surface_band.any() and ocean_band.any()):
        raise ValueError("The physical plotting bands contain no zonal rows")
    edge_temperature = np.average(
        surface[:, surface_band], axis=1,
        weights=fields["surface_weights"][surface_band],
    )
    band_ocean = np.average(
        ocean[:, ocean_band], axis=1,
        weights=fields["ocean_weights"][ocean_band],
    )
    centres, ocean_mean, _ = phase_composite(band_ocean, phase)
    _, edge_mean, _ = phase_composite(edge_temperature, phase)
    _, ice_mean, ice_spread = phase_composite(fields["ice_anomaly"], phase)
    surface_mode = mode_coefficients(psi, surface)
    ocean_mode = mode_coefficients(psi, ocean)
    period = 2 * np.pi / rates[index].imag

    figure = plt.figure(figsize=(10, 7), layout="constrained")
    outer = figure.add_gridspec(1, 2, width_ratios=(1, 1.2))
    left = outer[0, 0].subgridspec(2, 1, height_ratios=(1, 1.35))
    spectrum_axis = figure.add_subplot(left[0])
    scatter_axis = figure.add_subplot(left[1])
    right = outer[0, 1].subgridspec(3, 2, width_ratios=(1, 0.04),
                                    height_ratios=(0.75, 1, 1.35))
    ice_axis = figure.add_subplot(right[0, 0])
    surface_axis = figure.add_subplot(right[1, 0], sharex=ice_axis)
    ocean_axis = figure.add_subplot(right[2, 0], sharex=ice_axis)

    # Show the complete direct single-lag spectrum, as in the original single-fit
    # analysis. Local-slope resonance ratios require multiple lagged fits.

    spectrum_axis.scatter(rates.real / np.abs(rates.real[1]), rates.imag, s=6,
                          c="b", alpha=1, linewidths=0)
    # spectrum_axis.scatter([rates[index].real], [rates[index].imag],
    #                       c="black", s=35, zorder=3, label=r"$\psi_1$")
    # spectrum_axis.scatter([rates[index].real], [-rates[index].imag],
    #                       c="black", s=35, zorder=3)
    spectrum_axis.axhline(0, color="0.7", lw=0.7)
    spectrum_axis.axvline(0, color="0.7", lw=0.7)
    spectrum_axis.set(xlabel=r"Re $\lambda$ ",
                      ylabel=r"Im $\lambda$ (yr$^{-1}$)",
                      title=f"(a) Direct lag-{lag} KDMD spectrum")
    spectrum_axis.set_xlim(left=-18,right=0.2)
    _handles, _labels = spectrum_axis.get_legend_handles_labels()
    if _handles:
        spectrum_axis.legend(frameon=False)

    order = np.random.default_rng(0).permutation(psi.size)
    scatter_axis.scatter(band_ocean[order], edge_temperature[order],
                         c=phase[order], cmap=PHASE_MAP, vmin=0, vmax=2*np.pi,
                         s=3, alpha=0.7, linewidths=0, rasterized=True)
    scatter_axis.plot(np.r_[ocean_mean, ocean_mean[0]],
                      np.r_[edge_mean, edge_mean[0]], "w-", lw=2)
    scatter_axis.plot(np.r_[ocean_mean, ocean_mean[0]],
                      np.r_[edge_mean, edge_mean[0]], "k--", lw=0.8)
    scatter_axis.set(xlabel=rf"ocean $\theta$, 22–27°S, 0–{fields['maximum_depth']:g} m (mK)",
                     ylabel=r"edge-band $T_s$, 28–45°S (K)",
                     title=r"(b) South Atlantic states colored by arg $\psi_1$")

    phase_twice = np.r_[centres, centres + 2*np.pi]
    ice_twice = np.r_[ice_mean, ice_mean]
    spread_twice = np.r_[ice_spread, ice_spread]
    ice_lower = ice_twice - spread_twice
    ice_upper = ice_twice + spread_twice
    ice_points = np.column_stack((phase_twice, ice_twice))
    ice_segments = np.stack((ice_points[:-1], ice_points[1:]), axis=1)
    ice_segment_phase = np.mod(
        (phase_twice[:-1] + phase_twice[1:]) / 2, 2 * np.pi
    )
    ice_shading = np.stack((
        np.column_stack((phase_twice[:-1], ice_lower[:-1])),
        np.column_stack((phase_twice[:-1], ice_upper[:-1])),
        np.column_stack((phase_twice[1:], ice_upper[1:])),
        np.column_stack((phase_twice[1:], ice_lower[1:])),
    ), axis=1)
    ice_axis.add_collection(PolyCollection(
        ice_shading,
        facecolors=PHASE_MAP(ice_segment_phase / (2 * np.pi)),
        edgecolors="none",
        alpha=0.25,
        zorder=2,
    ))
    ice_axis.add_collection(LineCollection(
        ice_segments,
        colors=PHASE_MAP(ice_segment_phase / (2 * np.pi)),
        linewidths=2,
        zorder=3,
    ))
    ice_axis.autoscale_view()
    ice_axis.axhline(0, color="0.7", lw=0.6)
    ice_axis.set(ylabel=r"Atlantic ice area ($10^6$ km$^2$)",
                 title="(c) South Atlantic T21 sea-ice area ±1 SD")

    phase_grid = np.linspace(0, 4*np.pi, 289)
    for axis, colour_axis, lat, coefficients, band, unit, letter in (
        (surface_axis, figure.add_subplot(right[1, 1]), surface_lat,
         surface_mode, (-45, -28), "K", "(d)"),
        (ocean_axis, figure.add_subplot(right[2, 1]), ocean_lat,
         ocean_mode, (-27, -22), "mK", "(e)"),
    ):
        harmonics = 2 * np.real(np.outer(np.exp(1j * phase_grid), coefficients))
        limit = float(np.nanpercentile(np.abs(harmonics), 99))
        mesh = axis.pcolormesh(phase_grid, lat, harmonics.T,
                               cmap="RdBu_r", vmin=-limit, vmax=limit,
                               shading="nearest", rasterized=True)
        # for boundary in band:
        #     axis.axhline(boundary, color="black", lw=0.6, ls="--")
        axis.set(ylabel="latitude (°)", title=f"{letter} South Atlantic zonal {'surface T' if unit == 'K' else 'ocean θ'} mode")
        axis.set_ylim(-45, 0)
        figure.colorbar(mesh, cax=colour_axis, label=unit)
    ice_axis.tick_params(labelbottom=False)
    surface_axis.tick_params(labelbottom=False)
    ocean_axis.set_xticks(np.arange(0, 4*np.pi + 0.01, np.pi),
                         ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    ocean_axis.set_xlabel(r"phase arg $\psi_1$ (0 = Atlantic ice maximum)")
    ocean_axis.set_xlim(0, 4*np.pi)
    figure.suptitle(
        rf"PlaSim $\mu={fields['mu'].replace('p', '.')}$ · South Atlantic zonal KDMD · lag={lag} yr · "
        rf"TSVD={threshold:g} · rank={rank} · period={period:.1f} yr",
        fontsize=12,
    )
    return figure


def make_harmonic_contribution_figure(
    contributions: dict, *, lag: int, threshold: float, rank: int,
) -> plt.Figure:
    """Show what the direct second KDMD mode adds on the leading phase clock."""
    centres = contributions["centres"]
    ice = contributions["ice"]
    loop = contributions["loop"]
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.15))
    ice_axis = figure.add_subplot(grid[0, 0])
    loop_axis = figure.add_subplot(grid[0, 1])

    def periodic(values):
        return np.r_[values[-1], values, values[0]]

    phase_extended = np.r_[centres[-1] - 2*np.pi, centres, centres[0] + 2*np.pi]
    ice_axis.fill_between(
        phase_extended,
        periodic(ice["observed"] - ice["spread"]),
        periodic(ice["observed"] + ice["spread"]),
        color="0.82", alpha=0.7,
    )
    ice_axis.plot(phase_extended, periodic(ice["observed"]),
                  color="black", lw=2, label="Ice composite ±1 SD")
    r2_first, r2_both = ice["variance_fractions"]
    ice_axis.plot(phase_extended, periodic(ice["fundamental"]),
                  color="#315c9b", lw=2, ls="--",
                  label=rf"$I_1$ ($R^2={r2_first:.2f}$)")
    ice_axis.plot(phase_extended, periodic(ice["combined"]),
                  color="#c2185b", lw=2,
                  label=rf"$I_1+I_2$ ($R^2={r2_both:.2f}$)")
    ice_axis.set(
        xlim=(0, 2*np.pi), xlabel=r"$\phi=\arg\psi_1$",
        ylabel=r"South Atlantic ice area ($10^6$ km$^2$)",
        title="(a) Ice cycle",
    )
    ice_axis.set_xticks(
        np.arange(0, 2*np.pi + 0.01, np.pi/2),
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    ice_axis.legend(frameon=False, fontsize=8)

    loop_axis.scatter(
        loop["theta_annual"], loop["edge_annual"],
        color="0.7", alpha=0.25, s=3, linewidths=0, rasterized=True,
    )
    loop_axis.plot(periodic(loop["theta_observed"]),
                   periodic(loop["edge_observed"]),
                   color="black", lw=2)
    loop_axis.plot(periodic(loop["theta_first"]),
                   periodic(loop["edge_first"]),
                   color="#315c9b", lw=2, ls="--")
    combined_loop = np.column_stack((loop["theta_combined"], loop["edge_combined"]))
    closed_loop = np.vstack((combined_loop, combined_loop[0]))
    segments = np.stack((closed_loop[:-1], closed_loop[1:]), axis=1)
    segment_phase = np.mod(centres + np.pi / centres.size, 2*np.pi)
    loop_axis.add_collection(LineCollection(
        segments, colors=PHASE_MAP(segment_phase / (2*np.pi)),
        linewidths=2.5, zorder=4,
    ))
    loop_axis.set(
        xlabel=rf"$\theta$, 22–27°S, 0–{contributions['maximum_depth']:g} m (mK)",
        ylabel=r"Edge-band $T_s$, 28–45°S (K)",
        title="(b) South Atlantic loop",
    )
    loop_axis.legend(handles=[
        Line2D([], [], color="black", lw=2, label="Binned data"),
        Line2D([], [], color="#315c9b", lw=2, ls="--", label=r"$\ell=1$"),
        Line2D([], [], color="#c2185b", lw=2, label=r"$\ell=1+2$ (phase color)"),
    ], frameon=False, fontsize=8)

    for slot, key, unit, title in (
        (grid[1, 0], "surface", "K", "(c) Surface $T_s$: second harmonic"),
        (grid[1, 1], "ocean", "mK", "(d) Ocean $\\theta$: second harmonic"),
    ):
        subgrid = slot.subgridspec(1, 3, width_ratios=(1, 0.28, 0.05))
        axis = figure.add_subplot(subgrid[0, 0])
        ratio_axis = figure.add_subplot(subgrid[0, 1], sharey=axis)
        color_axis = figure.add_subplot(subgrid[0, 2])
        profile = contributions[key]
        field = profile["second"]
        limit = float(np.nanmax(np.abs(field)))
        if limit <= 0:
            raise ValueError(f"The {key} second-harmonic composite is zero")
        mesh = axis.pcolormesh(
            centres, profile["lat"], field.T,
            cmap="RdBu_r", vmin=-limit, vmax=limit,
            shading="nearest", rasterized=True,
        )
        ratio_axis.plot(profile["ratio"], profile["lat"], color="black", lw=1.5)
        ratio_axis.axvline(1, color="0.6", lw=0.8, ls=":")
        markers = (
            [profile["lat"][np.argmin(abs(profile["lat"] - target))]
             for target in (-30.5, -36.0)]
            if key == "surface" else [profile["node_lat"]]
        )
        for latitude in markers:
            axis.axhline(latitude, color="black", lw=0.8, ls="--")
            ratio_axis.axhline(latitude, color="black", lw=0.8, ls="--")
        axis.set(
            xlim=(0, 2*np.pi), ylim=(-45, 0),
            xlabel=r"$\phi=\arg\psi_1$", ylabel="latitude (°)", title=title,
        )
        axis.set_xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ratio_axis.set(xlabel=r"rms$_2$/rms$_1$", ylim=(-45, 0))
        ratio_axis.tick_params(labelleft=False)
        figure.colorbar(mesh, cax=color_axis, label=unit)
    figure.suptitle(
        rf"PlaSim $\mu={contributions['mu'].replace('p', '.')}$ · South Atlantic zonal KDMD · "
        rf"lag={lag} yr · TSVD={threshold:g} · rank={rank}",
        fontsize=12,
    )
    return figure


def mode_cycle(modes: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Return 2 Re(a1 e^{i phi}) + 2 Re(a2 e^{2 i phi}) for each phase.

    This takes |psi1| = 1 and psi2 = psi1**2 on the phase clock, which matches
    the unit-RMS normalization and the harmonic alignment of the fit.
    """
    waves = np.stack((np.exp(1j * phase), np.exp(2j * phase)))
    flat = modes.reshape(modes.shape[0], -1)
    cycle = 2 * np.real(waves.T @ flat)
    return cycle.reshape((phase.size,) + modes.shape[1:])


def make_state_mode_figure(comparison: dict) -> plt.Figure:
    """Draw the zonal state rows over one cycle: two modes against the composite."""
    figure, axes = plt.subplots(
        2, 4, figsize=(12, 6.5), layout="constrained",
        gridspec_kw={"width_ratios": (1, 1, 1, 0.05)},
    )
    phase_grid = np.linspace(0, 2 * np.pi, 145)
    for row, (name, title, unit) in enumerate((
        ("surface", "zonal surface T", "K"),
        ("ocean", "zonal θ 0–700 m", "mK"),
    )):
        block = comparison[name]
        panels = (
            ("KDMD modes ψ₁ + ψ₂", phase_grid, mode_cycle(block["direct"], phase_grid)),
            ("regression on ψ₁, ψ₂", phase_grid,
             mode_cycle(block["regression"], phase_grid)),
            ("observed phase composite", block["centres"], block["observed"]),
        )
        limit = float(np.nanpercentile(np.abs(panels[2][2]), 99))
        for column, (label, phase, values) in enumerate(panels):
            axis = axes[row, column]
            mesh = axis.pcolormesh(
                phase, block["lat"], values.T, cmap="RdBu_r",
                vmin=-limit, vmax=limit, shading="nearest", rasterized=True,
            )
            axis.set(xlim=(0, 2 * np.pi), ylim=(-45, 0), title=f"{label}: {title}")
            axis.set_xticks(
                np.linspace(0, 2 * np.pi, 5),
                ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
            )
            if column == 0:
                axis.set_ylabel("Latitude (°)")
            else:
                axis.tick_params(labelleft=False)
            if row == 1:
                axis.set_xlabel(r"phase arg $\psi_1$ (0 = SA ice maximum)")
        figure.colorbar(mesh, cax=axes[row, 3], label=unit)
        axes[row, 0].text(
            0.02, 0.03,
            f"corr(KDMD, regression) = {block['correlation']:.3f}\n"
            f"corr(KDMD, composite) = {block['composite_correlation']:.3f}",
            transform=axes[row, 0].transAxes, fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    figure.suptitle(
        f"μ={comparison['mu'].replace('p', '.')}: zonal state over one cycle "
        f"(period {comparison['period']:.1f} yr)"
    )
    return figure


def _wrap_longitude(longitude: np.ndarray) -> np.ndarray:
    return (np.asarray(longitude) + 180.0) % 360.0 - 180.0


def _lsg_to_regular(values: np.ndarray, lsg_lon: np.ndarray) -> np.ndarray:
    """Spread staggered 5° LSG cells over a regular 2.5° longitude grid.

    LSG rows alternate between 0° and 2.5° offsets, so every cell covers
    exactly two 2.5° columns; the result can be drawn with one pcolormesh.
    """
    regular = np.full(values.shape[:-1] + (144,), np.nan)
    first = np.floor((_wrap_longitude(lsg_lon) - 2.5 + 180.0) / 2.5 + 0.5).astype(int) % 144
    rows = np.arange(values.shape[-2])[:, None]
    regular[..., rows, first] = values
    regular[..., rows, (first + 1) % 144] = values
    return regular


def phase_window_composite(
    values: np.ndarray, phase: np.ndarray, centres: np.ndarray,
    half_width: float = np.pi / 18,
) -> np.ndarray:
    """Average values over the years whose phase lies within half_width of each centre."""
    composites = []
    for centre in centres:
        selected = np.abs(np.angle(np.exp(1j * (phase - centre)))) <= half_width
        if not selected.any():
            raise ValueError("A phase window contains no years")
        composites.append(values[selected].mean(axis=0))
    return np.asarray(composites)


def make_map_mode_figure(
    maps: dict, modes: dict[str, np.ndarray], *, period: float,
    leading_psi: np.ndarray | None = None,
    phase_count: int = 4, lat_range: tuple[float, float] = (-75.0, 5.0),
    lon_range: tuple[float, float] = (-70.0, 25.0), basin_only: bool = True,
    mu: str = "",
) -> plt.Figure:
    """Draw each map's two-mode cycle at equally spaced phases of psi1.

    Columns are phases (0 = SA ice maximum). With ``leading_psi``, each
    observable gets a second row: the observed composite of the anomaly over
    years within ±10° of that phase, on the same colour scale, and the row
    label gives the pattern correlation between the two rows. With
    ``basin_only``, cells outside the South Atlantic basin masks are hidden.
    """
    phases = np.arange(phase_count) * 2 * np.pi / phase_count
    names = list(modes)
    kinds = ("KDMD ψ₁+ψ₂", "composite") if leading_psi is not None else ("KDMD ψ₁+ψ₂",)
    psi_phase = None if leading_psi is None else np.angle(leading_psi)
    aspect = (lat_range[1] - lat_range[0]) / (lon_range[1] - lon_range[0])
    row_height = min(3.3, max(1.2, 3.3 * aspect))
    figure, axes = plt.subplots(
        len(names) * len(kinds), phase_count + 1,
        figsize=(3.3 * phase_count + 0.6, len(names) * len(kinds) * row_height + 0.8),
        layout="constrained",
        gridspec_kw={"width_ratios": (1,) * phase_count + (0.04,)},
        squeeze=False,
    )
    t21_lon = _wrap_longitude(maps["t21_lon"])
    order = np.argsort(t21_lon)
    t21_lon = t21_lon[order]
    t21_lat = maps["t21_lat"]
    lon_edges = np.r_[t21_lon - 2.8125, t21_lon[-1] + 2.8125]
    lat_edges = np.r_[
        t21_lat[0] - (t21_lat[1] - t21_lat[0]) / 2,
        (t21_lat[1:] + t21_lat[:-1]) / 2,
        t21_lat[-1] + (t21_lat[-1] - t21_lat[-2]) / 2,
    ]
    lsm = maps["lsm"][:, order]
    lsg_lat = maps["lsg_lat"][:, 0]
    lsg_lat_edges = np.r_[lsg_lat + 1.25, lsg_lat[-1] - 1.25]
    regular_lon_edges = np.arange(-180.0, 180.01, 2.5)
    regular_lon = regular_lon_edges[:-1] + 1.25
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("0.88")

    def to_display(values, grid):
        # Put a (..., lat, lon) field on the plotted grid, hiding cells
        # outside the basin when requested.
        if grid == "t21":
            if basin_only:
                values = np.where(maps["t21_basin"], values, np.nan)
            return values[..., order]
        if basin_only:
            values = np.where(maps["lsg_basin"], values, np.nan)
        return _lsg_to_regular(values, maps["lsg_lon"])

    for field_index, name in enumerate(names):
        field = maps["fields"][name]
        grid = field["grid"]
        rows = {"KDMD ψ₁+ψ₂": to_display(mode_cycle(modes[name], phases), grid)}
        if psi_phase is not None:
            observed = field["scale"] * phase_window_composite(
                field["anomaly"], psi_phase, phases,
            )
            observed = np.where(field["wet"][None], observed, np.nan)
            rows["composite"] = to_display(observed, grid)
        lat_axis = t21_lat if grid == "t21" else lsg_lat
        lon_axis = t21_lon if grid == "t21" else regular_lon
        window = np.ix_(
            (lat_axis >= lat_range[0]) & (lat_axis <= lat_range[1]),
            (lon_axis >= lon_range[0]) & (lon_axis <= lon_range[1]),
        )
        shown = np.concatenate([values[:, window[0], window[1]].ravel() for values in rows.values()])
        shown = shown[np.isfinite(shown)]
        limit = float(np.percentile(np.abs(shown), 99.5)) if shown.size else 1.0
        limit = limit or 1.0
        correlation = ""
        if psi_phase is not None:
            first = rows["KDMD ψ₁+ψ₂"][:, window[0], window[1]].ravel()
            second = rows["composite"][:, window[0], window[1]].ravel()
            valid = np.isfinite(first) & np.isfinite(second)
            if valid.any():
                value = np.sum(first[valid] * second[valid]) / np.sqrt(
                    np.sum(first[valid]**2) * np.sum(second[valid]**2)
                )
                correlation = f"\nr = {value:.2f}"
        for kind_index, kind in enumerate(kinds):
            row = field_index * len(kinds) + kind_index
            for column, phase in enumerate(phases):
                axis = axes[row, column]
                values = np.ma.masked_invalid(rows[kind][column])
                if grid == "t21":
                    mesh = axis.pcolormesh(
                        lon_edges, lat_edges, values, cmap=colormap,
                        vmin=-limit, vmax=limit, rasterized=True,
                    )
                else:
                    mesh = axis.pcolormesh(
                        regular_lon_edges, lsg_lat_edges, values, cmap=colormap,
                        vmin=-limit, vmax=limit, rasterized=True,
                    )
                axis.contour(t21_lon, t21_lat, lsm, levels=[0.5], colors="0.3", linewidths=0.5)
                axis.set(xlim=lon_range, ylim=lat_range)
                axis.tick_params(labelsize="small")
                if column:
                    axis.tick_params(labelleft=False)
                else:
                    axis.set_ylabel(
                        f"{field['label']}\n{kind}"
                        + (correlation if kind_index == len(kinds) - 1 else ""),
                        fontsize="small",
                    )
                if row < axes.shape[0] - 1:
                    axis.tick_params(labelbottom=False)
                if row == 0:
                    axis.set_title(
                        f"phase {phase / (2 * np.pi):.2f} cycle "
                        f"(+{phase / (2 * np.pi) * period:.1f} yr)",
                        fontsize="small",
                    )
            figure.colorbar(mesh, cax=axes[row, -1], label=field["unit"])
    figure.suptitle(
        f"μ={mu.replace('p', '.')}: lat–lon modes over one cycle "
        f"(period {period:.1f} yr; phase 0 = SA ice maximum"
        + ("; composites over ±10° of phase)" if psi_phase is not None else ")")
    )
    return figure


def make_selected_eigenmode_figure(
    fields: dict[str, np.ndarray], selected_mode: dict,
) -> plt.Figure:
    """Plot one selected KDMD eigenfunction and its observable modes."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    psi = selected_mode["eigenfunction"]
    is_real = selected_mode["is_real_mode"]
    axes[0].plot(fields["years"], psi.real, color="#315c9b", lw=0.8,
                 label="real")
    if not is_real:
        axes[0].plot(fields["years"], psi.imag, color="#c2185b", lw=0.8,
                     label="imaginary")
    axes[0].set(xlabel="year", ylabel=r"$\psi$ (unit RMS)",
                title="Eigenfunction")
    axes[0].legend(frameon=False)

    for axis, latitude, mode, unit, title in (
        (axes[1], fields["surface_lat"], selected_mode["surface_mode"],
         "K", r"South Atlantic $T_s$ mode"),
        (axes[2], fields["ocean_lat"], selected_mode["ocean_mode"] * 1e3,
         "mK", r"South Atlantic $\theta$ mode"),
    ):
        axis.plot(mode.real, latitude, ".-", color="#315c9b", label="real")
        if not is_real:
            axis.plot(mode.imag, latitude, ".--", color="#c2185b",
                      label="imaginary")
        axis.axvline(0, color="0.7", lw=0.7)
        axis.set(xlabel=unit, ylabel="latitude (°)", title=title)
        axis.legend(frameon=False)
    eigenvalue = selected_mode["eigenvalue"]
    figure.suptitle(
        f"KDMD mode {selected_mode['index']} · "
        f"λ={eigenvalue.real:+.5f}{eigenvalue.imag:+.5f}i yr⁻¹ · "
        f"South Atlantic μ={fields['mu'].replace('p', '.')}",
    )
    return figure


def make_indexed_spectrum_figure(
    rates: np.ndarray, leading_index: int, harmonic_index: int | None,
    label_count: int = 36,
) -> plt.Figure:
    """Show all rates and label the least damped modes for selection."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    count = min(label_count, rates.size)
    for axis in axes:
        axis.axhline(0, color="0.8", lw=0.7)
        axis.axvline(0, color="0.8", lw=0.7)
        axis.set(xlabel=r"Re $\lambda$ (yr$^{-1}$)",
                 ylabel=r"Im $\lambda$ (yr$^{-1}$)")
    axes[0].scatter(rates.real, rates.imag, s=7, color="0.55",
                    alpha=0.6, linewidths=0)
    axes[0].set_title("All fitted eigenvalues")
    axes[1].scatter(rates[:count].real, rates[:count].imag, s=18,
                    color="0.3", linewidths=0)
    real_indices = np.flatnonzero(np.abs(rates[:count].imag) < 1e-10)
    axes[1].scatter(rates[real_indices].real, rates[real_indices].imag,
                    s=40, marker="s", color="#315c9b", label="real rate")
    for index in range(count):
        axes[1].annotate(str(index), (rates[index].real, rates[index].imag),
                         xytext=(3, 3), textcoords="offset points", fontsize=7)
    for index, color, label in (
        (leading_index, "#c2185b", "leading"),
        (harmonic_index, "#e9a23b", "harmonic"),
    ):
        if index is not None:
            axes[0].scatter(rates[index].real, rates[index].imag, s=55,
                            facecolors="none", edgecolors=color, lw=1.4,
                            label=label)
            if index < count:
                axes[1].scatter(rates[index].real, rates[index].imag, s=85,
                                facecolors="none", edgecolors=color, lw=1.4)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title(f"Modes 0–{count - 1} (zero-based indices)")
    return figure


def main() -> None:
    """Fit one Koopman operator and save its figure and leading eigenfunction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="repo", choices=sorted(RAW_MAP_ROOTS),
                        help="Archive root holding the raw maps and basin masks")
    parser.add_argument("--mu", default=MU, help="PlaSim μ with raw maps and basin masks")
    parser.add_argument("--lag", type=int, default=5, help="KDMD snapshot lag in years")
    parser.add_argument("--start-year", type=int, default=6500)
    parser.add_argument("--maximum-depth", type=float, default=700.0)
    parser.add_argument("--factorization-tsvd", type=float, default=1e-5)
    parser.add_argument("--tsvd", type=float, default=5e-5)
    parser.add_argument("--maximum-training-snapshots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures/koopman"))
    args = parser.parse_args()
    if args.lag <= 0 or args.tsvd <= 0 or args.factorization_tsvd <= 0:
        parser.error("--lag and both TSVD thresholds must be positive")
    root = RAW_MAP_ROOTS[args.root]
    if args.mu not in available_mu_values(root):
        parser.error(f"No mu={args.mu} raw maps and basin masks under {root}")
    fields = load_fields(root, args.start_year, args.maximum_depth, args.mu)
    spectrum, rates, rank, bandwidths = fit_koopman(
        fields, args.lag, args.tsvd, args.maximum_training_snapshots, args.seed,
        args.factorization_tsvd,
    )
    index, psi = leading_eigenfunction(spectrum, rates, fields)
    figure = make_figure(fields, rates, index, psi, lag=args.lag,
                         threshold=args.tsvd, rank=rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"mu{args.mu}_eigenfunction_prl_south_atlantic_lag{args.lag}"
    figure.savefig(stem.with_suffix(".png"), dpi=250)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    np.savez_compressed(
        stem.with_suffix(".npz"), years=fields["years"], psi=psi,
        eigenvalue=rates[index], eigenvalues=rates, lag=args.lag,
        tsvd=args.tsvd, factorization_tsvd=args.factorization_tsvd,
        rank=rank, surface_bandwidth=bandwidths[0],
        ocean_bandwidth=bandwidths[1], ice_anomaly=fields["ice_anomaly"],
        source_archive=str(source_file(root, args.mu)),
        source_masks=str(basin_mask_file(root, args.mu)),
        ice_area_definition="South Atlantic T21 SIC integrated over basin ocean area",
    )
    print(f"Saved {stem}.png, .pdf, and .npz")
    print(f"South Atlantic zonal state: {fields['state'].shape[1]} features, {fields['years'].size} years")
    print(f"Leading mode {index}: lambda={rates[index]:.6g} yr^-1, period={2*np.pi/rates[index].imag:.2f} yr")


if __name__ == "__main__":
    main()
