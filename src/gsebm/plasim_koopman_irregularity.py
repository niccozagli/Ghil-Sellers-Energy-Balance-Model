"""Irregularity of the South Atlantic oscillation around its Koopman cycle.

The Marimo app is ``analysis/Koopman_irregularity.py``. The regular cycle is
the leading Koopman eigenfunction psi1 (and its harmonic psi2) from
``plasim_koopman_single``. Everything here works on what is left after that
cycle is removed, conditioned on the phase arg(psi1):

- three residuals of an observable (two-mode Koopman, phase composite, and
  phase-and-amplitude composite);
- phase-binned residual statistics (variance, skewness, lag-one memory);
- irregularity of the clock itself (phase advance, phase diffusion, cycle
  lengths);
- timing jitter of sea-ice cells switching on and off;
- a phase-dependent linear model of the residual and its Floquet multipliers.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gsebm import plasim_koopman_single as single
from gsebm.plasim_koopman_single import source_file

# Final 4000 stationary years of each run. Runs branched at year 15000 drift
# for the first kyr. 1245 and 1235 end before their corrupt block 13690–13699
# (archives extracted on 2026-10-05).
ANALYSIS_WINDOWS = {
    "1245": (9690, 13689),
    "1242p5": (20630, 24629),
    "1240": (13380, 17379),
    "1237p5": (20270, 24269),
    "1235": (9690, 13689),
    "1233p75": (20530, 24529),
    "1232p5": (20430, 24429),
}


def wrapped_phase(psi: np.ndarray) -> np.ndarray:
    """Return arg(psi) in [0, 2 pi)."""
    return np.mod(np.angle(psi), 2 * np.pi)


def phase_bin_index(phase: np.ndarray, bins: int) -> np.ndarray:
    """Return the bin index of each phase in [0, 2 pi)."""
    return np.minimum((np.asarray(phase) / (2 * np.pi) * bins).astype(int), bins - 1)


def bin_centres(bins: int) -> np.ndarray:
    """Return the centres of equally spaced phase bins."""
    return (np.arange(bins) + 0.5) * 2 * np.pi / bins


def amplitude_terciles(psi: np.ndarray) -> np.ndarray:
    """Return 0, 1, 2 for the lower, middle, and upper third of |psi|."""
    amplitude = np.abs(psi)
    return np.searchsorted(np.quantile(amplitude, [1 / 3, 2 / 3]), amplitude, side="right")


def _group_means(values: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    """Mean of values (axis 0) for each integer group; empty groups are NaN."""
    flat = values.reshape(values.shape[0], -1)
    sums = np.zeros((count, flat.shape[1]))
    np.add.at(sums, groups, flat)
    numbers = np.bincount(groups, minlength=count).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / numbers[:, None]
    return means.reshape((count,) + values.shape[1:])


def cycle_residuals(
    values: np.ndarray, leading_psi: np.ndarray, harmonic_psi: np.ndarray | None,
    modes: np.ndarray | None, bins: int = 36,
) -> dict[str, np.ndarray]:
    """Remove the regular cycle from an anomaly observable in three ways.

    ``values`` has one row per year. ``modes`` are the complex psi1 and psi2
    modes of ``values`` in the same units, shape ``(2,) + values.shape[1:]``.

    - ``koopman``: ``G - 2Re(a1 psi1) - 2Re(a2 psi2)`` (centered);
    - ``composite``: ``G - C(arg psi1)``, all harmonics of the mean cycle, so
      a switch that always happens at the same phase is removed exactly;
    - ``amplitude``: ``G - C(arg psi1, |psi1| tercile)``, also removing the
      dependence of the cycle shape on the cycle strength.

    Returns the three residuals and their variance fraction of ``G``.
    """
    values = np.asarray(values, dtype=float)
    phase_index = phase_bin_index(wrapped_phase(leading_psi), bins)
    residuals = {}
    if modes is not None:
        reconstruction = 2 * np.real(
            np.multiply.outer(leading_psi, modes[0])
            + (0 if harmonic_psi is None else np.multiply.outer(harmonic_psi, modes[1]))
        )
        koopman = values - reconstruction
        residuals["koopman"] = koopman - koopman.mean(axis=0)
    residuals["composite"] = values - _group_means(values, phase_index, bins)[phase_index]
    joint = phase_index * 3 + amplitude_terciles(leading_psi)
    joint_mean = _group_means(values, joint, 3 * bins)
    # Sparse joint bins fall back to the phase-only composite.
    empty = np.bincount(joint, minlength=3 * bins) < 3
    joint_mean[empty] = _group_means(values, phase_index, bins)[np.flatnonzero(empty) // 3]
    residuals["amplitude"] = values - joint_mean[joint]
    total = np.sum(values**2)
    fractions = {
        name: float(np.sum(residual**2) / total) if total > 0 else np.nan
        for name, residual in residuals.items()
    }
    return {**residuals, "fractions": fractions}


def phase_conditioned_stats(
    values: np.ndarray, phase: np.ndarray, bins: int = 36,
) -> dict[str, np.ndarray]:
    """Per phase bin: count, mean, variance and its standard error, skewness.

    ``values`` has one row per year and any trailing shape.
    """
    values = np.asarray(values, dtype=float)
    index = phase_bin_index(phase, bins)
    count = np.bincount(index, minlength=bins).astype(float)
    mean = _group_means(values, index, bins)
    deviation = values - mean[index]
    variance = _group_means(deviation**2, index, bins)
    third = _group_means(deviation**3, index, bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        skewness = third / variance**1.5
        shape = (bins,) + (1,) * (values.ndim - 1)
        standard_error = variance * np.sqrt(2 / np.maximum(count - 1, 1)).reshape(shape)
    return {
        "centres": bin_centres(bins), "count": count, "mean": mean,
        "variance": variance, "variance_se": standard_error, "skewness": skewness,
    }


def lag_one_by_phase(
    series: np.ndarray, phase: np.ndarray, bins: int = 18,
) -> np.ndarray:
    """Regression slope of r(t+1) on r(t) for the years starting in each phase bin."""
    series = np.asarray(series, dtype=float)
    index = phase_bin_index(phase[:-1], bins)
    numerator = np.bincount(index, series[1:] * series[:-1], minlength=bins)
    denominator = np.bincount(index, series[:-1]**2, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def phase_advance_stats(
    psi: np.ndarray, years: np.ndarray, bins: int = 36,
) -> dict[str, np.ndarray]:
    """Mean and spread of the one-year phase advance and of |psi| by phase."""
    advance = np.angle(psi[1:] * np.conj(psi[:-1])) / np.diff(years)
    phase = wrapped_phase(psi)
    rate = phase_conditioned_stats(advance, phase[:-1], bins)
    amplitude = phase_conditioned_stats(np.abs(psi), phase, bins)
    return {
        "centres": rate["centres"],
        "advance_mean": rate["mean"], "advance_std": np.sqrt(rate["variance"]),
        "amplitude_mean": amplitude["mean"],
        "amplitude_std": np.sqrt(amplitude["variance"]),
        "advance_std_total": float(np.std(advance)),
        "amplitude_cv": float(np.std(np.abs(psi)) / np.mean(np.abs(psi))),
    }


def phase_diffusion(
    psi: np.ndarray, maximum_lag: int = 100, fit_range: tuple[int, int] = (5, 100),
) -> dict:
    """Variance of the unwrapped phase increment against lag, and its slope.

    For a noisy oscillator, ``Var[theta(t+tau) - theta(t)] ~ 2 D tau``. ``D``
    (rad² yr⁻¹) is the phase diffusion coefficient; ``1/D`` is the time over
    which the oscillation loses its phase memory.
    """
    theta = np.unwrap(np.angle(psi))
    lags = np.arange(1, maximum_lag + 1)
    variance = np.array([np.var(theta[lag:] - theta[:-lag]) for lag in lags])
    selected = (lags >= fit_range[0]) & (lags <= fit_range[1])
    slope, intercept = np.polyfit(lags[selected], variance[selected], 1)
    return {
        "lags": lags, "variance": variance,
        "diffusion": float(slope / 2), "intercept": float(intercept),
    }


def cycle_crossings(psi: np.ndarray, years: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Fractional years at which the unwrapped phase passes phase + 2 pi k."""
    theta = np.unwrap(np.angle(psi))
    levels = np.arange(
        np.ceil((theta.min() - phase) / (2 * np.pi)),
        np.floor((theta.max() - phase) / (2 * np.pi)) + 1,
    ) * 2 * np.pi + phase
    times = []
    for level in levels:
        above = np.flatnonzero((theta[:-1] < level) & (theta[1:] >= level))
        if above.size:
            # The first upward pass; brief backward steps do not add cycles.
            k = above[0]
            fraction = (level - theta[k]) / (theta[k + 1] - theta[k])
            times.append(years[k] + fraction * (years[k + 1] - years[k]))
    return np.asarray(times)


def cycle_lengths(psi: np.ndarray, years: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Lengths (years) of complete cycles between successive phase passes."""
    return np.diff(cycle_crossings(psi, years, phase))


def residual_versus_cycle_length(
    residual: np.ndarray, psi: np.ndarray, years: np.ndarray,
    target_phase: float = np.pi, half_width: float = np.pi / 9,
) -> dict[str, np.ndarray | float]:
    """Correlate the residual near one phase with the length of that cycle.

    Cycles run between phase-zero passes. For each cycle, the residual is
    averaged over its years within ``half_width`` of ``target_phase``.
    """
    edges = cycle_crossings(psi, years)
    phase = wrapped_phase(psi)
    near = np.abs(np.angle(np.exp(1j * (phase - target_phase)))) <= half_width
    values, lengths = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        selected = near & (years >= start) & (years < stop)
        if selected.any():
            values.append(float(np.mean(residual[selected])))
            lengths.append(stop - start)
    values, lengths = np.asarray(values), np.asarray(lengths)
    correlation = (
        float(np.corrcoef(values, lengths)[0, 1]) if values.size > 2 else np.nan
    )
    return {"residual": values, "length": lengths, "correlation": correlation}


def ice_flip_phases(
    concentration: np.ndarray, psi: np.ndarray, threshold: float = 0.5,
    minimum_flips: int = 3,
) -> dict[str, np.ndarray]:
    """Phase of psi1 at which each cell's annual sea ice switches on and off.

    ``concentration`` is absolute annual cover (0–1), shape ``(years, ...)``.
    The phase of a switch between years t-1 and t is the phase of
    ``psi[t-1] + psi[t]``. For each cell: circular mean and circular standard
    deviation (``sqrt(-2 ln R)``) of the onset and retreat phases, and the
    number of onsets per complete cycle of psi1. Cells with fewer than
    ``minimum_flips`` onsets have NaN phases.
    """
    covered = np.asarray(concentration) >= threshold
    flat = covered.reshape(covered.shape[0], -1)
    midpoint = np.exp(1j * np.angle(psi[:-1] + psi[1:]))
    cycles = abs(np.unwrap(np.angle(psi))[-1] - np.unwrap(np.angle(psi))[0]) / (2 * np.pi)
    output = {}
    for name, switch in (
        ("onset", ~flat[:-1] & flat[1:]), ("retreat", flat[:-1] & ~flat[1:]),
    ):
        count = switch.sum(axis=0)
        resultant = (midpoint[:, None] * switch).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            length = np.abs(resultant) / count
            mean = np.mod(np.angle(resultant), 2 * np.pi)
            spread = np.sqrt(-2 * np.log(np.clip(length, 1e-12, 1.0)))
        few = count < minimum_flips
        output[f"{name}_phase"] = np.where(few, np.nan, mean).reshape(covered.shape[1:])
        output[f"{name}_std"] = np.where(few, np.nan, spread).reshape(covered.shape[1:])
        output[f"{name}_per_cycle"] = (count / max(cycles, 1e-12)).reshape(covered.shape[1:])
    output["cover_fraction"] = covered.mean(axis=0)
    return output


def load_absolute_map(
    root: Path, mu: str, years: np.ndarray, name: str = "sea_ice_concentration",
) -> np.ndarray:
    """Read one annual map in source units on the given contiguous years."""
    years = np.asarray(years, dtype=int)
    with h5py.File(source_file(root, mu), "r") as source:
        all_years = np.asarray(source["year"][:], dtype=int)
        first = int(np.searchsorted(all_years, years[0]))
        if not np.array_equal(all_years[first:first + years.size], years):
            raise ValueError("Map years do not match the Koopman state years")
        return np.asarray(source[name][first:first + years.size], dtype=float)


def slow_modes(
    rates: np.ndarray, leading_index: int, frequency_fraction: float = 0.25,
    count: int = 6, minimum_decay: float = 1e-4,
) -> np.ndarray:
    """Indices of the least damped stable modes with |Im λ| < fraction · ω1.

    One index per conjugate pair (Im λ >= 0). Modes decaying slower than
    ``minimum_decay`` (yr⁻¹) are the constant eigenfunction and are skipped.
    """
    omega = abs(rates[leading_index].imag)
    candidates = np.flatnonzero(
        (rates.real < -minimum_decay) & (rates.imag >= -1e-12)
        & (np.abs(rates.imag) < frequency_fraction * omega)
    )
    return candidates[np.argsort(-rates.real[candidates])][:count]


def slow_mode_projection(
    eigenfunctions: np.ndarray, residual: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Least-squares fit of a residual on the real and imaginary parts of the
    given eigenfunctions (columns). Returns the explained variance fraction,
    overall and per eigenfunction.
    """
    residual = np.asarray(residual, dtype=float).reshape(residual.shape[0], -1)
    residual = residual - residual.mean(axis=0)
    parts = [np.column_stack((f.real, f.imag)) for f in eigenfunctions.T]
    design = np.column_stack([np.ones(residual.shape[0])] + parts)
    coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    total = np.sum(residual**2)
    explained = 1 - np.sum((residual - design @ coefficients)**2) / total
    single = []
    for part in parts:
        local = np.column_stack((np.ones(residual.shape[0]), part))
        fit = local @ np.linalg.lstsq(local, residual, rcond=None)[0]
        single.append(1 - np.sum((residual - fit)**2) / total)
    return {"explained": float(explained), "per_mode": np.asarray(single)}


def residual_eofs(
    blocks: list[np.ndarray], weights: list[np.ndarray], count: int = 10,
) -> dict[str, np.ndarray]:
    """Joint area-weighted EOFs of several residual blocks (years × rows).

    Each block is scaled to unit total weighted variance so that no block
    dominates. Returns principal components (years × count), the patterns of
    each block in its own units, and the explained variance fractions.
    """
    scaled, scales = [], []
    for values, weight in zip(blocks, weights, strict=True):
        weighted = (values - values.mean(axis=0)) * np.sqrt(weight)[None]
        scale = np.sqrt(np.sum(weighted**2) / weighted.shape[0])
        scaled.append(weighted / scale)
        scales.append(scale)
    matrix = np.column_stack(scaled)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    count = min(count, s.size)
    pcs = u[:, :count] * s[:count]
    patterns, start = [], 0
    for values, weight, scale in zip(blocks, weights, scales, strict=True):
        stop = start + values.shape[1]
        patterns.append(vt[:count, start:stop] * scale / np.sqrt(weight)[None])
        start = stop
    return {
        "pcs": pcs, "patterns": patterns,
        "fractions": s[:count]**2 / np.sum(s**2),
    }


def _phase_features(phase: np.ndarray, harmonics: int) -> np.ndarray:
    columns = [np.ones_like(phase)]
    for k in range(1, harmonics + 1):
        columns += [np.cos(k * phase), np.sin(k * phase)]
    return np.column_stack(columns)


def fit_phase_linear_model(
    pcs: np.ndarray, phase: np.ndarray, harmonics: int = 2,
) -> dict[str, np.ndarray]:
    """Fit r(t+1) = A(theta_t) r(t) + c(theta_t) + noise by least squares.

    ``A(theta) = sum_j f_j(theta) A_j`` with Fourier features
    ``f = [1, cos theta, sin theta, ...]``; ``c`` uses the same features.
    """
    pcs = np.asarray(pcs, dtype=float)
    features = _phase_features(phase[:-1], harmonics)
    count = pcs.shape[1]
    design = np.column_stack((
        (features[:, :, None] * pcs[:-1, None, :]).reshape(features.shape[0], -1),
        features,
    ))
    coefficients, *_ = np.linalg.lstsq(design, pcs[1:], rcond=None)
    nf = features.shape[1]
    matrices = coefficients[:nf * count].reshape(nf, count, count).transpose(0, 2, 1)
    noise = pcs[1:] - design @ coefficients
    return {
        "harmonics": harmonics, "matrices": matrices,
        "forcing": coefficients[nf * count:], "noise_covariance": np.cov(noise.T),
    }


def propagator(model: dict, phase: float | np.ndarray) -> np.ndarray:
    """A(theta) for one or more phases."""
    features = _phase_features(np.atleast_1d(np.asarray(phase, dtype=float)), model["harmonics"])
    matrices = np.tensordot(features, model["matrices"], axes=(1, 0))
    return matrices[0] if np.ndim(phase) == 0 else matrices


def floquet_multipliers(model: dict, omega: float, start_phase: float = 0.0) -> dict:
    """Compose A over one mean cycle; return multipliers and local growth.

    The cycle has ``N = round(2 pi / omega)`` annual steps along
    ``theta_n = start_phase + omega n``. ``multipliers`` are the eigenvalues
    of the product, sorted by modulus; ``per_cycle`` rescales their modulus
    to exactly one period, ``|mu|^(P/N)``; ``rate`` is ``ln|mu| / P``
    (yr⁻¹). ``local_growth`` is the spectral radius of ``A(theta)`` on a
    phase grid, showing where along the cycle disturbances decay slowest.
    """
    period = 2 * np.pi / omega
    steps = max(int(round(period)), 1)
    product = np.eye(model["matrices"].shape[1])
    for matrix in propagator(model, start_phase + omega * np.arange(steps)):
        product = matrix @ product
    multipliers = np.linalg.eigvals(product)
    multipliers = multipliers[np.argsort(-np.abs(multipliers))]
    grid = bin_centres(36)
    local = np.array([
        np.max(np.abs(np.linalg.eigvals(matrix))) for matrix in propagator(model, grid)
    ])
    modulus = np.abs(multipliers)
    return {
        "multipliers": multipliers, "steps": steps, "period": period,
        "per_cycle": modulus ** (period / steps),
        "rate": np.log(modulus) / steps,
        "phase_grid": grid, "local_growth": local,
    }


def fit_residual_koopman(
    pcs: np.ndarray, phase: np.ndarray, lag: int = 1,
    rank_threshold: float = 5e-5, factorization_rel_threshold: float = 1e-5,
    maximum_training_snapshots: int = 10_000, seed: int = 0, clock_harmonics: int = 8,
) -> dict:
    """KDMD on the phase-augmented residual state [PCs, cos theta, sin theta].

    The clock coordinates keep the phase dependence of the residual dynamics.
    The two kernel blocks are scaled by their median pairwise distances, as
    in ``plasim_koopman_single.fit_koopman``. Each eigenfunction is split by
    the share of its variance explained by ``exp(i k theta)``, ``|k| <=
    clock_harmonics``: clock modes (the phase itself, rates near ``i k
    omega``) have a share near 1; transverse modes, describing the return of
    the residual to the cycle, have a share near 0. ``residual_share`` is
    the share explained by a linear function of the PCs; products of a
    transverse mode with ``exp(i k theta)`` (rates ``λ + i k omega``) have a
    small share and are the same decay seen at another clock harmonic.
    """
    from scipy.spatial.distance import pdist

    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.algorithms import KernelDMD, WeightedGaussianKernel
    from koopman_response.algorithms.regularization import TSVDRegularizer

    pcs = np.asarray(pcs, dtype=float)
    state = np.column_stack((pcs, np.cos(phase), np.sin(phase)))
    indices = single.kdmd_training_indices(
        state.shape[0], lag, maximum_training_snapshots, seed,
    )
    origin = state[indices]
    count = pcs.shape[1]
    residual_bandwidth = float(np.median(pdist(origin[:, :count])))
    clock_bandwidth = float(np.median(pdist(origin[:, count:])))
    kernel = WeightedGaussianKernel(
        sigma=1.0,
        weights=np.r_[
            np.full(count, 1 / residual_bandwidth**2), np.full(2, 1 / clock_bandwidth**2),
        ],
    )
    kdmd = KernelDMD(kernel=kernel)
    kdmd.fit_snapshots(X=origin, Y=state[indices + lag])
    factorization = TSVDRegularizer()
    factorization.factorize(kdmd.G, method="eigh", symmetrize=False,
                            rel_threshold=factorization_rel_threshold)
    matrix, basis, singular_values = factorization.solve_from_factorization(
        kdmd.A, rel_threshold=rank_threshold,
    )
    del kdmd
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        matrix, kernel=kernel, reference_data=origin,
        U_r=basis, S_r=singular_values,
    )
    rates = spectrum.continuous_time_eigenvalues(lag)
    eigenfunctions = np.asarray(spectrum.evaluate_eigenfunctions(state, batch_size=256))
    clock = np.column_stack([
        np.exp(1j * k * phase) for k in range(-clock_harmonics, clock_harmonics + 1)
    ])
    centred = eigenfunctions - eigenfunctions.mean(axis=0)
    fit = clock @ np.linalg.lstsq(clock, centred, rcond=None)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        clock_share = 1 - np.sum(np.abs(centred - fit)**2, axis=0) / np.sum(np.abs(centred)**2, axis=0)
    # Correlation of each eigenfunction with the residual PCs (how much of
    # the residual it carries).
    design = np.column_stack((np.ones(pcs.shape[0]), pcs))
    pcs_fit = design @ np.linalg.lstsq(design, centred, rcond=None)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        residual_share = 1 - np.sum(np.abs(centred - pcs_fit)**2, axis=0) / np.sum(np.abs(centred)**2, axis=0)
    return {
        "lag": lag, "rank": singular_values.size, "rates": rates,
        "clock_share": clock_share, "residual_share": residual_share,
        "eigenfunctions": eigenfunctions, "bandwidths": (residual_bandwidth, clock_bandwidth),
    }


def transverse_modes(
    residual_koopman: dict, maximum_clock_share: float = 0.5,
    minimum_residual_share: float = 0.2, count: int = 8,
) -> np.ndarray:
    """Least damped stable modes (Im λ >= 0) that are not clock modes and
    are carried linearly by the residual PCs."""
    rates = residual_koopman["rates"]
    candidates = np.flatnonzero(
        (rates.real < -1e-4) & (rates.imag >= -1e-12)
        & (residual_koopman["clock_share"] < maximum_clock_share)
        & (residual_koopman["residual_share"] >= minimum_residual_share)
    )
    return candidates[np.argsort(-rates.real[candidates])][:count]


def transverse_mode_patterns(
    residual_koopman: dict, index: int, blocks: dict,
) -> dict[str, np.ndarray]:
    """Zonal residual patterns carried by one residual-KDMD eigenfunction.

    The eigenfunction is centred and scaled to unit RMS; each pattern is the
    real part of ``<conj(phi) r>``, in the block's units per unit phi.
    """
    phi = residual_koopman["eigenfunctions"][:, index]
    phi = phi - phi.mean()
    phi = phi / np.sqrt(np.mean(np.abs(phi)**2))
    patterns = {}
    for name, block in blocks.items():
        pattern = np.mean(np.conj(phi)[:, None] * block["residuals"]["amplitude"], axis=0)
        patterns[name] = pattern
    # Real modes: rotate so the pattern is real and its largest ocean value positive.
    reference = patterns.get("ocean", next(iter(patterns.values())))
    peak = reference[np.argmax(np.abs(reference))]
    rotation = np.conj(peak) / abs(peak) if abs(peak) else 1.0
    return {name: np.real(rotation * value) for name, value in patterns.items()} | {
        "eigenfunction": phi * np.conj(rotation),
    }


def lagged_correlation(
    first: np.ndarray, second: np.ndarray, lags: np.ndarray,
) -> np.ndarray:
    """corr(first(t), second(t + lag)) for each lag (years)."""
    output = []
    for lag in lags:
        if lag >= 0:
            a, b = first[:first.size - lag], second[lag:]
        else:
            a, b = first[-lag:], second[:second.size + lag]
        output.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(output)


def cycle_tangent_and_shape(
    values: np.ndarray, psi: np.ndarray, bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Per year, the mean-cycle slope dC/dtheta and the mean-cycle value C.

    ``C`` is the phase composite of ``values`` (years × rows) on arg(psi1);
    the slope is a periodic central difference across bins. If the true
    phase is ahead of the clock by ``delta`` (rad) and the cycle is scaled by
    ``1 + alpha``, the residual is about ``delta * slope + alpha * C``.
    """
    values = np.asarray(values, dtype=float)
    index = phase_bin_index(wrapped_phase(psi), bins)
    composite = _group_means(values, index, bins)
    slope = (np.roll(composite, -1, axis=0) - np.roll(composite, 1, axis=0)) / (4 * np.pi / bins)
    return slope[index], composite[index]


def estimate_clock_error(
    residuals: list[np.ndarray], slopes: list[np.ndarray], shapes: list[np.ndarray],
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-year clock error delta (rad, true phase minus clock) and amplitude error alpha.

    Weighted least squares, each year separately, of the stacked residual
    rows on ``[slope, shape]``. Each block is scaled by its total residual
    standard deviation so no block dominates.
    """
    rows_r, rows_t, rows_c = [], [], []
    for residual, slope, shape, weight in zip(residuals, slopes, shapes, weights, strict=True):
        scale = np.sqrt(np.average(np.var(residual, axis=0), weights=weight))
        root = np.sqrt(weight / weight.sum())[None] / scale
        rows_r.append(residual * root)
        rows_t.append(slope * root)
        rows_c.append(shape * root)
    r = np.concatenate(rows_r, axis=1)
    t = np.concatenate(rows_t, axis=1)
    c = np.concatenate(rows_c, axis=1)
    tt, cc, tc = np.sum(t * t, 1), np.sum(c * c, 1), np.sum(t * c, 1)
    tr, cr = np.sum(t * r, 1), np.sum(c * r, 1)
    determinant = tt * cc - tc**2
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = (cc * tr - tc * cr) / determinant
        alpha = (tt * cr - tc * tr) / determinant
    return np.nan_to_num(delta), np.nan_to_num(alpha)


def remove_clock_error(
    residual: np.ndarray, slope: np.ndarray, shape: np.ndarray,
    delta: np.ndarray, alpha: np.ndarray,
) -> np.ndarray:
    """Residual minus the part explained by the clock and amplitude errors."""
    extra = (1,) * (np.ndim(residual) - 1)
    return residual - delta.reshape(-1, *extra) * slope - alpha.reshape(-1, *extra) * shape


def latitude_lag_correlation(
    predictor: np.ndarray, field: np.ndarray, lags: np.ndarray,
) -> np.ndarray:
    """corr(field row(t), predictor(t + lag)) for every row and lag: (rows, lags)."""
    return np.array([
        lagged_correlation(field[:, row], predictor, lags) for row in range(field.shape[1])
    ])


def surrogate_threshold(
    predictor: np.ndarray, target: np.ndarray, lags: np.ndarray,
    count: int = 200, quantile: float = 0.95, seed: int = 0,
) -> float:
    """Quantile of max |lagged correlation| with phase-randomized predictors.

    The surrogates keep the predictor's power spectrum (so its memory) but
    destroy any relation with ``target``.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(predictor - predictor.mean())
    maxima = []
    for _ in range(count):
        phases = np.exp(2j * np.pi * rng.random(spectrum.size))
        phases[0] = 1.0
        surrogate = np.fft.irfft(spectrum * phases, n=predictor.size)
        maxima.append(np.max(np.abs(lagged_correlation(target, surrogate, lags))))
    return float(np.quantile(maxima, quantile))


def lagged_regression_maps(
    predictor: np.ndarray, field: np.ndarray, lags: np.ndarray,
) -> np.ndarray:
    """Slope of field(t + lag) on the standardized predictor(t): (lags, ...)."""
    x = (predictor - predictor.mean()) / predictor.std()
    output = []
    for lag in lags:
        if lag >= 0:
            a, b = x[:x.size - lag], field[lag:]
        else:
            a, b = x[-lag:], field[:field.shape[0] + lag]
        output.append(np.tensordot(a, b - b.mean(axis=0), axes=1) / a.size)
    return np.asarray(output)


def coupling_analysis(
    result: dict, lags: np.ndarray = np.arange(-15, 16), surrogates: int = 200,
    clock_exclusion: tuple[float, float] = (-40.0, -15.0),
    residual_kind: str = "amplitude", band: tuple[float, float] = (-27.0, -22.0),
    leading_lags: tuple[int, int] = (-8, -2),
) -> dict:
    """Ice-residual relation with the ocean residual, by latitude and lag.

    Repeated after removing the clock and amplitude errors of psi1, estimated
    per year from the zonal Ts rows and the ocean rows outside
    ``clock_exclusion`` (so a genuine reservoir anomaly is not mistaken for a
    clock error), and applied to the ice and every ocean row. An exclusion
    covering every ocean row uses the surface rows alone. ``ice_leading``
    is the most negative correlation over all rows at ``leading_lags``.
    """
    fields, psi = result["fields"], result["psi1"]
    ice = result["ice_residuals"][residual_kind][:, 0]
    ocean = result["blocks"]["ocean"]["residuals"][residual_kind]
    surface = result["blocks"]["surface"]["residuals"][residual_kind]
    ocean_lat = fields["ocean_lat"]
    ocean_slope, ocean_shape = cycle_tangent_and_shape(1e3 * fields["ocean_anomaly"], psi)
    surface_slope, surface_shape = cycle_tangent_and_shape(fields["surface_anomaly"], psi)
    ice_slope, ice_shape = cycle_tangent_and_shape(fields["ice_anomaly"][:, None], psi)
    outside = (ocean_lat < clock_exclusion[0]) | (ocean_lat > clock_exclusion[1])
    blocks = [(surface, surface_slope, surface_shape, fields["surface_weights"])]
    if outside.any():
        blocks.append((ocean[:, outside], ocean_slope[:, outside], ocean_shape[:, outside],
                       fields["ocean_weights"][outside]))
    delta, alpha = estimate_clock_error(*map(list, zip(*blocks)))
    ice_clean = remove_clock_error(ice[:, None], ice_slope, ice_shape, delta, alpha)[:, 0]
    ocean_clean = remove_clock_error(ocean, ocean_slope, ocean_shape, delta, alpha)
    output = {"lags": lags, "ocean_lat": ocean_lat, "delta": delta, "alpha": alpha}
    for name, x, field in (("raw", ice, ocean), ("clock removed", ice_clean, ocean_clean)):
        table = latitude_lag_correlation(x, field, lags)
        row, column = np.unravel_index(np.argmax(np.abs(table)), table.shape)
        reservoir = (ocean_lat >= band[0]) & (ocean_lat <= band[1])
        band_series = np.average(field[:, reservoir], axis=1, weights=fields["ocean_weights"][reservoir])
        band_correlation = lagged_correlation(band_series, x, lags)
        strongest = int(np.argmax(np.abs(band_correlation)))
        output[name] = {
            "table": table,
            "peak": (float(table[row, column]), float(ocean_lat[row]), int(lags[column])),
            "band_peak": (float(band_correlation[strongest]), int(lags[strongest])),
            "band_threshold": surrogate_threshold(x, band_series, lags, count=surrogates),
            "ice": x, "band": band_series,
        }
        leading = (lags >= leading_lags[0]) & (lags <= leading_lags[1])
        sub = table[:, leading]
        r_index, l_index = np.unravel_index(np.argmin(sub), sub.shape)
        output[name]["ice_leading"] = (
            float(sub[r_index, l_index]), float(ocean_lat[r_index]), int(lags[leading][l_index]),
        )
    explained = 1 - np.var(ice_clean) / np.var(ice)
    output["ice variance explained by clock error"] = float(explained)
    output["clock removed ocean"] = ocean_clean
    return output


def irregularity_summary(
    psi: np.ndarray, years: np.ndarray, band_residuals: dict[str, np.ndarray],
    bins: int = 18,
) -> dict[str, float]:
    """Scalar irregularity metrics for one stretch of years."""
    phase = wrapped_phase(psi)
    lengths = cycle_lengths(psi, years)
    clock = phase_advance_stats(psi, years, bins)
    summary = {
        "advance_std": clock["advance_std_total"],
        "amplitude_cv": clock["amplitude_cv"],
        "diffusion": phase_diffusion(
            psi, maximum_lag=min(100, years.size // 4),
            fit_range=(5, min(100, years.size // 4)),
        )["diffusion"],
        "cycle_mean": float(np.mean(lengths)) if lengths.size else np.nan,
        "cycle_cv": float(np.std(lengths) / np.mean(lengths)) if lengths.size > 1 else np.nan,
    }
    for name, series in band_residuals.items():
        stats = phase_conditioned_stats(series, phase, bins)
        peak = int(np.nanargmax(stats["variance"]))
        lag_one = lag_one_by_phase(series, phase, bins)
        summary[f"{name} variance"] = float(np.var(series))
        summary[f"{name} peak variance"] = float(stats["variance"][peak])
        summary[f"{name} peak phase"] = float(stats["centres"][peak] / (2 * np.pi))
        summary[f"{name} lag-one at peak"] = float(lag_one[peak])
        summary[f"{name} lag-one"] = float(
            np.sum(series[1:] * series[:-1]) / np.sum(series[:-1]**2)
        )
    return summary


def block_summaries(
    psi: np.ndarray, years: np.ndarray, band_residuals: dict[str, np.ndarray],
    block_years: int = 1000,
) -> list[dict[str, float]]:
    """``irregularity_summary`` on consecutive blocks (psi kept fixed)."""
    output = []
    for start in range(0, years.size - block_years + 1, block_years):
        selected = slice(start, start + block_years)
        output.append(irregularity_summary(
            psi[selected], years[selected],
            {name: series[selected] for name, series in band_residuals.items()},
        ))
    return output


# Band means used for the scalar residual series: (block, label, latitude range).
RESIDUAL_BANDS = (
    ("surface", "edge Ts 28–45°S", (-45.0, -28.0)),
    ("ocean", "θ 0–700 m 33–42°S", (-42.0, -33.0)),
    ("ocean", "θ 0–700 m 22–27°S", (-27.0, -22.0)),
)


def irregularity_analysis(
    root: Path, mu: str, *, window: tuple[int, int] | None = None,
    maximum_depth: float = 700.0, lag: int = 5,
    factorization_rel_threshold: float = 1e-5, rank_threshold: float = 5e-5,
    maximum_training_snapshots: int = 10_000, seed: int = 0,
    eof_count: int = 10, floquet_harmonics: int = 2, bins: int = 36,
    residual_koopman_lag: int = 1, state_blocks: tuple[str, ...] = ("surface", "ocean"),
) -> dict:
    """Fit KDMD on one window and compute every state-based irregularity result.

    The Koopman state is the zonal blocks in ``state_blocks`` (``"surface"``,
    ``"ocean"``, ``"ice"``; see ``plasim_koopman_single.combined_state``). Ocean
    values are in mK, surface values in K, ice area in 10⁶ km².
    """
    start, end = window or ANALYSIS_WINDOWS[mu]
    fields = single.load_fields(root, start - 1, maximum_depth, mu, end_year=end)
    fields = single.combined_state(fields, state_blocks)
    spectrum, rates, rank, _ = single.fit_koopman(
        fields, lag, rank_threshold, maximum_training_snapshots, seed,
        factorization_rel_threshold,
    )
    leading, psi1, harmonic, psi2, scales = single.leading_and_harmonic_eigenfunctions(
        spectrum, rates, fields,
    )
    omega = float(rates[leading].imag)
    settings = {"lag": lag, "maximum_training_snapshots": maximum_training_snapshots,
                "seed": seed}
    state_modes = single.state_mode_comparison(
        spectrum, fields, rates, leading, psi1, harmonic, psi2, scales, **settings,
    )
    indices = single.kdmd_training_indices(
        fields["state"].shape[0], lag, maximum_training_snapshots, seed,
    )
    ice_modes = single.observable_modes(
        spectrum, fields["state"], fields["ice_anomaly"][:, None], indices,
        [leading, harmonic], scales,
    )
    phase = wrapped_phase(psi1)
    blocks = {}
    for name, key, scale, unit, label in (
        ("surface", "surface_anomaly", 1.0, "K", "zonal Ts"),
        ("ocean", "ocean_anomaly", 1e3, "mK", "zonal θ 0–700 m"),
    ):
        blocks[name] = {
            "lat": fields[f"{name}_lat"], "weights": fields[f"{name}_weights"],
            "unit": unit, "label": label,
            "residuals": cycle_residuals(
                scale * fields[key], psi1, psi2, state_modes[name]["direct"], bins,
            ),
        }
    ice_residuals = cycle_residuals(
        fields["ice_anomaly"][:, None], psi1, psi2, ice_modes, bins,
    )
    band_residuals = {"SA ice area": ice_residuals["amplitude"][:, 0]}
    for block, label, (south, north) in RESIDUAL_BANDS:
        lat, weights = blocks[block]["lat"], blocks[block]["weights"]
        selected = (lat >= south) & (lat <= north)
        band_residuals[label] = np.average(
            blocks[block]["residuals"]["amplitude"][:, selected], axis=1,
            weights=weights[selected],
        )
    years = fields["years"]
    summary = irregularity_summary(psi1, years, band_residuals)
    for name, block in blocks.items():
        for kind, fraction in block["residuals"]["fractions"].items():
            summary[f"{name} residual fraction ({kind})"] = fraction

    slow = slow_modes(rates, leading)
    evaluated = spectrum.evaluate_eigenfunctions(fields["state"], batch_size=256)
    slow_functions = np.array(evaluated[:, slow], dtype=complex, copy=True)
    del evaluated
    eofs = residual_eofs(
        [blocks["surface"]["residuals"]["amplitude"], blocks["ocean"]["residuals"]["amplitude"]],
        [blocks["surface"]["weights"], blocks["ocean"]["weights"]], eof_count,
    )
    slow_projection = slow_mode_projection(slow_functions, eofs["pcs"])
    model = fit_phase_linear_model(eofs["pcs"], phase, floquet_harmonics)
    floquet = floquet_multipliers(model, omega)
    summary["Floquet |μ|max per cycle"] = float(floquet["per_cycle"][0])
    summary["Floquet slowest rate (yr⁻¹)"] = float(floquet["rate"][0])
    summary["slow modes explain residual EOFs"] = slow_projection["explained"]
    residual_koopman = fit_residual_koopman(
        eofs["pcs"], phase, residual_koopman_lag, rank_threshold,
        factorization_rel_threshold, maximum_training_snapshots, seed,
    )
    transverse = transverse_modes(residual_koopman)
    residual_koopman["transverse"] = transverse
    if transverse.size:
        residual_koopman["patterns"] = transverse_mode_patterns(
            residual_koopman, int(transverse[0]), blocks,
        )
        summary["residual KDMD slowest transverse rate (yr⁻¹)"] = float(
            residual_koopman["rates"][transverse[0]].real
        )
    clock_modes = np.flatnonzero(
        (residual_koopman["clock_share"] > 0.8) & (residual_koopman["rates"].imag > 1e-3)
    )
    if clock_modes.size:
        fundamental = clock_modes[np.argmin(np.abs(residual_koopman["rates"][clock_modes].imag - omega))]
        summary["residual KDMD clock-mode decay (yr⁻¹)"] = float(
            -residual_koopman["rates"][fundamental].real
        )
    next_cycle = {
        name: residual_versus_cycle_length(series, psi1, years)
        for name, series in band_residuals.items()
    }
    return {
        "mu": mu, "fields": fields, "spectrum": spectrum, "rates": rates, "rank": rank,
        "leading_index": leading, "harmonic_index": harmonic,
        "psi1": psi1, "psi2": psi2, "mode_scales": scales, "omega": omega,
        "period": 2 * np.pi / omega, "phase": phase,
        "state_modes": state_modes, "blocks": blocks, "ice_residuals": ice_residuals,
        "band_residuals": band_residuals, "summary": summary,
        "block_summaries": block_summaries(psi1, years, band_residuals),
        "slow_indices": slow, "slow_projection": slow_projection,
        "eofs": eofs, "model": model, "floquet": floquet, "next_cycle": next_cycle,
        "residual_koopman": residual_koopman, "settings": settings,
    }


RESERVOIR_BAND = "θ 0–700 m 22–27°S"


def weak_episodes(
    psi: np.ndarray, years: np.ndarray, smoothing_years: int = 101,
    threshold: float = 0.75,
) -> dict[str, np.ndarray]:
    """Stretches where the running mean of |psi1| falls below ``threshold``.

    |psi1| has unit RMS, so the threshold is relative to a typical cycle. The
    running mean uses only full windows (``mode="valid"``), padded at the
    ends with the nearest full-window value, so the record ends do not
    create spurious episodes.
    """
    amplitude = np.abs(np.asarray(psi))
    running = np.convolve(amplitude, np.ones(smoothing_years) / smoothing_years, mode="valid")
    half = smoothing_years // 2
    running = np.r_[np.full(half, running[0]), running,
                    np.full(amplitude.size - running.size - half, running[-1])]
    weak = running < threshold
    change = np.diff(np.r_[0, weak.astype(int), 0])
    starts, stops = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    return {
        "running_amplitude": running, "weak": weak,
        "episodes": np.array([(years[a], years[b - 1]) for a, b in zip(starts, stops)]).reshape(-1, 2),
    }


def headline_metrics(
    result: dict, *, eof_count: int | None = None,
    residual_koopman_lag: int | None = None, floquet_harmonics: int | None = None,
) -> dict[str, float]:
    """The main irregularity numbers of one ``irregularity_analysis`` result.

    With any keyword given, the residual EOFs, the Floquet model, and the
    residual KDMD are recomputed with that setting (no refit of the main
    KDMD).
    """
    summary = result["summary"]
    psi, years, phase = result["psi1"], result["fields"]["years"], result["phase"]
    fields = result["fields"]
    if eof_count is None and residual_koopman_lag is None and floquet_harmonics is None:
        floquet_rate = summary["Floquet slowest rate (yr⁻¹)"]
        transverse_rate = summary.get("residual KDMD slowest transverse rate (yr⁻¹)", np.nan)
    else:
        blocks = result["blocks"]
        eofs = residual_eofs(
            [blocks["surface"]["residuals"]["amplitude"], blocks["ocean"]["residuals"]["amplitude"]],
            [blocks["surface"]["weights"], blocks["ocean"]["weights"]],
            eof_count or result["eofs"]["pcs"].shape[1],
        )
        model = fit_phase_linear_model(
            eofs["pcs"], phase, floquet_harmonics or result["model"]["harmonics"],
        )
        floquet_rate = float(floquet_multipliers(model, result["omega"])["rate"][0])
        koopman = fit_residual_koopman(
            eofs["pcs"], phase, residual_koopman_lag or result["residual_koopman"]["lag"],
        )
        transverse = transverse_modes(koopman)
        transverse_rate = float(koopman["rates"][transverse[0]].real) if transverse.size else np.nan
    reservoir = result["band_residuals"][RESERVOIR_BAND]
    lat = result["blocks"]["ocean"]["lat"]
    band = (lat >= -27) & (lat <= -22)
    index = phase_bin_index(phase, 36)
    ocean = 1e3 * fields["ocean_anomaly"][:, band] @ fields["ocean_weights"][band] / fields["ocean_weights"][band].sum()
    cycle = _group_means(ocean, index, 36)[index]
    lags = np.arange(-15, 16)
    coupling = lagged_correlation(reservoir, result["band_residuals"]["SA ice area"], lags)
    strongest = int(np.argmax(np.abs(coupling)))
    edges = cycle_crossings(psi, years)
    lengths = np.diff(edges)
    amplitude = [np.abs(psi[(years >= a) & (years < b)]).mean() for a, b in zip(edges[:-1], edges[1:])]
    ocean_stats = phase_conditioned_stats(result["blocks"]["ocean"]["residuals"]["amplitude"], phase, 12)
    ocean_variance = ocean_stats["variance"] @ result["blocks"]["ocean"]["weights"]
    return {
        "fraction of weak-cycle years": float(weak_episodes(psi, years)["weak"].mean()),
        "period (yr)": result["period"],
        "phase diffusion D": summary["diffusion"],
        "cycle-length CV": summary["cycle_cv"],
        "|ψ₁| CV": summary["amplitude_cv"],
        "Floquet return time (yr)": -1 / floquet_rate,
        "residual-KDMD return time (yr)": -1 / transverse_rate,
        "reservoir residual / cycle std": float(np.std(reservoir) / np.std(cycle)),
        "ice→reservoir max |corr|": float(coupling[strongest]),
        "lag of max |corr| (yr)": float(lags[strongest]),
        "ice-area residual lag-one": summary["SA ice area lag-one"],
        "corr(cycle |ψ₁|, length)": float(np.corrcoef(amplitude, lengths)[0, 1]),
        "ocean variance max/min by phase": float(ocean_variance.max() / ocean_variance.min()),
    }


def robustness_sweep(
    root: Path, mu: str, base: dict | None = None, progress=None,
) -> dict[str, dict[str, float]]:
    """Headline metrics under changes of each analysis choice, one at a time.

    Refitted variants: main KDMD lag 3 and 7, each half of the window, and 18
    composite bins. Cheap variants (main fit kept): 5 and 15 residual EOFs,
    residual-KDMD lag 2 and 5, and 1 and 3 Floquet harmonics.
    """
    base = base or irregularity_analysis(root, mu)
    start, end = ANALYSIS_WINDOWS[mu]
    middle = (start + end) // 2
    variants = {"baseline": headline_metrics(base)}
    refits = {
        "main lag 3": {"lag": 3}, "main lag 7": {"lag": 7},
        "first half": {"window": (start, middle)},
        "second half": {"window": (middle + 1, end)},
        "18 composite bins": {"bins": 18},
    }
    for name, settings in refits.items():
        if progress:
            progress(f"{mu}: {name}")
        variants[name] = headline_metrics(irregularity_analysis(root, mu, **settings))
    cheap = {
        "5 residual EOFs": {"eof_count": 5}, "15 residual EOFs": {"eof_count": 15},
        "residual-KDMD lag 2": {"residual_koopman_lag": 2},
        "residual-KDMD lag 5": {"residual_koopman_lag": 5},
        "1 Floquet harmonic": {"floquet_harmonics": 1},
        "3 Floquet harmonics": {"floquet_harmonics": 3},
    }
    for name, settings in cheap.items():
        if progress:
            progress(f"{mu}: {name}")
        variants[name] = headline_metrics(base, **settings)
    return variants


def make_robustness_figure(sweeps: dict[str, dict[str, dict[str, float]]]) -> plt.Figure:
    """One panel per metric: every variant (dots) and baseline (diamond) per μ."""
    labels = list(sweeps)
    metrics = list(next(iter(sweeps.values()))["baseline"])
    columns = 4
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 2.5 * rows),
                                layout="constrained", squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        for position, label in enumerate(labels):
            values = [variant[metric] for name, variant in sweeps[label].items() if name != "baseline"]
            jitter = np.linspace(-0.15, 0.15, len(values))
            axis.plot(position + jitter, values, "o", color="#4472a0", alpha=0.6, ms=4)
            axis.plot(position, sweeps[label]["baseline"][metric], "D", color="#b44b38", ms=7)
        axis.set_xticks(range(len(labels)), [label.replace("p", ".") for label in labels])
        axis.set_xlim(-0.5, len(labels) - 0.5)
        axis.set_title(metric, fontsize="small")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    figure.suptitle("Robustness: baseline (diamond) and one-at-a-time variants (dots)")
    return figure


MAP_LAGS = np.array([-4, 0, 2, 4, 6, 8])
SURFACE_FLUXES = ("rss", "rls", "hfss", "hfls")  # all positive downward


def ice_regression_entry(
    root: Path, result: dict, coupling: dict, maps: dict, lags: np.ndarray = MAP_LAGS,
) -> dict:
    """Residual maps (and net surface heat flux) regressed on the ice residual.

    ``maps`` comes from ``plasim_koopman_single.load_map_fields`` on the
    analysis years. Uses both the raw and the clock-removed ice residual.
    """
    psi, years = result["psi1"], result["fields"]["years"]
    predictors = {kind: coupling[kind]["ice"] for kind in ("raw", "clock removed")}
    regressions = {}
    for name, field in maps["fields"].items():
        residual = cycle_residuals(field["scale"] * field["anomaly"], psi, None, None)["amplitude"]
        regressions[name] = {
            kind: lagged_regression_maps(x, residual, lags).astype(np.float32)
            for kind, x in predictors.items()
        } | {"label": field["label"], "unit": field["unit"], "grid": field["grid"], "wet": field["wet"]}
        del residual
    flux = sum(load_absolute_map(root, result["mu"], years, name) for name in SURFACE_FLUXES)
    residual = cycle_residuals(flux - flux.mean(axis=0), psi, None, None)["amplitude"]
    regressions["net_surface_flux"] = {
        kind: lagged_regression_maps(x, residual, lags).astype(np.float32)
        for kind, x in predictors.items()
    } | {"label": "Net surface heat flux (down +)", "unit": "W m⁻²", "grid": "t21",
         "wet": np.ones(flux.shape[1:], bool)}
    return {
        "mu": result["mu"], "map_lags": lags, "regressions": regressions,
        "grid": {key: maps[key] for key in (
            "t21_lat", "t21_lon", "lsm", "lsg_lat", "lsg_lon", "t21_basin", "lsg_basin")},
    }


SEAWATER_HEAT_CAPACITY = 1025.0 * 3990.0  # J m⁻³ K⁻¹


def gyre_heat_budget(
    root: Path, mu: str, years: np.ndarray, south_face: float = -31.25,
    north_face: float = -16.25, maximum_depth: float = 700.0,
) -> dict[str, np.ndarray]:
    """Annual heat budget (TW) of the South Atlantic upper ocean between two faces.

    Box: wet cells of the South Atlantic basin mask with latitude strictly
    between the two vector-row faces, layers down to ``maximum_depth``.

    - ``storage``: centred difference of the box heat content;
    - ``surface``: T21 net surface heat flux (rss + rls + hfss + hfls, down
      positive) over South Atlantic T21 ocean cells in the same band; over
      ice it is the flux into the ice surface, not the ocean;
    - ``advection``: convergence of the Atlantic ``vbar*thetabar`` proxy
      through the two faces, relative to the box-mean temperature so that net
      volume exchange (through the bottom) does not carry heat;
    - ``residual``: storage − surface − advection (vertical exchange at the
      box bottom, eddies and sub-annual covariance, grid offsets).

    The faces are E-grid vector rows, half a row from the scalar box edges;
    the residual absorbs that offset.
    """
    years = np.asarray(years, dtype=int)
    with h5py.File(source_file(root, mu), "r") as source, h5py.File(
        single.basin_mask_file(root, mu), "r"
    ) as masks:
        all_years = np.asarray(source["year"][:], dtype=int)
        first = int(np.flatnonzero(all_years == years[0])[0])
        stop = first + years.size
        if not np.array_equal(all_years[first:stop], years):
            raise ValueError("Budget years do not match the archive")
        depth_bounds = np.asarray(source["depth_bounds"][:], dtype=float)
        levels = np.flatnonzero(depth_bounds[:, 1] <= maximum_depth)
        lat = np.asarray(source["lat"][:], dtype=float)
        rows = np.flatnonzero((lat[:, 0] > south_face) & (lat[:, 0] < north_face))
        basin = np.asarray(masks["lsg_scalar_south_atlantic"][:], dtype=bool)
        volume = np.asarray(source["wet_cell_volume"][:], dtype=float)[levels][:, rows]
        volume = np.where(basin[rows][None] & (volume > 0), volume, 0.0)
        heat = np.empty(years.size)
        box_temperature = np.empty(years.size)
        for begin in range(0, years.size, 50):
            end = min(begin + 50, years.size)
            temperature = np.asarray(
                source["temperature_upper"][first + begin:first + end, levels.min():levels.max() + 1],
                dtype=float,
            )[:, :, rows]
            temperature = np.where(volume[None] > 0, np.nan_to_num(temperature), 0.0)
            total = np.einsum("tdrl,drl->t", temperature, volume)
            heat[begin:end] = SEAWATER_HEAT_CAPACITY * total
            box_temperature[begin:end] = total / volume.sum()
        vector_lat = np.asarray(source["lsg_vector_lat"][:], dtype=float)
        faces = [int(np.argmin(np.abs(vector_lat - value))) for value in (south_face, north_face)]
        transport = np.asarray(source["atlantic_temperature_transport_proxy"][first:stop], dtype=float)
        volume_transport = np.asarray(source["atlantic_volume_transport"][first:stop], dtype=float)
        selected = levels
        temperature_flux = transport[:, faces][:, :, selected].sum(axis=2)
        volume_flux = volume_transport[:, faces][:, :, selected].sum(axis=2)
        relative = temperature_flux - box_temperature[:, None] * volume_flux
        advection = SEAWATER_HEAT_CAPACITY * (relative[:, 0] - relative[:, 1])
        t21_lat = np.asarray(source["t21_lat"][:], dtype=float)
        weights = np.asarray(source["t21_gaussian_weight"][:], dtype=float)
        lsm = np.asarray(source["lsm"][:], dtype=float)
        t21_basin = np.asarray(masks["t21_south_atlantic"][:], dtype=bool)
        band_cells = (
            t21_basin & (lsm < 0.5)
            & ((t21_lat > south_face) & (t21_lat < north_face))[:, None]
        )
        # Gaussian weights sum to 2, so the cell area is R² Δλ w.
        area = 6.371e6**2 * (2 * np.pi / lsm.shape[1]) * weights[:, None] * np.ones_like(lsm)
        surface = np.zeros(years.size)
        for name in SURFACE_FLUXES:
            values = np.asarray(source[name][first:stop], dtype=float)
            surface += np.sum(values * np.where(band_cells, area, 0.0)[None], axis=(1, 2))
    storage = np.gradient(heat) / (365.25 * 86400)
    scale = 1e-12
    return {
        "storage": storage * scale, "surface": surface * scale,
        "advection": advection * scale,
        "residual": (storage - surface - advection) * scale,
        "heat_content": heat, "box_temperature": box_temperature,
        "rows": lat[rows, 0], "faces": vector_lat[faces],
        "box_volume": float(volume.sum()), "band_area": float(np.where(band_cells, area, 0.0).sum()),
    }


def budget_regression(
    budget: dict[str, np.ndarray], predictor: np.ndarray, psi: np.ndarray,
    lags: np.ndarray = np.arange(-6, 13),
) -> dict[str, np.ndarray]:
    """Phase × |psi1| composite removed from each budget term, then regressed
    on the standardized predictor at each lag (TW per std; lag > 0: term after
    predictor). Also the cumulative storage response (PJ) for scale."""
    output = {"lags": lags}
    for name in ("storage", "surface", "advection", "residual"):
        residual = cycle_residuals(budget[name][:, None], psi, None, None)["amplitude"][:, 0]
        output[name] = lagged_regression_maps(predictor, residual[:, None], lags)[:, 0]
    content = cycle_residuals(budget["heat_content"][:, None], psi, None, None)["amplitude"][:, 0]
    output["heat_content"] = lagged_regression_maps(predictor, content[:, None], lags)[:, 0] * 1e-15
    return output


def coupling_robustness(
    root: Path, mu: str, base: dict | None = None, surrogates: int = 100, progress=None,
) -> dict[str, dict]:
    """Clock-removed coupling under one-at-a-time changes.

    Refitted: main KDMD lag 3 and 7, each window half, 18 composite bins.
    Same fit: clock estimated from surface rows only, or with a narrower or
    wider excluded band; phase-only composite residual; 25–30°S band.
    """
    base = base or irregularity_analysis(root, mu)
    start, end = ANALYSIS_WINDOWS[mu]
    middle = (start + end) // 2

    def metrics(result, **options):
        coupling = coupling_analysis(result, surrogates=surrogates, **options)
        clean = coupling["clock removed"]
        return {
            "band correlation": clean["band_peak"][0], "band lag": clean["band_peak"][1],
            "band threshold": clean["band_threshold"],
            "ice-leading correlation": clean["ice_leading"][0],
            "ice-leading latitude": clean["ice_leading"][1],
            "ice-leading lag": clean["ice_leading"][2],
            "clock share of ice variance": coupling["ice variance explained by clock error"],
        }

    variants = {"baseline": metrics(base)}
    for name, options in {
        "clock from Ts only": {"clock_exclusion": (-91.0, 91.0)},
        "clock excl. 20–35°S": {"clock_exclusion": (-35.0, -20.0)},
        "clock excl. 10–45°S": {"clock_exclusion": (-45.0, -10.0)},
        "phase-only composite": {"residual_kind": "composite"},
        "band 25–30°S": {"band": (-30.0, -25.0)},
    }.items():
        if progress:
            progress(f"{mu}: {name}")
        variants[name] = metrics(base, **options)
    for name, settings in {
        "main lag 3": {"lag": 3}, "main lag 7": {"lag": 7},
        "first half": {"window": (start, middle)},
        "second half": {"window": (middle + 1, end)},
        "18 composite bins": {"bins": 18},
    }.items():
        if progress:
            progress(f"{mu}: {name}")
        variants[name] = metrics(irregularity_analysis(root, mu, **settings))
    return variants


def coupled_mode_search(
    result: dict, coupling: dict, frequency_fraction: float = 0.5, count: int = 10,
    band: tuple[float, float] = (-30.0, -25.0),
) -> dict:
    """Find the slow Koopman modes of the fit that carry the ice–gyre anomaly.

    For the ``count`` least damped stable modes with ``|Im λ| <
    frequency_fraction · ω1`` (constant mode excluded), the eigenfunction
    (real and imaginary parts, centred) is regressed on the clock-removed ice
    residual and on the clock-removed ocean residual averaged over ``band``.
    Their R² measure how much of each the mode carries. The direct KDMD modes
    of every state block (``observable_modes``) give the spatial pattern.
    """
    fields, spectrum, rates = result["fields"], result["spectrum"], result["rates"]
    indices = slow_modes(rates, result["leading_index"], frequency_fraction, count)
    evaluated = spectrum.evaluate_eigenfunctions(fields["state"], batch_size=256)
    functions = np.array(evaluated[:, indices], dtype=complex, copy=True)
    del evaluated
    lat = fields["ocean_lat"]
    selected = (lat >= band[0]) & (lat <= band[1])
    gyre = np.average(
        coupling["clock removed ocean"][:, selected], axis=1, weights=fields["ocean_weights"][selected],
    )
    targets = {"ice": coupling["clock removed"]["ice"], "gyre": gyre}
    explained = {name: [] for name in targets}
    for function in functions.T:
        design = np.column_stack((np.ones(function.size), function.real - function.real.mean(),
                                  function.imag - function.imag.mean()))
        for name, target in targets.items():
            fit = design @ np.linalg.lstsq(design, target, rcond=None)[0]
            explained[name].append(1 - np.var(target - fit) / np.var(target))
    joint = np.column_stack([np.ones(functions.shape[0])] + [
        np.column_stack((f.real - f.real.mean(), f.imag - f.imag.mean())) for f in functions.T
    ])
    joint_r2 = {
        name: float(1 - np.var(target - joint @ np.linalg.lstsq(joint, target, rcond=None)[0]) / np.var(target))
        for name, target in targets.items()
    }
    settings = result["settings"]
    training = single.kdmd_training_indices(
        fields["state"].shape[0], settings["lag"], settings["maximum_training_snapshots"],
        settings["seed"],
    )
    # Unit-RMS eigenfunctions, so each mode is the observable part per unit φ.
    scales = 1 / np.sqrt(np.mean(np.abs(functions)**2, axis=0))
    modes = single.observable_modes(
        spectrum, fields["state"], fields["state"], training, list(indices), 1 / scales,
    )
    patterns = {
        name: {"lat": block_lat, "modes": modes[:, begin:end]}
        for name, begin, end, _, block_lat in fields["state_blocks"]
    }
    return {
        "indices": indices, "rates": rates[indices],
        "ice_r2": np.asarray(explained["ice"]), "gyre_r2": np.asarray(explained["gyre"]),
        "joint_r2": joint_r2,
        "eigenfunctions": functions * scales[None], "patterns": patterns,
    }


def coupled_residual_koopman(
    result: dict, coupling: dict, eof_count: int = 10, lag: int = 1,
    band: tuple[float, float] = (-30.0, -25.0), minimum_share: float = 0.2,
) -> dict:
    """Koopman analysis of the clock-removed ice and ocean residuals together.

    The basin ice area per T21 row and the zonal θ 0–700 m rows, each with
    the phase × |psi1| composite and the per-year clock/amplitude error of
    ``coupling`` removed, are reduced to joint EOFs (blocks scaled equally),
    and ``fit_residual_koopman`` is applied with the clock coordinates. For
    each transverse mode (not a clock mode, carried by the EOFs): its decay
    time, the R² with which it explains the ice-area residual and the gyre
    (``band``) residual, and its regression patterns on ice rows and θ rows.
    The ``coupled`` mode maximizes the smaller of the two R².
    """
    fields, psi = result["fields"], result["psi1"]
    delta, alpha = coupling["delta"], coupling["alpha"]
    ice_rows = fields["ice_rows_anomaly"]
    varying = ice_rows.std(axis=0) > 1e-9
    ice_rows = ice_rows[:, varying]
    ice_lat = fields["ice_row_lat"][varying]
    ice_weights = fields["ice_row_weights"][varying]
    ice_residual = cycle_residuals(ice_rows, psi, None, None)["amplitude"]
    slope, shape = cycle_tangent_and_shape(ice_rows, psi)
    ice_clean = remove_clock_error(ice_residual, slope, shape, delta, alpha)
    ocean_clean = coupling["clock removed ocean"]
    eofs = residual_eofs(
        [ice_clean, ocean_clean], [ice_weights / ice_weights.sum(), fields["ocean_weights"]],
        eof_count,
    )
    koopman = fit_residual_koopman(eofs["pcs"], result["phase"], lag=lag)
    candidates = transverse_modes(koopman, minimum_residual_share=minimum_share, count=12)
    lat = fields["ocean_lat"]
    selected = (lat >= band[0]) & (lat <= band[1])
    targets = {
        "ice": coupling["clock removed"]["ice"],
        "gyre": np.average(ocean_clean[:, selected], axis=1, weights=fields["ocean_weights"][selected]),
    }
    modes = []
    for index in candidates:
        phi = koopman["eigenfunctions"][:, index]
        phi = phi - phi.mean()
        phi = phi / np.sqrt(np.mean(np.abs(phi)**2))
        design = np.column_stack((np.ones(phi.size), phi.real, phi.imag))
        r2 = {
            name: float(1 - np.var(target - design @ np.linalg.lstsq(design, target, rcond=None)[0])
                        / np.var(target))
            for name, target in targets.items()
        }
        ice_pattern = np.mean(np.conj(phi)[:, None] * ice_clean, axis=0)
        ocean_pattern = np.mean(np.conj(phi)[:, None] * ocean_clean, axis=0)
        peak = ocean_pattern[np.argmax(np.abs(ocean_pattern))]
        rotation = np.conj(peak) / abs(peak) if abs(peak) else 1.0
        rate = koopman["rates"][index]
        modes.append({
            "index": int(index), "rate": rate, "decay_time": float(-1 / rate.real),
            "period": float(2 * np.pi / rate.imag) if abs(rate.imag) > 1e-6 else np.inf,
            "ice_r2": r2["ice"], "gyre_r2": r2["gyre"],
            "ice_pattern": np.real(rotation * ice_pattern), "ocean_pattern": np.real(rotation * ocean_pattern),
        })
    coupled = max(modes, key=lambda mode: min(mode["ice_r2"], mode["gyre_r2"])) if modes else None
    return {
        "modes": modes, "coupled": coupled, "ice_lat": ice_lat, "ocean_lat": lat,
        "eof_fractions": eofs["fractions"], "rank": koopman["rank"], "lag": lag,
    }


def make_coupled_mode_figure(results: dict[str, dict]) -> plt.Figure:
    """Coupled residual Koopman mode by μ: decay time and R², and its patterns."""
    labels = [label for label in results if results[label]["coupled"] is not None]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    x = np.arange(len(labels))
    coupled = [results[label]["coupled"] for label in labels]
    axes[0].plot(x, [mode["decay_time"] for mode in coupled], "o-", color="#315c9b")
    axes[0].set(ylabel="decay time (yr)", title="Coupled ice–gyre residual mode")
    twin = axes[0].twinx()
    twin.plot(x, [mode["ice_r2"] for mode in coupled], "s--", color="#e9a23b", label="R² ice")
    twin.plot(x, [mode["gyre_r2"] for mode in coupled], "^--", color="#c2185b", label="R² gyre")
    twin.set(ylabel="R²", ylim=(0, 1))
    twin.legend(frameon=False, fontsize="small")
    axes[0].set_xticks(x, [label.replace("p", ".") for label in labels])
    for label, mode in zip(labels, coupled):
        unit = np.max(np.abs(mode["ocean_pattern"])) or 1.0
        axes[1].plot(mode["ice_pattern"] / unit * 1e3, results[label]["ice_lat"], "o-", ms=3,
                     label=label.replace("p", "."))
        axes[2].plot(mode["ocean_pattern"] / unit, results[label]["ocean_lat"], ms=3,
                     label=label.replace("p", "."))
    axes[1].set(xlabel="ice area per row (10³ km² per unit θ peak, mK)", ylabel="latitude (°)",
                title="Ice pattern", ylim=(-60, 0))
    axes[2].set(xlabel="θ 0–700 m (normalized to its peak)", title="Ocean pattern", ylim=(-60, 0))
    for axis in axes[1:]:
        axis.axvline(0, color="0.6", lw=0.6)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize="x-small")
    axes[0].grid(alpha=0.2)
    return figure


# ---------------------------------------------------------------- figures

_TICKS = (np.linspace(0, 1, 5), ["0", "¼", "½", "¾", "1"])


def _phase_axis(axis, label: bool = True) -> None:
    axis.set_xlim(0, 1)
    axis.set_xticks(*_TICKS)
    if label:
        axis.set_xlabel("phase of ψ₁ (cycles; 0 = SA ice maximum)")
    axis.grid(alpha=0.2)


def make_clock_figure(
    psi: np.ndarray, years: np.ndarray, omega: float, *, mu: str = "",
) -> plt.Figure:
    """Irregularity of the clock: phase advance, |psi|, diffusion, cycle lengths."""
    clock = phase_advance_stats(psi, years)
    diffusion = phase_diffusion(psi)
    lengths = cycle_lengths(psi, years)
    x = clock["centres"] / (2 * np.pi)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axis = axes[0, 0]
    axis.plot(x, clock["advance_mean"], ".-", color="#315c9b", label="mean")
    axis.fill_between(
        x, clock["advance_mean"] - clock["advance_std"],
        clock["advance_mean"] + clock["advance_std"], color="#315c9b", alpha=0.2,
        label="± std",
    )
    axis.axhline(omega, color="#b44b38", ls="--", label="ω")
    axis.set(ylabel="phase advance (rad yr⁻¹)", title="One-year phase advance")
    axis.legend(frameon=False, fontsize="small")
    _phase_axis(axis)
    axis = axes[0, 1]
    axis.plot(x, clock["amplitude_mean"], ".-", color="#315c9b", label="mean")
    axis.plot(x, clock["amplitude_std"], ".-", color="#c2185b", label="std")
    axis.set(ylabel="|ψ₁| (unit RMS)", title="Amplitude by phase")
    axis.legend(frameon=False, fontsize="small")
    _phase_axis(axis)
    axis = axes[1, 0]
    axis.plot(diffusion["lags"], diffusion["variance"], color="#315c9b")
    axis.plot(
        diffusion["lags"],
        diffusion["intercept"] + 2 * diffusion["diffusion"] * diffusion["lags"],
        "k--", lw=1, label=f"D = {diffusion['diffusion']:.2e} rad² yr⁻¹",
    )
    axis.set(
        xlabel="lag τ (yr)", ylabel="Var[θ(t+τ) − θ(t)] (rad²)",
        title="Phase diffusion",
    )
    axis.legend(frameon=False, fontsize="small")
    axis.grid(alpha=0.2)
    axis = axes[1, 1]
    if lengths.size:
        axis.hist(lengths, bins=20, color="#4472a0")
        axis.axvline(2 * np.pi / omega, color="#b44b38", ls="--", label="2π/ω")
        axis.set_title(
            f"Cycle lengths: {lengths.size} cycles, mean {lengths.mean():.1f} yr, "
            f"CV {lengths.std() / lengths.mean():.2f}", fontsize="small",
        )
        axis.legend(frameon=False, fontsize="small")
    axis.set(xlabel="cycle length (yr)", ylabel="cycles")
    figure.suptitle(f"μ={mu.replace('p', '.')}: irregularity of the ψ₁ clock")
    return figure


RESIDUAL_LABELS = {
    "koopman": "G − ψ₁,ψ₂ modes",
    "composite": "G − phase composite",
    "amplitude": "G − phase×|ψ₁| composite",
}


def make_state_residual_figure(
    state_residuals: dict, phase: np.ndarray, *, mu: str = "",
    kind: str = "amplitude", bins: int = 36,
) -> plt.Figure:
    """Phase × latitude residual standard deviation for the zonal Ts and θ rows.

    ``state_residuals[block]`` holds ``lat``, ``unit`` and the residual dict
    of ``cycle_residuals`` for that block.
    """
    names = list(state_residuals)
    figure, axes = plt.subplots(
        2, len(names), figsize=(5.5 * len(names), 7.5), layout="constrained",
        squeeze=False, gridspec_kw={"height_ratios": (1.3, 1)},
    )
    for column, name in enumerate(names):
        block = state_residuals[name]
        stats = phase_conditioned_stats(block["residuals"][kind], phase, bins)
        x = np.r_[0, (np.arange(bins) + 1) / bins]
        lat = block["lat"]
        lat_edges = np.r_[lat[0] - (lat[1] - lat[0]) / 2, (lat[1:] + lat[:-1]) / 2,
                          lat[-1] + (lat[-1] - lat[-2]) / 2]
        mesh = axes[0, column].pcolormesh(
            x, lat_edges, np.sqrt(stats["variance"]).T, cmap="magma_r",
            shading="flat", rasterized=True,
        )
        figure.colorbar(mesh, ax=axes[0, column], label=f"residual std ({block['unit']})")
        axes[0, column].set(ylabel="latitude (°)", title=f"{block['label']}: {RESIDUAL_LABELS[kind]}")
        _phase_axis(axes[0, column], label=False)
        weights = block["weights"]
        for residual_kind, color in zip(RESIDUAL_LABELS, ("#999999", "#315c9b", "#c2185b")):
            if residual_kind not in block["residuals"]:
                continue
            local = phase_conditioned_stats(block["residuals"][residual_kind], phase, bins)
            mean_variance = local["variance"] @ weights
            axes[1, column].plot(
                stats["centres"] / (2 * np.pi), mean_variance, ".-", color=color,
                label=f"{RESIDUAL_LABELS[residual_kind]} "
                f"({block['residuals']['fractions'][residual_kind]:.0%} of var)",
            )
        axes[1, column].set(ylabel=f"weighted mean variance ({block['unit']}²)")
        axes[1, column].legend(frameon=False, fontsize="x-small")
        _phase_axis(axes[1, column])
    figure.suptitle(f"μ={mu.replace('p', '.')}: residual variance by phase")
    return figure


def make_band_residual_figure(
    band_residuals: dict[str, np.ndarray], phase: np.ndarray, years: np.ndarray,
    *, mu: str = "", bins: int = 18,
) -> plt.Figure:
    """Variance, skewness, lag-one memory and spectrum of band-mean residuals."""
    from scipy.signal import welch

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    x = bin_centres(bins) / (2 * np.pi)
    for name, series in band_residuals.items():
        stats = phase_conditioned_stats(series, phase, bins)
        normal = stats["variance"] / np.var(series)
        line, = axes[0, 0].plot(x, normal, ".-", label=name)
        axes[0, 0].fill_between(
            x, normal - stats["variance_se"] / np.var(series),
            normal + stats["variance_se"] / np.var(series),
            color=line.get_color(), alpha=0.15,
        )
        axes[0, 1].plot(x, stats["skewness"], ".-", color=line.get_color(), label=name)
        axes[1, 0].plot(
            x, lag_one_by_phase(series, phase, bins), ".-",
            color=line.get_color(), label=name,
        )
        frequency, power = welch(series, fs=1.0, nperseg=min(1024, series.size), detrend="linear")
        positive = frequency > 0
        axes[1, 1].loglog(1 / frequency[positive], power[positive] / np.var(series),
                          color=line.get_color(), label=name)
    axes[0, 0].axhline(1, color="0.6", lw=0.8)
    axes[0, 0].set(ylabel="variance / total variance", title="Residual variance by phase")
    axes[0, 1].axhline(0, color="0.6", lw=0.8)
    axes[0, 1].set(ylabel="skewness", title="Residual skewness by phase")
    axes[1, 0].set(ylabel="slope r(t+1) on r(t)", title="Lag-one memory by starting phase")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        _phase_axis(axis)
        axis.legend(frameon=False, fontsize="small")
    axes[1, 1].set(xlabel="period (yr)", ylabel="normalized power (yr)",
                   title="Residual spectrum", xlim=(2, 1000))
    axes[1, 1].grid(alpha=0.2, which="both")
    axes[1, 1].legend(frameon=False, fontsize="small")
    figure.suptitle(
        f"μ={mu.replace('p', '.')}: band-mean residuals (phase×|ψ₁| composite removed), "
        f"{years[0]}–{years[-1]}"
    )
    return figure


def make_flip_figure(
    flips: dict[str, np.ndarray], maps: dict, *, period: float, mu: str = "",
    lat_range: tuple[float, float] = (-65.0, -10.0),
    lon_range: tuple[float, float] = (-70.0, 25.0),
) -> plt.Figure:
    """Onset/retreat phase, their jitter (years), and flips per cycle per T21 cell."""
    lon = (np.asarray(maps["t21_lon"]) + 180) % 360 - 180
    order = np.argsort(lon)
    lon = lon[order]
    lat = maps["t21_lat"]
    lon_edges = np.r_[lon - 2.8125, lon[-1] + 2.8125]
    lat_edges = np.r_[lat[0] - (lat[1] - lat[0]) / 2, (lat[1:] + lat[:-1]) / 2,
                      lat[-1] + (lat[-1] - lat[-2]) / 2]
    ocean = maps["lsm"] < 0.5
    years_per_radian = period / (2 * np.pi)
    panels = (
        ("onset_phase", "onset phase (cycles)", "twilight", 0, 1, 1 / (2 * np.pi)),
        ("retreat_phase", "retreat phase (cycles)", "twilight", 0, 1, 1 / (2 * np.pi)),
        ("onset_per_cycle", "onsets per cycle", "viridis", 0, 1.2, 1.0),
        ("onset_std", "onset jitter (yr)", "magma_r", 0, 0.25 * period, years_per_radian),
        ("retreat_std", "retreat jitter (yr)", "magma_r", 0, 0.25 * period, years_per_radian),
        ("cover_fraction", "years with SIC ≥ 0.5", "Blues", 0, 1, 1.0),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for axis, (name, label, cmap, vmin, vmax, scale) in zip(axes.flat, panels):
        values = np.where(ocean, flips[name] * scale, np.nan)[:, order]
        mesh = axis.pcolormesh(
            lon_edges, lat_edges, np.ma.masked_invalid(values), cmap=cmap,
            vmin=vmin, vmax=vmax, rasterized=True,
        )
        axis.contour(lon, lat, maps["lsm"][:, order], levels=[0.5], colors="0.3", linewidths=0.5)
        axis.set(xlim=lon_range, ylim=lat_range, title=label)
        figure.colorbar(mesh, ax=axis)
    figure.suptitle(
        f"μ={mu.replace('p', '.')}: annual SIC ≥ 0.5 switches on the ψ₁ clock "
        f"(period {period:.1f} yr; jitter = circular std)"
    )
    return figure


def make_map_residual_figure(
    maps: dict, psi: np.ndarray, *, mu: str = "", phase_count: int = 4,
    lat_range: tuple[float, float] = (-65.0, 0.0),
    lon_range: tuple[float, float] = (-70.0, 25.0), basin_only: bool = True,
) -> plt.Figure:
    """Residual std (phase × |psi1| composite removed) in phase windows of psi1.

    Columns are ``phase_count`` equal phase bins starting at the SA ice
    maximum; the last column is the std over all years.
    """
    phase = wrapped_phase(np.asarray(psi) * np.exp(1j * np.pi / phase_count))
    names = list(maps["fields"])
    figure, axes = plt.subplots(
        len(names), phase_count + 2, squeeze=False, layout="constrained",
        figsize=(2.9 * (phase_count + 1) + 0.6, 2.6 * len(names) + 0.6),
        gridspec_kw={"width_ratios": (1,) * (phase_count + 1) + (0.05,)},
    )
    t21_lon = (np.asarray(maps["t21_lon"]) + 180) % 360 - 180
    order = np.argsort(t21_lon)
    t21_lon = t21_lon[order]
    t21_lat = maps["t21_lat"]
    lon_edges = np.r_[t21_lon - 2.8125, t21_lon[-1] + 2.8125]
    lat_edges = np.r_[t21_lat[0] - (t21_lat[1] - t21_lat[0]) / 2,
                      (t21_lat[1:] + t21_lat[:-1]) / 2,
                      t21_lat[-1] + (t21_lat[-1] - t21_lat[-2]) / 2]
    lsg_lat = maps["lsg_lat"][:, 0]
    lsg_lat_edges = np.r_[lsg_lat + 1.25, lsg_lat[-1] - 1.25]
    regular_lon_edges = np.arange(-180.0, 180.01, 2.5)
    colormap = plt.get_cmap("magma_r").copy()
    colormap.set_bad("0.88")
    for row, name in enumerate(names):
        field = maps["fields"][name]
        residual = cycle_residuals(
            field["scale"] * field["anomaly"], psi, None, None,
        )["amplitude"]
        stats = phase_conditioned_stats(residual, phase, phase_count)
        panels = np.concatenate((
            np.sqrt(stats["variance"]), np.std(residual, axis=0)[None],
        ))
        del residual
        wet = field["wet"]
        if field["grid"] == "t21":
            if basin_only:
                wet = wet & maps["t21_basin"]
            panels = np.where(wet[None], panels, np.nan)[..., order]
            x_edges, y_edges = lon_edges, lat_edges
        else:
            if basin_only:
                wet = wet & maps["lsg_basin"]
            panels = single._lsg_to_regular(np.where(wet[None], panels, np.nan), maps["lsg_lon"])
            x_edges, y_edges = regular_lon_edges, lsg_lat_edges
        shown = panels[np.isfinite(panels)]
        limit = float(np.percentile(shown, 99)) if shown.size else 1.0
        for column in range(phase_count + 1):
            axis = axes[row, column]
            mesh = axis.pcolormesh(
                x_edges, y_edges, np.ma.masked_invalid(panels[column]),
                cmap=colormap, vmin=0, vmax=limit or 1.0, rasterized=True,
            )
            axis.contour(t21_lon, t21_lat, maps["lsm"][:, order], levels=[0.5],
                         colors="0.3", linewidths=0.5)
            axis.set(xlim=lon_range, ylim=lat_range)
            axis.tick_params(labelsize="small", labelleft=column == 0,
                             labelbottom=row == len(names) - 1)
            if row == 0:
                axis.set_title(
                    "all years" if column == phase_count
                    else f"phase {column / phase_count:.2f} ± {0.5 / phase_count:.2f}",
                    fontsize="small",
                )
        axes[row, 0].set_ylabel(field["label"], fontsize="small")
        figure.colorbar(mesh, cax=axes[row, -1], label=f"std ({field['unit']})")
    figure.suptitle(
        f"μ={mu.replace('p', '.')}: residual std after removing the phase × |ψ₁| composite"
    )
    return figure


def make_floquet_figure(
    results: dict[str, dict], *, title: str = "",
) -> plt.Figure:
    """Floquet multipliers and local growth along the cycle, one entry per label."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    axes[0].plot(circle.real, circle.imag, color="0.6", lw=0.8)
    for label, result in results.items():
        multipliers = result["multipliers"]
        points = axes[0].scatter(
            multipliers.real, multipliers.imag, s=20,
            label=f"{label}: |μ|max = {np.abs(multipliers[0]):.3f}",
        )
        axes[1].plot(
            result["phase_grid"] / (2 * np.pi), result["local_growth"], ".-",
            color=points.get_facecolor()[0], label=label,
        )
    axes[0].set(aspect="equal", xlabel="Re μ", ylabel="Im μ",
                title="Multipliers of one mean cycle")
    axes[0].legend(frameon=False, fontsize="small")
    axes[0].grid(alpha=0.2)
    axes[1].axhline(1, color="0.6", lw=0.8)
    axes[1].set(ylabel="spectral radius of A(θ)", title="One-year growth along the cycle")
    _phase_axis(axes[1])
    axes[1].legend(frameon=False, fontsize="small")
    figure.suptitle(title)
    return figure


def make_residual_koopman_figure(results: dict[str, dict]) -> plt.Figure:
    """Residual-KDMD spectrum (coloured by clock share) and the zonal patterns
    of the slowest transverse mode, one entry per μ (``irregularity_analysis``
    outputs)."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    markers = ("o", "s", "^", "D")
    for (label, result), marker in zip(results.items(), markers):
        koopman = result["residual_koopman"]
        rates = koopman["rates"]
        shown = (rates.imag >= 0) & (rates.real > -0.4)
        points = axes[0].scatter(
            rates.real[shown], rates.imag[shown], c=koopman["clock_share"][shown],
            cmap="viridis", vmin=0, vmax=1, s=18, marker=marker,
            label=f"μ={label.replace('p', '.')}",
        )
        if "patterns" not in koopman:
            continue
        index = koopman["transverse"][0]
        axes[0].scatter(rates.real[index], rates.imag[index], s=120, facecolor="none",
                        edgecolor="#c2185b", marker=marker)
        text = f"μ={label.replace('p', '.')}: τ = {-1 / rates.real[index]:.1f} yr"
        for axis, block, unit in ((axes[1], "surface", "K"), (axes[2], "ocean", "mK")):
            axis.plot(koopman["patterns"][block], result["blocks"][block]["lat"],
                      marker=marker, ms=3, label=text)
            axis.set(xlabel=f"pattern ({unit} per unit φ)", ylabel="latitude (°)")
    figure.colorbar(points, ax=axes[0], label="variance share explained by the clock e^{ikθ}")
    axes[0].set(xlabel="Re λ (yr⁻¹)", ylabel="Im λ (yr⁻¹)",
                title="Residual KDMD on [residual EOFs, cos θ₁, sin θ₁]")
    axes[0].legend(frameon=False, fontsize="small")
    axes[1].set_title("Slowest transverse mode: zonal Ts")
    axes[2].set_title("Slowest transverse mode: zonal θ 0–700 m")
    for axis in axes[1:]:
        axis.axvline(0, color="0.6", lw=0.8)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize="small")
    return figure


def make_coupling_figure(
    results: dict[str, dict], first: str = "θ 0–700 m 22–27°S",
    second: str = "SA ice area", maximum_lag: int = 40,
) -> plt.Figure:
    """Lagged correlation of two band residuals, one line per μ."""
    lags = np.arange(-maximum_lag, maximum_lag + 1)
    figure, axis = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for label, result in results.items():
        axis.plot(lags, lagged_correlation(
            result["band_residuals"][first], result["band_residuals"][second], lags,
        ), label=f"μ={label.replace('p', '.')}")
    axis.axhline(0, color="0.6", lw=0.8)
    axis.axvline(0, color="0.6", lw=0.8)
    axis.set(xlabel=f"lag L (yr; L > 0: {second} later)",
             ylabel="correlation",
             title=f"corr({first} residual(t), {second} residual(t+L))")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return figure


def make_amplitude_history_figure(results: dict[str, dict]) -> plt.Figure:
    """|psi1| (running mean, weak episodes shaded) and cycle lengths through time."""
    figure, axes = plt.subplots(len(results), 1, figsize=(12, 2.8 * len(results)),
                                layout="constrained", squeeze=False)
    for axis, (label, result) in zip(axes[:, 0], results.items()):
        psi, years = result["psi1"], result["fields"]["years"]
        episodes = weak_episodes(psi, years)
        axis.plot(years, np.abs(psi), color="0.75", lw=0.5, label="|ψ₁|")
        axis.plot(years, episodes["running_amplitude"], color="#315c9b", lw=1.5,
                  label="101-yr running mean")
        axis.axhline(0.75, color="#315c9b", ls=":", lw=0.8)
        for start, stop in episodes["episodes"]:
            axis.axvspan(start, stop, color="#c2185b", alpha=0.15, lw=0)
        twin = axis.twinx()
        edges = cycle_crossings(psi, years)
        twin.plot((edges[:-1] + edges[1:]) / 2, np.diff(edges), "o-", color="#e9a23b",
                  ms=3, lw=0.8, label="cycle length")
        twin.set_ylabel("cycle length (yr)")
        axis.set(ylabel="|ψ₁| (unit RMS)", xlim=(years[0], years[-1]),
                 title=f"μ={label.replace('p', '.')}: {episodes['weak'].mean():.0%} of years "
                 "in weak-cycle episodes (shaded)")
        axis.legend(frameon=False, fontsize="small", loc="upper left")
        twin.legend(frameon=False, fontsize="small", loc="upper right")
    axes[-1, 0].set_xlabel("model year")
    return figure


def make_latitude_lag_figure(couplings: dict[str, dict], kind: str = "clock removed") -> plt.Figure:
    """corr(ocean residual row(t), ice residual(t+lag)) by latitude and lag, one panel per μ.

    Negative lags: ice earlier. The cross marks the strongest value; the
    dashed lines bound the 22–27°S reservoir band.
    """
    labels = list(couplings)
    figure, axes = plt.subplots(
        1, len(labels), figsize=(2.6 * len(labels) + 1, 4.2), layout="constrained",
        sharey=True, squeeze=False,
    )
    for axis, label in zip(axes[0], labels):
        coupling = couplings[label]
        lags, lat = coupling["lags"], coupling["ocean_lat"]
        table = coupling[kind]["table"]
        mesh = axis.pcolormesh(lags, lat, table, cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                               shading="nearest", rasterized=True)
        value, peak_lat, peak_lag = coupling[kind]["peak"]
        axis.plot(peak_lag, peak_lat, "kx", ms=8)
        for edge in (-27, -22):
            axis.axhline(edge, color="0.3", ls="--", lw=0.6)
        axis.axvline(0, color="0.5", lw=0.6)
        axis.set(xlabel="lag (yr; < 0: ice first)", ylim=(-60, 0),
                 title=f"μ={label.replace('p', '.')}: {value:+.2f}\nat {peak_lat:.0f}°, {peak_lag:+d} yr")
    axes[0, 0].set_ylabel("latitude of ocean row (°)")
    figure.colorbar(mesh, ax=axes[0], label="correlation", shrink=0.8)
    figure.suptitle(f"Ice residual vs zonal θ 0–700 m residual ({kind})")
    return figure


def make_coupling_trend_figure(couplings: dict[str, dict]) -> plt.Figure:
    """Reservoir-band and best-row correlations against μ, raw and with the
    clock error removed, with the 95% surrogate threshold."""
    labels = list(couplings)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    for kind, color in (("raw", "0.55"), ("clock removed", "#c2185b")):
        band = [couplings[label][kind]["band_peak"][0] for label in labels]
        threshold = [couplings[label][kind]["band_threshold"] for label in labels]
        peak = [couplings[label][kind]["peak"] for label in labels]
        axes[0].plot(x, band, "o-", color=color, label=kind)
        axes[0].fill_between(x, -np.array(threshold), threshold, color=color, alpha=0.12)
        axes[1].plot(x, [item[0] for item in peak], "o-", color=color, label=kind)
        axes[2].plot(x, [item[1] for item in peak], "o-", color=color, label=f"{kind}: latitude")
    axes[0].set(ylabel="correlation", title="22–27°S band (shaded: 95% surrogate range)")
    axes[1].set(ylabel="correlation", title="Strongest row and lag")
    axes[2].set(ylabel="latitude (°)", title="Latitude of the strongest row")
    for axis in axes:
        axis.set_xticks(x, [label.replace("p", ".") for label in labels])
        axis.set_xlabel("μ")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize="small")
        axis.axhline(0, color="0.6", lw=0.6)
    return figure


def make_coupling_timeseries_figure(
    couplings: dict[str, dict], years: dict[str, np.ndarray], lag: int = 4, smoothing: int = 5,
) -> plt.Figure:
    """Clock-removed ice residual (sign flipped, shifted by ``lag``) and the
    reservoir residual, each standardized and smoothed, plus a lag scatter."""
    labels = list(couplings)
    figure, axes = plt.subplots(
        len(labels), 2, figsize=(14, 2.3 * len(labels)), layout="constrained",
        gridspec_kw={"width_ratios": (4, 1)}, squeeze=False,
    )
    kernel = np.ones(smoothing) / smoothing
    for row, label in enumerate(labels):
        item = couplings[label]["clock removed"]
        ice = (item["ice"] - item["ice"].mean()) / item["ice"].std()
        band = (item["band"] - item["band"].mean()) / item["band"].std()
        y = years[label]
        axis = axes[row, 0]
        axis.plot(y + lag, -np.convolve(ice, kernel, "same"), color="#4472a0", lw=0.8,
                  label=f"−ice residual, shifted +{lag} yr")
        axis.plot(y, np.convolve(band, kernel, "same"), color="#c2185b", lw=0.8,
                  label="reservoir θ residual (22–27°S)")
        axis.set(xlim=(y[0], y[0] + 1500), ylabel=f"μ={label.replace('p', '.')}\nstd units")
        axis.grid(alpha=0.2)
        if row == 0:
            axis.legend(frameon=False, fontsize="small", ncol=2)
        scatter = axes[row, 1]
        scatter.plot(ice[:-lag], band[lag:], ".", ms=1.5, alpha=0.3, color="0.3", rasterized=True)
        slope = np.polyfit(ice[:-lag], band[lag:], 1)[0]
        scatter.plot([-3, 3], [-3 * slope, 3 * slope], color="#c2185b")
        scatter.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5),
                    title=f"r = {np.corrcoef(ice[:-lag], band[lag:])[0, 1]:+.2f}", xlabel="ice(t)",
                    ylabel=f"reservoir(t+{lag})")
    axes[-1, 0].set_xlabel(f"model year (first 1500 yr of each window; {smoothing}-yr running mean)")
    return figure


def make_regression_map_figure(
    entry: dict, *, kind: str = "clock removed", fields: tuple[str, ...] | None = None,
    lat_range: tuple[float, float] = (-60.0, 0.0), lon_range: tuple[float, float] = (-70.0, 25.0),
) -> plt.Figure:
    """Residual maps regressed on the standardized ice residual at each lag.

    Positive lags: the map is later than the ice. The sign is per +1 std of
    ice, so a warm anomaly after *less* ice appears with negative values.
    """
    grid = entry["grid"]
    names = list(fields or entry["regressions"])
    lags = entry["map_lags"]
    t21_lon = (np.asarray(grid["t21_lon"]) + 180) % 360 - 180
    order = np.argsort(t21_lon)
    t21_lon = t21_lon[order]
    t21_lat = grid["t21_lat"]
    lon_edges = np.r_[t21_lon - 2.8125, t21_lon[-1] + 2.8125]
    lat_edges = np.r_[t21_lat[0] - (t21_lat[1] - t21_lat[0]) / 2, (t21_lat[1:] + t21_lat[:-1]) / 2,
                      t21_lat[-1] + (t21_lat[-1] - t21_lat[-2]) / 2]
    lsg_lat = grid["lsg_lat"][:, 0]
    lsg_lat_edges = np.r_[lsg_lat + 1.25, lsg_lat[-1] - 1.25]
    regular_lon_edges = np.arange(-180.0, 180.01, 2.5)
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("0.88")
    figure, axes = plt.subplots(
        len(names), lags.size + 1, squeeze=False, layout="constrained",
        figsize=(2.4 * lags.size + 0.6, 2.4 * len(names) + 0.6),
        gridspec_kw={"width_ratios": (1,) * lags.size + (0.05,)},
    )
    for row, name in enumerate(names):
        item = entry["regressions"][name]
        values = np.asarray(item[kind], dtype=float)
        if item["grid"] == "t21":
            values = np.where((grid["lsm"] < 0.5)[None], values, np.nan)[..., order]
            x_edges, y_edges, lat_axis = lon_edges, lat_edges, t21_lat
        else:
            values = single._lsg_to_regular(np.where(item["wet"][None], values, np.nan), grid["lsg_lon"])
            x_edges, y_edges, lat_axis = regular_lon_edges, lsg_lat_edges, lsg_lat
        window = (lat_axis >= lat_range[0]) & (lat_axis <= lat_range[1])
        shown = values[:, window][np.isfinite(values[:, window])]
        limit = float(np.percentile(np.abs(shown), 99)) if shown.size else 1.0
        for column, lag in enumerate(lags):
            axis = axes[row, column]
            mesh = axis.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values[column]),
                                   cmap=colormap, vmin=-limit, vmax=limit, rasterized=True)
            axis.contour(t21_lon, t21_lat, grid["lsm"][:, order], levels=[0.5], colors="0.3", linewidths=0.5)
            axis.set(xlim=lon_range, ylim=lat_range)
            axis.tick_params(labelsize="x-small", labelleft=column == 0, labelbottom=row == len(names) - 1)
            if row == 0:
                axis.set_title(f"lag {lag:+d} yr", fontsize="small")
        axes[row, 0].set_ylabel(item["label"], fontsize="small")
        figure.colorbar(mesh, cax=axes[row, -1], label=f"{item['unit']} per std of ice")
    figure.suptitle(
        f"μ={entry['mu'].replace('p', '.')}: residual maps regressed on the SA ice-area residual "
        f"({kind}); lag > 0: map after ice"
    )
    return figure


BUDGET_COLORS = {"storage": "black", "surface": "#e9a23b", "advection": "#315c9b", "residual": "0.6"}


def make_budget_figure(
    regressions: dict[str, dict], kind: str = "clock removed",
    title: str = "per std of the SA ice-area residual",
) -> plt.Figure:
    """Gyre heat-budget terms regressed on a predictor, one panel per μ.

    ``regressions[mu]`` is the output of ``budget_regression``. Lag > 0:
    the term after the predictor. The cumulative storage (heat content) is
    on the right axis.
    """
    labels = list(regressions)
    figure, axes = plt.subplots(
        1, len(labels), figsize=(2.9 * len(labels) + 0.6, 3.8), layout="constrained",
        sharey=True, squeeze=False,
    )
    for axis, label in zip(axes[0], labels):
        item = regressions[label][kind]
        for name, color in BUDGET_COLORS.items():
            axis.plot(item["lags"], item[name], color=color, lw=1.4 if name != "residual" else 1,
                      ls="--" if name == "residual" else "-", label=name)
        twin = axis.twinx()
        twin.plot(item["lags"], item["heat_content"], color="#c2185b", lw=1, ls=":",
                  label="heat content (PJ)")
        twin.tick_params(axis="y", colors="#c2185b", labelsize="small")
        axis.axhline(0, color="0.7", lw=0.6)
        axis.axvline(0, color="0.7", lw=0.6)
        axis.set(xlabel="lag (yr; > 0: after)", title=f"μ={label.replace('p', '.')}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("TW")
    axes[0, 0].legend(frameon=False, fontsize="x-small", loc="lower left")
    twin.legend(frameon=False, fontsize="x-small", loc="lower right")
    figure.suptitle(
        "South Atlantic upper-ocean (0–700 m, 17.5–30°S) heat budget, residuals regressed "
        f"{title} ({kind})"
    )
    return figure


def make_comparison_figure(summaries: dict[str, dict]) -> plt.Figure:
    """Each scalar metric against μ: full window (diamond) and blocks (dots).

    ``summaries[mu] = {"full": dict, "blocks": list[dict]}``.
    """
    labels = list(summaries)
    metrics = [key for key in summaries[labels[0]]["full"]
               if all(key in summaries[label]["full"] for label in labels)]
    columns = 4
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.4 * rows), layout="constrained",
        squeeze=False,
    )
    for axis, metric in zip(axes.flat, metrics):
        for position, label in enumerate(labels):
            blocks = [block.get(metric, np.nan) for block in summaries[label]["blocks"]]
            axis.plot(np.full(len(blocks), position), blocks, "o", color="#4472a0", alpha=0.5, ms=4)
            axis.plot(position, summaries[label]["full"][metric], "D", color="#b44b38", ms=6)
        axis.set_xticks(range(len(labels)), [label.replace("p", ".") for label in labels])
        axis.set_xlim(-0.5, len(labels) - 0.5)
        axis.set_title(metric, fontsize="small")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    figure.suptitle("Irregularity metrics by μ (diamond: full window; dots: 1000-yr blocks)")
    return figure
