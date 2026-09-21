"""Diffusion-map bases and reduced EDMD diagnostics for sampled trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DiffusionMapResult:
    """Variable-bandwidth diffusion-map basis evaluated on training samples."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    epsilon: float
    epsilon_candidates: np.ndarray
    kernel_sums: np.ndarray
    log_slopes: np.ndarray
    intrinsic_dimension: float
    density: np.ndarray
    local_bandwidth: np.ndarray
    graph_components: int
    neighbor_count: int


@dataclass(frozen=True)
class EDMDResult:
    """Finite-dimensional EDMD approximation at one sampling lag."""

    lag: int
    basis_size: int
    gram: np.ndarray
    cross_gram: np.ndarray
    koopman: np.ndarray
    eigenvalues: np.ndarray
    rates: np.ndarray
    right_eigenvectors: np.ndarray
    stationary_index: int
    retained_rank: int
    gram_singular_values: np.ndarray


def weighted_embedding(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return coordinates whose Euclidean metric is the weighted field metric."""
    values = np.asarray(data, dtype=float)
    metric_weights = np.asarray(weights, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape (n_samples, n_features).")
    if metric_weights.shape != (values.shape[1],):
        raise ValueError("weights must have one entry per data feature.")
    if not np.isfinite(values).all() or not np.isfinite(metric_weights).all():
        raise ValueError("data and weights must be finite.")
    if np.any(metric_weights <= 0.0):
        raise ValueError("weights must be strictly positive.")
    return values * np.sqrt(metric_weights)[None, :]


def _symmetric_neighbor_distances(
    embedded: np.ndarray, neighbor_count: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    sample_count = embedded.shape[0]
    if sample_count < 4:
        raise ValueError("At least four samples are required for diffusion maps.")
    if not 2 <= neighbor_count < sample_count:
        raise ValueError("neighbor_count must lie between 2 and n_samples - 1.")

    distances, indices = cKDTree(embedded).query(
        embedded, k=neighbor_count + 1, workers=-1
    )
    local_scale = distances[:, -1]
    if np.any(~np.isfinite(local_scale)) or np.any(local_scale <= 0.0):
        raise ValueError("Nearest-neighbour local scales must be finite and positive.")

    rows = np.repeat(np.arange(sample_count), neighbor_count)
    columns = indices[:, 1:].reshape(-1)
    squared = distances[:, 1:].reshape(-1) ** 2
    directed = sparse.csr_matrix(
        (squared, (rows, columns)), shape=(sample_count, sample_count)
    )
    # The minimum retains an edge selected from either endpoint and uses the
    # shorter numerical copy when an edge is present in both directed lists.
    transposed = directed.T.tocsr()
    both = directed.minimum(transposed)
    either = directed.maximum(transposed)
    symmetric = either.copy()
    shared_rows, shared_columns = both.nonzero()
    symmetric[shared_rows, shared_columns] = both[shared_rows, shared_columns]
    symmetric.eliminate_zeros()
    return symmetric.tocsr(), local_scale


def _kernel_scaling(
    scaled_edge_distances: np.ndarray,
    sample_count: int,
    epsilon_candidates: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    kernel_sums = np.asarray(
        [
            sample_count
            + np.exp(-scaled_edge_distances / epsilon).sum()
            for epsilon in epsilon_candidates
        ],
        dtype=float,
    )
    log_epsilon = np.log(epsilon_candidates)
    log_slopes = np.gradient(np.log(kernel_sums), log_epsilon)
    selected_index = int(np.argmax(log_slopes))
    intrinsic_dimension = float(2.0 * log_slopes[selected_index])
    return (
        float(epsilon_candidates[selected_index]),
        kernel_sums,
        log_slopes,
        intrinsic_dimension,
    )


def fit_variable_bandwidth_diffusion_map(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    n_eigenfunctions: int = 200,
    neighbor_count: int = 256,
    local_scale_neighbors: int = 32,
    beta: float = -0.5,
    epsilon_powers: Iterable[float] = tuple(np.linspace(-4.0, 4.0, 17)),
) -> DiffusionMapResult:
    """Fit a sparse variable-bandwidth diffusion map.

    The Gaussian-area metric is represented by ``weights``. A pilot
    self-tuning kernel estimates sampling density and intrinsic dimension;
    the final kernel uses ``rho = q**beta`` and ``alpha = -d/4``.
    """
    embedded = weighted_embedding(data, weights)
    sample_count, ambient_dimension = embedded.shape
    if not 2 <= local_scale_neighbors <= neighbor_count:
        raise ValueError(
            "local_scale_neighbors must be between 2 and neighbor_count."
        )
    if not 2 <= n_eigenfunctions < sample_count:
        raise ValueError("n_eigenfunctions must lie between 2 and n_samples - 1.")

    squared_graph, neighbor_scale = _symmetric_neighbor_distances(
        embedded, neighbor_count
    )
    _, local_indices = cKDTree(embedded).query(
        embedded, k=local_scale_neighbors + 1, workers=-1
    )
    # Use an RMS local radius, which is less noisy than a single order statistic.
    local_points = embedded[local_indices[:, 1:]]
    local_squared = np.sum((local_points - embedded[:, None, :]) ** 2, axis=2)
    local_scale = np.sqrt(local_squared.mean(axis=1))
    local_scale /= np.exp(np.mean(np.log(local_scale)))

    rows, columns = squared_graph.nonzero()
    edge_squared = np.asarray(squared_graph[rows, columns]).ravel()
    pilot_scaled = edge_squared / (4.0 * local_scale[rows] * local_scale[columns])
    positive_pilot = pilot_scaled[pilot_scaled > 0.0]
    if positive_pilot.size == 0:
        raise ValueError("The neighbour graph contains no positive distances.")
    powers = np.asarray(tuple(epsilon_powers), dtype=float)
    if powers.ndim != 1 or powers.size < 3 or not np.isfinite(powers).all():
        raise ValueError("epsilon_powers must contain at least three finite values.")
    pilot_center = float(np.median(positive_pilot))
    pilot_candidates = pilot_center * np.power(2.0, powers)
    pilot_epsilon, _, _, dimension_estimate = _kernel_scaling(
        pilot_scaled, sample_count, pilot_candidates
    )
    dimension_estimate = float(np.clip(dimension_estimate, 1.0, ambient_dimension))

    pilot_values = np.exp(-pilot_scaled / pilot_epsilon)
    pilot_kernel = sparse.csr_matrix(
        (pilot_values, (rows, columns)), shape=squared_graph.shape
    )
    pilot_kernel.setdiag(1.0)
    pilot_mass = np.asarray(pilot_kernel.sum(axis=1)).ravel()
    density = pilot_mass / np.power(local_scale, dimension_estimate)
    density_floor = float(np.quantile(density, 0.01))
    density = np.maximum(density, max(density_floor, np.finfo(float).tiny))
    density /= np.exp(np.mean(np.log(density)))

    bandwidth = np.power(density, beta)
    bandwidth /= np.exp(np.mean(np.log(bandwidth)))
    final_scaled = edge_squared / (4.0 * bandwidth[rows] * bandwidth[columns])
    final_center = float(np.median(final_scaled[final_scaled > 0.0]))
    epsilon_candidates = final_center * np.power(2.0, powers)
    epsilon, kernel_sums, log_slopes, _ = _kernel_scaling(
        final_scaled, sample_count, epsilon_candidates
    )

    kernel_values = np.exp(-final_scaled / epsilon)
    kernel = sparse.csr_matrix(
        (kernel_values, (rows, columns)), shape=squared_graph.shape
    )
    kernel.setdiag(1.0)
    component_count = int(
        connected_components(kernel, directed=False, return_labels=False)
    )
    if component_count != 1:
        raise ValueError(
            f"The diffusion-map neighbour graph has {component_count} components; "
            "increase neighbor_count."
        )

    kernel_density = np.asarray(kernel.sum(axis=1)).ravel() / np.power(
        bandwidth, dimension_estimate
    )
    if np.any(kernel_density <= 0.0) or not np.isfinite(kernel_density).all():
        raise ValueError("The variable-bandwidth density estimate is invalid.")
    alpha = -dimension_estimate / 4.0
    density_factor = np.power(kernel_density, -alpha)
    normalized_kernel = sparse.diags(density_factor) @ kernel @ sparse.diags(
        density_factor
    )
    row_mass = np.asarray(normalized_kernel.sum(axis=1)).ravel()
    inverse_sqrt_mass = 1.0 / np.sqrt(row_mass)
    symmetric_operator = (
        sparse.diags(inverse_sqrt_mass)
        @ normalized_kernel
        @ sparse.diags(inverse_sqrt_mass)
    ).tocsr()

    eigenvalues, symmetric_vectors = eigsh(
        symmetric_operator,
        k=n_eigenfunctions,
        which="LA",
        tol=1.0e-10,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    basis = inverse_sqrt_mass[:, None] * symmetric_vectors[:, order]
    basis /= np.sqrt(np.mean(basis**2, axis=0))[None, :]
    for index in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, index])))
        if basis[pivot, index] < 0.0:
            basis[:, index] *= -1.0

    return DiffusionMapResult(
        basis=basis,
        eigenvalues=eigenvalues,
        epsilon=epsilon,
        epsilon_candidates=epsilon_candidates,
        kernel_sums=kernel_sums,
        log_slopes=log_slopes,
        intrinsic_dimension=dimension_estimate,
        density=density,
        local_bandwidth=bandwidth,
        graph_components=component_count,
        neighbor_count=neighbor_count,
    )


def fit_edmd_from_basis(
    basis_values: np.ndarray,
    *,
    lag: int,
    basis_size: int,
    common_origin_count: int | None = None,
    rcond: float = 1.0e-10,
) -> EDMDResult:
    """Fit an EDMD operator using precomputed basis values along a trajectory."""
    basis = np.asarray(basis_values)
    if basis.ndim != 2 or not np.isfinite(basis).all():
        raise ValueError("basis_values must be a finite two-dimensional array.")
    if not 1 <= lag < basis.shape[0]:
        raise ValueError("lag must lie between 1 and n_samples - 1.")
    if not 2 <= basis_size <= basis.shape[1]:
        raise ValueError("basis_size is outside the available diffusion basis.")
    maximum_origins = basis.shape[0] - lag
    origin_count = maximum_origins if common_origin_count is None else common_origin_count
    if not 2 <= origin_count <= maximum_origins:
        raise ValueError("common_origin_count is incompatible with the requested lag.")
    if not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie strictly between zero and one.")

    phi_x = basis[:origin_count, :basis_size]
    phi_y = basis[lag : lag + origin_count, :basis_size]
    gram = phi_x.conj().T @ phi_x / origin_count
    cross_gram = phi_x.conj().T @ phi_y / origin_count
    singular_values = np.linalg.svd(gram, compute_uv=False)
    threshold = rcond * singular_values[0]
    retained_rank = int(np.count_nonzero(singular_values > threshold))
    koopman = np.linalg.pinv(gram, rcond=rcond) @ cross_gram
    eigenvalues, right_eigenvectors = np.linalg.eig(koopman)
    stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    rates = np.log(eigenvalues.astype(complex)) / float(lag)
    return EDMDResult(
        lag=lag,
        basis_size=basis_size,
        gram=gram,
        cross_gram=cross_gram,
        koopman=koopman,
        eigenvalues=eigenvalues,
        rates=rates,
        right_eigenvectors=right_eigenvectors,
        stationary_index=stationary_index,
        retained_rank=retained_rank,
        gram_singular_values=singular_values,
    )


def leading_nonstationary_indices(result: EDMDResult, count: int = 5) -> np.ndarray:
    """Return the slowest finite nonstationary EDMD modes."""
    candidates = np.flatnonzero(np.arange(result.eigenvalues.size) != result.stationary_index)
    candidates = candidates[
        np.isfinite(result.rates[candidates].real)
        & np.isfinite(result.rates[candidates].imag)
    ]
    order = np.argsort(result.rates[candidates].real)[::-1]
    if candidates.size < count:
        raise ValueError(f"Only {candidates.size} nonstationary modes are available.")
    return candidates[order[:count]]


def match_rates(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Match candidate complex rates one-to-one to reference rates."""
    reference_values = np.asarray(reference, dtype=complex)
    candidate_values = np.asarray(candidates, dtype=complex)
    if reference_values.ndim != 1 or candidate_values.ndim != 1:
        raise ValueError("reference and candidates must be one-dimensional.")
    if candidate_values.size < reference_values.size:
        raise ValueError("There are fewer candidate rates than reference rates.")
    rows, columns = linear_sum_assignment(
        np.abs(reference_values[:, None] - candidate_values[None, :])
    )
    matched = np.empty(reference_values.size, dtype=complex)
    matched[rows] = candidate_values[columns]
    return matched


def project_observable(
    result: EDMDResult,
    basis_values: np.ndarray,
    observable: np.ndarray,
    *,
    origin_count: int,
    rcond: float = 1.0e-10,
) -> np.ndarray:
    """Project a centered scalar observable into an EDMD trial space."""
    values = np.asarray(observable, dtype=float)
    if values.shape != (basis_values.shape[0],) or not np.isfinite(values).all():
        raise ValueError("observable must be finite and aligned with basis_values.")
    phi_x = np.asarray(basis_values[:origin_count, : result.basis_size])
    centered = values[:origin_count] - values[:origin_count].mean()
    inner = phi_x.conj().T @ centered / origin_count
    return np.linalg.pinv(result.gram, rcond=rcond) @ inner


def modal_covariance(
    result: EDMDResult,
    coefficients: np.ndarray,
    lags: np.ndarray,
    *,
    stable_only: bool = False,
) -> np.ndarray:
    """Evaluate the EDMD modal covariance at integer multiples of the fit lag."""
    requested_lags = np.asarray(lags, dtype=int)
    if requested_lags.ndim != 1 or np.any(requested_lags < 0):
        raise ValueError("lags must be a one-dimensional array of nonnegative integers.")
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (result.basis_size,):
        raise ValueError("coefficients do not match the EDMD basis size.")
    left_factor = coefficients.conj().T @ result.gram
    if stable_only:
        eigenvalues = result.eigenvalues.astype(complex).copy()
        retained = np.abs(eigenvalues) < 1.0 - 1.0e-10
        retained[result.stationary_index] = False
        inverse_vectors = np.linalg.pinv(result.right_eigenvectors)
        left_amplitudes = left_factor @ result.right_eigenvectors
        right_amplitudes = inverse_vectors @ coefficients
        modal_amplitudes = left_amplitudes * right_amplitudes
        covariance = np.asarray(
            [
                np.sum(modal_amplitudes[retained] * eigenvalues[retained] ** lag)
                for lag in requested_lags
            ]
        )
    else:
        covariance = np.empty(requested_lags.size, dtype=complex)
        for index, lag_multiple in enumerate(requested_lags):
            covariance[index] = (
                left_factor
                @ np.linalg.matrix_power(result.koopman, int(lag_multiple))
                @ coefficients
            )
    return np.real_if_close(covariance)


def modal_psd(
    covariance: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Return a one-sided PSD from a finite covariance sequence."""
    covariance_values = np.asarray(covariance)
    frequency_values = np.asarray(frequencies, dtype=float)
    if covariance_values.ndim != 1 or covariance_values.size < 2:
        raise ValueError("covariance must contain at least two nonnegative lags.")
    if frequency_values.ndim != 1 or np.any((frequency_values < 0) | (frequency_values > 0.5)):
        raise ValueError("frequencies must lie in [0, 0.5] cycles per sample.")
    lag = np.arange(1, covariance_values.size)
    two_sided = covariance_values[0].real + 2.0 * np.real(
        np.exp(-2j * np.pi * frequency_values[:, None] * lag[None, :])
        @ covariance_values[1:]
    )
    one_sided = two_sided.copy()
    interior = (frequency_values > 0.0) & (frequency_values < 0.5)
    one_sided[interior] *= 2.0
    return one_sided
