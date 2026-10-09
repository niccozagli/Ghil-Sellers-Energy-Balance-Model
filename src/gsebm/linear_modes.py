"""Linear modes of variability: EOF truncation and linear inverse models (LIM).

A LIM fits the propagator of a stationary time series at one lag,
A(τ) = C(τ) C(0)⁻¹ with C(τ) = ⟨x(t + τ) x(t)ᵀ⟩, on the leading EOFs of
area-weighted anomalies. For a linear system dx/dt = L x + noise, A(τ) =
exp(L τ), so the rates s = log λ(τ) / τ estimate the eigenvalues of L and
should not depend on τ.

Modes are picked by their alignment with a reference pattern (for example
the forced response dX*/dμ), not by their amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EOFBasis:
    """Leading EOFs of area-weighted anomalies.

    `patterns` has shape (k, n_features) and holds field patterns in the
    original units, orthonormal under the area-weighted inner product.
    `pcs` has shape (n_samples, k).
    """

    mean: np.ndarray
    patterns: np.ndarray
    pcs: np.ndarray
    variance_fraction: np.ndarray


@dataclass(frozen=True)
class LIMResult:
    """LIM fitted at one lag (in seconds), modes sorted slowest first."""

    lag_seconds: float
    eigenvalues: np.ndarray
    rates: np.ndarray
    field_patterns: np.ndarray


def weighted_inner(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> complex:
    """Area-weighted inner product Σ w conj(a) b."""
    return complex(np.sum(np.asarray(weights) * np.conj(a) * b))


def pattern_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """|cos| of the area-weighted angle between two (possibly complex) patterns."""
    norm = np.sqrt(weighted_inner(a, a, weights).real * weighted_inner(b, b, weights).real)
    return float(abs(weighted_inner(a, b, weights)) / norm)


def eof_basis(data: np.ndarray, weights: np.ndarray, count: int) -> EOFBasis:
    """Return the leading `count` EOFs of `data` (n_samples, n_features).

    Anomalies are taken from the time mean; EOFs are computed in the metric
    Σ w x², so that the PCs are area-weighted projections.
    """
    values = np.asarray(data, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = values.mean(axis=0)
    scaled = (values - mean) * np.sqrt(w)
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    patterns = vt[:count] / np.sqrt(w)
    return EOFBasis(
        mean=mean,
        patterns=patterns,
        pcs=scaled @ vt[:count].T,
        variance_fraction=singular[:count] ** 2 / np.sum(singular**2),
    )


def fit_lim(pcs: np.ndarray, lag: int, sample_seconds: float,
            patterns: np.ndarray | None = None) -> LIMResult:
    """Fit A = C(lag) C(0)⁻¹ on `pcs` (n_samples, k) and return its modes.

    `lag` is in samples. Rates are log(λ) / τ in s⁻¹ (complex). If EOF
    `patterns` are given, the eigenvectors are mapped back to field space.
    """
    x = np.asarray(pcs, dtype=float)
    x0, x1 = x[:-lag], x[lag:]
    c0 = x0.T @ x0 / x0.shape[0]
    ctau = x1.T @ x0 / x0.shape[0]
    eigenvalues, vectors = np.linalg.eig(ctau @ np.linalg.inv(c0))
    tau = lag * sample_seconds
    rates = np.log(eigenvalues.astype(complex)) / tau
    order = np.argsort(-rates.real)
    fields = vectors[:, order].T
    if patterns is not None:
        fields = fields @ np.asarray(patterns)
    return LIMResult(
        lag_seconds=tau,
        eigenvalues=eigenvalues[order],
        rates=rates[order],
        field_patterns=fields,
    )


def best_aligned_mode(result: "LIMResult | KDMDResult", reference: np.ndarray,
                      weights: np.ndarray) -> tuple[int, float]:
    """Index of the mode whose field pattern best aligns with `reference`, and its |cos|."""
    cosines = [pattern_cosine(p, reference, weights) for p in result.field_patterns]
    index = int(np.argmax(cosines))
    return index, cosines[index]


@dataclass(frozen=True)
class KDMDResult:
    """Gaussian-kernel KDMD at one lag, modes sorted slowest first.

    `field_patterns` are the Koopman modes of the state itself (the linear
    observable), so they are comparable with LIM eigenvectors.
    """

    lag_seconds: float
    rates: np.ndarray
    field_patterns: np.ndarray
    rank: int
    bandwidth: float
    eigenfunction_variance: np.ndarray
    origins: np.ndarray | None = None
    eigenfunctions: np.ndarray | None = None


def fit_kdmd(anomalies: np.ndarray, weights: np.ndarray, lag: int, sample_seconds: float,
             training_count: int, seed: int, rel_threshold: float = 1e-3) -> KDMDResult:
    """Fit KDMD with a weighted Gaussian kernel on snapshot pairs of `anomalies`.

    Pairs (x_t, x_{t+lag}) are drawn at `training_count` random origins. The
    kernel is exp(-Σ w (x - y)² / 2σ²) with σ the median pairwise weighted
    distance (over at most 5000 training states); the Gram matrix is
    factorized partially (eigsh) and truncated at `rel_threshold` (TSVD). The stationary mode (eigenvalue closest to 1 with a
    constant eigenfunction) is dropped. Each remaining mode k contributes
    a_k φ_k to the state, with mean |φ_k|² in `eigenfunction_variance`.
    `eigenfunctions[:, k]` holds φ_k at the training origins (indices
    `origins`), from the factorization G ≈ U S Uᵀ: φ = U S^½ v.
    """
    from scipy.spatial.distance import pdist
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.algorithms import KernelDMD, WeightedGaussianKernel
    from koopman_response.algorithms.regularization import TSVDRegularizer

    values = np.asarray(anomalies, dtype=float)
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    origins = np.sort(rng.choice(values.shape[0] - lag, size=training_count, replace=False))
    x, y = values[origins], values[origins + lag]
    subset = x[rng.choice(training_count, size=min(training_count, 5000), replace=False)]
    bandwidth = float(np.median(pdist(subset * np.sqrt(w))))
    kdmd = KernelDMD(kernel=WeightedGaussianKernel(sigma=bandwidth, weights=w))
    kdmd.fit_snapshots(X=x, Y=y, show_progress=False)
    tsvd = TSVDRegularizer()
    tsvd.factorize(kdmd.G, method="eigsh", symmetrize=False, rel_threshold=rel_threshold)
    kdmd.G = None
    reduced, u_r, s_r = tsvd.solve_from_factorization(kdmd.A, rel_threshold=rel_threshold)
    kdmd.A = None
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(reduced, U_r=u_r, S_r=s_r)
    tau = lag * sample_seconds
    rates = np.log(spectrum.eigenvalues.astype(complex)) / tau
    projection = u_r.conj().T @ x / np.sqrt(s_r)[:, None]
    modes = spectrum.left_eigvecs.conj().T @ projection
    # mean |phi_k|² over the training states: diag(V* S_r V) / N
    variance = np.real(np.einsum("ik,i,ik->k", spectrum.right_eigvecs.conj(), s_r,
                                 spectrum.right_eigvecs)) / training_count
    stationary = int(np.argmin(np.abs(spectrum.eigenvalues - 1.0)))
    order = [k for k in np.argsort(-rates.real) if k != stationary]
    phi = (u_r * np.sqrt(s_r)[None, :]) @ spectrum.right_eigvecs
    return KDMDResult(lag_seconds=tau, rates=rates[order], field_patterns=modes[order],
                      rank=int(s_r.size), bandwidth=bandwidth,
                      eigenfunction_variance=variance[order],
                      origins=origins, eigenfunctions=phi[:, order])


def dominant_mode_along(result: KDMDResult, reference: np.ndarray,
                        weights: np.ndarray, min_cosine: float = 0.9) -> tuple[int, float]:
    """Mode carrying the largest variance along `reference`, and its |cos| with it.

    The variance along the unit reference direction r̂ from mode k is
    |⟨r̂, a_k⟩|² ⟨|φ_k|²⟩. Unlike `best_aligned_mode`, this ignores modes that
    are aligned with r but carry almost no variance. Only modes with
    |cos| ≥ `min_cosine` are eligible, which excludes badly fitted modes whose
    eigenfunction variance is inflated.
    """
    norm = np.sqrt(weighted_inner(reference, reference, weights).real)
    along = np.array([abs(weighted_inner(reference, a, weights)) / norm
                      for a in result.field_patterns])
    cosines = np.array([pattern_cosine(a, reference, weights) for a in result.field_patterns])
    score = np.where(cosines >= min_cosine, along**2 * result.eigenfunction_variance, -np.inf)
    if not np.isfinite(score).any():
        raise ValueError(f"No mode has |cos| ≥ {min_cosine} with the reference.")
    index = int(np.argmax(score))
    return index, float(cosines[index])
