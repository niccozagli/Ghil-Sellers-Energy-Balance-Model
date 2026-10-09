"""Tests for EOF/LIM mode estimates and EBM linear stability."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

from gsebm import linear_modes as lm


def ornstein_uhlenbeck(operator: np.ndarray, steps: int, dt: float, seed: int) -> np.ndarray:
    """Exact discrete sampling of dx = L x dt + dW."""
    rng = np.random.default_rng(seed)
    propagator = expm(operator * dt)
    # stationary covariance increment for unit noise: Q = C - A C Aᵀ
    covariance = np.linalg.solve(
        np.kron(np.eye(2), operator) + np.kron(operator, np.eye(2)), -np.eye(2).ravel()
    ).reshape(2, 2)
    noise = np.linalg.cholesky(covariance - propagator @ covariance @ propagator.T)
    x = np.zeros((steps, 2))
    for k in range(1, steps):
        x[k] = propagator @ x[k - 1] + noise @ rng.standard_normal(2)
    return x


class LIMTests(unittest.TestCase):
    def test_rates_of_a_linear_process_are_recovered_at_every_lag(self) -> None:
        operator = np.array([[-0.1, 0.05], [0.0, -1.0]])
        x = ornstein_uhlenbeck(operator, 200_000, 0.25, seed=0)

        for lag in (1, 4, 8):
            result = lm.fit_lim(x, lag, sample_seconds=0.25)
            np.testing.assert_allclose(np.sort(result.rates.real), [-1.0, -0.1], rtol=0.1)

    def test_best_aligned_mode_is_the_slow_eigenvector(self) -> None:
        operator = np.array([[-0.1, 0.05], [0.0, -1.0]])
        x = ornstein_uhlenbeck(operator, 100_000, 0.25, seed=1)
        result = lm.fit_lim(x, 4, sample_seconds=0.25)

        index, cosine = lm.best_aligned_mode(result, np.array([1.0, 0.0]), np.ones(2))

        self.assertAlmostEqual(result.rates[index].real, -0.1, delta=0.01)
        self.assertGreater(cosine, 0.99)


class KDMDTests(unittest.TestCase):
    def test_mode_along_the_slow_direction_has_the_slow_rate(self) -> None:
        operator = np.array([[-0.1, 0.05], [0.0, -1.0]])
        x = ornstein_uhlenbeck(operator, 40_000, 0.25, seed=3)

        result = lm.fit_kdmd(x, np.ones(2), 4, 0.25, training_count=2000, seed=0)
        index, cosine = lm.dominant_mode_along(result, np.array([1.0, 0.0]), np.ones(2))

        self.assertAlmostEqual(result.rates[index].real, -0.1, delta=0.03)
        self.assertGreater(cosine, 0.95)
        # the slow eigenfunction tracks the slow coordinate at the training states
        phi = result.eigenfunctions[:, index].real
        self.assertGreater(abs(np.corrcoef(phi, x[result.origins, 0])[0, 1]), 0.9)


class EOFTests(unittest.TestCase):
    def test_patterns_are_orthonormal_in_the_weighted_metric(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))
        weights = np.array([0.5, 1.0, 1.5, 2.0, 0.2, 0.8])

        basis = lm.eof_basis(data, weights, 4)

        gram = (basis.patterns * weights) @ basis.patterns.T
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(basis.pcs, (data - data.mean(0)) * weights @ basis.patterns.T,
                                   atol=1e-10)


class StabilityTests(unittest.TestCase):
    def test_warm_equilibrium_slow_mode_is_the_forced_response(self) -> None:
        from gsebm.ebm_stability import stability

        result = stability(1.0, np.full(205, 300.0))  # 280 K converges to the edge state
        weights = np.gradient(result.x)

        self.assertTrue(np.all(result.rates.real < 0))
        self.assertAlmostEqual(result.rates[0].real, -0.152, delta=0.005)
        self.assertGreater(
            lm.pattern_cosine(result.modes[0], result.forced_response, weights), 0.99)


if __name__ == "__main__":
    unittest.main()
