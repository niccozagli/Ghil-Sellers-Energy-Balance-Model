"""Tests for diffusion-map and reduced EDMD utilities."""

import unittest

import numpy as np

from gsebm.reduced_dynamics import (
    fit_edmd_from_basis,
    fit_variable_bandwidth_diffusion_map,
    leading_nonstationary_indices,
    match_rates,
    modal_covariance,
    modal_psd,
    weighted_embedding,
)


class ReducedDynamicsTest(unittest.TestCase):
    def test_weighted_embedding_reproduces_weighted_distances(self) -> None:
        data = np.array([[1.0, 2.0], [4.0, 6.0], [-1.0, 3.0]])
        weights = np.array([0.25, 2.0])
        embedded = weighted_embedding(data, weights)
        expected = np.sum(weights * (data[0] - data[1]) ** 2)
        actual = np.sum((embedded[0] - embedded[1]) ** 2)
        self.assertAlmostEqual(actual, expected)

    def test_diffusion_map_has_constant_leading_function(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
        data = np.column_stack((np.cos(angles), np.sin(angles)))
        result = fit_variable_bandwidth_diffusion_map(
            data,
            np.ones(2),
            n_eigenfunctions=8,
            neighbor_count=24,
            local_scale_neighbors=8,
        )
        self.assertEqual(result.graph_components, 1)
        self.assertTrue(np.isfinite(result.basis).all())
        self.assertTrue(np.all(result.density > 0.0))
        self.assertTrue(np.all(np.diff(result.eigenvalues) <= 1.0e-12))
        self.assertAlmostEqual(result.eigenvalues[0], 1.0, places=8)
        np.testing.assert_allclose(
            result.basis[:, 0],
            np.full(data.shape[0], result.basis[0, 0]),
            atol=1.0e-7,
        )

    def test_edmd_recovers_diagonal_linear_map(self) -> None:
        sample_count = 80
        first = np.power(0.92, np.arange(sample_count))
        second = np.power(0.75, np.arange(sample_count))
        basis = np.column_stack((np.ones(sample_count), first, second))
        result = fit_edmd_from_basis(
            basis,
            lag=1,
            basis_size=3,
            rcond=1.0e-12,
        )
        recovered = np.sort_complex(result.eigenvalues)
        expected = np.sort_complex(np.array([1.0, 0.92, 0.75]))
        np.testing.assert_allclose(recovered, expected, atol=1.0e-9)
        self.assertEqual(result.retained_rank, 3)
        self.assertEqual(result.stationary_index, np.argmin(abs(result.eigenvalues - 1)))

    def test_edmd_truncates_a_singular_gram_matrix(self) -> None:
        coordinate = np.linspace(-1.0, 1.0, 30)
        basis = np.column_stack((np.ones(30), coordinate, coordinate))
        result = fit_edmd_from_basis(basis, lag=1, basis_size=3)
        self.assertEqual(result.retained_rank, 2)
        self.assertTrue(np.isfinite(result.koopman).all())

    def test_leading_modes_and_matching(self) -> None:
        sample_count = 60
        basis = np.column_stack(
            [np.ones(sample_count)]
            + [np.power(value, np.arange(sample_count)) for value in (0.95, 0.8, 0.6)]
        )
        result = fit_edmd_from_basis(basis, lag=1, basis_size=4, rcond=1.0e-12)
        indices = leading_nonstationary_indices(result, count=3)
        np.testing.assert_allclose(
            np.sort(result.eigenvalues[indices].real),
            [0.6, 0.8, 0.95],
            atol=1.0e-8,
        )
        reference = np.array([-0.1 + 0.2j, -0.2 + 0.0j])
        candidates = np.array([-0.19 + 0.0j, -0.11 + 0.21j, -2.0 + 0.0j])
        np.testing.assert_allclose(
            match_rates(reference, candidates), candidates[[1, 0]]
        )

    def test_covariance_and_psd_for_stable_mode(self) -> None:
        sample_count = 50
        coordinate = np.power(0.8, np.arange(sample_count))
        basis = np.column_stack((np.ones(sample_count), coordinate))
        result = fit_edmd_from_basis(
            basis, lag=1, basis_size=2, rcond=1.0e-12
        )
        coefficients = np.array([0.0, 1.0])
        lags = np.arange(200)
        covariance = np.asarray(modal_covariance(result, coefficients, lags)).real
        expected = result.gram[1, 1] * np.power(0.8, lags)
        np.testing.assert_allclose(covariance, expected, atol=1.0e-9)
        frequencies = np.linspace(0.0, 0.5, 40)
        psd = modal_psd(covariance, frequencies)
        self.assertTrue(np.isfinite(psd).all())
        self.assertTrue(np.all(psd >= -1.0e-10))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            weighted_embedding(np.ones((4, 2)), np.array([1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "lag"):
            fit_edmd_from_basis(np.ones((5, 2)), lag=0, basis_size=2)
        with self.assertRaisesRegex(ValueError, "frequencies"):
            modal_psd(np.ones(3), np.array([0.6]))

    def test_disconnected_neighbor_graph_is_rejected(self) -> None:
        first_cluster = np.column_stack(
            (np.linspace(0.0, 0.03, 6), np.zeros(6))
        )
        second_cluster = first_cluster + np.array([10.0, 10.0])
        data = np.vstack((first_cluster, second_cluster))
        with self.assertRaisesRegex(ValueError, "components"):
            fit_variable_bandwidth_diffusion_map(
                data,
                np.ones(2),
                n_eigenfunctions=3,
                neighbor_count=3,
                local_scale_neighbors=2,
            )


if __name__ == "__main__":
    unittest.main()
