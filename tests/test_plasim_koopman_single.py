"""Tests for the Koopman mode estimates used on observables."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_koopman_single as koopman


class ModeEstimateTests(unittest.TestCase):
    def test_regression_recovers_both_modes(self) -> None:
        rng = np.random.default_rng(1)
        years = np.arange(2000)
        radius = 1 + 0.2 * rng.standard_normal(years.size)
        leading = radius * np.exp(2j * np.pi * years / 50)
        harmonic = leading**2 / np.sqrt(np.mean(np.abs(leading)**4))
        a1 = np.array([1.0 - 0.5j, 0.3 + 0.2j])
        a2 = np.array([0.2j, -0.1 + 0.05j])
        observables = (
            2 * np.real(np.outer(leading, a1)) + 2 * np.real(np.outer(harmonic, a2))
            + 0.01 * rng.standard_normal((years.size, 2))
        )

        modes = koopman.regression_modes(leading, harmonic, observables)

        np.testing.assert_allclose(modes[0], a1, atol=5e-3)
        np.testing.assert_allclose(modes[1], a2, atol=5e-3)

    def test_composite_fourier_of_a_cosine(self) -> None:
        centres = (np.arange(36) + 0.5) * 2 * np.pi / 36
        composite = np.column_stack((np.cos(centres), np.sin(centres)))

        first = koopman.composite_fourier(composite, centres, 1)

        np.testing.assert_allclose(first, [0.5, -0.5j], atol=1e-12)
        self.assertAlmostEqual(
            complex(koopman.composite_fourier(np.cos(centres), centres, 2)), 0.0,
        )

    def test_time_to_peak_and_pattern_correlation(self) -> None:
        omega = 2 * np.pi / 40
        modes = np.exp(-1j * omega * np.array([3.0, -5.0, 25.0]))

        np.testing.assert_allclose(
            koopman.time_to_peak(modes, omega), [3.0, -5.0, -15.0],
        )
        weights = np.array([1.0, 2.0, 0.5])
        self.assertAlmostEqual(
            koopman.pattern_correlation(modes, (2 - 1j) * modes, weights), 1.0,
        )


class MapHelperTests(unittest.TestCase):
    def test_mode_cycle_combines_both_harmonics(self) -> None:
        modes = np.array([[0.5, 1j], [0.25, 0.0]])
        phase = np.array([0.0, np.pi / 2])

        cycle = koopman.mode_cycle(modes, phase)

        np.testing.assert_allclose(cycle, [[1.5, 0.0], [-0.5, -2.0]], atol=1e-12)

    def test_staggered_lsg_rows_fill_two_regular_columns(self) -> None:
        lon = np.array([[2.5, 7.5], [5.0, 10.0]])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        regular = koopman._lsg_to_regular(values, lon)

        # Column j spans [-180 + 2.5 j, -177.5 + 2.5 j]; 0° is column 72.
        np.testing.assert_array_equal(regular[0, 72:76], [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_array_equal(regular[1, 73:77], [3.0, 3.0, 4.0, 4.0])
        self.assertEqual(np.isfinite(regular).sum(), 8)


class ObservableModesTests(unittest.TestCase):
    def test_matches_column_by_column_koopman_modes(self) -> None:
        rng = np.random.default_rng(0)
        years = np.arange(400)
        phase = 2 * np.pi * years / 37
        surface = np.column_stack([
            np.cos(phase + shift) for shift in (0.0, 0.4, 0.9)
        ]) + 0.1 * rng.standard_normal((years.size, 3))
        ocean = np.column_stack([
            np.sin(phase + shift) for shift in (0.0, 0.7)
        ]) + 0.1 * rng.standard_normal((years.size, 2))
        fields = {
            "state": np.column_stack((surface, ocean)),
            "surface_anomaly": surface,
            "surface_weights": np.full(3, 1 / 3),
            "ocean_weights": np.full(2, 1 / 2),
        }
        spectrum, rates, _, _ = koopman.fit_koopman(fields, 1, 1e-3, 1000, 0)
        indices = koopman.kdmd_training_indices(years.size, 1, 1000, 0)
        observables = np.column_stack((surface[:, 0] ** 2, ocean))
        selected = [0, rates.size - 1]
        scales = np.array([2.0, 0.5 + 0.5j])

        modes = koopman.observable_modes(
            spectrum, fields["state"], observables, indices, selected, scales,
        )

        expected = np.column_stack([
            spectrum.koopman_modes(observables[indices, column])[selected]
            for column in range(observables.shape[1])
        ]) / scales[:, None]
        np.testing.assert_allclose(modes, expected, rtol=1e-10, atol=1e-12)
        with self.assertRaises(ValueError):
            koopman.observable_modes(
                spectrum, fields["state"], observables, indices[::-1], selected,
            )


class CombinedStateTests(unittest.TestCase):
    def test_blocks_are_stacked_and_constant_ice_rows_dropped(self) -> None:
        rng = np.random.default_rng(3)
        fields = {
            "surface_anomaly": rng.standard_normal((50, 3)),
            "surface_lat": np.array([-40.0, -30.0, -20.0]),
            "surface_weights": np.array([1.0, 1.0, 2.0]),
            "ocean_anomaly": rng.standard_normal((50, 2)),
            "ocean_lat": np.array([-35.0, -25.0]),
            "ocean_weights": np.array([0.5, 0.5]),
            "ice_rows_anomaly": np.column_stack((rng.standard_normal(50), np.zeros(50))),
            "ice_row_lat": np.array([-36.0, -20.0]),
            "ice_row_weights": np.array([1.0, 1.0]),
        }

        combined = koopman.combined_state(fields, ("ice", "ocean"))

        self.assertEqual(combined["state"].shape, (50, 3))
        names = [block[0] for block in combined["state_blocks"]]
        self.assertEqual(names, ["ice", "ocean"])
        np.testing.assert_allclose(combined["state_blocks"][0][3], [1.0])
        np.testing.assert_array_equal(combined["state_blocks"][0][4], [-36.0])


if __name__ == "__main__":
    unittest.main()
