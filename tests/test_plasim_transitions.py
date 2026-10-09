"""Tests for the transition-anatomy diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_transitions as transitions


class SectorTests(unittest.TestCase):
    def test_sectors_partition_the_globe(self) -> None:
        lon = np.arange(64) * 5.625
        total = sum(transitions.sector_mask(lon, name).astype(int) for name in transitions.SECTORS)

        np.testing.assert_array_equal(total, 1)
        self.assertTrue(transitions.sector_mask(np.array([-30.0]), "atlantic")[0])
        self.assertTrue(transitions.sector_mask(np.array([180.0]), "pacific")[0])


class OceanMeanTests(unittest.TestCase):
    def test_layer_mean_uses_only_selected_rows_and_levels(self) -> None:
        lsg_lat = np.array([-30.0, 0.0, 30.0])
        depth = np.array([50.0, 500.0, 2000.0])
        volume = np.ones((3, 3))
        theta = np.zeros((2, 3, 3))
        theta[:, 1, :2] = 5.0  # warm tropical upper ocean

        tropical = transitions.ocean_layer_mean(theta, volume, lsg_lat, depth, (0, 700), (-20, 20))
        deep = transitions.ocean_layer_mean(theta, volume, lsg_lat, depth, (1000, 1e4))

        np.testing.assert_allclose(tropical, 5.0)
        np.testing.assert_allclose(deep, 0.0)


class RunningMeanTests(unittest.TestCase):
    def test_constant_series_stays_constant_at_the_ends(self) -> None:
        np.testing.assert_allclose(transitions.running_mean(np.full(50, 275.0), 31), 275.0)

    def test_missing_values_are_skipped(self) -> None:
        values = np.array([1.0, np.nan, 3.0, 5.0])

        np.testing.assert_allclose(transitions.running_mean(values, 3), [1.0, 2.0, 4.0, 4.0])


class OnsetTests(unittest.TestCase):
    def test_onset_finds_a_step_after_a_trend(self) -> None:
        rng = np.random.default_rng(0)
        years = np.arange(1000, 2000)
        values = 0.001 * (years - 1000) + 0.05 * rng.standard_normal(years.size)
        values[years >= 1600] -= 2.0

        onset = transitions.onset_year(years, values, (1100, 1500))

        self.assertIsNotNone(onset)
        self.assertLessEqual(abs(onset - 1600), 10)

    def test_no_onset_without_a_change(self) -> None:
        rng = np.random.default_rng(1)
        years = np.arange(1000, 2000)
        values = 0.002 * years + rng.standard_normal(years.size)

        self.assertIsNone(transitions.onset_year(years, values, (1100, 1500)))

    def test_variability_of_an_ar1(self) -> None:
        rng = np.random.default_rng(2)
        x = np.zeros(20000)
        for k in range(1, x.size):
            x[k] = 0.8 * x[k - 1] + rng.standard_normal()

        std, lag_one = transitions.variability(x + 0.01 * np.arange(x.size))

        self.assertAlmostEqual(lag_one, 0.8, delta=0.02)
        self.assertAlmostEqual(std, 1 / np.sqrt(1 - 0.64), delta=0.05)


if __name__ == "__main__":
    unittest.main()
