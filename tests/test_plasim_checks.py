"""Tests for the grid-signature and energy-closure checks."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_checks as checks


class PartialRowTests(unittest.TestCase):
    def test_only_rows_strictly_between_bounds_are_partial(self) -> None:
        lat = np.array([-40.0, -30.0, -20.0, 20.0, 30.0])
        concentration = np.array([1.0, 0.5, 0.02, 0.3, np.nan])

        rows = checks.partial_rows(concentration, lat)

        np.testing.assert_array_equal(rows["S"], [-30.0])
        np.testing.assert_array_equal(rows["N"], [20.0])


class JumpTests(unittest.TestCase):
    def test_contributions_sum_to_the_hemispheric_change(self) -> None:
        years = np.arange(10)
        lat = np.array([-30.0, -20.0, 20.0])
        atlantic = np.zeros((10, 3))
        atlantic[5:, 1] = 2.0  # one southern row fills after year 4
        pacific = np.zeros((10, 3))
        pacific[5:, 0] = 0.5
        rows = checks.IceRows("x", years, lat, {"atlantic": atlantic, "pacific": pacific},
                              np.zeros((10, 3)))

        parts = checks.jump_contributions(rows, (0, 4), (5, 9))

        self.assertEqual(parts[0][:2], ("atlantic", -20.0))
        self.assertAlmostEqual(parts[0][2], 2.0)
        self.assertAlmostEqual(sum(part[2] for part in parts), 2.5)



class EnergyTests(unittest.TestCase):
    def test_period_difference_is_event_minus_quiet(self) -> None:
        years = np.arange(100)
        sphere = 4 * np.pi * checks.EARTH_RADIUS_M**2
        toa = np.where(years >= 50, -2.0, 0.0) * sphere
        zeros = np.zeros(100)
        data = checks.EnergySeries("x", years, toa, zeros, toa, zeros)

        difference = checks.period_difference(data, (50, 99), (0, 49))

        self.assertAlmostEqual(difference["toa_net"], -2.0)
        self.assertAlmostEqual(difference["ocean_storage"], -2.0)
        self.assertAlmostEqual(difference["ocean_uptake"], 0.0)


if __name__ == "__main__":
    unittest.main()
