"""Tests for empirical latitude-dependent data."""

import unittest

from gsebm.empirical import (
    default_empirical_data,
    latitude_grid_x1,
    latitude_grid_x2,
    mirror_about_equator,
    raw_empirical_data,
)


class EmpiricalDataTest(unittest.TestCase):
    def test_latitude_grids_have_expected_lengths(self) -> None:
        x1 = latitude_grid_x1()
        x2 = latitude_grid_x2()

        self.assertEqual(len(x1), 19)
        self.assertEqual(len(x2), 18)
        self.assertEqual(x1[0], -1.0)
        self.assertEqual(x1[-1], 1.0)
        self.assertEqual(x2[0], -85.0 / 90.0)
        self.assertEqual(x2[-1], 85.0 / 90.0)
        self.assertIn(0.0, x1)
        self.assertNotIn(0.0, x2)

    def test_mirror_about_equator_keeps_or_skips_equator(self) -> None:
        self.assertEqual(mirror_about_equator((1.0, 2.0, 3.0), includes_equator=True), (1.0, 2.0, 3.0, 2.0, 1.0))
        self.assertEqual(mirror_about_equator((1.0, 2.0, 3.0), includes_equator=False), (1.0, 2.0, 3.0, 3.0, 2.0, 1.0))

    def test_raw_empirical_data_uses_expected_k22_variant(self) -> None:
        positive = raw_empirical_data(remove_negative_k2=True)
        signed = raw_empirical_data(remove_negative_k2=False)

        self.assertEqual(positive.latent_heat_flux_coefficient[0], 0.003)
        self.assertEqual(signed.latent_heat_flux_coefficient[0], 0.0)
        self.assertLess(signed.latent_heat_flux_coefficient[-1], 0.0)
        self.assertGreaterEqual(min(positive.latent_heat_flux_coefficient), 0.0)

    def test_default_empirical_data_has_expected_shapes(self) -> None:
        data = default_empirical_data()

        self.assertEqual(len(data.x1), 19)
        self.assertEqual(len(data.x2), 18)
        self.assertEqual(len(data.heat_capacity), 19)
        self.assertEqual(len(data.solar_irradiance), 19)
        self.assertEqual(len(data.b_parameter), 18)
        self.assertEqual(len(data.surface_height_offset), 18)
        self.assertEqual(len(data.sensible_heat_flux_coefficient), 18)
        self.assertEqual(len(data.latent_heat_flux_coefficient), 18)

    def test_default_empirical_data_is_symmetric(self) -> None:
        data = default_empirical_data()

        self.assertEqual(data.heat_capacity[0], data.heat_capacity[-1])
        self.assertEqual(data.heat_capacity[1], data.heat_capacity[-2])
        self.assertEqual(data.b_parameter[0], data.b_parameter[-1])
        self.assertEqual(data.b_parameter[3], data.b_parameter[-4])
        self.assertEqual(
            data.sensible_heat_flux_coefficient[2],
            data.sensible_heat_flux_coefficient[-3],
        )


if __name__ == "__main__":
    unittest.main()
