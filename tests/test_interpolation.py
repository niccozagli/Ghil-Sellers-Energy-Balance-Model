"""Tests for interpolation and IVP-oriented preprocessing."""

import unittest

import numpy as np

from gsebm.empirical import (
    build_empirical_interpolants,
    default_empirical_data,
    prepare_ivp_empirical_fields,
)


class InterpolationTest(unittest.TestCase):
    def test_unconstrained_interpolants_match_tabulated_values_on_native_grids(self) -> None:
        data = default_empirical_data()
        interpolants = build_empirical_interpolants(data, constrain_boundary_slopes=False)

        x1 = np.asarray(data.x1, dtype=float)
        x2 = np.asarray(data.x2, dtype=float)

        np.testing.assert_allclose(interpolants.heat_capacity(x1), data.heat_capacity)
        np.testing.assert_allclose(interpolants.solar_irradiance(x1), data.solar_irradiance)
        np.testing.assert_allclose(interpolants.b_parameter(x2), data.b_parameter)
        np.testing.assert_allclose(interpolants.surface_height_offset(x2), data.surface_height_offset)
        np.testing.assert_allclose(
            interpolants.sensible_heat_flux_coefficient(x2),
            data.sensible_heat_flux_coefficient,
        )
        np.testing.assert_allclose(
            interpolants.latent_heat_flux_coefficient(x2),
            data.latent_heat_flux_coefficient,
        )

    def test_constrained_interpolants_have_zero_boundary_slopes(self) -> None:
        data = default_empirical_data()
        interpolants = build_empirical_interpolants(data, constrain_boundary_slopes=True)

        for spline in (
            interpolants.heat_capacity,
            interpolants.solar_irradiance,
            interpolants.b_parameter,
            interpolants.surface_height_offset,
            interpolants.sensible_heat_flux_coefficient,
            interpolants.latent_heat_flux_coefficient,
        ):
            self.assertAlmostEqual(float(spline(-1.0, 1)), 0.0, places=10)
            self.assertAlmostEqual(float(spline(1.0, 1)), 0.0, places=10)

    def test_prepare_ivp_empirical_fields_samples_arrays_on_solver_grid(self) -> None:
        x_grid = np.linspace(-0.998, 0.998, 25)
        fields = prepare_ivp_empirical_fields(x_grid)

        self.assertEqual(fields.x.shape, x_grid.shape)
        self.assertEqual(fields.heat_capacity.shape, x_grid.shape)
        self.assertEqual(fields.solar_irradiance.shape, x_grid.shape)
        self.assertEqual(fields.b_parameter.shape, x_grid.shape)
        self.assertEqual(fields.surface_height_offset.shape, x_grid.shape)
        self.assertEqual(fields.sensible_heat_flux_coefficient.shape, x_grid.shape)
        self.assertEqual(fields.latent_heat_flux_coefficient.shape, x_grid.shape)
        self.assertTrue(np.isfinite(fields.heat_capacity).all())
        self.assertTrue(np.isfinite(fields.latent_heat_flux_coefficient).all())

    def test_prepare_ivp_empirical_fields_respects_negative_k2_switch(self) -> None:
        x_grid = np.linspace(-0.5, 0.5, 11)
        positive = prepare_ivp_empirical_fields(x_grid, remove_negative_k2=True)
        signed = prepare_ivp_empirical_fields(x_grid, remove_negative_k2=False)

        self.assertGreaterEqual(float(np.min(positive.latent_heat_flux_coefficient)), 0.0)
        self.assertLess(float(np.min(signed.latent_heat_flux_coefficient)), 0.0)


if __name__ == "__main__":
    unittest.main()
