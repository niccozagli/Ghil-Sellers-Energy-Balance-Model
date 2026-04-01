"""Smoke tests for the package scaffold."""

import unittest

import gsebm
from gsebm.parameters import ModelParameters, RunSettings


class PackageSmokeTest(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertTrue(gsebm.__version__)

    def test_default_model_parameters_have_expected_values(self) -> None:
        params = ModelParameters()

        self.assertEqual(params.mu, 1.0)
        self.assertEqual(params.albedo_min, 0.25)
        self.assertEqual(params.albedo_max, 0.6)
        self.assertEqual(params.c1, 0.009)
        self.assertEqual(params.c2, 0.0065)
        self.assertEqual(params.c3, 1.9e-15)
        self.assertAlmostEqual(params.c4, 1489082559.0935206)
        self.assertEqual(params.c5, 5350.0)
        self.assertEqual(params.sig, 1.356e-12)
        self.assertEqual(params.m1, 0.5)
        self.assertEqual(params.um, 283.16)

    def test_default_run_settings_have_expected_values(self) -> None:
        settings = RunSettings()

        self.assertEqual(settings.final_time, 1e9)
        self.assertEqual(settings.ivp_initial_temperature, 280.0)
        self.assertEqual(settings.bvp_initial_temperature, 0.0)
        self.assertEqual(settings.temperature_derivative_method, 1)
        self.assertTrue(settings.remove_negative_k2)
        self.assertTrue(settings.constrain_empirical_function_slopes)
        self.assertEqual(settings.symmetry_mode, 0)
        self.assertEqual(settings.delta, 1e-3)
        self.assertEqual(settings.time_output_count, 101)
        self.assertEqual(settings.interior_grid_count, 201)
        self.assertEqual(settings.pole_workaround_points_per_side, 2)
        self.assertEqual(settings.bvp_perturbation_amplitude, 10.0)
        self.assertEqual(settings.bvp_sampling_count, 1001)

    def test_invalid_temperature_derivative_method_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RunSettings(temperature_derivative_method=4)

    def test_invalid_albedo_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelParameters(albedo_min=0.7, albedo_max=0.6)


if __name__ == "__main__":
    unittest.main()
