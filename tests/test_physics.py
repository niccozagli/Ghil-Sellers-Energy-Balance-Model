"""Tests for local physics relationships."""

import unittest

import numpy as np

from gsebm.parameters import ModelParameters
from gsebm.physics import (
    absorbed_shortwave_radiation,
    clamp_albedo,
    latitude_weight,
    moisture_factor,
    net_radiative_energy_transport,
    outgoing_longwave_radiation,
    surface_albedo,
    total_diffusivity,
)


class PhysicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ModelParameters()

    def test_latitude_weight_matches_geometry(self) -> None:
        self.assertAlmostEqual(latitude_weight(0.0), 1.0)
        self.assertAlmostEqual(latitude_weight(-1.0), 0.0, places=12)
        self.assertAlmostEqual(latitude_weight(1.0), 0.0, places=12)

    def test_clamp_albedo_enforces_cutoffs(self) -> None:
        result = clamp_albedo(
            np.array([0.1, 0.4, 0.8]),
            albedo_min=self.params.albedo_min,
            albedo_max=self.params.albedo_max,
        )
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.6]))

    def test_surface_albedo_uses_cold_branch_and_upper_cutoff(self) -> None:
        result = surface_albedo(250.0, 2.912, 150.5, self.params)
        self.assertEqual(result, self.params.albedo_max)

    def test_surface_albedo_uses_warm_branch_without_extra_cooling_term(self) -> None:
        result = surface_albedo(300.0, 2.912, 150.5, self.params)
        expected = 2.912 - self.params.c1 * self.params.um
        self.assertAlmostEqual(result, expected)

    def test_moisture_factor_matches_formula(self) -> None:
        temperature = 280.0
        expected = self.params.c4 / temperature**2 * np.exp(-self.params.c5 / temperature)
        self.assertAlmostEqual(moisture_factor(temperature, self.params), expected)

    def test_total_diffusivity_matches_k1_plus_gk2(self) -> None:
        temperature = 280.0
        k1_value = 1.5e-5
        k2_value = 0.02
        expected = k1_value + k2_value * moisture_factor(temperature, self.params)
        self.assertAlmostEqual(total_diffusivity(temperature, k1_value, k2_value, self.params), expected)

    def test_outgoing_longwave_radiation_matches_formula(self) -> None:
        temperature = 280.0
        expected = self.params.sig * temperature**4 * (
            1.0 - self.params.m1 * np.tanh(self.params.c3 * temperature**6)
        )
        self.assertAlmostEqual(outgoing_longwave_radiation(temperature, self.params), expected)

    def test_net_radiative_energy_transport_matches_balance(self) -> None:
        temperature = 280.0
        q_value = 9.61e-3
        albedo = 0.35
        expected = absorbed_shortwave_radiation(q_value, albedo, self.params) - outgoing_longwave_radiation(
            temperature,
            self.params,
        )
        self.assertAlmostEqual(net_radiative_energy_transport(temperature, q_value, albedo, self.params), expected)


if __name__ == "__main__":
    unittest.main()
