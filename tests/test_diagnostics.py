"""Tests for diagnostic quantities derived from model states."""

import unittest

import numpy as np

from gsebm.diagnostics import (
    CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER,
    meridional_heat_transfer_rate_watts_per_square_meter,
    pde_meridional_flux,
)
from gsebm.parameters import default_model_parameters
from gsebm.physics import total_diffusivity


class DiagnosticsTest(unittest.TestCase):
    def test_pde_meridional_flux_matches_ivp_flux_definition(self) -> None:
        params = default_model_parameters()
        latitude = np.array([0.0, 0.5])
        temperature = np.array([280.0, 270.0])
        temperature_x = np.array([2.0, -1.0])
        k1 = np.array([3.0e-5, 2.0e-5])
        k2 = np.array([0.0, 1.0e-2])

        expected = (
            np.cos(0.5 * np.pi * latitude)
            * total_diffusivity(temperature, k1, k2, params)
            * temperature_x
            * (2.0 / np.pi) ** 2
        )
        np.testing.assert_allclose(
            pde_meridional_flux(latitude, temperature, temperature_x, k1, k2, params),
            expected,
        )

    def test_meridional_heat_transfer_rate_matches_matlab_conversion(self) -> None:
        params = default_model_parameters()
        latitude = np.array([0.0])
        temperature = np.array([280.0])
        temperature_x = np.array([2.0])
        k1 = np.array([3.0e-5])
        k2 = np.array([0.0])

        flux = pde_meridional_flux(latitude, temperature, temperature_x, k1, k2, params)
        expected = (
            -flux
            * np.pi
            / 2.0
            * CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER
        )
        np.testing.assert_allclose(
            meridional_heat_transfer_rate_watts_per_square_meter(
                latitude, temperature, temperature_x, k1, k2, params
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
