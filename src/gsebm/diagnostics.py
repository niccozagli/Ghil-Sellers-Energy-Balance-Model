"""Diagnostics derived from model states and fluxes."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from gsebm.parameters import ModelParameters
from gsebm.physics import latitude_weight, total_diffusivity

ArrayLike: TypeAlias = float | np.ndarray

CALORIES_PER_SECOND_PER_WATT = 0.239
SQUARE_CENTIMETERS_PER_SQUARE_METER = 1.0e4
CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER = (
    SQUARE_CENTIMETERS_PER_SQUARE_METER / CALORIES_PER_SECOND_PER_WATT
)


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def pde_meridional_flux(
    latitude: ArrayLike,
    temperature: ArrayLike,
    temperature_x: ArrayLike,
    k1_value: ArrayLike,
    k2_value: ArrayLike,
    params: ModelParameters,
) -> np.ndarray:
    """Return the ``f`` term from the IVP PDE in cgs units.

    This is the quantity called ``f`` in the MATLAB ``ivp_eq_GSEBM.m``
    function:

        f = cos(pi x / 2) * k(T, x) * dT/dx * (2 / pi)^2
    """

    diffusivity = _as_float_array(total_diffusivity(temperature, k1_value, k2_value, params))
    return (
        _as_float_array(latitude_weight(latitude))
        * diffusivity
        * _as_float_array(temperature_x)
        * (2.0 / np.pi) ** 2
    )


def meridional_heat_transfer_rate_watts_per_square_meter(
    latitude: ArrayLike,
    temperature: ArrayLike,
    temperature_x: ArrayLike,
    k1_value: ArrayLike,
    k2_value: ArrayLike,
    params: ModelParameters,
) -> np.ndarray:
    """Return meridional heat-transfer rate specified to half the globe.

    This follows the final MATLAB diagnostic exactly:

        j = -f * pi / 2 * 1 / 0.239 * 1e4

    where ``f`` is the PDE flux term in cgs units. The output is in
    ``W m^-2``.
    """

    flux = pde_meridional_flux(latitude, temperature, temperature_x, k1_value, k2_value, params)
    return -flux * (np.pi / 2.0) * CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER
