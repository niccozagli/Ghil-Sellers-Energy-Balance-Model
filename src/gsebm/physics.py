"""Core physical relationships used by the model.

These functions implement the local physics terms that appear in the
latitude-dependent energy balance equation:

    c(x) dT/dt = transport(T, x) - longwave_loss(T) + absorbed_solar(T, x)

The module keeps the formulas separate from interpolation and solver code.
It works with already-evaluated local quantities such as ``b(x)``, ``z(x)``,
``Q(x)``, ``k1(x)``, and ``k2(x)``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from gsebm.parameters import ModelParameters

ArrayLike: TypeAlias = float | np.ndarray


def _as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _to_python_scalar_if_needed(value: np.ndarray) -> ArrayLike:
    if value.ndim == 0:
        return float(value)
    return value


def latitude_weight(latitude: ArrayLike) -> ArrayLike:
    """Return the geometric weight ``cos(pi x / 2)``."""

    result = np.cos(np.pi * _as_float_array(latitude) / 2.0)
    return _to_python_scalar_if_needed(result)


def clamp_albedo(albedo: ArrayLike, albedo_min: float, albedo_max: float) -> ArrayLike:
    """Clamp albedo to the physically allowed interval."""

    result = np.clip(_as_float_array(albedo), albedo_min, albedo_max)
    return _to_python_scalar_if_needed(result)


def surface_albedo(
    temperature: ArrayLike,
    b_value: ArrayLike,
    z_value: ArrayLike,
    params: ModelParameters,
) -> ArrayLike:
    """Return the temperature-dependent albedo.

    The temperature dependence introduces the ice-albedo positive feedback.
    The empirical ``b(x)`` term and the height-offset contribution enter
    before the albedo is clipped to its lower and upper cutoffs.
    """

    temperature_array = _as_float_array(temperature)
    b_array = _as_float_array(b_value)
    z_array = _as_float_array(z_value)
    temperature_anomaly = np.minimum(temperature_array - params.c2 * z_array - params.um, 0.0)
    raw_albedo = b_array - params.c1 * (params.um + temperature_anomaly)
    return clamp_albedo(raw_albedo, albedo_min=params.albedo_min, albedo_max=params.albedo_max)


def moisture_factor(temperature: ArrayLike, params: ModelParameters) -> ArrayLike:
    """Return the moisture-dependent factor ``g(T)`` in the diffusivity law."""

    temperature_array = _as_float_array(temperature)
    result = params.c4 / temperature_array**2 * np.exp(-params.c5 / temperature_array)
    return _to_python_scalar_if_needed(result)


def total_diffusivity(
    temperature: ArrayLike,
    k1_value: ArrayLike,
    k2_value: ArrayLike,
    params: ModelParameters,
) -> ArrayLike:
    """Return ``k1(x) + g(T) k2(x)`` for meridional heat transport."""

    k1_array = _as_float_array(k1_value)
    k2_array = _as_float_array(k2_value)
    result = k1_array + k2_array * _as_float_array(moisture_factor(temperature, params))
    return _to_python_scalar_if_needed(result)


def outgoing_longwave_radiation(temperature: ArrayLike, params: ModelParameters) -> ArrayLike:
    """Return the longwave radiative loss term.

    The Stefan-Boltzmann ``sigma T^4`` contribution is modified by the
    cloud-feedback factor ``1 - m1 tanh(c3 T^6)``.
    """

    temperature_array = _as_float_array(temperature)
    result = params.sig * temperature_array**4 * (
        1.0 - params.m1 * np.tanh(params.c3 * temperature_array**6)
    )
    return _to_python_scalar_if_needed(result)


def absorbed_shortwave_radiation(
    q_value: ArrayLike,
    albedo: ArrayLike,
    params: ModelParameters,
) -> ArrayLike:
    """Return the absorbed shortwave forcing ``mu Q(x) (1 - alpha)``."""

    q_array = _as_float_array(q_value)
    albedo_array = _as_float_array(albedo)
    result = params.mu * q_array * (1.0 - albedo_array)
    return _to_python_scalar_if_needed(result)


def net_radiative_energy_transport(
    temperature: ArrayLike,
    q_value: ArrayLike,
    albedo: ArrayLike,
    params: ModelParameters,
) -> ArrayLike:
    """Return absorbed shortwave radiation minus outgoing longwave loss."""

    result = _as_float_array(absorbed_shortwave_radiation(q_value, albedo, params)) - _as_float_array(
        outgoing_longwave_radiation(temperature, params)
    )
    return _to_python_scalar_if_needed(result)
