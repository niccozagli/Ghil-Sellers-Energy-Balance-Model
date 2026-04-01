"""Latitude-dependent empirical data.

The model uses a small set of tabulated latitude-dependent fields:

- heat capacity ``C(x)``
- solar irradiance ``Q(x)``
- the empirical ``b(x)`` term used in the albedo law
- a height-offset field ``z(x)``
- sensible and latent heat transport coefficients ``k1(x)`` and ``k2(x)``

These tables are observational inputs for the continuous model. They are
defined on coarse data grids and later interpolated before being evaluated
on a numerical solver grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

FloatSeries: TypeAlias = tuple[float, ...]


@dataclass(frozen=True)
class HalfEmpiricalData:
    """Empirical data defined on one hemisphere before equatorial mirroring."""

    heat_capacity: FloatSeries  # [cal cm^-2 K^-1]
    solar_irradiance: FloatSeries  # [cal cm^-2 s^-1]
    b_parameter: FloatSeries  # [1]
    surface_height_offset: FloatSeries  # [m]
    sensible_heat_flux_coefficient: FloatSeries  # [cal K^-1 cm^-2 s^-1]
    latent_heat_flux_coefficient: FloatSeries  # [cal dyn^-1 s^-1]


@dataclass(frozen=True)
class EmpiricalData:
    """Empirical data on the full normalized latitude domain ``[-1, 1]``."""

    x1: FloatSeries  # [1]
    x2: FloatSeries  # [1]
    heat_capacity: FloatSeries  # [cal cm^-2 K^-1]
    solar_irradiance: FloatSeries  # [cal cm^-2 s^-1]
    b_parameter: FloatSeries  # [1]
    surface_height_offset: FloatSeries  # [m]
    sensible_heat_flux_coefficient: FloatSeries  # [cal K^-1 cm^-2 s^-1]
    latent_heat_flux_coefficient: FloatSeries  # [cal dyn^-1 s^-1]

    def __post_init__(self) -> None:
        if len(self.x1) != len(self.heat_capacity):
            raise ValueError("x1 and heat_capacity must have the same length.")
        if len(self.x1) != len(self.solar_irradiance):
            raise ValueError("x1 and solar_irradiance must have the same length.")
        if len(self.x2) != len(self.b_parameter):
            raise ValueError("x2 and b_parameter must have the same length.")
        if len(self.x2) != len(self.surface_height_offset):
            raise ValueError("x2 and surface_height_offset must have the same length.")
        if len(self.x2) != len(self.sensible_heat_flux_coefficient):
            raise ValueError(
                "x2 and sensible_heat_flux_coefficient must have the same length."
            )
        if len(self.x2) != len(self.latent_heat_flux_coefficient):
            raise ValueError(
                "x2 and latent_heat_flux_coefficient must have the same length."
            )


@dataclass(frozen=True)
class EmpiricalInterpolants:
    """Continuous latitude-dependent empirical fields."""

    heat_capacity: CubicSpline
    solar_irradiance: CubicSpline
    b_parameter: CubicSpline
    surface_height_offset: CubicSpline
    sensible_heat_flux_coefficient: CubicSpline
    latent_heat_flux_coefficient: CubicSpline


@dataclass(frozen=True)
class IVPEmpiricalFields:
    """Empirical fields sampled on a fixed IVP solver grid."""

    x: np.ndarray  # [1]
    heat_capacity: np.ndarray  # [cal cm^-2 K^-1]
    solar_irradiance: np.ndarray  # [cal cm^-2 s^-1]
    b_parameter: np.ndarray  # [1]
    surface_height_offset: np.ndarray  # [m]
    sensible_heat_flux_coefficient: np.ndarray  # [cal K^-1 cm^-2 s^-1]
    latent_heat_flux_coefficient: np.ndarray  # [cal dyn^-1 s^-1]


def latitude_grid_x1() -> FloatSeries:
    """Return the data grid that includes the equator.

    This grid is used for the tabulated heat-capacity and solar-forcing
    values.
    """

    return tuple(step / 90.0 for step in range(-90, 91, 10))


def latitude_grid_x2() -> FloatSeries:
    """Return the data grid that excludes the equator.

    This grid is used for the remaining empirical fields, for which no data
    point is given at the equator.
    """

    south = tuple(step / 90.0 for step in range(-85, 0, 10))
    north = tuple(step / 90.0 for step in range(5, 86, 10))
    return south + north


def mirror_about_equator(values: FloatSeries, *, includes_equator: bool) -> FloatSeries:
    """Mirror one-hemisphere values onto the full latitude domain.

    The empirical fields are treated as hemispherically symmetric. When the
    source grid already contains the equator, the central value is not
    duplicated in the mirrored output.
    """

    if not values:
        raise ValueError("values must not be empty.")
    if includes_equator:
        if len(values) < 2:
            raise ValueError("values must contain at least two entries.")
        return values + values[-2::-1]
    return values + values[::-1]


def raw_empirical_data(remove_negative_k2: bool = True) -> HalfEmpiricalData:
    """Return the tabulated one-hemisphere empirical data.

    The ``remove_negative_k2`` switch selects a nonnegative variant of the
    latent-transport coefficient near the equator. This follows the default
    model setup used to avoid negative total diffusivity in later steps.
    """

    heat_capacity = (500.0, 1000.0, 1500.0, 4725.0, 5625.0, 5812.0, 5813.0, 5625.0, 6000.0, 5625.0)
    solar_irradiance = tuple(
        value * 1e-2 for value in (0.426, 0.440, 0.484, 0.579, 0.696, 0.804, 0.894, 0.961, 1.003, 1.017)
    )
    # This is the empirical b(x) term used inside the albedo law. It is not
    # itself the final physical albedo.
    b_parameter = (2.912, 2.96, 2.934, 2.914, 2.915, 2.868, 2.821, 2.804, 2.805)
    surface_height_offset = (1204.5, 820.0, 295.0, 150.5, 193.5, 301.0, 261.0, 133.5, 156.0)
    sensible_heat_flux_coefficient = tuple(
        value * 1e-5
        for value in (0.47113, 0.61988, 1.19933, 1.50214, 1.51063, 1.69562, 2.02342, 3.20611, 4.80401)
    )
    if remove_negative_k2:
        latent_heat_flux_coefficient = tuple(
            value * 1e-2 for value in (0.3, 0.9314, 1.9772, 3.4348, 4.8316, 3.7359, 0.6903, 0.2, 0.1)
        )
    else:
        latent_heat_flux_coefficient = tuple(
            value * 1e-2 for value in (0.0, 0.9314, 1.9772, 3.4348, 4.8316, 3.7359, 0.6903, -2.5401, -10.5975)
        )

    return HalfEmpiricalData(
        heat_capacity=heat_capacity,
        solar_irradiance=solar_irradiance,
        b_parameter=b_parameter,
        surface_height_offset=surface_height_offset,
        sensible_heat_flux_coefficient=sensible_heat_flux_coefficient,
        latent_heat_flux_coefficient=latent_heat_flux_coefficient,
    )


def default_empirical_data(remove_negative_k2: bool = True) -> EmpiricalData:
    """Return the default empirical data on the full latitude domain.

    The returned arrays still live on empirical data grids. They are inputs
    to later interpolation steps, not the numerical discretization grid of
    the PDE solver.
    """

    raw = raw_empirical_data(remove_negative_k2=remove_negative_k2)
    return EmpiricalData(
        x1=latitude_grid_x1(),
        x2=latitude_grid_x2(),
        heat_capacity=mirror_about_equator(raw.heat_capacity, includes_equator=True),
        solar_irradiance=mirror_about_equator(raw.solar_irradiance, includes_equator=True),
        b_parameter=mirror_about_equator(raw.b_parameter, includes_equator=False),
        surface_height_offset=mirror_about_equator(raw.surface_height_offset, includes_equator=False),
        sensible_heat_flux_coefficient=mirror_about_equator(
            raw.sensible_heat_flux_coefficient,
            includes_equator=False,
        ),
        latent_heat_flux_coefficient=mirror_about_equator(
            raw.latent_heat_flux_coefficient,
            includes_equator=False,
        ),
    )


def _as_float_array(values: FloatSeries | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _zero_slope_spline(x: FloatSeries | np.ndarray, y: FloatSeries | np.ndarray) -> CubicSpline:
    """Build a cubic spline with vanishing endpoint slopes."""

    return CubicSpline(_as_float_array(x), _as_float_array(y), bc_type=((1, 0.0), (1, 0.0)))


def build_empirical_interpolants(
    empirical_data: EmpiricalData | None = None,
    *,
    remove_negative_k2: bool = True,
    constrain_boundary_slopes: bool = True,
) -> EmpiricalInterpolants:
    """Build continuous empirical fields over latitude.

    When ``constrain_boundary_slopes`` is enabled, the fields originally
    tabulated on ``x2`` are first transferred to ``x1`` with a monotone
    PCHIP interpolator. Final cubic splines are then built with zero slopes
    at the endpoints to match the intended boundary behavior.
    """

    data = empirical_data or default_empirical_data(remove_negative_k2=remove_negative_k2)
    x1 = _as_float_array(data.x1)
    x2 = _as_float_array(data.x2)

    if constrain_boundary_slopes:
        b_on_x1 = PchipInterpolator(x2, _as_float_array(data.b_parameter))(x1)
        z_on_x1 = PchipInterpolator(x2, _as_float_array(data.surface_height_offset))(x1)
        k1_on_x1 = PchipInterpolator(x2, _as_float_array(data.sensible_heat_flux_coefficient))(x1)
        k2_on_x1 = PchipInterpolator(x2, _as_float_array(data.latent_heat_flux_coefficient))(x1)
        return EmpiricalInterpolants(
            heat_capacity=_zero_slope_spline(x1, data.heat_capacity),
            solar_irradiance=_zero_slope_spline(x1, data.solar_irradiance),
            b_parameter=_zero_slope_spline(x1, b_on_x1),
            surface_height_offset=_zero_slope_spline(x1, z_on_x1),
            sensible_heat_flux_coefficient=_zero_slope_spline(x1, k1_on_x1),
            latent_heat_flux_coefficient=_zero_slope_spline(x1, k2_on_x1),
        )

    return EmpiricalInterpolants(
        heat_capacity=CubicSpline(x1, _as_float_array(data.heat_capacity)),
        solar_irradiance=CubicSpline(x1, _as_float_array(data.solar_irradiance)),
        b_parameter=CubicSpline(x2, _as_float_array(data.b_parameter)),
        surface_height_offset=CubicSpline(x2, _as_float_array(data.surface_height_offset)),
        sensible_heat_flux_coefficient=CubicSpline(x2, _as_float_array(data.sensible_heat_flux_coefficient)),
        latent_heat_flux_coefficient=CubicSpline(x2, _as_float_array(data.latent_heat_flux_coefficient)),
    )


def sample_empirical_fields(
    x_grid: FloatSeries | np.ndarray,
    interpolants: EmpiricalInterpolants,
) -> IVPEmpiricalFields:
    """Sample continuous empirical fields on a fixed spatial grid."""

    x = _as_float_array(x_grid)
    return IVPEmpiricalFields(
        x=x,
        heat_capacity=np.asarray(interpolants.heat_capacity(x), dtype=float),
        solar_irradiance=np.asarray(interpolants.solar_irradiance(x), dtype=float),
        b_parameter=np.asarray(interpolants.b_parameter(x), dtype=float),
        surface_height_offset=np.asarray(interpolants.surface_height_offset(x), dtype=float),
        sensible_heat_flux_coefficient=np.asarray(
            interpolants.sensible_heat_flux_coefficient(x),
            dtype=float,
        ),
        latent_heat_flux_coefficient=np.asarray(
            interpolants.latent_heat_flux_coefficient(x),
            dtype=float,
        ),
    )


def prepare_ivp_empirical_fields(
    x_grid: FloatSeries | np.ndarray,
    empirical_data: EmpiricalData | None = None,
    *,
    remove_negative_k2: bool = True,
    constrain_boundary_slopes: bool = True,
) -> IVPEmpiricalFields:
    """Prepare empirical fields for IVP integration on a fixed solver grid.

    Build the interpolants once, then sample all latitude-only quantities
    once on the solver grid.
    """

    interpolants = build_empirical_interpolants(
        empirical_data=empirical_data,
        remove_negative_k2=remove_negative_k2,
        constrain_boundary_slopes=constrain_boundary_slopes,
    )
    return sample_empirical_fields(x_grid, interpolants)
