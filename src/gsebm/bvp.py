"""Boundary value problem components for steady-state solutions.

The BVP differs from the IVP in one important respect: the solver evaluates
the ODE on an adaptive mesh, so the latitude-dependent coefficients must be
available as continuous functions of latitude rather than as arrays sampled
once on a fixed grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline, PPoly

from gsebm.empirical import build_empirical_interpolants
from gsebm.ivp import IVPSolution, build_ivp_grid, solve_temperature_ivp
from gsebm.parameters import ModelParameters, RunSettings, default_model_parameters, default_run_settings
from gsebm.physics import net_radiative_energy_transport, surface_albedo

ArrayLike: TypeAlias = float | np.ndarray
InitialGuessKind: TypeAlias = Literal["scalar", "ivp", "custom"]
Interpolant: TypeAlias = CubicSpline | PPoly


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _zero_slope_spline(x: np.ndarray, y: np.ndarray) -> CubicSpline:
    return CubicSpline(x, y, bc_type=((1, 0.0), (1, 0.0)))


@dataclass(frozen=True)
class BVPFields:
    """Continuous latitude-dependent empirical fields for the BVP."""

    solar_irradiance: Interpolant
    b_parameter: Interpolant
    surface_height_offset: Interpolant
    sensible_heat_flux_coefficient: Interpolant
    latent_heat_flux_coefficient: Interpolant
    sensible_heat_flux_derivative: Interpolant
    latent_heat_flux_derivative: Interpolant


@dataclass(frozen=True)
class BVPInitialGuess:
    """Initial guess for the BVP state vector."""

    x: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if self.values.shape != (2, self.x.size):
            raise ValueError("values must have shape (2, x.size).")


@dataclass(frozen=True)
class BVPProblem:
    """Steady-state ODE system and boundary conditions."""

    fields: BVPFields
    params: ModelParameters

    def ode(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the first-order ODE system for the stationary model."""

        temperature = _as_float_array(y[0])
        temperature_x = _as_float_array(y[1])
        solar_irradiance = _as_float_array(self.fields.solar_irradiance(x))
        b_parameter = _as_float_array(self.fields.b_parameter(x))
        height_offset = _as_float_array(self.fields.surface_height_offset(x))
        k1 = _as_float_array(self.fields.sensible_heat_flux_coefficient(x))
        k2 = _as_float_array(self.fields.latent_heat_flux_coefficient(x))
        k1_x = _as_float_array(self.fields.sensible_heat_flux_derivative(x))
        k2_x = _as_float_array(self.fields.latent_heat_flux_derivative(x))

        albedo = _as_float_array(surface_albedo(temperature, b_parameter, height_offset, self.params))
        g = self.params.c4 / temperature**2 * np.exp(-self.params.c5 / temperature)
        diffusivity = k1 + k2 * g
        radiative_transport = _as_float_array(
            net_radiative_energy_transport(temperature, solar_irradiance, albedo, self.params)
        )

        saturated_geometry = -np.pi / 2.0 * np.minimum(np.tan(np.pi * _as_float_array(x) / 2.0), 1.0e10)
        x_gradient_term = (k1_x + k2_x * g) / diffusivity
        temperature_gradient_term = (
            k2 * g / temperature**2 * (self.params.c5 - temperature / 2.0) / diffusivity
        )

        return np.vstack(
            (
                temperature_x,
                -(np.pi / 2.0) ** 2 * radiative_transport / diffusivity
                - (saturated_geometry + x_gradient_term) * temperature_x
                - temperature_gradient_term * temperature_x**2,
            )
        )

    @staticmethod
    def boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Return zero-slope boundary conditions at the two poles."""

        return np.asarray((ya[1], yb[1]), dtype=float)


@dataclass(frozen=True)
class BVPSolution:
    """Result of the steady-state BVP solve."""

    x: np.ndarray
    temperature: np.ndarray
    temperature_x: np.ndarray
    solver_result: Any


def build_bvp_grid(settings: RunSettings | None = None) -> np.ndarray:
    """Build the default latitude mesh for the BVP solve."""

    return build_ivp_grid(settings)


def build_bvp_fields(
    *,
    settings: RunSettings | None = None,
) -> BVPFields:
    """Build continuous empirical fields and their x-derivatives."""

    run_settings = settings or default_run_settings()
    interpolants = build_empirical_interpolants(
        remove_negative_k2=run_settings.remove_negative_k2,
        constrain_boundary_slopes=run_settings.constrain_empirical_function_slopes,
    )
    return BVPFields(
        solar_irradiance=interpolants.solar_irradiance,
        b_parameter=interpolants.b_parameter,
        surface_height_offset=interpolants.surface_height_offset,
        sensible_heat_flux_coefficient=interpolants.sensible_heat_flux_coefficient,
        latent_heat_flux_coefficient=interpolants.latent_heat_flux_coefficient,
        sensible_heat_flux_derivative=interpolants.sensible_heat_flux_coefficient.derivative(),
        latent_heat_flux_derivative=interpolants.latent_heat_flux_coefficient.derivative(),
    )


def build_bvp_problem(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
) -> BVPProblem:
    """Build the steady-state BVP problem."""

    return BVPProblem(
        fields=build_bvp_fields(settings=settings),
        params=params or default_model_parameters(),
    )


def _build_guess_from_profile(
    x_grid: np.ndarray,
    profile_x: np.ndarray,
    profile_temperature: np.ndarray,
) -> BVPInitialGuess:
    temperature_spline = _zero_slope_spline(profile_x, profile_temperature)
    derivative_spline = temperature_spline.derivative()
    return BVPInitialGuess(
        x=x_grid,
        values=np.vstack(
            (
                _as_float_array(temperature_spline(x_grid)),
                _as_float_array(derivative_spline(x_grid)),
            )
        ),
    )


def build_bvp_initial_guess(
    x_grid: np.ndarray | None = None,
    *,
    kind: InitialGuessKind | None = None,
    scalar_value: float | None = None,
    custom_temperature: np.ndarray | None = None,
    custom_x: np.ndarray | None = None,
    ivp_solution: IVPSolution | None = None,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    ivp_solver_method: str = "BDF",
) -> BVPInitialGuess:
    """Build the initial guess for the BVP solve.

    Supported modes are:

    - ``"scalar"``: constant temperature with zero slope
    - ``"ivp"``: final IVP state plus a cosine perturbation
    - ``"custom"``: user-supplied profile, differentiated analytically
    """

    run_settings = settings or default_run_settings()
    x = build_bvp_grid(run_settings) if x_grid is None else _as_float_array(x_grid)

    if kind is None:
        kind = "scalar" if run_settings.bvp_initial_temperature != 0.0 else "ivp"

    if kind == "scalar":
        value = run_settings.bvp_initial_temperature if scalar_value is None else float(scalar_value)
        return BVPInitialGuess(
            x=x,
            values=np.vstack((np.full_like(x, value, dtype=float), np.zeros_like(x))),
        )

    if kind == "custom":
        if custom_temperature is None:
            raise ValueError("custom_temperature is required when kind='custom'.")
        if custom_x is None:
            raise ValueError("custom_x is required when kind='custom'.")
        profile_x = _as_float_array(custom_x)
        profile_temperature = _as_float_array(custom_temperature)
        if profile_x.shape != profile_temperature.shape:
            raise ValueError("custom_x and custom_temperature must have the same shape.")
        return _build_guess_from_profile(x, profile_x, profile_temperature)

    if kind != "ivp":
        raise ValueError("kind must be 'scalar', 'ivp', or 'custom'.")

    if ivp_solution is None:
        ivp_solution = solve_temperature_ivp(
            params=params,
            settings=run_settings,
            x_grid=x,
            initial_condition_kind="scalar",
            initial_scalar_value=run_settings.ivp_initial_temperature,
            method=ivp_solver_method,
        )
    profile_temperature = (
        _as_float_array(ivp_solution.temperature[-1])
        + run_settings.bvp_perturbation_amplitude * np.cos(np.pi * _as_float_array(ivp_solution.x))
    )
    return _build_guess_from_profile(x, _as_float_array(ivp_solution.x), profile_temperature)


def solve_temperature_bvp(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    x_grid: np.ndarray | None = None,
    initial_guess_kind: InitialGuessKind | None = None,
    initial_scalar_value: float | None = None,
    custom_initial_temperature: np.ndarray | None = None,
    custom_initial_x: np.ndarray | None = None,
    ivp_solution: IVPSolution | None = None,
    ivp_solver_method: str = "BDF",
    tolerance: float = 1e-3,
    max_nodes: int = 10000,
) -> BVPSolution:
    """Solve the stationary boundary value problem."""

    run_settings = settings or default_run_settings()
    x = build_bvp_grid(run_settings) if x_grid is None else _as_float_array(x_grid)
    problem = build_bvp_problem(params=params, settings=run_settings)
    initial_guess = build_bvp_initial_guess(
        x,
        kind=initial_guess_kind,
        scalar_value=initial_scalar_value,
        custom_temperature=custom_initial_temperature,
        custom_x=custom_initial_x,
        ivp_solution=ivp_solution,
        params=params,
        settings=run_settings,
        ivp_solver_method=ivp_solver_method,
    )
    result = solve_bvp(
        problem.ode,
        problem.boundary_conditions,
        initial_guess.x,
        initial_guess.values,
        tol=tolerance,
        max_nodes=max_nodes,
    )
    if not result.success:
        raise RuntimeError(f"BVP solve failed: {result.message}")
    return BVPSolution(
        x=np.asarray(result.x, dtype=float),
        temperature=np.asarray(result.y[0], dtype=float),
        temperature_x=np.asarray(result.y[1], dtype=float),
        solver_result=result,
    )
