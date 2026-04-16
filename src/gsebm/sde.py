"""Stochastic time integration for the temperature equation.

The stochastic solver reuses the deterministic semi-discrete IVP operator
and advances the temperature with a fixed-step IMEX scheme: diffusion is
treated implicitly with the diffusivity frozen at the previous state, while
the radiative reaction and additive noise are treated explicitly. The noise
is white in time and spatially correlated through Gaussian kernels centered
on a coarse latitude grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from gsebm.initial_conditions import build_initial_temperature
from gsebm.ivp import build_ivp_operator
from gsebm.parameters import (
    ModelParameters,
    RunSettings,
    StochasticRunSettings,
    default_model_parameters,
    default_run_settings,
    default_stochastic_run_settings,
)


def _as_float_array(values: float | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _latitude_degrees_from_x(x: np.ndarray) -> np.ndarray:
    return 90.0 * _as_float_array(x)


def build_noise_latitude_grid(step_degrees: float = 5.0) -> np.ndarray:
    """Return the coarse latitude grid used to define the noise field."""

    if step_degrees <= 0.0:
        raise ValueError("step_degrees must be positive.")
    interval_count = round(180.0 / step_degrees)
    if not np.isclose(interval_count * step_degrees, 180.0):
        raise ValueError("step_degrees must divide 180 degrees.")
    return np.linspace(-90.0, 90.0, int(interval_count) + 1, dtype=float)


@dataclass(frozen=True)
class SpatialNoiseProcess:
    """Spatially correlated additive noise on the IVP latitude grid."""

    x: np.ndarray
    coarse_latitude_degrees: np.ndarray
    length_scale_degrees: float
    normalized_basis: np.ndarray

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one spatial noise realization with unit pointwise variance."""

        coefficients = rng.standard_normal(self.coarse_latitude_degrees.size)
        return self.normalized_basis @ coefficients


@dataclass(frozen=True)
class SDESolution:
    """Result of the stochastic temperature integration."""

    x: np.ndarray
    t: np.ndarray
    temperature: np.ndarray
    step_count: int


def build_spatial_noise_process(
    x_grid: np.ndarray,
    *,
    coarse_step_degrees: float = 5.0,
    length_scale_degrees: float = 5.0,
) -> SpatialNoiseProcess:
    """Build the normalized spatial noise process on the solver grid.

    The Gaussian basis is normalized row by row so the resulting field has
    unit variance at each interior latitude when the coarse coefficients are
    iid standard Gaussians. The first and last latitude rows are forced to
    zero so the additive forcing vanishes at the two pole points; the
    interior rows are then re-normalized after this boundary tapering.
    """

    x = _as_float_array(x_grid)
    latitude = _latitude_degrees_from_x(x)
    coarse_latitude = build_noise_latitude_grid(coarse_step_degrees)
    distances = latitude[:, np.newaxis] - coarse_latitude[np.newaxis, :]
    basis = np.exp(-0.5 * (distances / length_scale_degrees) ** 2)
    basis[0, :] = 0.0
    basis[-1, :] = 0.0
    row_norm = np.sqrt(np.sum(basis**2, axis=1))
    normalized_basis = np.zeros_like(basis)
    nonzero_rows = row_norm > 0.0
    normalized_basis[nonzero_rows] = basis[nonzero_rows] / row_norm[nonzero_rows, np.newaxis]
    return SpatialNoiseProcess(
        x=x,
        coarse_latitude_degrees=coarse_latitude,
        length_scale_degrees=float(length_scale_degrees),
        normalized_basis=normalized_basis,
    )


def solve_temperature_sde(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    stochastic_settings: StochasticRunSettings | None = None,
    x_grid: np.ndarray | None = None,
    initial_condition_kind: str | None = None,
    initial_scalar_value: float | None = None,
    custom_initial_temperature: np.ndarray | None = None,
    initial_x: np.ndarray | None = None,
    constrain_initial_profile_slopes: bool = False,
    noise_process: SpatialNoiseProcess | None = None,
) -> SDESolution:
    """Solve the stochastic temperature equation with a semi-implicit IMEX step."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()
    stochastic_run_settings = stochastic_settings or default_stochastic_run_settings()

    if run_settings.final_time < 0.0:
        raise ValueError("final_time must be nonnegative.")

    operator = build_ivp_operator(x_grid=x_grid, params=model_parameters, settings=run_settings)
    y = build_initial_temperature(
        operator.x,
        kind=initial_condition_kind,
        scalar_value=initial_scalar_value,
        custom_temperature=custom_initial_temperature,
        initial_x=initial_x,
        constrain_boundary_slopes=constrain_initial_profile_slopes,
        settings=run_settings,
    )
    process = noise_process or build_spatial_noise_process(
        operator.x,
        coarse_step_degrees=stochastic_run_settings.noise_grid_step_degrees,
        length_scale_degrees=stochastic_run_settings.noise_length_scale_degrees,
    )
    rng = np.random.default_rng(stochastic_run_settings.noise_seed)

    current_time = 0.0
    step_count = 0
    saved_times = [0.0]
    saved_temperatures = [np.asarray(y, dtype=float).copy()]

    while current_time < run_settings.final_time:
        dt = min(stochastic_run_settings.dt, run_settings.final_time - current_time)
        reaction = operator.reaction_tendency(y)
        y_star = y + reaction * dt
        if stochastic_run_settings.noise_amplitude != 0.0:
            y_star = y_star + (
                stochastic_run_settings.noise_amplitude
                * process.sample(rng)
                * np.sqrt(dt)
            )
        diffusion_operator = operator.frozen_diffusion_operator(y)
        y = solve_banded(
            (1, 1),
            diffusion_operator.to_banded_matrix(diagonal_shift=1.0, scale=-dt),
            y_star,
            check_finite=False,
        )
        current_time += dt
        step_count += 1

        if not np.all(np.isfinite(y)):
            raise RuntimeError(
                "Stochastic integration produced non-finite temperatures. "
                "Try reducing dt or the noise amplitude."
            )

        if (
            step_count % stochastic_run_settings.save_every == 0
            or current_time >= run_settings.final_time
        ):
            saved_times.append(current_time)
            saved_temperatures.append(np.asarray(y, dtype=float).copy())

    return SDESolution(
        x=operator.x,
        t=np.asarray(saved_times, dtype=float),
        temperature=np.asarray(saved_temperatures, dtype=float),
        step_count=step_count,
    )
