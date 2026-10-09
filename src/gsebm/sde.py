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
class OscillatorForcing:
    """External heating q(t) g(x) driven by a noisy damped oscillator.

    g(x) = exp(-(x - center_x)² / 2 width_x²) on normalized latitude x, and
    q (K s⁻¹) follows dq = p dt, dp = (-ω² q - 2γ p) dt + σ dW with
    ω = 2π / period_seconds, γ = ω / (2 quality) and σ set so that the
    stationary standard deviation of q is `q_std`. The oscillator does not
    see the temperature.
    """

    center_x: float
    width_x: float
    period_seconds: float
    quality: float
    q_std: float

    def __post_init__(self) -> None:
        if self.width_x <= 0.0 or self.period_seconds <= 0.0 or self.quality <= 0.0:
            raise ValueError("width_x, period_seconds and quality must be positive.")
        if self.q_std < 0.0:
            raise ValueError("q_std must be nonnegative.")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period_seconds

    @property
    def damping(self) -> float:
        return self.omega / (2.0 * self.quality)

    @property
    def noise_scale(self) -> float:
        # stationary Var(q) = σ² / (4 γ ω²)
        return self.q_std * np.sqrt(4.0 * self.damping * self.omega**2)

    state_names = ("q", "p")

    def pattern(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((_as_float_array(x) - self.center_x) / self.width_x) ** 2)

    def start(self) -> np.ndarray:
        return np.zeros(2)

    def heating(self, state: np.ndarray) -> float:
        return float(state[0])

    def advance(self, state: np.ndarray, global_mean_temperature: float, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        """Symplectic Euler: update p first, then q with the new p."""
        q, p = state
        p = (
            p
            - (self.omega**2 * q + 2.0 * self.damping * p) * dt
            + self.noise_scale * np.sqrt(dt) * rng.standard_normal()
        )
        return np.array([q + p * dt, p])


@dataclass(frozen=True)
class LimitCycleForcing:
    """External heating q(t) g(x) from a noisy limit cycle that feels the climate.

    q = q_amplitude · a · cos θ, with g(x) as in `OscillatorForcing`:

    * da = (1 - a²) a dt / amplitude_relaxation_seconds + σ_a dW, with σ_a
      set so that a fluctuates by about `amplitude_std` around 1;
    * dθ = ω dt + sqrt(2 phase_diffusion) dW + slips, where
      ω = 2π / period_seconds · exp(frequency_sensitivity · (T_g - reference_temperature))
      and T_g is the instantaneous area-mean temperature (the coupling);
    * slips: θ jumps by ±slip_size (random sign) at Poisson rate slip_rate.
    """

    center_x: float
    width_x: float
    period_seconds: float
    reference_temperature: float
    frequency_sensitivity: float
    amplitude_relaxation_seconds: float
    amplitude_std: float
    phase_diffusion: float
    slip_rate: float
    slip_size: float
    q_amplitude: float

    state_names = ("a", "theta")

    def __post_init__(self) -> None:
        if min(self.width_x, self.period_seconds, self.amplitude_relaxation_seconds) <= 0.0:
            raise ValueError("width_x, period_seconds and amplitude_relaxation_seconds must be positive.")
        if min(self.amplitude_std, self.phase_diffusion, self.slip_rate, self.q_amplitude) < 0.0:
            raise ValueError("noise levels, slip_rate and q_amplitude must be nonnegative.")

    def pattern(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((_as_float_array(x) - self.center_x) / self.width_x) ** 2)

    def frequency(self, global_mean_temperature: float) -> float:
        return 2.0 * np.pi / self.period_seconds * np.exp(
            self.frequency_sensitivity * (global_mean_temperature - self.reference_temperature))

    def start(self) -> np.ndarray:
        return np.array([1.0, 0.0])

    def heating(self, state: np.ndarray) -> float:
        return float(self.q_amplitude * state[0] * np.cos(state[1]))

    def advance(self, state: np.ndarray, global_mean_temperature: float, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        a, theta = state
        relaxation = 1.0 / self.amplitude_relaxation_seconds
        # linearized amplitude relaxation rate is 2/relaxation time: Var(a) = σ² / (4 / τ)
        sigma_a = self.amplitude_std * np.sqrt(4.0 * relaxation)
        noise = rng.standard_normal(2)
        a = a + (1.0 - a * a) * a * relaxation * dt + sigma_a * np.sqrt(dt) * noise[0]
        theta = (
            theta
            + self.frequency(global_mean_temperature) * dt
            + np.sqrt(2.0 * self.phase_diffusion * dt) * noise[1]
        )
        if rng.random() < self.slip_rate * dt:
            theta += self.slip_size * (1.0 if rng.random() < 0.5 else -1.0)
        return np.array([a, theta])


@dataclass(frozen=True)
class SDESolution:
    """Result of the stochastic temperature integration.

    `forcing` holds the saved forcing state (one column per entry of the
    forcing's `state_names`, e.g. q and p in K s⁻¹ for `OscillatorForcing`)
    when a forcing was used, otherwise None.
    """

    x: np.ndarray
    t: np.ndarray
    temperature: np.ndarray
    step_count: int
    forcing: np.ndarray | None = None


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
    oscillator: OscillatorForcing | LimitCycleForcing | None = None,
    oscillator_seed: int | None = None,
) -> SDESolution:
    """Solve the stochastic temperature equation with a semi-implicit IMEX step.

    An optional `oscillator` adds q(t) g(x) to the explicit reaction step and
    is advanced with the area-mean temperature of the previous state;
    its own noise uses a separate generator seeded by `oscillator_seed`, so
    the temperature noise sequence is the same with or without it.
    """

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
    if oscillator is not None:
        oscillator_rng = np.random.default_rng(oscillator_seed)
        forcing_pattern = oscillator.pattern(operator.x)
        area = operator.control_widths / operator.control_widths.sum()
        forcing_state = oscillator.start()
        saved_forcing = [forcing_state.copy()]

    while current_time < run_settings.final_time:
        dt = min(stochastic_run_settings.dt, run_settings.final_time - current_time)
        reaction = operator.reaction_tendency(y)
        y_star = y + reaction * dt
        if oscillator is not None:
            y_star = y_star + oscillator.heating(forcing_state) * forcing_pattern * dt
            forcing_state = oscillator.advance(forcing_state, float(area @ y), dt, oscillator_rng)
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
            if oscillator is not None:
                saved_forcing.append(forcing_state.copy())

    return SDESolution(
        x=operator.x,
        t=np.asarray(saved_times, dtype=float),
        temperature=np.asarray(saved_temperatures, dtype=float),
        step_count=step_count,
        forcing=None if oscillator is None else np.asarray(saved_forcing, dtype=float),
    )
