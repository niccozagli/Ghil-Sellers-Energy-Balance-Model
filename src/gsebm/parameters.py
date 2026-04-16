"""Model constants and default run settings."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from gsebm.time import DAY, YEAR


@dataclass(frozen=True)
class ModelParameters:
    """Physical parameters for the Ghil-Sellers energy balance model."""

    mu: float = 1.0  # [1]
    albedo_min: float = 0.25  # [1]
    albedo_max: float = 0.6  # [1]
    c1: float = 0.009  # [K^-1]
    c2: float = 0.0065  # [K m^-1]
    c3: float = 1.9e-15  # [K^-6]
    c4: float = 6.105 * 0.75 * exp(19.6)  # [dyn K cm^-2]
    c5: float = 5350.0  # [K]
    sig: float = 1.356e-12  # [cal cm^-2 s^-1 K^-4]
    m1: float = 0.5  # [1]
    um: float = 283.16  # [K]

    def __post_init__(self) -> None:
        if self.albedo_min > self.albedo_max:
            raise ValueError("albedo_min must be less than or equal to albedo_max.")


@dataclass(frozen=True)
class RunSettings:
    """Default numerical setup for model runs."""

    final_time: float = 35.0 * YEAR  # [s]
    ivp_initial_temperature: float = 280.0  # [K]
    bvp_initial_temperature: float = 0.0  # [K]
    remove_negative_k2: bool = True  # [1]
    constrain_empirical_function_slopes: bool = True  # [1]
    delta: float = 1e-3  # [1]
    time_output_count: int = 101  # [count]
    interior_grid_count: int = 201  # [count]
    bvp_perturbation_amplitude: float = 10.0  # [K]

    def __post_init__(self) -> None:
        if self.time_output_count < 2:
            raise ValueError("time_output_count must be at least 2.")
        if self.interior_grid_count < 1:
            raise ValueError("interior_grid_count must be positive.")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")


@dataclass(frozen=True)
class StochasticRunSettings:
    """Numerical controls for stochastic time integration.

    The additive noise enters the temperature equation through per-step
    increments of the form ``noise_amplitude * xi(x) * sqrt(dt)``, where
    ``xi(x)`` has unit pointwise variance on the interior latitude grid.
    """

    dt: float = DAY  # [s]
    noise_amplitude: float = 0.0  # [K s^-1/2]
    noise_grid_step_degrees: float = 5.0  # [deg]
    noise_length_scale_degrees: float = 5.0  # [deg]
    noise_seed: int | None = None  # [1]
    save_every: int = 1  # [step]

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.noise_grid_step_degrees <= 0.0:
            raise ValueError("noise_grid_step_degrees must be positive.")
        if self.noise_length_scale_degrees <= 0.0:
            raise ValueError("noise_length_scale_degrees must be positive.")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1.")


def default_model_parameters() -> ModelParameters:
    """Return the default physical parameter set."""

    return ModelParameters()


def default_run_settings() -> RunSettings:
    """Return the default numerical settings."""

    return RunSettings()


def default_stochastic_run_settings() -> StochasticRunSettings:
    """Return the default stochastic integration settings."""

    return StochasticRunSettings()
