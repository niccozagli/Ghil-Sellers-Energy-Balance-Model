"""Initial-condition data and profile builders."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from gsebm.parameters import RunSettings, default_run_settings

ArrayLike: TypeAlias = float | np.ndarray
InitialConditionKind: TypeAlias = Literal["scalar", "default", "custom"]

DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE: tuple[float, ...] = (
    247.3625,
    252.0740,
    262.5715,
    271.2980,
    278.9325,
    285.7530,
    291.4090,
    296.0815,
    298.7815,
    299.3510,
    298.7815,
    296.0815,
    291.4090,
    285.7530,
    278.9325,
    271.2980,
    262.5715,
    252.0740,
    247.3625,
)
DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE_X: tuple[float, ...] = tuple(
    step / 90.0 for step in range(-90, 91, 10)
)


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def default_initial_profile_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the default observational initial profile."""

    return (
        _as_float_array(DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE_X),
        _as_float_array(DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE),
    )


def build_initial_temperature(
    x_grid: np.ndarray,
    *,
    kind: InitialConditionKind | None = None,
    scalar_value: float | None = None,
    custom_temperature: np.ndarray | None = None,
    initial_x: np.ndarray | None = None,
    constrain_boundary_slopes: bool = False,
    settings: RunSettings | None = None,
) -> np.ndarray:
    """Build the initial temperature profile on the IVP grid.

    Supported modes are:

    - ``"scalar"``: use a uniform temperature profile
    - ``"default"``: use the default observational profile
    - ``"custom"``: interpolate user-supplied tabulated data
    """

    run_settings = settings or default_run_settings()
    if kind is None:
        kind = "scalar"

    if kind == "scalar":
        value = run_settings.ivp_initial_temperature if scalar_value is None else float(scalar_value)
        return np.full_like(x_grid, value, dtype=float)

    if kind == "default":
        x_data, t_data = default_initial_profile_data()

    elif kind == "custom":
        if custom_temperature is None:
            raise ValueError("custom_temperature is required when kind='custom'.")
        if initial_x is None:
            raise ValueError("initial_x is required when kind='custom'.")
        x_data = _as_float_array(initial_x)
        t_data = _as_float_array(custom_temperature)
        if x_data.shape != t_data.shape:
            raise ValueError("initial_x and custom_temperature must have the same shape.")

    else:
        raise ValueError("kind must be 'scalar', 'default', or 'custom'.")

    if constrain_boundary_slopes:
        interpolant = CubicSpline(x_data, t_data, bc_type=((1, 0.0), (1, 0.0)))
    else:
        interpolant = PchipInterpolator(x_data, t_data)
    return np.asarray(interpolant(x_grid), dtype=float)
