"""Initial value problem components for the time-dependent model.

The IVP is treated with a method-of-lines discretization on a fixed
latitude grid. The transport term is kept in divergence form so the pole
boundary condition can be imposed as zero meridional heat flux at the outer
faces of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from scipy.integrate import solve_ivp

from gsebm.empirical import IVPEmpiricalFields, prepare_ivp_empirical_fields
from gsebm.initial_conditions import build_initial_temperature
from gsebm.parameters import ModelParameters, RunSettings, default_model_parameters, default_run_settings
from gsebm.physics import (
    latitude_weight,
    net_radiative_energy_transport,
    surface_albedo,
    total_diffusivity,
)

ArrayLike: TypeAlias = float | np.ndarray


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class TridiagonalOperator:
    """Tridiagonal linear operator represented by its three diagonals."""

    lower: np.ndarray
    main: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        if self.main.ndim != 1:
            raise ValueError("main diagonal must be one-dimensional.")
        if self.lower.shape != (self.main.size - 1,):
            raise ValueError("lower diagonal must have length main.size - 1.")
        if self.upper.shape != (self.main.size - 1,):
            raise ValueError("upper diagonal must have length main.size - 1.")

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the tridiagonal operator to a state vector."""

        vector = _as_float_array(values)
        if vector.shape != self.main.shape:
            raise ValueError("values must match the operator dimension.")
        result = self.main * vector
        result[:-1] += self.upper * vector[1:]
        result[1:] += self.lower * vector[:-1]
        return result

    def to_banded_matrix(
        self,
        *,
        diagonal_shift: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Return the operator in SciPy's tridiagonal banded format."""

        banded = np.zeros((3, self.main.size), dtype=float)
        banded[0, 1:] = scale * self.upper
        banded[1, :] = diagonal_shift + scale * self.main
        banded[2, :-1] = scale * self.lower
        return banded


@dataclass(frozen=True)
class IVPOperator:
    """Semi-discrete IVP operator on a fixed latitude grid.

    The x-dependent empirical fields are assumed to have been precomputed on
    the solver grid. Transport is discretized through face fluxes, with zero
    flux imposed at the two outer faces.
    """

    x: np.ndarray
    empirical_fields: IVPEmpiricalFields
    params: ModelParameters

    def __post_init__(self) -> None:
        if self.x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if self.x.size < 2:
            raise ValueError("x must contain at least two points.")
        if not np.all(np.diff(self.x) > 0.0):
            raise ValueError("x must be strictly increasing.")
        if self.empirical_fields.x.shape != self.x.shape:
            raise ValueError("empirical field grid must match x.")

    @property
    def geometric_weight(self) -> np.ndarray:
        return _as_float_array(latitude_weight(self.x))

    @property
    def thermal_capacity(self) -> np.ndarray:
        return self.empirical_fields.heat_capacity * self.geometric_weight

    @property
    def face_positions(self) -> np.ndarray:
        interior = 0.5 * (self.x[:-1] + self.x[1:])
        return np.concatenate(([self.x[0]], interior, [self.x[-1]]))

    @property
    def control_widths(self) -> np.ndarray:
        return np.diff(self.face_positions)

    @property
    def face_geometric_weight(self) -> np.ndarray:
        return _as_float_array(latitude_weight(self.face_positions))

    def local_albedo(self, temperature: np.ndarray) -> np.ndarray:
        return _as_float_array(
            surface_albedo(
                temperature,
                self.empirical_fields.b_parameter,
                self.empirical_fields.surface_height_offset,
                self.params,
            )
        )

    def local_diffusivity(self, temperature: np.ndarray) -> np.ndarray:
        return _as_float_array(
            total_diffusivity(
                temperature,
                self.empirical_fields.sensible_heat_flux_coefficient,
                self.empirical_fields.latent_heat_flux_coefficient,
                self.params,
            )
        )

    def source_term(self, temperature: np.ndarray) -> np.ndarray:
        """Return the radiative source term multiplied by the geometric factor."""

        albedo = self.local_albedo(temperature)
        net_transport = _as_float_array(
            net_radiative_energy_transport(
                temperature,
                self.empirical_fields.solar_irradiance,
                albedo,
                self.params,
            )
        )
        return net_transport * self.geometric_weight

    def face_flux(self, temperature: np.ndarray) -> np.ndarray:
        """Return meridional heat flux on cell faces.

        The outer face fluxes are set to zero to impose the no-flux pole
        boundary condition. Interior face diffusivities are computed by
        arithmetic averaging of neighboring nodal values.
        """

        diffusivity = self.local_diffusivity(temperature)
        gradients = np.diff(temperature) / np.diff(self.x)
        face_diffusivity = 0.5 * (diffusivity[:-1] + diffusivity[1:])
        flux = np.zeros(self.x.size + 1, dtype=float)
        flux[1:-1] = (2.0 / np.pi) ** 2 * self.face_geometric_weight[1:-1] * face_diffusivity * gradients
        return flux

    def transport_term(self, temperature: np.ndarray) -> np.ndarray:
        """Return the discrete divergence of meridional heat flux."""

        flux = self.face_flux(temperature)
        return np.diff(flux) / self.control_widths

    def frozen_diffusion_operator(self, temperature: np.ndarray) -> TridiagonalOperator:
        """Return the frozen-diffusion tendency operator.

        The diffusivity is frozen at ``temperature`` and the returned linear
        operator acts on a state vector in temperature units, producing the
        diffusion contribution to ``dT/dt``. It uses the same zero-flux pole
        treatment as ``transport_term`` and includes division by the local
        thermal capacity.
        """

        diffusivity = self.local_diffusivity(temperature)
        segment_widths = np.diff(self.x)
        face_diffusivity = 0.5 * (diffusivity[:-1] + diffusivity[1:])
        face_coefficients = (
            (2.0 / np.pi) ** 2
            * self.face_geometric_weight[1:-1]
            * face_diffusivity
            / segment_widths
        )

        lower = np.zeros(self.x.size - 1, dtype=float)
        main = np.zeros(self.x.size, dtype=float)
        upper = np.zeros(self.x.size - 1, dtype=float)
        tendency_scale = 1.0 / (self.control_widths * self.thermal_capacity)

        main[0] = -face_coefficients[0] * tendency_scale[0]
        upper[0] = face_coefficients[0] * tendency_scale[0]

        for index in range(1, self.x.size - 1):
            left_face = face_coefficients[index - 1]
            right_face = face_coefficients[index]
            lower[index - 1] = left_face * tendency_scale[index]
            main[index] = -(left_face + right_face) * tendency_scale[index]
            upper[index] = right_face * tendency_scale[index]

        lower[-1] = face_coefficients[-1] * tendency_scale[-1]
        main[-1] = -face_coefficients[-1] * tendency_scale[-1]

        return TridiagonalOperator(lower=lower, main=main, upper=upper)

    def diffusion_tendency(
        self,
        frozen_temperature: np.ndarray,
        state_temperature: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the frozen-diffusion tendency applied to a state vector."""

        operator = self.frozen_diffusion_operator(frozen_temperature)
        state = frozen_temperature if state_temperature is None else _as_float_array(state_temperature)
        return operator.apply(state)

    def reaction_tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the radiative contribution to ``dT/dt``."""

        return self.source_term(temperature) / self.thermal_capacity

    def rhs(self, _time: float, temperature: np.ndarray) -> np.ndarray:
        """Return the semi-discrete temperature tendency."""

        return self.diffusion_tendency(temperature) + self.reaction_tendency(temperature)


@dataclass(frozen=True)
class IVPSolution:
    """Result of the IVP time integration."""

    x: np.ndarray
    t: np.ndarray
    temperature: np.ndarray
    solver_result: Any


def build_ivp_grid(settings: RunSettings | None = None) -> np.ndarray:
    """Build the fixed IVP latitude grid with near-pole workaround points."""

    run_settings = settings or default_run_settings()
    delta = run_settings.delta
    interior = np.linspace(-1.0 + 2.0 * delta, 1.0 - 2.0 * delta, run_settings.interior_grid_count)
    left = np.array([-1.0 + delta, -1.0 + 1.01 * delta], dtype=float)
    right = np.array([1.0 - 1.01 * delta, 1.0 - delta], dtype=float)
    return np.concatenate((left, interior, right))


def build_ivp_time_grid(settings: RunSettings | None = None) -> np.ndarray:
    """Build the output time grid for the IVP integration."""

    run_settings = settings or default_run_settings()
    return np.linspace(0.0, run_settings.final_time, run_settings.time_output_count)


def build_ivp_operator(
    x_grid: np.ndarray | None = None,
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
) -> IVPOperator:
    """Build the semi-discrete IVP operator with precomputed empirical fields."""

    run_settings = settings or default_run_settings()
    x = build_ivp_grid(run_settings) if x_grid is None else _as_float_array(x_grid)
    fields = prepare_ivp_empirical_fields(
        x,
        remove_negative_k2=run_settings.remove_negative_k2,
        constrain_boundary_slopes=run_settings.constrain_empirical_function_slopes,
    )
    return IVPOperator(
        x=x,
        empirical_fields=fields,
        params=params or default_model_parameters(),
    )


def solve_temperature_ivp(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    x_grid: np.ndarray | None = None,
    initial_condition_kind: str | None = None,
    initial_scalar_value: float | None = None,
    custom_initial_temperature: np.ndarray | None = None,
    initial_x: np.ndarray | None = None,
    constrain_initial_profile_slopes: bool = False,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> IVPSolution:
    """Solve the time-dependent IVP on a fixed latitude grid."""

    run_settings = settings or default_run_settings()
    operator = build_ivp_operator(x_grid=x_grid, params=params, settings=run_settings)
    t_eval = build_ivp_time_grid(run_settings)
    y0 = build_initial_temperature(
        operator.x,
        kind=initial_condition_kind,
        scalar_value=initial_scalar_value,
        custom_temperature=custom_initial_temperature,
        initial_x=initial_x,
        constrain_boundary_slopes=constrain_initial_profile_slopes,
        settings=run_settings,
    )
    result = solve_ivp(
        operator.rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"IVP integration failed: {result.message}")
    return IVPSolution(
        x=operator.x,
        t=np.asarray(result.t, dtype=float),
        temperature=np.asarray(result.y.T, dtype=float),
        solver_result=result,
    )
