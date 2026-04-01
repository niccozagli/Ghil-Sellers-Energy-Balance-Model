"""Python port of the Ghil-Sellers Energy Balance Model."""

from gsebm.parameters import (
    ModelParameters,
    RunSettings,
    default_model_parameters,
    default_run_settings,
)
from gsebm.empirical import (
    EmpiricalData,
    EmpiricalInterpolants,
    HalfEmpiricalData,
    IVPEmpiricalFields,
    build_empirical_interpolants,
    default_empirical_data,
    latitude_grid_x1,
    latitude_grid_x2,
    prepare_ivp_empirical_fields,
    raw_empirical_data,
    sample_empirical_fields,
)
from gsebm.physics import (
    absorbed_shortwave_radiation,
    clamp_albedo,
    latitude_weight,
    moisture_factor,
    net_radiative_energy_transport,
    outgoing_longwave_radiation,
    surface_albedo,
    total_diffusivity,
)
from gsebm.initial_conditions import (
    DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE,
    DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE_X,
    build_initial_temperature,
    default_initial_profile_data,
)
from gsebm.ivp import (
    IVPOperator,
    IVPSolution,
    build_ivp_grid,
    build_ivp_operator,
    build_ivp_time_grid,
    solve_temperature_ivp,
)

__all__ = [
    "__version__",
    "ModelParameters",
    "RunSettings",
    "EmpiricalData",
    "EmpiricalInterpolants",
    "HalfEmpiricalData",
    "IVPEmpiricalFields",
    "build_empirical_interpolants",
    "default_model_parameters",
    "default_run_settings",
    "default_empirical_data",
    "latitude_grid_x1",
    "latitude_grid_x2",
    "prepare_ivp_empirical_fields",
    "raw_empirical_data",
    "sample_empirical_fields",
    "absorbed_shortwave_radiation",
    "clamp_albedo",
    "latitude_weight",
    "moisture_factor",
    "net_radiative_energy_transport",
    "outgoing_longwave_radiation",
    "surface_albedo",
    "total_diffusivity",
    "DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE",
    "DEFAULT_OBSERVATIONAL_INITIAL_TEMPERATURE_X",
    "default_initial_profile_data",
    "IVPOperator",
    "IVPSolution",
    "build_initial_temperature",
    "build_ivp_grid",
    "build_ivp_operator",
    "build_ivp_time_grid",
    "solve_temperature_ivp",
]

__version__ = "0.1.0"
