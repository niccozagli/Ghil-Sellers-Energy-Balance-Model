"""Python port of the Ghil-Sellers Energy Balance Model."""

from gsebm.parameters import (
    ModelParameters,
    RunSettings,
    default_model_parameters,
    default_run_settings,
)
from gsebm.empirical import (
    EmpiricalData,
    HalfEmpiricalData,
    default_empirical_data,
    latitude_grid_x1,
    latitude_grid_x2,
    raw_empirical_data,
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

__all__ = [
    "__version__",
    "ModelParameters",
    "RunSettings",
    "EmpiricalData",
    "HalfEmpiricalData",
    "default_model_parameters",
    "default_run_settings",
    "default_empirical_data",
    "latitude_grid_x1",
    "latitude_grid_x2",
    "raw_empirical_data",
    "absorbed_shortwave_radiation",
    "clamp_albedo",
    "latitude_weight",
    "moisture_factor",
    "net_radiative_energy_transport",
    "outgoing_longwave_radiation",
    "surface_albedo",
    "total_diffusivity",
]

__version__ = "0.1.0"
