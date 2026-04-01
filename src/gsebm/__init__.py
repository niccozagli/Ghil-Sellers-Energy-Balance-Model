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
]

__version__ = "0.1.0"
