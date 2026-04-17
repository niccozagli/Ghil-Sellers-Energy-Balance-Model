"""Diagnostics derived from model states, fluxes, and saved datasets."""

from __future__ import annotations

from dataclasses import fields
from typing import TypeAlias

import numpy as np
import xarray as xr

from gsebm.bvp import BVPProblem, build_bvp_problem
from gsebm.ivp import IVPOperator, build_ivp_operator
from gsebm.parameters import ModelParameters, RunSettings
from gsebm.physics import latitude_weight, surface_albedo, total_diffusivity

ArrayLike: TypeAlias = float | np.ndarray

CALORIES_PER_SECOND_PER_WATT = 0.239
SQUARE_CENTIMETERS_PER_SQUARE_METER = 1.0e4
CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER = (
    SQUARE_CENTIMETERS_PER_SQUARE_METER / CALORIES_PER_SECOND_PER_WATT
)


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _float_or_bool_or_int(value: object) -> object:
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _attrs_to_dataclass_kwargs(dataset: xr.Dataset, prefix: str, dataclass_type: type) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for field in fields(dataclass_type):
        key = f"{prefix}_{field.name}"
        if key in dataset.attrs:
            kwargs[field.name] = _float_or_bool_or_int(dataset.attrs[key])
    return kwargs


def model_parameters_from_dataset_attrs(dataset: xr.Dataset) -> ModelParameters:
    """Reconstruct model parameters from dataset attributes."""

    return ModelParameters(**_attrs_to_dataclass_kwargs(dataset, "param", ModelParameters))


def run_settings_from_dataset_attrs(dataset: xr.Dataset) -> RunSettings:
    """Reconstruct run settings from dataset attributes."""

    return RunSettings(**_attrs_to_dataclass_kwargs(dataset, "setting", RunSettings))


def build_ivp_operator_from_dataset(dataset: xr.Dataset) -> IVPOperator:
    """Rebuild the IVP operator on the dataset latitude grid."""

    latitude = _as_float_array(dataset["latitude"].values)
    return build_ivp_operator(
        x_grid=latitude,
        params=model_parameters_from_dataset_attrs(dataset),
        settings=run_settings_from_dataset_attrs(dataset),
    )


def build_bvp_problem_from_dataset(dataset: xr.Dataset) -> BVPProblem:
    """Rebuild the BVP problem from dataset attributes."""

    return build_bvp_problem(
        params=model_parameters_from_dataset_attrs(dataset),
        settings=run_settings_from_dataset_attrs(dataset),
    )


def pde_meridional_flux(
    latitude: ArrayLike,
    temperature: ArrayLike,
    temperature_x: ArrayLike,
    k1_value: ArrayLike,
    k2_value: ArrayLike,
    params: ModelParameters,
) -> np.ndarray:
    """Return the ``f`` term from the IVP PDE in cgs units.

        f = cos(pi x / 2) * k(T, x) * dT/dx * (2 / pi)^2
    """

    diffusivity = _as_float_array(total_diffusivity(temperature, k1_value, k2_value, params))
    return (
        _as_float_array(latitude_weight(latitude))
        * diffusivity
        * _as_float_array(temperature_x)
        * (2.0 / np.pi) ** 2
    )


def meridional_heat_transfer_rate_watts_per_square_meter(
    latitude: ArrayLike,
    temperature: ArrayLike,
    temperature_x: ArrayLike,
    k1_value: ArrayLike,
    k2_value: ArrayLike,
    params: ModelParameters,
) -> np.ndarray:
    """Return meridional heat-transfer rate specified to half the globe.

        j = -f * pi / 2 * 1 / 0.239 * 1e4

    where ``f`` is the PDE flux term in cgs units. The output is in
    ``W m^-2``.
    """

    flux = pde_meridional_flux(latitude, temperature, temperature_x, k1_value, k2_value, params)
    return -flux * (np.pi / 2.0) * CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER


def latitude_weighted_mean(
    data: xr.DataArray | xr.Dataset,
    *,
    x: xr.DataArray | None = None,
    dim: str = "latitude",
    xmin: float = -1.0,
    xmax: float = 1.0,
) -> xr.DataArray | xr.Dataset:
    """Return the latitude-weighted mean over ``xmin <= x <= xmax``.

    The weights are ``cos(pi x / 2)``, consistent with the model's
    normalized latitude coordinate.
    """

    coordinate = data[dim] if x is None else x
    if dim not in coordinate.dims:
        raise ValueError(f"x coordinate must depend on the '{dim}' dimension.")
    if xmin > xmax:
        raise ValueError("xmin must be less than or equal to xmax.")

    region_mask = (coordinate >= xmin) & (coordinate <= xmax)
    regional_data = data.where(region_mask, drop=True)
    regional_coordinate = coordinate.where(region_mask, drop=True)
    weights = xr.apply_ufunc(latitude_weight, regional_coordinate)
    return regional_data.weighted(weights).mean(dim=dim)


def warm_cold_state_albedo_from_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return warm/cold albedo fields derived from a saved IVP dataset."""

    operator = build_ivp_operator_from_dataset(dataset)
    warm_temperature = _as_float_array(dataset["warm_state_temperature"].values)
    cold_temperature = _as_float_array(dataset["cold_state_temperature"].values)

    warm_albedo = surface_albedo(
        warm_temperature,
        operator.empirical_fields.b_parameter[None, :],
        operator.empirical_fields.surface_height_offset[None, :],
        operator.params,
    )
    cold_albedo = surface_albedo(
        cold_temperature,
        operator.empirical_fields.b_parameter[None, :],
        operator.empirical_fields.surface_height_offset[None, :],
        operator.params,
    )

    return xr.Dataset(
        data_vars={
            "warm_state_albedo": (
                ("time", "latitude"),
                warm_albedo,
                {"units": "1", "long_name": "warm-state albedo"},
            ),
            "cold_state_albedo": (
                ("time", "latitude"),
                cold_albedo,
                {"units": "1", "long_name": "cold-state albedo"},
            ),
        },
        coords={"time": dataset["time"], "latitude": dataset["latitude"]},
    )


def edge_state_albedo_from_dataset(dataset: xr.Dataset) -> xr.DataArray:
    """Return edge-state albedo derived from a saved BVP dataset."""

    problem = build_bvp_problem_from_dataset(dataset)
    latitude = _as_float_array(dataset["latitude"].values)
    temperature = _as_float_array(dataset["edge_state_temperature"].values)
    albedo = surface_albedo(
        temperature,
        problem.fields.b_parameter(latitude),
        problem.fields.surface_height_offset(latitude),
        problem.params,
    )
    return xr.DataArray(
        albedo,
        dims=("latitude",),
        coords={"latitude": dataset["latitude"]},
        name="edge_state_albedo",
        attrs={"units": "1", "long_name": "edge-state albedo"},
    )


def warm_cold_state_heat_transfer_from_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return warm/cold meridional heat-transfer rates from a saved IVP dataset."""

    operator = build_ivp_operator_from_dataset(dataset)
    latitude = _as_float_array(dataset["latitude"].values)
    warm_temperature = _as_float_array(dataset["warm_state_temperature"].values)
    cold_temperature = _as_float_array(dataset["cold_state_temperature"].values)
    warm_temperature_x = np.gradient(warm_temperature, latitude, axis=1)
    cold_temperature_x = np.gradient(cold_temperature, latitude, axis=1)

    warm_heat_transfer = meridional_heat_transfer_rate_watts_per_square_meter(
        latitude[None, :],
        warm_temperature,
        warm_temperature_x,
        operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
        operator.empirical_fields.latent_heat_flux_coefficient[None, :],
        operator.params,
    )
    cold_heat_transfer = meridional_heat_transfer_rate_watts_per_square_meter(
        latitude[None, :],
        cold_temperature,
        cold_temperature_x,
        operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
        operator.empirical_fields.latent_heat_flux_coefficient[None, :],
        operator.params,
    )

    return xr.Dataset(
        data_vars={
            "warm_state_heat_transfer": (
                ("time", "latitude"),
                warm_heat_transfer,
                {"units": "W m^-2", "long_name": "warm-state meridional heat-transfer rate"},
            ),
            "cold_state_heat_transfer": (
                ("time", "latitude"),
                cold_heat_transfer,
                {"units": "W m^-2", "long_name": "cold-state meridional heat-transfer rate"},
            ),
        },
        coords={"time": dataset["time"], "latitude": dataset["latitude"]},
    )


def edge_state_heat_transfer_from_dataset(dataset: xr.Dataset) -> xr.DataArray:
    """Return edge-state meridional heat-transfer rate from a saved BVP dataset."""

    problem = build_bvp_problem_from_dataset(dataset)
    latitude = _as_float_array(dataset["latitude"].values)
    temperature = _as_float_array(dataset["edge_state_temperature"].values)
    temperature_x = _as_float_array(dataset["edge_state_temperature_derivative"].values)
    heat_transfer = meridional_heat_transfer_rate_watts_per_square_meter(
        latitude,
        temperature,
        temperature_x,
        problem.fields.sensible_heat_flux_coefficient(latitude),
        problem.fields.latent_heat_flux_coefficient(latitude),
        problem.params,
    )
    return xr.DataArray(
        heat_transfer,
        dims=("latitude",),
        coords={"latitude": dataset["latitude"]},
        name="edge_state_heat_transfer",
        attrs={"units": "W m^-2", "long_name": "edge-state meridional heat-transfer rate"},
    )
