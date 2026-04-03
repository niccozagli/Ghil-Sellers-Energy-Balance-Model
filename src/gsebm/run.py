"""High-level workflows for running model configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import xarray as xr

from gsebm.bvp import BVPSolution, solve_temperature_bvp
from gsebm.ivp import IVPSolution, solve_temperature_ivp
from gsebm.parameters import ModelParameters, RunSettings, default_model_parameters, default_run_settings
from gsebm.paths import get_data_dir


@dataclass(frozen=True)
class WarmColdStateSolutions:
    """Warm-state and cold-state IVP solutions for one configuration."""

    params: ModelParameters
    settings: RunSettings
    warm_state: IVPSolution
    cold_state: IVPSolution


@dataclass(frozen=True)
class EdgeStateSolution:
    """Edge-state BVP solution for one configuration."""

    params: ModelParameters
    settings: RunSettings
    edge_state: BVPSolution


def _flatten_prefixed_attributes(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run_warm_cold_state(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    warm_initial_temperature: float = 300.0,
    cold_initial_temperature: float = 250.0,
    ivp_method: str = "BDF",
) -> WarmColdStateSolutions:
    """Solve the warm-state and cold-state IVP branches."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()

    warm_state = solve_temperature_ivp(
        params=model_parameters,
        settings=run_settings,
        initial_condition_kind="scalar",
        initial_scalar_value=warm_initial_temperature,
        method=ivp_method,
    )
    cold_state = solve_temperature_ivp(
        params=model_parameters,
        settings=run_settings,
        initial_condition_kind="scalar",
        initial_scalar_value=cold_initial_temperature,
        method=ivp_method,
    )

    return WarmColdStateSolutions(
        params=model_parameters,
        settings=run_settings,
        warm_state=warm_state,
        cold_state=cold_state,
    )


def run_edge_state(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    edge_initial_temperature: float = 260.0,
    bvp_tolerance: float = 1e-3,
    bvp_max_nodes: int = 10000,
) -> EdgeStateSolution:
    """Solve the edge-state BVP branch."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()

    edge_state = solve_temperature_bvp(
        params=model_parameters,
        settings=run_settings,
        initial_guess_kind="scalar",
        initial_scalar_value=edge_initial_temperature,
        tolerance=bvp_tolerance,
        max_nodes=bvp_max_nodes,
    )

    return EdgeStateSolution(
        params=model_parameters,
        settings=run_settings,
        edge_state=edge_state,
    )


def warm_cold_state_dataset(
    solutions: WarmColdStateSolutions,
    *,
    warm_initial_temperature: float,
    cold_initial_temperature: float,
    ivp_method: str,
) -> xr.Dataset:
    """Return the warm/cold IVP results as an ``xarray.Dataset``."""

    warm_state = solutions.warm_state
    cold_state = solutions.cold_state

    if warm_state.x.shape != cold_state.x.shape:
        raise ValueError("warm_state and cold_state must share the same latitude grid.")
    if warm_state.t.shape != cold_state.t.shape:
        raise ValueError("warm_state and cold_state must share the same time grid.")

    attrs: dict[str, object] = {
        "title": "Warm-state and cold-state IVP solutions",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "ivp_method": ivp_method,
        "warm_initial_temperature": float(warm_initial_temperature),
        "cold_initial_temperature": float(cold_initial_temperature),
    }
    attrs.update(_flatten_prefixed_attributes("param", asdict(solutions.params)))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(solutions.settings)))

    dataset = xr.Dataset(
        data_vars={
            "warm_state_temperature": (
                ("time", "latitude"),
                warm_state.temperature,
                {"units": "K", "long_name": "warm-state temperature"},
            ),
            "cold_state_temperature": (
                ("time", "latitude"),
                cold_state.temperature,
                {"units": "K", "long_name": "cold-state temperature"},
            ),
        },
        coords={
            "time": (
                "time",
                warm_state.t,
                {"units": "s", "long_name": "time"},
            ),
            "latitude": (
                "latitude",
                warm_state.x,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )
    return dataset


def save_warm_cold_state_dataset(
    solutions: WarmColdStateSolutions,
    *,
    filename: str = "warm_cold_state.nc",
    warm_initial_temperature: float,
    cold_initial_temperature: float,
    ivp_method: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the warm/cold IVP dataset to ``data/<filename>``."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset = warm_cold_state_dataset(
        solutions,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path


def edge_state_dataset(
    solution: EdgeStateSolution,
    *,
    edge_initial_temperature: float,
    bvp_tolerance: float,
    bvp_max_nodes: int,
) -> xr.Dataset:
    """Return the edge-state BVP result as an ``xarray.Dataset``."""

    edge_state = solution.edge_state

    attrs: dict[str, object] = {
        "title": "Edge-state BVP solution",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "edge_initial_temperature": float(edge_initial_temperature),
        "bvp_tolerance": float(bvp_tolerance),
        "bvp_max_nodes": int(bvp_max_nodes),
    }
    attrs.update(_flatten_prefixed_attributes("param", asdict(solution.params)))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(solution.settings)))

    dataset = xr.Dataset(
        data_vars={
            "edge_state_temperature": (
                ("latitude",),
                edge_state.temperature,
                {"units": "K", "long_name": "edge-state temperature"},
            ),
            "edge_state_temperature_derivative": (
                ("latitude",),
                edge_state.temperature_x,
                {"units": "K", "long_name": "temperature derivative with respect to normalized latitude"},
            ),
        },
        coords={
            "latitude": (
                "latitude",
                edge_state.x,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )
    return dataset


def save_edge_state_dataset(
    solution: EdgeStateSolution,
    *,
    filename: str = "edge_state.nc",
    edge_initial_temperature: float,
    bvp_tolerance: float,
    bvp_max_nodes: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the edge-state BVP dataset to ``data/<filename>``."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset = edge_state_dataset(
        solution,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path
