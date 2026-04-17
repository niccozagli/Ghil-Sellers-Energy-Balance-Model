"""High-level workflows for running model configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import xarray as xr
import numpy as np

from gsebm.bvp import BVPSolution, solve_temperature_bvp
from gsebm.ivp import IVPSolution, solve_temperature_ivp
from gsebm.parameters import (
    ModelParameters,
    RunSettings,
    StochasticRunSettings,
    default_model_parameters,
    default_run_settings,
    default_stochastic_run_settings,
)
from gsebm.paths import get_data_dir
from gsebm.sde import SDESolution, solve_temperature_sde


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


@dataclass(frozen=True)
class WarmColdMuBifurcationSolutions:
    """Warm/cold IVP branches over a sequence of ``mu`` values."""

    base_params: ModelParameters
    settings: RunSettings
    mu_values: tuple[float, ...]
    warm_states: tuple[IVPSolution, ...]
    cold_states: tuple[IVPSolution, ...]


@dataclass(frozen=True)
class EdgeMuBifurcationSolutions:
    """Edge-state BVP branch over a sequence of ``mu`` values."""

    base_params: ModelParameters
    settings: RunSettings
    mu_values: tuple[float, ...]
    edge_states: tuple[BVPSolution, ...]


@dataclass(frozen=True)
class StochasticStateSolution:
    """Stochastic temperature trajectory for one configuration."""

    params: ModelParameters
    settings: RunSettings
    stochastic_settings: StochasticRunSettings
    state: SDESolution


def _flatten_prefixed_attributes(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _solve_temperature_bvp_with_retries(
    *,
    params: ModelParameters,
    settings: RunSettings,
    tolerance: float,
    max_nodes: int,
    context: str,
    **solve_kwargs: object,
) -> BVPSolution:
    """Retry BVP solves with relaxed controls when the mesh node cap is exceeded."""

    attempt_controls = (
        (tolerance, max_nodes),
        (tolerance * 2.0, max_nodes),
        (tolerance * 5.0, max_nodes),
        (tolerance * 10.0, max_nodes * 2),
        (tolerance * 10.0, max_nodes * 4),
    )
    last_error: RuntimeError | None = None

    for attempt_tolerance, attempt_max_nodes in attempt_controls:
        try:
            return solve_temperature_bvp(
                params=params,
                settings=settings,
                tolerance=attempt_tolerance,
                max_nodes=attempt_max_nodes,
                **solve_kwargs,
            )
        except RuntimeError as error:
            last_error = error

    raise RuntimeError(
        f"BVP solve failed for {context} after retries up to tol={attempt_controls[-1][0]:.3g} "
        f"and max_nodes={attempt_controls[-1][1]}: {last_error}"
    ) from last_error


def _continue_edge_state_to_mu(
    *,
    previous_mu: float,
    previous_edge_state: BVPSolution,
    target_mu: float,
    base_params: ModelParameters,
    settings: RunSettings,
    tolerance: float,
    max_nodes: int,
    max_refinements: int,
) -> BVPSolution:
    """Continue the edge branch to ``target_mu``, refining the step on failure."""

    target_params = replace(base_params, mu=target_mu)
    try:
        return _solve_temperature_bvp_with_retries(
            params=target_params,
            settings=settings,
            initial_guess_kind="custom",
            custom_initial_temperature=previous_edge_state.temperature,
            custom_initial_x=previous_edge_state.x,
            tolerance=tolerance,
            max_nodes=max_nodes,
            context=f"mu={target_mu:.6g}",
        )
    except RuntimeError:
        if max_refinements <= 0:
            raise
        midpoint_mu = 0.5 * (previous_mu + target_mu)
        if midpoint_mu == previous_mu or midpoint_mu == target_mu:
            raise
        midpoint_state = _continue_edge_state_to_mu(
            previous_mu=previous_mu,
            previous_edge_state=previous_edge_state,
            target_mu=midpoint_mu,
            base_params=base_params,
            settings=settings,
            tolerance=tolerance,
            max_nodes=max_nodes,
            max_refinements=max_refinements - 1,
        )
        return _continue_edge_state_to_mu(
            previous_mu=midpoint_mu,
            previous_edge_state=midpoint_state,
            target_mu=target_mu,
            base_params=base_params,
            settings=settings,
            tolerance=tolerance,
            max_nodes=max_nodes,
            max_refinements=max_refinements - 1,
        )


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


def run_stochastic_state(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    stochastic_settings: StochasticRunSettings | None = None,
    initial_temperature: float = 280.0,
) -> StochasticStateSolution:
    """Solve one stochastic temperature trajectory from a scalar initial state."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()
    stochastic_run_settings = stochastic_settings or default_stochastic_run_settings()

    state = solve_temperature_sde(
        params=model_parameters,
        settings=run_settings,
        stochastic_settings=stochastic_run_settings,
        initial_condition_kind="scalar",
        initial_scalar_value=initial_temperature,
    )
    return StochasticStateSolution(
        params=model_parameters,
        settings=run_settings,
        stochastic_settings=stochastic_run_settings,
        state=state,
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


def stochastic_state_dataset(
    solution: StochasticStateSolution,
    *,
    initial_temperature: float,
) -> xr.Dataset:
    """Return one stochastic temperature trajectory as an ``xarray.Dataset``."""

    state = solution.state

    attrs: dict[str, object] = {
        "title": "Stochastic temperature trajectory",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "initial_temperature": float(initial_temperature),
    }
    stochastic_attrs = asdict(solution.stochastic_settings)
    if stochastic_attrs["noise_seed"] is None:
        stochastic_attrs["noise_seed"] = "None"
    attrs.update(_flatten_prefixed_attributes("param", asdict(solution.params)))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(solution.settings)))
    attrs.update(_flatten_prefixed_attributes("stochastic", stochastic_attrs))

    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "latitude"),
                state.temperature,
                {"units": "K", "long_name": "stochastic temperature"},
            ),
        },
        coords={
            "time": (
                "time",
                state.t,
                {"units": "s", "long_name": "time"},
            ),
            "latitude": (
                "latitude",
                state.x,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )


def save_stochastic_state_dataset(
    solution: StochasticStateSolution,
    *,
    filename: str = "stochastic_state.nc",
    initial_temperature: float,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the stochastic trajectory dataset to ``data/<filename>``."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset = stochastic_state_dataset(
        solution,
        initial_temperature=initial_temperature,
    )
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path


def run_warm_cold_mu_bifurcation(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    mu_values: tuple[float, ...] | list[float],
    warm_initial_temperature: float = 300.0,
    cold_initial_temperature: float = 250.0,
    ivp_method: str = "BDF",
) -> WarmColdMuBifurcationSolutions:
    """Solve the warm/cold IVP branches over a ``mu`` sweep."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()
    mu_tuple = tuple(float(mu_value) for mu_value in mu_values)
    if not mu_tuple:
        raise ValueError("mu_values must contain at least one value.")

    warm_states: list[IVPSolution] = []
    cold_states: list[IVPSolution] = []

    for mu_value in mu_tuple:
        mu_params = replace(model_parameters, mu=mu_value)
        warm_cold = run_warm_cold_state(
            params=mu_params,
            settings=run_settings,
            warm_initial_temperature=warm_initial_temperature,
            cold_initial_temperature=cold_initial_temperature,
            ivp_method=ivp_method,
        )
        warm_states.append(warm_cold.warm_state)
        cold_states.append(warm_cold.cold_state)

    return WarmColdMuBifurcationSolutions(
        base_params=model_parameters,
        settings=run_settings,
        mu_values=mu_tuple,
        warm_states=tuple(warm_states),
        cold_states=tuple(cold_states),
    )


def warm_cold_mu_bifurcation_dataset(
    solutions: WarmColdMuBifurcationSolutions,
    *,
    warm_initial_temperature: float,
    cold_initial_temperature: float,
    ivp_method: str,
) -> xr.Dataset:
    """Return the warm/cold ``mu`` sweep as an ``xarray.Dataset``."""

    warm_reference = solutions.warm_states[0]
    cold_reference = solutions.cold_states[0]
    for warm_state in solutions.warm_states[1:]:
        if warm_state.x.shape != warm_reference.x.shape or warm_state.t.shape != warm_reference.t.shape:
            raise ValueError("All warm-state IVP solutions must share the same latitude and time grids.")
    for cold_state in solutions.cold_states:
        if cold_state.x.shape != warm_reference.x.shape or cold_state.t.shape != warm_reference.t.shape:
            raise ValueError("All cold-state IVP solutions must share the same latitude and time grids.")

    attrs: dict[str, object] = {
        "title": "Warm/cold IVP branches over a mu bifurcation sweep",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "mu_name": "relative solar strength",
        "warm_initial_temperature": float(warm_initial_temperature),
        "cold_initial_temperature": float(cold_initial_temperature),
        "ivp_method": ivp_method,
    }
    base_param_attrs = asdict(solutions.base_params)
    base_param_attrs.pop("mu", None)
    attrs.update(_flatten_prefixed_attributes("param", base_param_attrs))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(solutions.settings)))

    warm_stack = np.stack([state.temperature for state in solutions.warm_states], axis=0)
    cold_stack = np.stack([state.temperature for state in solutions.cold_states], axis=0)
    mu_array = np.asarray(solutions.mu_values, dtype=float)

    return xr.Dataset(
        data_vars={
            "warm_state_temperature": (
                ("mu", "time", "latitude"),
                warm_stack,
                {"units": "K", "long_name": "warm-state temperature"},
            ),
            "cold_state_temperature": (
                ("mu", "time", "latitude"),
                cold_stack,
                {"units": "K", "long_name": "cold-state temperature"},
            ),
        },
        coords={
            "mu": (
                "mu",
                mu_array,
                {"units": "1", "long_name": "relative solar strength"},
            ),
            "time": (
                "time",
                warm_reference.t,
                {"units": "s", "long_name": "time"},
            ),
            "latitude": (
                "latitude",
                warm_reference.x,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )


def save_warm_cold_mu_bifurcation_dataset(
    solutions: WarmColdMuBifurcationSolutions,
    *,
    filename: str = "warm_cold_mu_bifurcation.nc",
    warm_initial_temperature: float,
    cold_initial_temperature: float,
    ivp_method: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the warm/cold ``mu`` bifurcation dataset to ``data/<filename>``."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset = warm_cold_mu_bifurcation_dataset(
        solutions,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path


def run_edge_mu_bifurcation(
    *,
    params: ModelParameters | None = None,
    settings: RunSettings | None = None,
    mu_values: tuple[float, ...] | list[float],
    edge_initial_temperature: float = 260.0,
    bvp_tolerance: float = 1e-3,
    bvp_max_nodes: int = 20000,
    max_continuation_refinements: int = 8,
) -> EdgeMuBifurcationSolutions:
    """Solve the edge-state BVP branch over a ``mu`` sweep by continuation."""

    model_parameters = params or default_model_parameters()
    run_settings = settings or default_run_settings()
    mu_tuple = tuple(float(mu_value) for mu_value in mu_values)
    if not mu_tuple:
        raise ValueError("mu_values must contain at least one value.")

    edge_states: list[BVPSolution | None] = [None] * len(mu_tuple)
    sorted_indices = sorted(range(len(mu_tuple)), key=lambda index: mu_tuple[index])
    center_sorted_position = min(
        range(len(sorted_indices)),
        key=lambda position: abs(mu_tuple[sorted_indices[position]] - model_parameters.mu),
    )
    center_index = sorted_indices[center_sorted_position]

    center_params = replace(model_parameters, mu=mu_tuple[center_index])
    center_edge = _solve_temperature_bvp_with_retries(
        params=center_params,
        settings=run_settings,
        initial_guess_kind="scalar",
        initial_scalar_value=edge_initial_temperature,
        tolerance=bvp_tolerance,
        max_nodes=bvp_max_nodes,
        context=f"mu={mu_tuple[center_index]:.6g}",
    )
    edge_states[center_index] = center_edge

    previous_edge_state = center_edge
    previous_mu = mu_tuple[center_index]
    for position in range(center_sorted_position + 1, len(sorted_indices)):
        mu_index = sorted_indices[position]
        continued_edge = _continue_edge_state_to_mu(
            previous_mu=previous_mu,
            previous_edge_state=previous_edge_state,
            target_mu=mu_tuple[mu_index],
            base_params=model_parameters,
            settings=run_settings,
            tolerance=bvp_tolerance,
            max_nodes=bvp_max_nodes,
            max_refinements=max_continuation_refinements,
        )
        edge_states[mu_index] = continued_edge
        previous_edge_state = continued_edge
        previous_mu = mu_tuple[mu_index]

    previous_edge_state = center_edge
    previous_mu = mu_tuple[center_index]
    for position in range(center_sorted_position - 1, -1, -1):
        mu_index = sorted_indices[position]
        continued_edge = _continue_edge_state_to_mu(
            previous_mu=previous_mu,
            previous_edge_state=previous_edge_state,
            target_mu=mu_tuple[mu_index],
            base_params=model_parameters,
            settings=run_settings,
            tolerance=bvp_tolerance,
            max_nodes=bvp_max_nodes,
            max_refinements=max_continuation_refinements,
        )
        edge_states[mu_index] = continued_edge
        previous_edge_state = continued_edge
        previous_mu = mu_tuple[mu_index]

    if any(edge_state is None for edge_state in edge_states):
        raise RuntimeError("Edge-state continuation did not populate all mu values.")

    return EdgeMuBifurcationSolutions(
        base_params=model_parameters,
        settings=run_settings,
        mu_values=mu_tuple,
        edge_states=tuple(edge_state for edge_state in edge_states if edge_state is not None),
    )


def edge_mu_bifurcation_dataset(
    solutions: EdgeMuBifurcationSolutions,
    *,
    edge_initial_temperature: float,
    bvp_tolerance: float,
    bvp_max_nodes: int,
) -> xr.Dataset:
    """Return the edge-state ``mu`` sweep as an ``xarray.Dataset``."""

    edge_reference = solutions.edge_states[0]
    for edge_state in solutions.edge_states:
        if edge_state.solver_result.sol is None:
            raise ValueError("Each edge-state BVP solution must provide a dense solution interpolant.")

    attrs: dict[str, object] = {
        "title": "Edge-state BVP branch over a mu bifurcation sweep",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "mu_name": "relative solar strength",
        "edge_initial_temperature": float(edge_initial_temperature),
        "bvp_tolerance": float(bvp_tolerance),
        "bvp_max_nodes": int(bvp_max_nodes),
    }
    base_param_attrs = asdict(solutions.base_params)
    base_param_attrs.pop("mu", None)
    attrs.update(_flatten_prefixed_attributes("param", base_param_attrs))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(solutions.settings)))

    mu_array = np.asarray(solutions.mu_values, dtype=float)
    common_latitude = np.asarray(edge_reference.x, dtype=float)
    edge_temperature_stack = np.stack(
        [state.solver_result.sol(common_latitude)[0] for state in solutions.edge_states],
        axis=0,
    )
    edge_derivative_stack = np.stack(
        [state.solver_result.sol(common_latitude)[1] for state in solutions.edge_states],
        axis=0,
    )

    return xr.Dataset(
        data_vars={
            "edge_state_temperature": (
                ("mu", "latitude"),
                edge_temperature_stack,
                {"units": "K", "long_name": "edge-state temperature"},
            ),
            "edge_state_temperature_derivative": (
                ("mu", "latitude"),
                edge_derivative_stack,
                {"units": "K", "long_name": "temperature derivative with respect to normalized latitude"},
            ),
        },
        coords={
            "mu": (
                "mu",
                mu_array,
                {"units": "1", "long_name": "relative solar strength"},
            ),
            "latitude": (
                "latitude",
                common_latitude,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )


def save_edge_mu_bifurcation_dataset(
    solutions: EdgeMuBifurcationSolutions,
    *,
    filename: str = "edge_mu_bifurcation.nc",
    edge_initial_temperature: float,
    bvp_tolerance: float,
    bvp_max_nodes: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Write the edge-state ``mu`` bifurcation dataset to ``data/<filename>``."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset = edge_mu_bifurcation_dataset(
        solutions,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path
