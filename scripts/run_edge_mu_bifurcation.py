"""Run an edge-state ``mu`` bifurcation sweep and save the results to NetCDF."""

from __future__ import annotations

import numpy as np
import typer

from gsebm import (
    default_run_settings,
    run_edge_mu_bifurcation,
    save_edge_mu_bifurcation_dataset,
)

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    filename: str = typer.Option("edge_mu_bifurcation.nc", help="Output filename under the repository data directory."),
    mu_min: float = typer.Option(0.965, help="Lower bound of the mu sweep."),
    mu_max: float = typer.Option(1.115, help="Upper bound of the mu sweep."),
    mu_count: int = typer.Option(70, help="Number of mu values in the sweep."),
    edge_initial_temperature: float = typer.Option(260.0, help="Uniform edge-state BVP initial guess [K]."),
    bvp_tolerance: float = typer.Option(1e-3, help="Tolerance passed to solve_bvp."),
    bvp_max_nodes: int = typer.Option(50000, help="Maximum mesh nodes allowed for solve_bvp."),
) -> None:
    """Solve the edge branch over a ``mu`` sweep and save it."""

    if mu_count < 1:
        raise typer.BadParameter("mu_count must be at least 1.")
    mu_values = tuple(float(mu_value) for mu_value in np.linspace(mu_min, mu_max, mu_count))
    solutions = run_edge_mu_bifurcation(
        settings=default_run_settings(),
        mu_values=mu_values,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    output_path = save_edge_mu_bifurcation_dataset(
        solutions,
        filename=filename,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    typer.echo(output_path)


if __name__ == "__main__":
    app()
