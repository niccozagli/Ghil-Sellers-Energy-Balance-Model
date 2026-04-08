"""Run a warm/cold ``mu`` bifurcation sweep and save the results to NetCDF."""

from __future__ import annotations

import numpy as np
import typer

from gsebm import (
    YEAR,
    RunSettings,
    run_warm_cold_mu_bifurcation,
    save_warm_cold_mu_bifurcation_dataset,
)

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    filename: str = typer.Option("warm_cold_mu_bifurcation.nc", help="Output filename under the repository data directory."),
    mu_min: float = typer.Option(0.90, help="Lower bound of the mu sweep."),
    mu_max: float = typer.Option(1.15, help="Upper bound of the mu sweep."),
    mu_count: int = typer.Option(70, help="Number of mu values in the sweep."),
    final_time: float = typer.Option(350.0 * YEAR, help="Final IVP integration time [s]."),
    time_output_count: int = typer.Option(101, help="Number of saved IVP output times."),
    warm_initial_temperature: float = typer.Option(300.0, help="Uniform warm-state IVP initial condition [K]."),
    cold_initial_temperature: float = typer.Option(250.0, help="Uniform cold-state IVP initial condition [K]."),
    ivp_method: str = typer.Option("BDF", help="SciPy solve_ivp method for the IVP."),
) -> None:
    """Solve the warm/cold branches over a ``mu`` sweep and save them."""

    if mu_count < 1:
        raise typer.BadParameter("mu_count must be at least 1.")
    settings = RunSettings(final_time=final_time, time_output_count=time_output_count)
    mu_values = tuple(float(mu_value) for mu_value in np.linspace(mu_min, mu_max, mu_count))
    solutions = run_warm_cold_mu_bifurcation(
        settings=settings,
        mu_values=mu_values,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    output_path = save_warm_cold_mu_bifurcation_dataset(
        solutions,
        filename=filename,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    typer.echo(output_path)


if __name__ == "__main__":
    app()
