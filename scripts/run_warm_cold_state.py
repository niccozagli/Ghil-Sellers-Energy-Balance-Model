"""Run the warm-state and cold-state IVP solves for one parameter setup."""

from __future__ import annotations

import typer

from gsebm import YEAR, RunSettings, run_warm_cold_state, save_warm_cold_state_dataset

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    filename: str = typer.Option("warm_cold_state.nc", help="Output filename under the repository data directory."),
    final_time: float = typer.Option(35.0 * YEAR, help="Final IVP integration time [s]."),
    time_output_count: int = typer.Option(101, help="Number of saved IVP output times."),
    warm_initial_temperature: float = typer.Option(300.0, help="Uniform warm-state IVP initial condition [K]."),
    cold_initial_temperature: float = typer.Option(250.0, help="Uniform cold-state IVP initial condition [K]."),
    ivp_method: str = typer.Option("BDF", help="SciPy solve_ivp method for the IVP."),
) -> None:
    """Solve the warm-state and cold-state branches and save them to NetCDF."""

    settings = RunSettings(final_time=final_time, time_output_count=time_output_count)
    solutions = run_warm_cold_state(
        settings=settings,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    output_path = save_warm_cold_state_dataset(
        solutions,
        filename=filename,
        warm_initial_temperature=warm_initial_temperature,
        cold_initial_temperature=cold_initial_temperature,
        ivp_method=ivp_method,
    )
    typer.echo(output_path)


if __name__ == "__main__":
    app()
