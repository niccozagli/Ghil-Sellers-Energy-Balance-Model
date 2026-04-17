"""Run one stochastic temperature trajectory and save it to NetCDF."""

from __future__ import annotations

from dataclasses import replace

import typer

from gsebm import (
    DAY,
    YEAR,
    RunSettings,
    StochasticRunSettings,
    default_model_parameters,
    run_stochastic_state,
    save_stochastic_state_dataset,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help=(
        "Run one stochastic temperature trajectory and save it to NetCDF. "
        "CLI units: final_time in years, initial_temperature in kelvin, "
        "dt in days, noise_amplitude in K s^-1/2, and spatial noise scales in degrees."
    ),
)


@app.command()
def main(
    filename: str = typer.Option("stochastic_state.nc", help="Output filename for the NetCDF dataset."),
    output_dir: str | None = typer.Option(None, help="Directory in which to write the NetCDF file; defaults to the repository data directory."),
    mu: float = typer.Option(1.05, help="Relative solar strength parameter."),
    final_time: float = typer.Option(50.0, help="Final stochastic integration time in years."),
    initial_temperature: float = typer.Option(290.0, help="Uniform stochastic initial condition in kelvin."),
    dt: float = typer.Option(1.0, help="Fixed IMEX timestep in days."),
    noise_amplitude: float = typer.Option(1e-3, help="Additive noise amplitude in K s^-1/2."),
    noise_grid_step_degrees: float = typer.Option(3.0, help="Coarse latitude spacing of the sampled noise in degrees."),
    noise_length_scale_degrees: float = typer.Option(3.0, help="Gaussian kernel width for spatial smoothing in degrees."),
    noise_seed: int | None = typer.Option(None, help="Random seed for the stochastic forcing; omit for a fresh realization."),
    save_every: int = typer.Option(30, help="Save every N stochastic timesteps of length dt."),
) -> None:
    """Solve one stochastic trajectory and save it to NetCDF.

    CLI units: years for ``final_time``, kelvin for ``initial_temperature``,
    days for ``dt``, ``K s^-1/2`` for ``noise_amplitude``, and degrees for
    the spatial noise scales.
    """

    params = replace(default_model_parameters(), mu=mu)
    settings = RunSettings(final_time=final_time * YEAR)
    stochastic_settings = StochasticRunSettings(
        dt=dt * DAY,
        noise_amplitude=noise_amplitude,
        noise_grid_step_degrees=noise_grid_step_degrees,
        noise_length_scale_degrees=noise_length_scale_degrees,
        noise_seed=noise_seed,
        save_every=save_every,
    )
    solution = run_stochastic_state(
        params=params,
        settings=settings,
        stochastic_settings=stochastic_settings,
        initial_temperature=initial_temperature,
    )
    output_path = save_stochastic_state_dataset(
        solution,
        filename=filename,
        initial_temperature=initial_temperature,
        output_dir=output_dir,
    )
    typer.echo(output_path)


if __name__ == "__main__":
    app()
