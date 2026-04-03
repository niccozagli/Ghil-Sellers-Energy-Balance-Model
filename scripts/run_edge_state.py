"""Run the edge-state BVP solve for one parameter setup."""

from __future__ import annotations

import typer

from gsebm import RunSettings, run_edge_state, save_edge_state_dataset

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    filename: str = typer.Option("edge_state.nc", help="Output filename under the repository data directory."),
    final_time: float = typer.Option(1.0e9, help="Final integration time parameter stored in the run settings [s]."),
    time_output_count: int = typer.Option(101, help="Number of IVP output times stored in the run settings."),
    edge_initial_temperature: float = typer.Option(260.0, help="Uniform edge-state BVP initial guess [K]."),
    bvp_tolerance: float = typer.Option(1e-3, help="Tolerance passed to solve_bvp."),
    bvp_max_nodes: int = typer.Option(10000, help="Maximum mesh nodes allowed for solve_bvp."),
) -> None:
    """Solve the edge-state branch and save it to NetCDF."""

    settings = RunSettings(final_time=final_time, time_output_count=time_output_count)
    solution = run_edge_state(
        settings=settings,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    output_path = save_edge_state_dataset(
        solution,
        filename=filename,
        edge_initial_temperature=edge_initial_temperature,
        bvp_tolerance=bvp_tolerance,
        bvp_max_nodes=bvp_max_nodes,
    )
    typer.echo(output_path)


if __name__ == "__main__":
    app()
