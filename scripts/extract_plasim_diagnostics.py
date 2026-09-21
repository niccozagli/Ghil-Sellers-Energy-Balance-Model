"""Extract annual scalar and zonal diagnostics from one PLASIM experiment.

Usage template
--------------
Process the ``spinup`` output for one experiment:

    PYTHONPATH=src uv run python scripts/extract_plasim_diagnostics.py \\
      --experiment-dir /Volumes/Nicco/Plasim/experiments/CONTROL_360ppm_T21L10_10000Y_MU_1312

The experiment directory must contain ``output/<state>`` atmospheric files
named ``*_PLA.<start>-<end>.nc`` and ``diag/<state>`` LSG files named
``*_DIAG.<start>-<end>.txt``. Use ``--state`` to choose another state,
``--workers`` to set parallelism, and ``--output-dir`` to change the
destination.

By default, the command writes these non-overwriting NetCDF products:

    data/Plasim/<experiment>/<experiment>_<state>_diagnostics.nc
    data/Plasim/<experiment>/<experiment>_<state>_zonal_temperatures.nc

The diagnostics product contains global, regional, hemispheric, radiation,
AMOC, and sea-ice quantities. The zonal product contains surface and 2-metre
temperature, ocean-only sea-ice concentration and thickness, and surface
albedo and net TOA energy imbalance on the ``time x lat`` grid. It also stores
the zonal fraction of ocean longitudes whose local annual sea-ice cover is at
least 0.5.
"""

from __future__ import annotations

from pathlib import Path
import re

import typer

from gsebm.paths import get_data_dir
from gsebm.plasim_diagnostics import (
    build_plasim_diagnostic_datasets,
    find_plasim_input_files,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    experiment_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="PLASIM experiment directory containing output/<state> and diag/<state>.",
    ),
    state: str = typer.Option("spinup", help="Experiment state subdirectory to process."),
    workers: int = typer.Option(4, min=1, help="Number of worker processes."),
    output_dir: Path | None = typer.Option(
        None,
        file_okay=False,
        dir_okay=True,
        help="Output directory; defaults to data/Plasim/<experiment name>.",
    ),
) -> None:
    """Create scalar diagnostics and zonal-field NetCDF datasets."""
    atmospheric_paths, diagnostic_paths = find_plasim_input_files(experiment_dir, state)
    destination = output_dir or get_data_dir() / "Plasim" / experiment_dir.name
    diagnostics_path = destination / f"{experiment_dir.name}_{state}_diagnostics.nc"
    zonal_temperatures_path = (
        destination / f"{experiment_dir.name}_{state}_zonal_temperatures.nc"
    )
    existing_outputs = [
        path for path in (diagnostics_path, zonal_temperatures_path) if path.exists()
    ]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing output: {paths}")

    typer.echo(
        "Processing "
        f"{len(atmospheric_paths)} atmospheric files and "
        f"{len(diagnostic_paths)} LSG diagnostic files with up to {workers} workers."
    )
    products = build_plasim_diagnostic_datasets(
        atmospheric_paths, diagnostic_paths, workers=workers
    )
    metadata = {
        "experiment_name": experiment_dir.name,
        "experiment_state": state,
    }
    mu_match = re.search(r"_MU_(\d+)$", experiment_dir.name)
    if mu_match is not None:
        metadata["mu"] = mu_match.group(1)
    products.diagnostics.attrs.update(metadata)
    products.zonal_temperatures.attrs.update(metadata)
    destination.mkdir(parents=True, exist_ok=True)
    products.diagnostics.to_netcdf(diagnostics_path, engine="scipy")
    products.zonal_temperatures.to_netcdf(zonal_temperatures_path, engine="scipy")
    typer.echo(diagnostics_path)
    typer.echo(zonal_temperatures_path)


if __name__ == "__main__":
    app()
