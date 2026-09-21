"""Extract annual regional ocean heat diagnostics from PLASIM-LSG output.

Example
-------
PYTHONPATH=src uv run python scripts/extract_plasim_ocean_diagnostics.py \
    --experiment-dir /Volumes/Nicco/Plasim/experiments/CONTROL_360ppm_T21L10_10000Y_MU_1240
"""

from __future__ import annotations

from pathlib import Path
import re

import typer

from gsebm.paths import get_data_dir
from gsebm.plasim_diagnostics import (
    build_lsg_ocean_diagnostics,
    find_lsg_ocean_files,
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
        help="PLASIM experiment directory containing output/<state> LSG files.",
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
    """Create a compact regional, depth-resolved ocean diagnostic dataset."""
    lsg_paths = find_lsg_ocean_files(experiment_dir, state)
    destination = output_dir or get_data_dir() / "Plasim" / experiment_dir.name
    output_path = (
        destination / f"{experiment_dir.name}_{state}_ocean_diagnostics.nc"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    typer.echo(
        f"Processing {len(lsg_paths)} LSG files with up to {workers} workers."
    )
    diagnostics = build_lsg_ocean_diagnostics(lsg_paths, workers=workers)
    metadata = {
        "experiment_name": experiment_dir.name,
        "experiment_state": state,
    }
    mu_match = re.search(r"_MU_(\d+)$", experiment_dir.name)
    if mu_match is not None:
        metadata["mu"] = mu_match.group(1)
    diagnostics.attrs.update(metadata)
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics.to_netcdf(output_path, engine="scipy")
    typer.echo(output_path)


if __name__ == "__main__":
    app()
