"""Inventory and extract annual PlaSim maps without temporal analysis.

Examples
--------
PYTHONPATH=src uv run python scripts/extract_plasim_raw_maps.py --all --dry-run
PYTHONPATH=src uv run python scripts/extract_plasim_raw_maps.py --experiment-dir /Volumes/Nicco/Plasim/experiments/CONTROL_360ppm_T21L10_10000Y_MU_1240
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time

import h5netcdf
import typer

from gsebm.plasim_raw_maps import (
    SCHEMA_VERSION,
    extract_raw_maps,
    inventory_experiment,
    plan_extraction,
    raw_map_root,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    experiment_dir: Path | None = typer.Option(None, help="One experiment directory."),
    all_runs: bool = typer.Option(False, "--all", help="Inventory every experiment below --experiments-root."),
    mu_min: float | None = typer.Option(None, help="Lowest μ to include with --all."),
    mu_max: float | None = typer.Option(None, help="Highest μ to include with --all."),
    experiments_root: Path = typer.Option(Path("/Volumes/Nicco/Plasim/experiments")),
    output_root: Path | None = typer.Option(None, help="Archive root; defaults to PLASIM_RAW_MAP_ROOT or data/Plasim."),
    state: str = typer.Option("spinup", help="Source state. Only spinup is supported by this archive schema."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print coverage and planned actions without writing outputs."),
    refresh: bool = typer.Option(False, "--refresh", help="Rebuild even when a current archive covers all source years."),
    workers: int = typer.Option(4, min=1, help="Number of parallel per-file readers."),
    report_path: Path | None = typer.Option(
        None, help="Inventory report file; defaults to <archive root>/raw_map_extraction_inventory.json."
    ),
) -> None:
    """Build compressed raw map archives for new or extended spinup records."""
    if state != "spinup":
        raise typer.BadParameter("This archive currently supports --state spinup only.")
    if all_runs == (experiment_dir is not None):
        raise typer.BadParameter("Choose exactly one of --all or --experiment-dir.")
    if not all_runs and (mu_min is not None or mu_max is not None):
        raise typer.BadParameter("--mu-min and --mu-max require --all.")
    if mu_min is not None and mu_max is not None and mu_min > mu_max:
        raise typer.BadParameter("--mu-min must not exceed --mu-max.")

    def mu_value(path: Path) -> float:
        match = re.search(r"_MU_(\d+(?:p\d+)?)", path.name)
        return float(match.group(1).replace("p", ".")) if match else float("inf")

    def priority(path: Path) -> tuple[int, float, str]:
        mu = mu_value(path)
        return (0 if 1230.0 <= mu <= 1245.0 else 1, mu, path.name)

    experiments = (
        sorted(
            (
                path for path in experiments_root.glob("CONTROL_*_MU_*")
                if (mu_min is None or mu_value(path) >= mu_min)
                and (mu_max is None or mu_value(path) <= mu_max)
            ),
            key=priority,
        ) if all_runs else [experiment_dir]
    )
    root = output_root or raw_map_root()
    if not experiments:
        raise typer.BadParameter(f"No experiments found under {experiments_root}")
    if not dry_run and len(root.parts) >= 3 and root.parts[:2] == ("/", "Volumes"):
        volume = Path("/Volumes") / root.parts[2]
        if not os.path.ismount(volume):
            raise typer.BadParameter(f"External archive volume is not mounted: {volume}")
    report: list[dict[str, object]] = []
    for experiment in experiments:
        assert experiment is not None
        entry: dict[str, object] = {"experiment": experiment.name}
        try:
            inventory = inventory_experiment(experiment, state)
            destination = root / inventory.experiment
            archive = destination / f"{inventory.experiment}_{state}_raw_maps.nc"
            mask = destination / f"{inventory.experiment}_{state}_basin_masks.nc"
            plan = plan_extraction(inventory, archive, mask, refresh=refresh)
            entry.update(
                first_year=inventory.first_year,
                last_year=inventory.last_year,
                source_files=len(inventory.lsg_paths),
                extracted_years=plan.existing_years,
                action=plan.action,
                new_years=(
                    inventory.year_count - (plan.existing_years[1] - plan.existing_years[0] + 1)
                    if plan.action in ("append", "resume_build") and plan.existing_years
                    else inventory.year_count if plan.action == "rebuild" else 0
                ),
            )
            if plan.action != "skip" and not dry_run:
                started = time.monotonic()
                paths = extract_raw_maps(
                    inventory, destination, workers=workers, refresh=refresh,
                    progress=lambda completed, total: typer.echo(
                        f"{inventory.experiment}: {completed}/{total} source blocks"
                    ),
                )
                with h5netcdf.File(paths[0], "r") as completed:
                    missing_optional = json.loads(completed.attrs["missing_optional_pla_maps"])
                entry.update(
                    archive=str(paths[0]),
                    mask=str(paths[1]),
                    size_bytes=paths[0].stat().st_size,
                    seconds=round(time.monotonic() - started, 1),
                    schema_version=SCHEMA_VERSION,
                    missing_optional_pla_maps=missing_optional,
                )
        except FileNotFoundError as error:
            entry.update(action="skip_missing_source", reason=str(error))
        except (ValueError, OSError, KeyError) as error:
            entry.update(action="failed", error=str(error))
        report.append(entry)
        typer.echo(json.dumps(entry))
    if not dry_run:
        report_path = report_path or root / "raw_map_extraction_inventory.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        typer.echo(report_path)
    if any(entry["action"] == "failed" for entry in report if entry.get("source_files")):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
