"""Compute PlaSim-LSG indicators on the cluster, and check which runs need it.

The quantities are defined in Plasim_Indicators.md. For each run the product
is three files in <output root>/<experiment>/:

    <experiment>_<phase>_indicators_zonal.nc      zonal means and scalars
    <experiment>_<phase>_indicators_sections.nc   ocean sections (rows x levels)
    <experiment>_<phase>_indicators_manifest.json what was used, checks, provenance

Two subcommands, run from the repository clone:

    python scripts/slurm/compute_plasim_indicators.py status \
        --experiments-root <experiments> [--experiments-root <more>] --output-root <root>

lists every experiment with output and says whether its product is
"up to date", "missing" or has "new decades" (--stale prints only those
that are not up to date, as "<experiment dir> <phase>" lines).

    python scripts/slurm/compute_plasim_indicators.py run \
        --output-root <root> [--phase spinup] <experiments>/CONTROL_..._MU_1245

recomputes each given run in full from its complete decades. The output
root defaults to $PLASIM_INDICATOR_ROOT. Nothing is written inside the
experiment directories.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from gsebm.plasim_indicators import run_indicators, run_status, write_indicator_files


def phases_with_output(experiment: Path) -> list[str]:
    """Phases (subfolders of output/) that hold at least one file."""
    output = experiment / "output"
    if not output.is_dir():
        return []
    return sorted(path.name for path in output.iterdir() if path.is_dir() and any(path.iterdir()))


def status_command(args: argparse.Namespace) -> int:
    rows = []
    for root in args.experiments_root:
        for experiment in sorted(path for path in root.iterdir() if path.is_dir()):
            for phase in phases_with_output(experiment):
                status = run_status(experiment, phase, args.output_root / experiment.name)
                rows.append((experiment, phase, status))
    for experiment, phase, status in rows:
        if args.stale:
            if status["status"] != "up to date":
                print(f"{experiment} {phase}")
            continue
        detail = ""
        if status["status"] == "new decades":
            detail = f"  ({status['new_decades']} after {status['product_last_label']})"
        elif status["status"] == "up to date":
            detail = f"  (to {status['product_last_label']})"
        print(f"{experiment.name:45s} {phase:11s} {status['status']}{detail}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    def report(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    command = " ".join(sys.argv)
    failed = 0
    for experiment in args.experiments:
        beklas = args.beklas or experiment / "run" / "beklas"
        try:
            zonal, sections, record = run_indicators(experiment, args.phase, beklas, report)
            written = write_indicator_files(zonal, sections, record, args.output_root / experiment.name, command)
        except Exception as error:
            print(f"== {experiment.name}: FAILED {type(error).__name__}: {error}", flush=True)
            failed += 1
            continue
        summary = {"experiment": experiment.name, "phase": args.phase,
                   "first_label": record["decades_used"][0][0], "last_label": record["decades_used"][-1][1],
                   "decades": len(record["decades_used"]), "left_out": record["left_out"],
                   "checks": record["checks"], "files": [str(path) for path in written]}
        print(json.dumps(summary), flush=True)
    return 1 if failed else 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=os.environ.get("PLASIM_INDICATOR_ROOT"),
                        help="Directory holding one product folder per experiment "
                             "(default $PLASIM_INDICATOR_ROOT).")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", parents=[common],
                                 help="Say which runs have no product or new decades.")
    status.add_argument("--experiments-root", type=Path, action="append", required=True,
                        help="Directory of experiment folders; may be given more than once.")
    status.add_argument("--stale", action="store_true",
                        help="Print only runs that are not up to date, as '<experiment dir> <phase>'.")

    run = commands.add_parser("run", parents=[common], help="Recompute the product of the given runs in full.")
    run.add_argument("experiments", nargs="+", type=Path, help="Experiment directories.")
    run.add_argument("--phase", default="spinup", help="Subfolder of output/ (stationary for the base run).")
    run.add_argument("--beklas", type=Path, default=None,
                     help="LSG basin file (default <experiment>/run/beklas).")

    args = parser.parse_args()
    if args.output_root is None:
        parser.error("give --output-root or set PLASIM_INDICATOR_ROOT")
    return status_command(args) if args.command == "status" else run_command(args)


if __name__ == "__main__":
    sys.exit(main())
