"""Read-only inventory of PlaSim-LSG experiment directories on the cluster.

For each experiment it records namelists, initial conditions, every yearly
output block (time length, internal model years), every restart bundle (model
dates of REST and LSGREST), the active state in run/ and the Slurm logs, and
compares the file-name labels with the model's own years. It writes
<root>/<experiment>/<experiment>_inventory_<UTC stamp>.json and prints a
summary. Nothing is written inside the experiment directories.

Run from the repository clone on a login node:

    .venv/bin/python scripts/slurm/inventory_plasim_runs.py \
        --inventory-root <root> <experiments>/CONTROL_360ppm_T21L10_10000Y_MU_1245

The root defaults to $PLASIM_INVENTORY_ROOT.
"""
import argparse
import os
from pathlib import Path
import sys

from gsebm.plasim_run_inventory import format_summary, inventory_run, write_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="+", type=Path, help="Experiment directories.")
    parser.add_argument("--inventory-root", type=Path, default=os.environ.get("PLASIM_INVENTORY_ROOT"),
                        help="Directory holding one subfolder per experiment (default $PLASIM_INVENTORY_ROOT).")
    parser.add_argument("--phase", default="spinup", help="Subfolder of output/, restart/ and diag/.")
    parser.add_argument("--workers", type=int, default=1, help="Processes reading files in parallel.")
    parser.add_argument("--hash-restarts", action="store_true", help="Also record sha256 of every restart file.")
    args = parser.parse_args()
    if args.inventory_root is None:
        parser.error("give --inventory-root or set PLASIM_INVENTORY_ROOT")

    failed = 0
    for experiment in args.experiments:
        try:
            inventory = inventory_run(experiment, args.phase, args.hash_restarts,
                                      progress=lambda message: print(message, file=sys.stderr, flush=True),
                                      workers=args.workers)
        except Exception as error:
            print(f"== {experiment.name}: FAILED {type(error).__name__}: {error}", flush=True)
            failed += 1
            continue
        path = write_inventory(inventory, args.inventory_root)
        print(format_summary(inventory), flush=True)
        print(f"  written {path}\n", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
