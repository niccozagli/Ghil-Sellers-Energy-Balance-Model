"""Reduce single PlaSim-LSG runs to their series and latitude files (Plasim_Indicators.md §6).

Each argument is a product folder (the folder holding a run's
``<run>_<phase>_indicators_zonal.nc`` and ``..._sections.nc``; the phase is
read from the file names). The script writes ``<run>_<phase>_indicators_series.nc``
and ``..._latitude.nc`` into the same folder, replacing older ones. Folders
without a readable product are reported and skipped; the script then exits
with status 1. For experiments of many runs use collect_plasim_experiment.py.
Run it after the indicators, on the machine that holds the products:

    PYTHONPATH=src python scripts/slurm/reduce_plasim_runs.py <product folder> [<product folder> ...]
"""
import argparse
from pathlib import Path
import sys

from gsebm.plasim_indicators import write_run_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+", help="product folders, one per run")
    args = parser.parse_args()

    command = " ".join(sys.argv)
    skipped = 0
    for folder in args.folders:
        try:
            written = write_run_files(folder, command)
        except OSError as error:  # no product, or an unreadable one (e.g. a truncated copy)
            print(f"skipped ({error}): {folder}")
            skipped += 1
            continue
        print("written " + ", ".join(path.name for path in written))
    print(f"{len(args.folders) - skipped} reduced, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
