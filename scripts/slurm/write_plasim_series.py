"""Write the annual series file of PlaSim-LSG indicator products (Plasim_Indicators.md §6).

Each argument is a product folder (the folder holding a run's
``<run>_<phase>_indicators_zonal.nc`` and ``..._sections.nc``). The script
computes the run's annual global, hemispheric and depth-band indicators and
writes ``<run>_<phase>_indicators_series.nc`` into the same folder, replacing
an older one. Folders without both product files, or whose files cannot be
read, are reported and skipped; the script then exits with status 1. Run it
after the indicators, on the machine that holds the products:

    PYTHONPATH=src python scripts/slurm/write_plasim_series.py <product folder> [<product folder> ...]
"""
import argparse
from pathlib import Path
import sys

from gsebm.plasim_indicators import product_stem, write_series_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+", help="product folders, one per run")
    parser.add_argument("--phase", default="spinup")
    args = parser.parse_args()

    command = " ".join(sys.argv)
    skipped = 0
    for folder in args.folders:
        stem = folder / product_stem(folder.name, args.phase)
        if not all(stem.with_name(f"{stem.name}_{kind}.nc").is_file() for kind in ("zonal", "sections")):
            print(f"skipped (no zonal and sections files): {folder}")
            skipped += 1
            continue
        try:
            target = write_series_file(folder, args.phase, command)
        except OSError as error:  # an unreadable product, e.g. a truncated copy
            print(f"skipped (cannot read the product: {error}): {folder}")
            skipped += 1
            continue
        print(f"written {target}")
    print(f"{len(args.folders) - skipped} written, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
