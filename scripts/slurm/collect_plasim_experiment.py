"""Collect the selected observables of every member of a PlaSim-LSG experiment (Plasim_Indicators.md §7).

Reads the member list (``# parent:`` and ``# base_mu:`` header, one line per
member, an optional fifth column naming each member's source run), finds
each member's indicator product under --products, and writes
``<name>_series.nc`` and ``<name>_latitude.nc`` to --output, with the
members aligned by years since their step. --name defaults to the member
list's file name without its extension; --indices keeps only some members
(e.g. ``2-104:6,1``, as Slurm array indices). Members without a readable
product are reported and left out; the script then exits with status 1.

    PYTHONPATH=src python scripts/slurm/collect_plasim_experiment.py <member list> \
        --products <indicator products root> --output <folder> [--name NAME] [--indices LIST] [--no-latitude]
"""
import argparse
from pathlib import Path
import sys

from gsebm.plasim_experiments import collect_experiment, write_experiment_files
from gsebm.plasim_step_members import read_member_list


def parse_indices(text: str) -> set[int]:
    """Slurm-style index list: ``1-5,8,10-20:2``."""
    indices = set()
    for part in text.split(","):
        span, _, step = part.partition(":")
        first, _, last = span.partition("-")
        indices.update(range(int(first), int(last or first) + 1, int(step or 1)))
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("member_list", type=Path)
    parser.add_argument("--products", type=Path, required=True, help="folder holding one product folder per run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", help="file name stem (default: the member list's name)")
    parser.add_argument("--indices", help="keep only these member-list indices, e.g. 2-104:6")
    parser.add_argument("--phase", default="spinup")
    parser.add_argument("--no-latitude", action="store_true", help="write only the series file")
    args = parser.parse_args()

    header, members = read_member_list(args.member_list)
    if args.indices:
        keep = parse_indices(args.indices)
        members = [m for m in members if m["index"] in keep]
    name = args.name or args.member_list.stem
    series, fields, missing = collect_experiment(members, header, args.products, args.phase,
                                                 latitude=not args.no_latitude)
    attributes = {"member_list": args.member_list.name, "base_mu": header["base_mu"], "phase": args.phase,
                  "indices": args.indices or "all", "command": " ".join(sys.argv)}
    for path in write_experiment_files(series, fields, args.output, name, attributes):
        print(f"written {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{series.sizes['member']} members, up to {series.sizes['year']} years; {len(missing)} without a product")
    for experiment in missing:
        print(f"  no product: {experiment}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
