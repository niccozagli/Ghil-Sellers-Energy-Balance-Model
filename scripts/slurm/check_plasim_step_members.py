"""Check the members of one PlaSim-LSG step ensemble against its member list and parent.

Reads the newest inventory of every member (under --inventory-root) and of the
parent (under --parent-root), both written by inventory_plasim_runs.py, and
prints one line per member:

    index  experiment  role  delta  start label  start model year  years run  last label  problems

then the namelist entries that differ from the parent's run/ (only
planet_namelist:GSOL0 should) and the executables found. The member list
gives `# parent:` and `# base_mu:` in its header, or one parent per member in
a fifth column (see gsebm.plasim_step_members). Nothing is written.

    PYTHONPATH=src python scripts/slurm/check_plasim_step_members.py \
        --inventory-root <member inventories> --parent-root <parent inventories> step_members_1235.txt
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

from gsebm.plasim_run_inventory import latest_inventory
from gsebm.plasim_step_members import member_problems, namelist_differences, parent_restart_year, read_member_list


def role(gsol0: float, base_mu: float) -> str:
    """plus, minus or control, from the sign of GSOL0 - base."""
    if gsol0 > base_mu:
        return "plus"
    if gsol0 < base_mu:
        return "minus"
    return "control"


def years_run(inventory: dict, start_label: int) -> tuple[int, int | None]:
    """Complete years from the start without a gap, and the last such label."""
    complete = inventory["summary"]["complete_label_years"]
    if not complete or complete[0][0] != start_label + 1:
        return 0, None
    return complete[0][1] - start_label, complete[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("member_list", type=Path)
    parser.add_argument("--inventory-root", type=Path, required=True, help="Member inventories, one folder per member.")
    parser.add_argument("--parent-root", type=Path, required=True, help="Parent inventories, one folder per run.")
    parser.add_argument("--phase", default="spinup")
    args = parser.parse_args()

    header, members = read_member_list(args.member_list)
    base_mu = float(header["base_mu"])
    print(f"== {args.member_list.name}: parent {header['parent']}, base GSOL0 {base_mu}, {len(members)} members")

    failed = 0
    differences: Counter = Counter()
    executables: Counter = Counter()
    parents: dict = {}
    for member in members:
        parent = member.get("parent", header["parent"])
        if parent not in parents:
            found = latest_inventory(args.parent_root, parent, args.phase)
            if found is None:
                print(f"no inventory of the parent {parent} under {args.parent_root}")
                return 1
            print(f"parent {parent}: inventory {found[0].name}")
            parents[parent] = found[1]
        parent_inventory = parents[parent]
        name = member["experiment"]
        start = member["start_label"]
        line = (f"{member['index']:4d} {name} {role(member['gsol0'], base_mu):7s} "
                f"{abs(member['gsol0'] - base_mu):5.2f} {start:6d} {str(parent_restart_year(parent_inventory, start)):>6s}")
        found = latest_inventory(args.inventory_root, name, args.phase)
        if found is None:
            print(f"{line}  NO INVENTORY")
            failed += 1
            continue
        inventory = found[1]
        problems = member_problems(inventory, member, parent_inventory)
        years, last = years_run(inventory, start)
        print(f"{line} {years:5d} {str(last):>6s}  " + ("ok" if not problems else "; ".join(problems)))
        failed += bool(problems)
        differences.update(namelist_differences(inventory, parent_inventory))
        executables.update(row["sha256"][:12] for row in inventory["run_state"]["executables"])

    print(f"members with problems: {failed} of {len(members)}")
    print("namelist entries differing from the parent (members):",
          dict(differences) if differences else "none")
    print("executables (sha256[:12], members):", dict(executables))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
