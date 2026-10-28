"""Checks of PlaSim-LSG step-response members against their member list and parent.

A step ensemble is described by a plain member list (one line per member:
``idx experiment start_label GSOL0``, as read by ``setup_step_members.sh``)
with two header comments, ``# parent: <experiment>`` and ``# base_mu: <GSOL0>``.
Each member is a run started from the parent's restart bundle labelled
``start_label`` with only ``GSOL0`` changed.

The checks read the JSON inventories written by
``scripts/slurm/inventory_plasim_runs.py`` (``plasim_run_inventory``): one
inventory of the member and one of the parent. Nothing is read from the run
directories here.
"""

from __future__ import annotations

from pathlib import Path


def read_member_list(path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Return the header (``# key: value`` lines) and the members of a member list.

    Each member is ``{"index", "experiment", "start_label", "gsol0"}``. Other
    comment lines are ignored.
    """
    header: dict[str, str] = {}
    members: list[dict[str, object]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, colon, value = line.lstrip("# ").partition(":")
            if colon and " " not in key.strip():
                header[key.strip()] = value.strip()
            continue
        index, experiment, start_label, gsol0 = line.split()
        members.append({"index": int(index), "experiment": experiment,
                        "start_label": int(start_label), "gsol0": float(gsol0)})
    return header, members


def parent_restart_year(parent_inventory: dict[str, object], label: int) -> int | None:
    """Model end of year of the parent's restart bundle with this label, or ``None``.

    This is the model year the member starts from: its first output year
    should be this year + 1.
    """
    for row in parent_inventory["restarts"]:
        if row["label"] == label:
            return row["rest_end_of_year"]
    return None


def first_output_model_year(inventory: dict[str, object], label: int) -> int | None:
    """Model year (from the time coordinate) of the PLA block that starts at ``label``."""
    for row in inventory["outputs"]:
        if row["stream"] == "PLA" and row["label_first"] == label:
            return row.get("internal_first")
    return None


def member_problems(inventory: dict[str, object], member: dict[str, object],
                    parent_inventory: dict[str, object]) -> list[str]:
    """List what is wrong with one member; an empty list means it passed every check."""
    problems = []
    start = member["start_label"]
    complete = inventory["summary"]["complete_label_years"]

    # The member's output must begin right after the start bundle and run without gaps.
    if not complete:
        problems.append("no complete four-stream output")
    else:
        if complete[0][0] != start + 1:
            problems.append(f"first complete label {complete[0][0]}, expected {start + 1}")
        if len(complete) > 1:
            problems.append(f"gaps in complete output: {complete}")

    # The first output year must follow the parent's restart in model years, i.e. the
    # member really started from that bundle's date (labels can carry an offset).
    start_year = parent_restart_year(parent_inventory, start)
    first_year = first_output_model_year(inventory, start + 1)
    first_year_ok = False
    if start_year is None:
        problems.append(f"parent has no restart labelled {start}")
    elif first_year is None:
        problems.append(f"no PLA block labelled from {start + 1}")
    elif first_year != start_year + 1:
        problems.append(f"first model year {first_year}, parent restart ends {start_year}")
    else:
        first_year_ok = True

    # Only GSOL0 differs from the parent; it must be the value in the member list.
    gsol0 = inventory["namelists"].get("planet_namelist", {}).get("values", {}).get("GSOL0")
    if gsol0 is None or float(gsol0) != member["gsol0"]:
        problems.append(f"GSOL0 {gsol0} in planet_namelist, expected {member['gsol0']}")

    # One label offset for the whole member (an offset inherited from the parent is expected).
    if len(inventory["summary"]["output_label_offsets"]) > 1:
        problems.append("label offset changes between output blocks")

    # The inventory's own problems; the label-offset note is expected when the first year matched.
    for problem in inventory["summary"]["problems"]:
        if first_year_ok and problem.startswith("labels differ from model years"):
            continue
        problems.append(problem)
    return problems


def namelist_differences(inventory: dict[str, object], reference: dict[str, object]) -> list[str]:
    """Namelist entries (``file:KEY``) whose values differ between two inventories.

    A namelist file present in only one of them is reported as ``file``.
    """
    differences = []
    for name in sorted(set(inventory["namelists"]) | set(reference["namelists"])):
        if name not in inventory["namelists"] or name not in reference["namelists"]:
            differences.append(name)
            continue
        ours = inventory["namelists"][name]["values"]
        theirs = reference["namelists"][name]["values"]
        for key in sorted(set(ours) | set(theirs)):
            if ours.get(key) != theirs.get(key):
                differences.append(f"{name}:{key}")
    return differences
