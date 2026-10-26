"""Read-only inventory of one PlaSim-LSG experiment directory.

The inventory lists what a run directory holds and compares three year
records for every file: the label in its name, the atmosphere model date
(restart ``nstep`` or output time) and the LSG model date (restart header
``nddr``). Nothing is written next to the run; ``write_inventory`` saves one
JSON file per call under an inventory root,
one subfolder per experiment.

Conventions read from the PlaSim 1.0 source:

- ``REST`` (``plasim_restart``): first record ``nstep``, the step count since
  year 0; ``calmod.f90:step2cal30`` turns it into a 360-day calendar date. The
  restart written at the end of year ``Y`` holds 1 January of ``Y + 1``.
- ``LSGREST`` (``kleiin1``): first record ``nddr`` (512 int64); ``nddr(12)`` is
  the year and ``nddr(11) = -(100 * month + day)`` (``lsgmod.f90:outback``). The
  restart written at the end of year ``Y`` holds 30 December of ``Y``.
- Yearly output: ``time`` in ``day as %Y%m%d.%f``, so the internal year is
  ``floor(time / 1e4)``.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import struct

from typing import Callable

import h5py
import numpy as np

SCHEMA = "plasim-run-inventory/1"
STREAMS = ("PLA", "OCE", "ICE", "LSG")
RESTART_KINDS = ("REST", "LSGREST", "KLEISWI")
TRANSIENT_FILES = (
    "Abort_Message", "plasim_output", "ocean_output", "ice_output", "lsg_output",
    "plasim_diag", "plasim_status",
)
TIME_UNITS = "day as %Y%m%d.%f"

_OUTPUT = re.compile(r"(.+)_(PLA|OCE|ICE|LSG)\.(\d+)-(\d+)\.nc")
_RESTART = re.compile(r"(.+)_(REST|LSGREST|KLEISWI)\.(\d+)\.12")
_DIAG = re.compile(r"(.+)_DIAG\.(\d+)-(\d+)\.txt")
_NAMELIST_ENTRY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*,?\s*$")
_SLURM_CONTROL = re.compile(r"^(PHASE|FIRST_YEAR|SEGMENT_YEARS|N_SEGMENTS)=(\S+)")


def sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, hash_file: bool = False) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {
        "file": path.name,
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if hash_file:
        record["sha256"] = sha256(path)
    return record


def _fortran_records(path: Path, count: int) -> list[bytes]:
    """Read the first ``count`` little-endian sequential unformatted records."""
    records = []
    with path.open("rb") as file:
        for _ in range(count):
            head = file.read(4)
            if len(head) < 4:
                break
            (size,) = struct.unpack("<i", head)
            data = file.read(size)
            if len(data) != size or file.read(4) != head:
                raise ValueError(f"Truncated Fortran record in {path}")
            records.append(data)
    if len(records) != count:
        raise ValueError(f"Expected {count} Fortran records in {path}, found {len(records)}")
    return records


def read_namelist(path: Path) -> dict[str, str]:
    """Return ``KEY = value`` entries of a PlaSim namelist file as strings."""
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        match = _NAMELIST_ENTRY.match(line)
        if match is not None:
            values[match.group(1).upper()] = match.group(2)
    return values


def steps_per_day(plasim_namelist: dict[str, str]) -> int:
    """Return PlaSim ``ntspd`` from ``MPSTEP`` (minutes) or ``NTSPD``."""
    mpstep = int(float(plasim_namelist.get("MPSTEP", "0")))
    if mpstep > 0:
        if 1440 % mpstep:
            raise ValueError(f"MPSTEP={mpstep} does not divide a day")
        return 1440 // mpstep
    ntspd = int(float(plasim_namelist.get("NTSPD", "0")))
    if ntspd > 0:
        return ntspd
    raise ValueError("plasim_namelist sets neither MPSTEP nor NTSPD")


def atmosphere_restart_date(path: Path, ntspd: int, days_per_year: int = 360) -> dict[str, object]:
    """Read ``nstep`` from a PlaSim restart and return its model date.

    ``end_of_year`` is the year that just finished when the date is
    1 January at step 0, else ``None``.
    """
    name, value = _fortran_records(path, 2)
    if name.decode("ascii", errors="replace").strip() != "nstep":
        raise ValueError(f"First restart variable is not nstep in {path}")
    (nstep,) = struct.unpack("<i" if len(value) == 4 else "<q", value)
    day, step = divmod(nstep, ntspd)
    year, day_of_year = divmod(day, days_per_year)
    month, day_of_month = divmod(day_of_year, days_per_year // 12)
    month, day_of_month = month + 1, day_of_month + 1
    return {
        "nstep": nstep, "year": year, "month": month, "day": day_of_month, "step_of_day": step,
        "end_of_year": year - 1 if (month, day_of_month, step) == (1, 1, 0) else None,
    }


def lsg_restart_date(path: Path) -> dict[str, object]:
    """Read the LSG restart header date; ``end_of_year`` is set on 30 December."""
    (header,) = _fortran_records(path, 1)
    if len(header) != 512 * 8:
        raise ValueError(f"Unexpected LSG header length {len(header)} in {path}")
    nddr = struct.unpack("<512q", header)
    month, day = divmod(-nddr[10], 100)
    year = int(nddr[11])
    return {"year": year, "month": month, "day": day,
            "end_of_year": year if (month, day) == (12, 30) else None}


def read_kleiswi(path: Path) -> list[int] | str:
    """Return the two LSG restart switches (``idin``, ``inorm``) or the raw text."""
    text = path.read_text(errors="replace").strip()
    try:
        return [int(value) for value in text.split()]
    except ValueError:
        return text


def output_block(path: Path, first: int, last: int) -> dict[str, object]:
    """Describe one yearly-mean output block and its internal model years."""
    record = _file_record(path) | {"label_first": first, "label_last": last}
    try:
        # h5py reads only the time variable; h5netcdf would load all metadata.
        with h5py.File(path, "r") as file:
            time = file["time"]
            units = time.attrs.get("units", b"")
            units = units.decode() if isinstance(units, bytes) else str(units)
            values = np.asarray(time[()], dtype="float64")
    except Exception as error:
        return record | {"status": "unreadable", "error": f"{type(error).__name__}: {error}"}
    record["ntime"] = int(values.size)
    if units != TIME_UNITS:
        return record | {"status": "bad_time", "error": f"time units {units!r}"}
    if values.size == 0 or not np.isfinite(values).all():
        return record | {"status": "bad_time", "error": "empty or non-finite time"}
    internal = np.floor(values / 1e4).astype(int)
    record |= {"time_first": float(values[0]), "time_last": float(values[-1]),
               "internal_first": int(internal[0]), "internal_last": int(internal[-1])}
    if not np.array_equal(np.diff(internal), np.ones(internal.size - 1, dtype=int)):
        return record | {"status": "bad_time", "error": "internal years are not consecutive"}
    record["label_minus_internal"] = first - int(internal[0])
    record["status"] = "ok" if values.size == last - first + 1 else "short"
    return record


def _ranges(years: list[int]) -> list[list[int]]:
    """Collapse sorted integers into inclusive ``[first, last]`` ranges."""
    ranges: list[list[int]] = []
    for year in years:
        if ranges and year == ranges[-1][1] + 1:
            ranges[-1][1] = year
        else:
            ranges.append([year, year])
    return ranges


def _runs(rows: list[tuple[int, int, object]], step: int = 1) -> list[dict[str, object]]:
    """Group ``(first, last, value)`` rows with equal value that follow on by ``step``."""
    runs: list[dict[str, object]] = []
    for first, last, value in rows:
        if runs and runs[-1]["value"] == value and first == runs[-1]["label_last"] + step:
            runs[-1]["label_last"] = last
        else:
            runs.append({"label_first": first, "label_last": last, "value": value})
    return runs


def _restart_state(directory: Path, ntspd: int, days_per_year: int, hash_files: bool) -> dict[str, object]:
    """Read the atmosphere/LSG restart pair and switches found in one directory."""
    names = {"REST": "plasim_restart", "LSGREST": "kleiin1", "KLEISWI": "kleiswi"}
    state: dict[str, object] = {}
    for kind, name in names.items():
        path = directory / name
        if path.is_file():
            state[kind] = _describe_restart(kind, path, ntspd, days_per_year, hash_files)
    return state


def _describe_restart(kind: str, path: Path, ntspd: int, days_per_year: int, hash_file: bool) -> dict[str, object]:
    record = _file_record(path, hash_file)
    try:
        if kind == "REST":
            record["date"] = atmosphere_restart_date(path, ntspd, days_per_year)
        elif kind == "LSGREST":
            record["date"] = lsg_restart_date(path)
        else:
            record["switches"] = read_kleiswi(path)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _parallel(function, tasks: list[tuple], workers: int, report: Callable[[str], None], label: str) -> list:
    """Apply ``function`` to argument tuples in order, optionally in processes."""
    results = []
    if workers <= 1:
        iterator = (function(*task) for task in tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(workers)
        iterator = pool.map(function, *zip(*tasks), chunksize=16) if tasks else iter(())
    try:
        for index, result in enumerate(iterator, 1):
            results.append(result)
            if index % 250 == 0:
                report(f"  {label} {index}/{len(tasks)}")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return results


def _end_of_year(record: dict[str, object] | None) -> int | None:
    if record is None or "date" not in record:
        return None
    return record["date"]["end_of_year"]


def _slurm_logs(directory: Path) -> list[dict[str, object]]:
    """Summarise Slurm logs: first started and last completed segment, errors."""
    logs = []
    for path in sorted(directory.glob("slurm-*.out")) + sorted(directory.glob("slurm-*.err")):
        record = _file_record(path)
        lines = path.read_text(errors="replace").splitlines()
        started = [line for line in lines if line.startswith("Starting PlaSim segment")]
        completed = [line for line in lines if line.startswith("Completed ") and "-" in line.split()[1]]
        errors = [line for line in lines if "ERROR" in line or "error" in line.lower()]
        record |= {"first_started": started[0] if started else None,
                   "last_completed": completed[-1] if completed else None,
                   "error_lines": errors[-5:], "last_line": lines[-1] if lines else None}
        logs.append(record)
    return logs


def inventory_run(
    experiment: Path, phase: str = "spinup", hash_restarts: bool = False,
    progress: Callable[[str], None] | None = None, workers: int = 1,
) -> dict[str, object]:
    """Return the inventory of one experiment directory as a JSON-ready dict.

    ``progress``, if given, receives a short message every few hundred files.
    ``workers > 1`` reads output and restart files in that many processes,
    which helps when each file access on the cluster file system is slow.
    """
    report = progress or (lambda message: None)
    experiment = Path(experiment)
    name = experiment.name
    run = experiment / "run"
    if not run.is_dir():
        raise FileNotFoundError(run)

    namelists = {}
    for path in sorted(run.glob("*_namelist")):
        namelists[path.name] = {"sha256": sha256(path), "values": read_namelist(path)}
    plasim = namelists.get("plasim_namelist", {}).get("values", {})
    ntspd = steps_per_day(plasim)
    days_per_year = int(float(plasim.get("N_DAYS_PER_YEAR", "360")))

    slurm = []
    for path in sorted((experiment / "slurm").glob("*")):
        if path.is_file():
            controls = {m.group(1): m.group(2) for line in path.read_text(errors="replace").splitlines()
                        if (m := _SLURM_CONTROL.match(line))}
            slurm.append(_file_record(path, True) | {"controls": controls})
    segment_years = int(next((s["controls"]["SEGMENT_YEARS"] for s in slurm if "SEGMENT_YEARS" in s["controls"]),
                             plasim.get("N_RUN_YEARS", "10")))

    initial = []
    for path in sorted((experiment / "initial_conditions").glob("*")):
        match = _RESTART.fullmatch(path.name)
        kind = match.group(2) if match else None
        if kind:
            initial.append(_describe_restart(kind, path, ntspd, days_per_year, True)
                           | {"kind": kind, "source_experiment": match.group(1), "label": int(match.group(3))})
        elif path.is_file():
            initial.append(_file_record(path, True) | {"kind": None})

    unrecognised: dict[str, list[str]] = {"output": [], "restart": [], "diag": []}

    output_tasks, output_meta = [], []
    for path in sorted((experiment / "output" / phase).glob("*")):
        match = _OUTPUT.fullmatch(path.name)
        if match is None or match.group(1) != name:
            unrecognised["output"].append(path.name)
            continue
        output_tasks.append((path, int(match.group(3)), int(match.group(4))))
        output_meta.append(match.group(2))
    report(f"{name}: reading {len(output_tasks)} output blocks with {workers} worker(s)")
    blocks = _parallel(output_block, output_tasks, workers, report, "output")
    outputs = [{"stream": stream} | block for stream, block in zip(output_meta, blocks)]
    outputs.sort(key=lambda row: (STREAMS.index(row["stream"]), row["label_first"]))

    restart_tasks, restart_meta = [], []
    for path in sorted((experiment / "restart" / phase).glob("*")):
        match = _RESTART.fullmatch(path.name)
        if match is None or match.group(1) != name:
            unrecognised["restart"].append(path.name)
            continue
        restart_tasks.append((match.group(2), path, ntspd, days_per_year, hash_restarts))
        restart_meta.append((int(match.group(3)), match.group(2)))
    report(f"{name}: reading {len(restart_tasks)} restart files with {workers} worker(s)")
    restarts: dict[int, dict[str, object]] = {}
    for (label, kind), record in zip(restart_meta, _parallel(_describe_restart, restart_tasks, workers, report, "restart")):
        restarts.setdefault(label, {"label": label})[kind] = record
    for row in restarts.values():
        rest, lsg = _end_of_year(row.get("REST")), _end_of_year(row.get("LSGREST"))
        row["rest_end_of_year"], row["lsg_end_of_year"] = rest, lsg
        row["label_minus_rest"] = None if rest is None else row["label"] - rest
        row["label_minus_lsg"] = None if lsg is None else row["label"] - lsg
    restart_rows = [restarts[label] for label in sorted(restarts)]

    diag_blocks = []
    diag_dir = experiment / "diag" / phase
    for path in sorted(diag_dir.glob("*")):
        match = _DIAG.fullmatch(path.name)
        if match is not None and match.group(1) == name:
            diag_blocks.append(_file_record(path) | {"label_first": int(match.group(2)), "label_last": int(match.group(3))})
        elif not path.name.startswith("slurm-"):
            unrecognised["diag"].append(path.name)
    diag_blocks.sort(key=lambda row: row["label_first"])

    run_state = _restart_state(run, ntspd, days_per_year, True)
    run_state["transient_files"] = [f for f in TRANSIENT_FILES if (run / f).exists()]
    executables = sorted(run.glob("*.x"))
    run_state["executables"] = [_file_record(path, True) for path in executables]

    inventory = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "experiment": name,
        "path": str(experiment.resolve()),
        "phase": phase,
        "steps_per_day": ntspd,
        "days_per_year": days_per_year,
        "segment_years": segment_years,
        "namelists": namelists,
        "slurm_scripts": slurm,
        "slurm_logs": _slurm_logs(diag_dir) if diag_dir.is_dir() else [],
        "initial_conditions": initial,
        "outputs": outputs,
        "restarts": restart_rows,
        "diag_blocks": diag_blocks,
        "run_state": run_state,
        "unrecognised_files": unrecognised,
    }
    inventory["summary"] = summarise(inventory)
    return inventory


def summarise(inventory: dict[str, object]) -> dict[str, object]:
    """Derive coverage, gaps, label offsets and problems from an inventory."""
    outputs = inventory["outputs"]
    restarts = inventory["restarts"]
    segment = inventory["segment_years"]
    problems: list[str] = []

    streams = {}
    for stream in STREAMS:
        rows = [row for row in outputs if row["stream"] == stream]
        years = sorted(y for row in rows for y in range(row["label_first"], row["label_last"] + 1))
        overlaps = sorted(y for y, n in Counter(years).items() if n > 1)
        bad = [f"{row['file']}: {row['status']}" for row in rows if row["status"] != "ok"]
        streams[stream] = {"blocks": len(rows), "label_ranges": _ranges(sorted(set(years))),
                           "overlapping_years": _ranges(overlaps), "bad_blocks": bad}
        problems += bad
        if len(streams[stream]["label_ranges"]) > 1:
            problems.append(f"{stream} label gaps: {streams[stream]['label_ranges']}")

    ok_blocks = {}
    for row in outputs:
        if row["status"] == "ok":
            ok_blocks.setdefault((row["label_first"], row["label_last"]), {})[row["stream"]] = row
    complete = sorted(key for key, rows in ok_blocks.items() if len(rows) == len(STREAMS))
    complete_years = _ranges([y for first, last in complete for y in range(first, last + 1)])

    blocks = sorted({(row["label_first"], row["label_last"]) for row in outputs})
    by_block = {key: {row["stream"]: row.get("label_minus_internal") for row in outputs
                      if (row["label_first"], row["label_last"]) == key} for key in blocks}
    output_offsets = _runs([(first, last, {s: by_block[(first, last)].get(s) for s in STREAMS})
                            for first, last in blocks])
    for key, offsets in by_block.items():
        values = {value for value in offsets.values() if value is not None}
        if len(values) > 1:
            problems.append(f"streams disagree on label offset for {key[0]}-{key[1]}: {offsets}")

    restart_labels = [row["label"] for row in restarts]
    restart_gaps = [[a + segment, b - segment] for a, b in zip(restart_labels, restart_labels[1:]) if b - a != segment]
    incomplete = [row["label"] for row in restarts if not all(kind in row for kind in RESTART_KINDS)]
    disagree = [row["label"] for row in restarts if row["rest_end_of_year"] != row["lsg_end_of_year"]]
    restart_offsets = _runs([(row["label"], row["label"], {"rest": row["label_minus_rest"], "lsg": row["label_minus_lsg"]})
                             for row in restarts], step=segment)
    sizes = {kind: {} for kind in RESTART_KINDS}
    for row in restarts:
        for kind in RESTART_KINDS:
            if kind in row:
                size = str(row[kind]["size"])
                sizes[kind][size] = sizes[kind].get(size, 0) + 1
    if restart_gaps:
        problems.append(f"restart label gaps: {restart_gaps}")
    if incomplete:
        problems.append(f"incomplete restart bundles at labels {incomplete}")
    if disagree:
        problems.append(f"REST and LSGREST years disagree at labels {disagree}")
    for kind, counts in sizes.items():
        if len(counts) > 1:
            problems.append(f"{kind} sizes differ: {counts}")
    nonzero = [run for run in output_offsets if any(v not in (0, None) for v in run["value"].values())]
    nonzero += [run for run in restart_offsets if any(v not in (0, None) for v in run["value"].values())]
    if nonzero:
        problems.append(f"labels differ from model years in {len(nonzero)} run(s); see label offsets")

    state = inventory["run_state"]
    rest, lsg = _end_of_year(state.get("REST")), _end_of_year(state.get("LSGREST"))
    if rest != lsg:
        problems.append(f"run/ plasim_restart ends year {rest} but kleiin1 ends year {lsg}")
    if state["transient_files"]:
        problems.append(f"transient files in run/: {state['transient_files']}")
    for kind in ("output", "restart", "diag"):
        if inventory["unrecognised_files"][kind]:
            problems.append(f"{len(inventory['unrecognised_files'][kind])} unrecognised file(s) in {kind}")
    last_restart = restarts[-1] if restarts else None
    if last_restart is not None and rest is not None and last_restart["rest_end_of_year"] != rest:
        problems.append(f"run/ state ends model year {rest}, newest restart bundle ends {last_restart['rest_end_of_year']}")

    return {
        "streams": streams,
        "complete_label_years": complete_years,
        "output_label_offsets": output_offsets,
        "restart_labels": [restart_labels[0], restart_labels[-1]] if restart_labels else None,
        "restart_count": len(restart_labels),
        "restart_label_gaps": restart_gaps,
        "restart_label_offsets": restart_offsets,
        "restart_sizes": sizes,
        "run_state_end_of_year": {"REST": rest, "LSGREST": lsg},
        "problems": problems,
    }


def write_inventory(inventory: dict[str, object], inventory_root: Path) -> Path:
    """Write ``<root>/<experiment>/<experiment>_inventory_<UTC stamp>.json``."""
    output_dir = Path(inventory_root) / inventory["experiment"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(inventory["created"]).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{inventory['experiment']}_inventory_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(inventory, indent=1) + "\n")
    return path


def format_summary(inventory: dict[str, object]) -> str:
    """Return a short human-readable summary of an inventory."""
    summary = inventory["summary"]
    planet = inventory["namelists"].get("planet_namelist", {}).get("values", {})
    radmod = inventory["namelists"].get("radmod_namelist", {}).get("values", {})
    lines = [f"== {inventory['experiment']}",
             f"  GSOL0 {planet.get('GSOL0')}  CO2 {radmod.get('CO2')}  steps/day {inventory['steps_per_day']}"
             f"  days/year {inventory['days_per_year']}  segment {inventory['segment_years']} y"]
    for row in inventory["initial_conditions"]:
        if row.get("kind") in ("REST", "LSGREST"):
            lines.append(f"  initial {row['file']}: model end of year {row.get('date', {}).get('end_of_year')}")
    for stream, info in summary["streams"].items():
        lines.append(f"  {stream}: {info['blocks']} blocks, labels {info['label_ranges']}, {len(info['bad_blocks'])} bad")
    lines.append(f"  complete four-stream label years: {summary['complete_label_years']}")
    lines.append("  output label - internal year:")
    lines += [f"    {r['label_first']}-{r['label_last']}: {r['value']}" for r in summary["output_label_offsets"]]
    lines.append(f"  restarts: {summary['restart_count']} labels {summary['restart_labels']}, gaps {summary['restart_label_gaps']}")
    lines.append("  restart label - model end-of-year:")
    lines += [f"    {r['label_first']}-{r['label_last']}: {r['value']}" for r in summary["restart_label_offsets"]]
    lines.append(f"  run/ state ends model year {summary['run_state_end_of_year']}")
    lines.append(f"  problems ({len(summary['problems'])}):")
    lines += [f"    {problem}" for problem in summary["problems"][:40]]
    if len(summary["problems"]) > 40:
        lines.append(f"    ... {len(summary['problems']) - 40} more in the JSON file")
    return "\n".join(lines)
