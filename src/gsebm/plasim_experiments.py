"""Experiment files: selected observables of every member of a PlaSim-LSG experiment (Plasim_Indicators.md §7).

An experiment is a set of runs listed in a member list (``plasim_step_members``):
a step ensemble, or a continuation whose members each start from another run.
On the cluster, ``collect_experiment`` reads every member's indicator product
and keeps only selected observables, aligned by years since the member's
step (year 1 is the first year after its start label):

- the annual series of ``annual_series`` (global, hemispheric and depth-band
  indicators), as (member, year);
- the fields by latitude of ``latitude_fields``, as (member, year,
  latitude): surface air temperature and sea-ice fraction on the T21 rows,
  and ocean temperature in the three depth bands on the LSG rows.

``write_experiment_files`` stores them as ``<name>_series.nc`` and
``<name>_latitude.nc``. Responses and every other analysis are computed from
these files on the laptop. To add an observable, add it here and rerun the
collector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from gsebm.plasim_indicators import annual_series, git_revision, latitude_fields, load_indicators, product_stem

EXPERIMENT_SCHEMA = "plasim-experiment/1"


def member_role(gsol0: float, base_mu: float) -> str:
    """plus, minus or control, from the sign of GSOL0 - base."""
    if gsol0 > base_mu:
        return "plus"
    if gsol0 < base_mu:
        return "minus"
    return "control"


def _stack(rows: list[xr.Dataset], length: int) -> xr.Dataset:
    """Stack per-member datasets of different lengths along ``member``, padding the years with NaN."""
    padded = [row.reindex(year=np.arange(1, length + 1)) for row in rows]
    return xr.concat(padded, dim="member").astype("float32")


def collect_experiment(members: list[dict[str, object]], header: dict[str, str], product_root: Path,
                       phase: str = "spinup", latitude: bool = True) -> tuple[xr.Dataset, xr.Dataset | None, list[str]]:
    """Series and (optionally) latitude fields of every member, aligned by years since its step.

    ``members`` and ``header`` come from ``read_member_list``. A member whose
    product is missing or unreadable is left out and named in the returned
    list; a product whose first label is not the start label + 1 raises.
    """
    base_mu = float(header["base_mu"])
    kept, series_rows, latitude_rows, missing = [], [], [], []
    for member in members:
        folder = Path(product_root) / member["experiment"]
        stem = folder / product_stem(member["experiment"], phase)
        paths = [stem.with_name(f"{stem.name}_{kind}.nc") for kind in ("zonal", "sections")]
        if not all(path.is_file() for path in paths):
            missing.append(member["experiment"])
            continue
        try:
            zonal, sections = (load_indicators(path) for path in paths)
        except OSError:  # an unreadable product, e.g. a truncated copy
            missing.append(member["experiment"])
            continue
        labels = zonal["year"].values
        if labels[0] != member["start_label"] + 1:
            raise ValueError(f"{member['experiment']}: first label {labels[0]}, expected {member['start_label'] + 1}")
        since_step = labels - member["start_label"]
        series_rows.append(annual_series(zonal, sections).drop_vars("model_year").assign_coords(year=since_step))
        if latitude:
            latitude_rows.append(latitude_fields(zonal, sections).drop_vars("model_year").assign_coords(year=since_step))
        kept.append(member)

    length = max(int(row["year"][-1]) for row in series_rows)
    member_coords = {
        "experiment": ("member", [m["experiment"] for m in kept]),
        "start_label": ("member", [m["start_label"] for m in kept]),
        "gsol0": ("member", [m["gsol0"] for m in kept]),
        "role": ("member", [member_role(m["gsol0"], base_mu) for m in kept]),
        "source": ("member", [m.get("parent", header.get("parent", "")) for m in kept]),
        "years_run": ("member", [int(row["year"][-1]) for row in series_rows]),
    }
    series = _stack(series_rows, length).assign_coords(member_coords)
    fields = _stack(latitude_rows, length).assign_coords(member_coords) if latitude else None
    return series, fields, missing


def write_experiment_files(series: xr.Dataset, fields: xr.Dataset | None, output_dir: Path, name: str,
                           attributes: dict[str, object]) -> list[Path]:
    """Write ``<name>_series.nc`` (and ``<name>_latitude.nc``), each to ``*.partial`` first and then renamed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit, modified = git_revision()
    provenance = {"schema": EXPERIMENT_SCHEMA, "name": name, "git_commit": commit, "git_modified": int(modified),
                  "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "definitions": "Plasim_Indicators.md §7", **attributes}
    written = []
    for dataset, suffix in ((series, "series"), (fields, "latitude")):
        if dataset is None:
            continue
        dataset.attrs.update(provenance)
        target = output_dir / f"{name}_{suffix}.nc"
        partial = target.with_name(target.name + ".partial")
        dataset.to_netcdf(partial, engine="h5netcdf",
                          encoding={v: {"zlib": True, "complevel": 4} for v in dataset.data_vars})
        partial.replace(target)
        written.append(target)
    return written
