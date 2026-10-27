"""Annual PlaSim-LSG indicators: zonal means, ocean sections and a few scalars.

Definitions are in ``Plasim_Indicators.md``; each function names its section.

Cluster side (run by ``scripts/slurm/compute_plasim_indicators.py``):

- ``find_decades`` lists the complete output decades of one run;
- ``run_indicators`` reads them in label order and returns two datasets
  (zonal and sections) plus the record of what was used;
- ``append_indicators`` adds the decades after an existing product's last
  one, with the same result as a full run;
- ``update_indicators`` chooses between the two (``PRODUCT_VERSION``) and
  ``write_indicator_files`` writes the files with a JSON manifest;
- ``run_status`` tells whether a run's product is up to date.

Laptop side: ``load_indicators``, ``area_mean``, ``area_total`` and
``volume_mean`` compute global, hemispheric and band indicators exactly from
the stored zonal means and sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
import xarray as xr

from gsebm.paths import get_repo_root
from gsebm.plasim_grids import (
    LsgGrid,
    T21Grid,
    lsg_grid_from_lsg,
    lsg_interfaces,
    LSG_LAYER_CENTRES_M,
    t21_grid_from_pla,
)
from gsebm.plasim_run_inventory import STREAMS, list_output_files, output_block, read_namelist

SCHEMA = "plasim-indicators/1"
# Raise by hand whenever a formula, mask or output variable changes: a product
# is only extended by appending when its version equals this one.
PRODUCT_VERSION = 1

PLA_FIELDS = ("tas", "ts", "rst", "rlut", "rsut", "rss", "rls", "hfss", "hfls", "sic", "sit")
OCE_FIELDS = ("sst", "fldoa")
ICE_FIELDS = ("cfluxra",)
LSG_SURFACE_FIELDS = ("zeta", "sice", "psi", "flukhea", "fluwat")
LSG_VOLUME_FIELDS = ("t", "s", "vtot")


# ---------------------------------------------------------------------------
# Which decades exist (Plasim_Indicators.md §1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decade:
    """One output block: the four files of a label range."""

    first: int
    last: int
    paths: dict[str, Path]  # stream -> file
    label_minus_model_year: int


def check_decade(first: int, last: int, paths: dict[str, Path]) -> tuple[str | None, int | None]:
    """Return ``(problem, label - model year)`` for one block; ``problem`` is None if complete.

    Complete: all four streams present, each passing ``output_block`` (one
    record per year, consecutive model years), with the same label offset.
    """
    missing = [stream for stream in STREAMS if stream not in paths]
    if missing:
        return f"missing streams {missing}", None
    offsets = set()
    for stream in STREAMS:
        block = output_block(paths[stream], first, last)
        if block["status"] != "ok":
            return f"{stream} {block['status']}", None
        offsets.add(block["label_minus_internal"])
    if len(offsets) != 1:
        return f"streams disagree on label - model year: {sorted(offsets)}", None
    return None, offsets.pop()


def _blocks_on_disk(experiment: Path, phase: str) -> dict[tuple[int, int], dict[str, Path]]:
    """Group the output files of a run by label range."""
    files, _ = list_output_files(experiment, phase)
    blocks: dict[tuple[int, int], dict[str, Path]] = {}
    for stream, path, first, last in files:
        blocks.setdefault((first, last), {})[stream] = path
    return dict(sorted(blocks.items()))


def find_decades(experiment: Path, phase: str) -> tuple[list[Decade], list[dict[str, object]]]:
    """Return the complete decades of a run in label order, and the blocks left out.

    Blocks may be left out only after the last complete one (a decade still
    being written). A gap inside the record, or a change of the label offset,
    raises, because it means damaged or mislabelled output.
    """
    complete: list[Decade] = []
    left_out: list[dict[str, object]] = []
    for (first, last), paths in _blocks_on_disk(Path(experiment), phase).items():
        problem, offset = check_decade(first, last, paths)
        if problem is not None:
            left_out.append({"first": first, "last": last, "reason": problem})
            continue
        if left_out:
            raise ValueError(f"{Path(experiment).name}: block {first}-{last} is complete but "
                             f"an earlier block was left out: {left_out}")
        if complete and first != complete[-1].last + 1:
            raise ValueError(f"{Path(experiment).name}: labels jump from {complete[-1].last} to {first}.")
        if complete and offset != complete[-1].label_minus_model_year:
            raise ValueError(f"{Path(experiment).name}: label - model year changes at {first}.")
        complete.append(Decade(first, last, paths, offset))
    if not complete:
        raise ValueError(f"{Path(experiment).name}: no complete output block in output/{phase}.")
    return complete, left_out


# ---------------------------------------------------------------------------
# Quantities for one decade (Plasim_Indicators.md §3)
# ---------------------------------------------------------------------------

def t21_zonal_mean(field: np.ndarray) -> np.ndarray:
    """Mean over the 64 longitudes of each T21 latitude (§3.1). Input (..., lat, lon)."""
    return np.asarray(field, dtype=float).mean(axis=-1)


def t21_ocean_zonal_mean(field: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Mean over the ocean points of each T21 latitude, NaN where there are none (§3.1).

    All points of a latitude have the same area, so this is a plain mean.
    """
    count = ocean.sum(axis=-1)
    total = np.where(ocean, field, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def lsg_surface_zonal_mean(field: np.ndarray, wet_surface: np.ndarray) -> np.ndarray:
    """Mean over the wet surface points of each LSG row, NaN where there are none (§3.2).

    All points of a row have the same area, so this is a plain mean.
    """
    return t21_ocean_zonal_mean(field, wet_surface)


def lsg_volume_section(field: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume-weighted zonal mean sum_i f V / sum_i V of each (level, row) (§3.5).

    ``field`` is (time, level, row, point); ``volume`` (level, row, point) is 0
    on dry cells and outside the basin. NaN where a (level, row) has no volume.
    """
    total = (np.where(volume > 0, field, 0.0) * volume).sum(axis=-1)
    weight = volume.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / weight, np.nan)


def layer_transport(vtot: np.ndarray, delta: np.ndarray, zonal_step: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Meridional volume transport of each level through each row, m3 s-1 (§3.5).

    T(k, j) = sum_i vtot delta dl(j) over the wet vector points i of row j
    (and of ``mask``, a (row, point) basin mask). Output (time, level, row).
    """
    flux = np.where(delta > 0, vtot, 0.0) * delta * zonal_step[:, None]
    if mask is not None:
        flux = flux * mask
    return flux.sum(axis=-1)


def overturning_streamfunction(transport: np.ndarray) -> np.ndarray:
    """psi(k, j) = -sum_{k' >= k} T(k', j) x 1e-6, in Sv (§3.5, lsgmod.f90 psimer).

    Integrated from the sea floor up; psi(k) is at the top interface of level
    k. The minus sign makes the NADW cell positive.
    """
    from_bottom = np.cumsum(transport[:, ::-1], axis=1)[:, ::-1]
    return -from_bottom * 1e-6


def freshwater_flux_north_of(fluwat: np.ndarray, area: np.ndarray, wet_surface: np.ndarray) -> np.ndarray:
    """Net surface freshwater flux into the ocean over rows 1..j, m3 s-1 (§3.5, §5).

    In a steady state the net southward volume transport across row j equals
    this. Output (time, row).
    """
    per_row = (np.where(wet_surface, fluwat, 0.0) * area).sum(axis=-1)
    return np.cumsum(per_row, axis=-1)


def salt_content(s: np.ndarray, ddz: np.ndarray, zeta: np.ndarray, sice: np.ndarray,
                 area: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Salt content sum s a h over wet cells, in permil m3 (§3.4, LSG's ``Salt inv``).

    h = ddz, except in the top layer: ddz(1) + zeta - sice. Output (time,).
    """
    thickness = np.broadcast_to(ddz, s.shape).copy()
    thickness[:, 0] = np.where(wet[0], ddz[0] + zeta - sice, 0.0)
    return (np.where(wet, s, 0.0) * thickness * area).sum(axis=(1, 2, 3))


def acc_transport(psi: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean barotropic streamfunction over South American land, even and odd rows, Sv (§3.3).

    ``psi`` (time, row, point) is LSG's own, 0 on Antarctica and in Sv. Rows
    are counted 1..76 as in the Fortran code.
    """
    row_number = np.arange(1, psi.shape[1] + 1)[:, None]
    even = land & (row_number % 2 == 0)
    odd = land & (row_number % 2 == 1)
    return psi[:, even].mean(axis=1), psi[:, odd].mean(axis=1)


def read_decade(decade: Decade) -> dict[str, np.ndarray]:
    """Read the fields used here from the four files of one decade, as float64 arrays."""
    wanted = {"PLA": PLA_FIELDS + ("lsm",), "OCE": OCE_FIELDS, "ICE": ICE_FIELDS,
              "LSG": LSG_SURFACE_FIELDS + LSG_VOLUME_FIELDS + ("wet",)}
    fields: dict[str, np.ndarray] = {}
    for stream, names in wanted.items():
        with xr.open_dataset(decade.paths[stream], decode_times=False, engine="h5netcdf") as dataset:
            for name in names:
                if name in fields:
                    raise ValueError(f"Field name {name!r} is read from two streams.")
                variable = dataset[name]
                if "lev" in variable.dims:  # LSG surface fields have a single level
                    variable = variable.isel(lev=0)
                fields[name] = np.asarray(variable.values, dtype=float)
    return fields


def decade_arrays(fields: dict[str, np.ndarray], t21: T21Grid, lsg: LsgGrid) -> dict[str, np.ndarray]:
    """Return every stored quantity of one decade, each with a leading year axis (§3)."""
    if not np.array_equal(fields["lsm"][0] < 0.5, t21.ocean):
        raise ValueError("PLA land-sea mask differs from the first decade.")
    if not np.array_equal(fields["wet"][0] > 0.5, lsg.wet):
        raise ValueError("LSG wet mask differs from the first decade.")

    arrays: dict[str, np.ndarray] = {}
    for name in ("tas", "ts", "rst", "rlut", "rsut"):
        arrays[f"{name}_zonal"] = t21_zonal_mean(fields[name])
    surface_net = fields["rss"] + fields["rls"] + fields["hfss"] + fields["hfls"]
    arrays["surface_net_flux_zonal"] = t21_zonal_mean(surface_net)
    for name in ("sst", "sic", "sit", "fldoa", "cfluxra"):
        arrays[f"{name}_ocean_zonal"] = t21_ocean_zonal_mean(fields[name], t21.ocean)

    wet_surface = lsg.wet[0]
    for name in ("zeta", "flukhea"):
        arrays[f"{name}_zonal"] = lsg_surface_zonal_mean(fields[name], wet_surface)
    arrays["salt_content"] = salt_content(fields["s"], lsg.ddz, fields["zeta"], fields["sice"],
                                          lsg.area, lsg.wet)
    arrays["zeta_abs_max"] = np.where(wet_surface, np.abs(fields["zeta"]), 0.0).max(axis=(1, 2))
    even, odd = acc_transport(fields["psi"], lsg.south_american_land)
    arrays["acc_even_rows"], arrays["acc_odd_rows"], arrays["acc"] = even, odd, 0.5 * (even + odd)

    atlantic_volume = lsg.volume * lsg.atlantic[None]
    arrays["t_section"] = lsg_volume_section(fields["t"], lsg.volume)
    arrays["s_section"] = lsg_volume_section(fields["s"], lsg.volume)
    arrays["s_atlantic_section"] = lsg_volume_section(fields["s"], atlantic_volume)
    transport_global = layer_transport(fields["vtot"], lsg.delta, lsg.zonal_step)
    transport_atlantic = layer_transport(fields["vtot"], lsg.delta, lsg.zonal_step, lsg.atlantic)
    arrays["psi_global"] = overturning_streamfunction(transport_global)
    arrays["psi_atlantic"] = overturning_streamfunction(transport_atlantic)

    # For the check of §5: in a time mean, net transport across a row = - freshwater flux north of it.
    arrays["net_meridional_transport"] = transport_global.sum(axis=1) * 1e-6
    arrays["freshwater_flux_north_of"] = freshwater_flux_north_of(fields["fluwat"], lsg.area, wet_surface) * 1e-6
    return arrays


# ---------------------------------------------------------------------------
# One run (Plasim_Indicators.md §3, §5)
# ---------------------------------------------------------------------------

# name: (dimensions after "year", units, long name)
VARIABLES = {
    "tas_zonal": (("t21_lat",), "K", "zonal mean of tas (lowest level brought to the surface on a dry adiabat)"),
    "ts_zonal": (("t21_lat",), "K", "zonal mean of ts (surface temperature)"),
    "rst_zonal": (("t21_lat",), "W m-2", "zonal mean of net solar flux at the top, positive down"),
    "rlut_zonal": (("t21_lat",), "W m-2", "zonal mean of net thermal flux at the top, negative up"),
    "rsut_zonal": (("t21_lat",), "W m-2", "zonal mean of upward solar flux at the top, negative"),
    "surface_net_flux_zonal": (("t21_lat",), "W m-2", "zonal mean of rss + rls + hfss + hfls, positive down"),
    "sst_ocean_zonal": (("t21_lat",), "K", "ocean zonal mean of slab sst"),
    "sic_ocean_zonal": (("t21_lat",), "1", "ocean zonal mean of sea-ice cover (fraction of year covered)"),
    "sit_ocean_zonal": (("t21_lat",), "m", "ocean zonal mean of sea-ice thickness, 0 when ice-free"),
    "fldoa_ocean_zonal": (("t21_lat",), "W m-2", "ocean zonal mean of heat flux from LSG to the slab"),
    "cfluxra_ocean_zonal": (("t21_lat",), "W m-2", "ocean zonal mean of heat flux holding ice at the 9 m cap"),
    "zeta_zonal": (("lsg_lat",), "m", "wet-point zonal mean of sea level"),
    "flukhea_zonal": (("lsg_lat",), "W m-2", "wet-point zonal mean of coupling heat flux into LSG"),
    "salt_content": ((), "permil m3", "salt content, top layer ddz + zeta - sice"),
    "zeta_abs_max": ((), "m", "largest |zeta| over wet points (LSG clips at 25 m)"),
    "acc_even_rows": ((), "Sv", "barotropic streamfunction on South American land, even rows"),
    "acc_odd_rows": ((), "Sv", "barotropic streamfunction on South American land, odd rows"),
    "acc": ((), "Sv", "Drake Passage transport, mean of even and odd rows"),
    "net_meridional_transport": (("lsg_lat",), "Sv", "net northward volume transport across the row, all vector points"),
    "freshwater_flux_north_of": (("lsg_lat",), "Sv", "net surface freshwater flux into the ocean over rows 1..j"),
    "t_section": (("lsg_level", "lsg_lat"), "K", "volume-weighted zonal mean potential temperature"),
    "s_section": (("lsg_level", "lsg_lat"), "permil", "volume-weighted zonal mean salinity"),
    "s_atlantic_section": (("lsg_level", "lsg_lat"), "permil", "volume-weighted Atlantic zonal mean salinity"),
    "psi_global": (("lsg_level", "lsg_lat"), "Sv", "global overturning streamfunction at the top of each level, NADW positive"),
    "psi_atlantic": (("lsg_level", "lsg_lat"), "Sv", "Atlantic overturning streamfunction at the top of each level, NADW positive"),
}
SECTION_VARIABLES = ("t_section", "s_section", "s_atlantic_section", "psi_global", "psi_atlantic")
FLOAT64_VARIABLES = ("salt_content",)  # its changes are ~1e-7 relative, below float32 resolution


def run_indicators(experiment: Path, phase: str, beklas_path: Path,
                   progress: Callable[[str], None] | None = None,
                   found: tuple[list[Decade], list[dict[str, object]]] | None = None,
                   ) -> tuple[xr.Dataset, xr.Dataset, dict[str, object]]:
    """Compute the zonal and sections datasets of one run from all its complete decades.

    Returns ``(zonal, sections, record)``; ``record`` lists the decades used
    and left out and the run-level checks of §5, for the manifest. ``found``
    is the result of ``find_decades``, if the caller already has it.
    """
    experiment = Path(experiment)
    decades, left_out = found or find_decades(experiment, phase)
    t21, lsg = _grids(decades[0], beklas_path)
    zonal, sections = _compute(decades, t21, lsg, experiment.name, progress)
    record = _record(experiment, phase, decades, left_out, zonal)
    record["mode"], record["computed_from"] = "full", decades[0].first
    return zonal, sections, record


def append_indicators(experiment: Path, phase: str, beklas_path: Path, manifest: dict[str, object],
                      zonal_old: xr.Dataset, sections_old: xr.Dataset,
                      progress: Callable[[str], None] | None = None,
                      found: tuple[list[Decade], list[dict[str, object]]] | None = None,
                      ) -> tuple[xr.Dataset, xr.Dataset, dict[str, object]]:
    """Add the decades after an existing product's last decade (same result as a full run).

    The product's decades must be the first decades on disk, and the static
    fields of the new decades (ocean fraction, wet areas and volumes) must
    equal the stored ones; otherwise this raises and the run needs a full
    recompute. The caller checks ``product_version``.
    """
    experiment = Path(experiment)
    decades, left_out = found or find_decades(experiment, phase)
    if not product_is_first_decades(manifest, decades):
        raise ValueError(f"{experiment.name}: the product's decades are not the first decades on disk.")
    new = decades[len(manifest["decades_used"]):]
    if not new:
        raise ValueError(f"{experiment.name}: no new decades to append.")

    t21, lsg = _grids(new[0], beklas_path)
    zonal_new, sections_new = _compute(new, t21, lsg, experiment.name, progress)
    for old, added, names in ((zonal_old, zonal_new, ZONAL_STATIC), (sections_old, sections_new, SECTIONS_STATIC)):
        for name in names:
            if not np.allclose(old[name].values, added[name].values, rtol=1e-6, equal_nan=True):
                raise ValueError(f"{experiment.name}: static field {name} differs from the stored product.")
    zonal = xr.concat([zonal_old, zonal_new], dim="year", data_vars="minimal", coords="minimal",
                      compat="override", join="exact")
    sections = xr.concat([sections_old, sections_new], dim="year", data_vars="minimal", coords="minimal",
                         compat="override", join="exact")
    record = _record(experiment, phase, decades, left_out, zonal)
    record["mode"], record["computed_from"] = "append", new[0].first
    record["history"] = list(manifest.get("history", []))
    return zonal, sections, record


ZONAL_STATIC = ("t21_gaussian_weight", "t21_band_area", "t21_ocean_fraction", "lsg_row_wet_area")
SECTIONS_STATIC = ("wet_volume", "wet_volume_atlantic")


def product_is_first_decades(manifest: dict[str, object], decades: list[Decade]) -> bool:
    """Whether the manifest's decades are exactly the first decades now on disk."""
    used = [tuple(labels) for labels in manifest["decades_used"]]
    return [(decade.first, decade.last) for decade in decades[:len(used)]] == used


def _grids(decade: Decade, beklas_path: Path) -> tuple[T21Grid, LsgGrid]:
    """Build the T21 and LSG geometry from the files of one decade (§2)."""
    with xr.open_dataset(decade.paths["PLA"], decode_times=False, engine="h5netcdf") as pla:
        t21 = t21_grid_from_pla(pla)
    with xr.open_dataset(decade.paths["LSG"], decode_times=False, engine="h5netcdf") as lsg_file:
        lsg = lsg_grid_from_lsg(lsg_file, beklas_path)
    return t21, lsg


def _compute(decades: list[Decade], t21: T21Grid, lsg: LsgGrid, name: str,
             progress: Callable[[str], None] | None) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute the given decades, in order, into zonal and sections datasets."""
    report = progress or (lambda message: None)
    pieces: dict[str, list[np.ndarray]] = {variable: [] for variable in VARIABLES}
    for number, decade in enumerate(decades, start=1):
        arrays = decade_arrays(read_decade(decade), t21, lsg)
        for variable in VARIABLES:
            pieces[variable].append(arrays[variable])
        if number % 50 == 0 or number == len(decades):
            report(f"{name}: {number}/{len(decades)} decades ({decade.last})")

    years = np.arange(decades[0].first, decades[-1].last + 1)
    values = {variable: np.concatenate(parts).astype("float64" if variable in FLOAT64_VARIABLES else "float32")
              for variable, parts in pieces.items()}
    if values["tas_zonal"].shape[0] != years.size:
        raise ValueError("Number of annual records differs from the label range.")
    return _datasets(values, years, decades[0].label_minus_model_year, t21, lsg)


def _record(experiment: Path, phase: str, decades: list[Decade], left_out: list[dict[str, object]],
            zonal: xr.Dataset) -> dict[str, object]:
    """What a product was built from, and its run-level checks, for the manifest."""
    record = {
        "experiment": experiment.name,
        "phase": phase,
        "decades_used": [[decade.first, decade.last] for decade in decades],
        "left_out": left_out,
        "label_minus_model_year": decades[0].label_minus_model_year,
        "checks": run_checks(zonal),
    }
    gsol0 = _gsol0(experiment)
    if gsol0 is not None:
        record["gsol0"] = gsol0
    return record


def _datasets(values: dict[str, np.ndarray], years: np.ndarray, offset: int,
              t21: T21Grid, lsg: LsgGrid) -> tuple[xr.Dataset, xr.Dataset]:
    """Arrange the stacked arrays into the zonal and sections datasets (§3)."""
    bottom, _ = lsg_interfaces()
    top = np.concatenate(([0.0], bottom[:-1]))
    coords = {
        "year": ("year", years, {"long_name": "label from the output file names"}),
        "model_year": ("year", years - offset, {"long_name": "year in the files' time coordinate"}),
    }
    zonal = xr.Dataset(coords=coords | {"t21_lat": ("t21_lat", t21.lat, {"units": "degrees_north"}),
                                        "lsg_lat": ("lsg_lat", lsg.row_lat, {"units": "degrees_north"})})
    sections = xr.Dataset(coords=coords | {
        "lsg_lat": ("lsg_lat", lsg.row_lat, {"units": "degrees_north"}),
        "lsg_level": ("lsg_level", LSG_LAYER_CENTRES_M[:-1], {"units": "m", "long_name": "layer centre du"}),
    })
    for name, (dims, units, long_name) in VARIABLES.items():
        target = sections if name in SECTION_VARIABLES else zonal
        target[name] = (("year",) + dims, values[name], {"units": units, "long_name": long_name})

    zonal["t21_gaussian_weight"] = ("t21_lat", t21.weight, {"units": "1", "long_name": "Gaussian weight, sum 2"})
    zonal["t21_band_area"] = ("t21_lat", t21.band_area, {"units": "m2"})
    zonal["t21_ocean_fraction"] = ("t21_lat", t21.ocean_fraction, {"units": "1"})
    zonal["lsg_row_wet_area"] = ("lsg_lat", (lsg.area * lsg.wet[0]).sum(axis=1), {"units": "m2"})
    sections["wet_volume"] = (("lsg_level", "lsg_lat"), lsg.volume.sum(axis=2), {"units": "m3"})
    sections["wet_volume_atlantic"] = (("lsg_level", "lsg_lat"), (lsg.volume * lsg.atlantic[None]).sum(axis=2),
                                       {"units": "m3"})
    sections["lsg_level_top"] = ("lsg_level", top, {"units": "m", "long_name": "dw(k-1); psi is given here"})
    sections["lsg_level_bottom"] = ("lsg_level", bottom, {"units": "m", "long_name": "dw(k)"})
    return zonal, sections


def run_checks(zonal: xr.Dataset) -> dict[str, float]:
    """Run-level checks of §5, from the stored quantities."""
    incoming = area_mean(zonal["rst_zonal"] - zonal["rsut_zonal"], zonal)
    salt = zonal["salt_content"].values.astype(float)
    balance = (zonal["net_meridional_transport"] + zonal["freshwater_flux_north_of"]).mean("year")
    return {
        "incoming_solar_mean": float(incoming.mean()),
        "incoming_solar_range": float(incoming.max() - incoming.min()),
        "salt_content_relative_range": float((salt.max() - salt.min()) / salt.mean()),
        "freshwater_balance_max_abs_sv": float(np.abs(balance).max()),
    }


def _gsol0(experiment: Path) -> float | None:
    """Solar constant GSOL0 from the run's planet_namelist, if it is there."""
    path = experiment / "run" / "planet_namelist"
    if not path.is_file():
        return None
    value = read_namelist(path).get("GSOL0")
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Files and status (Plasim_Indicators.md §3; manifest)
# ---------------------------------------------------------------------------

def product_stem(experiment_name: str, phase: str) -> str:
    """File name stem of a run's product: ``<run>_<phase>_indicators``."""
    return f"{experiment_name}_{phase}_indicators"


def git_revision() -> tuple[str, bool]:
    """Return the git commit of this code and whether tracked files are modified."""
    root = get_repo_root()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                 check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return commit, bool(changes)


def write_indicator_files(zonal: xr.Dataset, sections: xr.Dataset, record: dict[str, object],
                          output_dir: Path, command: str) -> list[Path]:
    """Write ``<stem>_zonal.nc``, ``<stem>_sections.nc`` and ``<stem>_manifest.json``.

    Each file is written to ``*.partial`` and then renamed, so a stopped job
    never leaves a half-written product under the final name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit, modified = git_revision()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "schema": SCHEMA, "product_version": PRODUCT_VERSION,
        "experiment": record["experiment"], "phase": record["phase"],
        "first_label": record["decades_used"][0][0], "last_label": record["decades_used"][-1][1],
        "decades": len(record["decades_used"]), "label_minus_model_year": record["label_minus_model_year"],
        "git_commit": commit, "git_modified": int(modified), "command": command, "created": created,
        "definitions": "Plasim_Indicators.md",
    }
    if "gsol0" in record:
        provenance["gsol0"] = record["gsol0"]

    stem = product_stem(record["experiment"], record["phase"])
    written = []
    for dataset, suffix in ((zonal, "zonal"), (sections, "sections")):
        dataset.attrs.update(provenance)
        target = output_dir / f"{stem}_{suffix}.nc"
        partial = output_dir / f"{stem}_{suffix}.nc.partial"
        encoding = {name: {"zlib": True, "complevel": 4} for name in dataset.data_vars}
        dataset.to_netcdf(partial, engine="h5netcdf", encoding=encoding)
        partial.replace(target)
        written.append(target)

    # One history entry per full run or append, so products built by several commits say so.
    history = list(record.get("history", []))
    history.append({"mode": record["mode"], "first_label": record["computed_from"],
                    "last_label": provenance["last_label"], "git_commit": commit, "git_modified": modified,
                    "command": command, "created": created})
    manifest = provenance | {key: record[key] for key in ("decades_used", "left_out", "checks")}
    manifest["history"] = history
    target = output_dir / f"{stem}_manifest.json"
    partial = output_dir / f"{stem}_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=1) + "\n")
    partial.replace(target)
    written.append(target)
    return written


def update_indicators(experiment: Path, phase: str, beklas_path: Path, output_dir: Path, command: str,
                      full: bool = False, progress: Callable[[str], None] | None = None) -> dict[str, object]:
    """Bring one run's product up to date and return a summary of what was done.

    - New complete decades after an existing product with this code's
      ``PRODUCT_VERSION``: append them (``append_indicators``).
    - No product, another ``product_version``, decades that no longer match
      the disk, or ``full``: recompute from the first decade (``run_indicators``).
    - Nothing new: leave the files as they are.
    """
    experiment, output_dir = Path(experiment), Path(output_dir)
    found = find_decades(experiment, phase)
    decades = found[0]
    manifest = read_manifest(output_dir, experiment.name, phase)
    can_append = (not full and manifest is not None
                  and manifest.get("product_version") == PRODUCT_VERSION
                  and product_is_first_decades(manifest, decades))
    summary: dict[str, object] = {"experiment": experiment.name, "phase": phase}

    if can_append and len(manifest["decades_used"]) == len(decades):
        return summary | {"mode": "up to date", "last_label": decades[-1].last}
    if can_append:
        stem = product_stem(experiment.name, phase)
        zonal_old = load_indicators(output_dir / f"{stem}_zonal.nc")
        sections_old = load_indicators(output_dir / f"{stem}_sections.nc")
        zonal, sections, record = append_indicators(experiment, phase, beklas_path, manifest,
                                                    zonal_old, sections_old, progress, found)
    else:
        zonal, sections, record = run_indicators(experiment, phase, beklas_path, progress, found)
    written = write_indicator_files(zonal, sections, record, output_dir, command)
    return summary | {
        "mode": record["mode"], "computed_from": record["computed_from"],
        "first_label": record["decades_used"][0][0], "last_label": record["decades_used"][-1][1],
        "decades": len(record["decades_used"]), "left_out": record["left_out"],
        "checks": record["checks"], "files": [str(path) for path in written],
    }


def read_manifest(output_dir: Path, experiment_name: str, phase: str) -> dict[str, object] | None:
    """Return a run's manifest, or None if the run has no product yet."""
    path = Path(output_dir) / f"{product_stem(experiment_name, phase)}_manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def product_status(manifest: dict[str, object] | None, new_complete: list[tuple[int, int]]) -> str:
    """``missing``, ``new decades`` or ``up to date``.

    ``new_complete`` lists the complete decades on disk after the manifest's
    last decade.
    """
    if manifest is None:
        return "missing"
    return "new decades" if new_complete else "up to date"


def run_status(experiment: Path, phase: str, output_dir: Path) -> dict[str, object]:
    """Compare a run's product with the decades on disk now.

    Only blocks after the product's last decade are checked with
    ``check_decade``, so a run that is up to date costs no data reads.
    """
    experiment = Path(experiment)
    manifest = read_manifest(output_dir, experiment.name, phase)
    blocks = _blocks_on_disk(experiment, phase)
    if manifest is None:
        new_complete = []
    else:
        last_used = manifest["decades_used"][-1][1]
        new_complete = [(first, last) for (first, last), paths in blocks.items()
                        if first > last_used and check_decade(first, last, paths)[0] is None]
    status = {"experiment": experiment.name, "phase": phase,
              "status": product_status(manifest, new_complete)}
    if manifest is not None:
        status["product_last_label"] = manifest["decades_used"][-1][1]
        status["new_decades"] = len(new_complete)
    return status


# ---------------------------------------------------------------------------
# Laptop side: global, hemispheric and band indicators (Plasim_Indicators.md §4)
# ---------------------------------------------------------------------------

def load_indicators(path: Path) -> xr.Dataset:
    """Load one product file into memory."""
    return xr.load_dataset(path, engine="h5netcdf")


def _region(latitude: xr.DataArray, region: str) -> xr.DataArray:
    """Rows of a region: ``global``, ``nh`` (latitude > 0) or ``sh`` (latitude < 0)."""
    if region == "global":
        return xr.ones_like(latitude, dtype=bool)
    if region == "nh":
        return latitude > 0
    if region == "sh":
        return latitude < 0
    raise ValueError(f"Unknown region {region!r}; use 'global', 'nh' or 'sh'.")


def area_mean(zonal_field: xr.DataArray, zonal: xr.Dataset, region: str = "global",
              ocean_only: bool = False) -> xr.DataArray:
    """Area mean of a T21 zonal-mean field over a region (§4).

    Weights are the Gaussian weights w_j, times the ocean fraction f_j for
    ocean-only fields.
    """
    weight = zonal["t21_gaussian_weight"]
    if ocean_only:
        weight = weight * zonal["t21_ocean_fraction"]
    rows = _region(zonal["t21_lat"], region)
    return zonal_field.where(rows).weighted(weight.where(rows, 0.0)).mean("t21_lat")


def area_total(zonal_field: xr.DataArray, zonal: xr.Dataset, region: str = "global") -> xr.DataArray:
    """Ocean area integral of an ocean-only T21 zonal mean: sum_j A_j f_j z_j (§4).

    For ``sic`` this is the sea-ice area (m2), for ``sit`` the ice volume (m3),
    for ``cfluxra`` the cap flux (W).
    """
    rows = _region(zonal["t21_lat"], region)
    ocean_area = zonal["t21_band_area"] * zonal["t21_ocean_fraction"]
    return (zonal_field.fillna(0.0) * ocean_area).where(rows, 0.0).sum("t21_lat")


def volume_mean(section: xr.DataArray, sections: xr.Dataset, levels: slice | None = None,
                region: str = "global", atlantic: bool = False) -> xr.DataArray:
    """Volume mean of an LSG section over a region and a set of levels (§4).

    ``levels`` selects levels by position, e.g. ``slice(0, 10)`` for the upper
    band 0-700 m (levels 1-10), ``slice(10, 17)`` for 700-2025 m,
    ``slice(17, 22)`` for 2025 m to the bottom.
    """
    volume = sections["wet_volume_atlantic" if atlantic else "wet_volume"]
    rows = _region(sections["lsg_lat"], region)
    if levels is not None:
        section = section.isel(lsg_level=levels)
        volume = volume.isel(lsg_level=levels)
    weight = volume.where(rows, 0.0)
    return section.weighted(weight.fillna(0.0)).mean(("lsg_level", "lsg_lat"))
