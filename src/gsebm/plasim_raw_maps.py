"""Stream annual PlaSim maps and compact mechanism fields to NetCDF4.

The archive retains native upper-ocean levels so zonal or basin means can be
defined after extraction. No temporal filtering or phase selection is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Callable

import h5netcdf
import numpy as np
import xarray as xr

from gsebm.plasim_diagnostics import (
    _layer_overlap_thickness,
    _lsg_ocean_geometry,
    _lsg_wet_cell_volumes,
    LSG_EARTH_RADIUS_M,
    LSG_HORIZONTAL_GRID_SPACING_DEGREES,
    find_mechanism_field_files,
    lsg_ocean_layer_maps_for_file,
    lsg_zonal_fields_for_file,
    plasim_surface_maps_for_file,
)


SCHEMA_VERSION = "2"
_WORKER_GEOMETRY: dict[str, tuple[tuple[str, ...], np.ndarray]] | None = None
UPPER_DEPTH_LIMIT_M = 1025.0
DEEP_BANDS_M = (("1025_2000m", 1025.0, 2000.0), ("2000_6000m", 2000.0, 6000.0))
S2_LSG_FIELDS = (
    "zonal_ocean_heat_content",
    "zonal_newtonian_coupling_heat_flux",
    "zonal_sea_surface_height",
    "zonal_potential_temperature",
    "zonal_meridional_volume_transport",
    "zonal_face_potential_temperature",
    "zonal_meridional_temperature_transport_proxy",
    "zonal_net_meridional_volume_transport",
    "zonal_salinity", "zonal_vertical_velocity", "zonal_convective_adjustment",
    "zonal_boundary_temperature", "zonal_sst_mismatch", "zonal_ice_mismatch",
    "zonal_lsg_ice_thickness", "zonal_lsg_ice_covered_fraction",
    "zonal_fresh_water_flux", "zonal_newtonian_coupling_fresh_water_flux",
    "zonal_ocean_heat_flux", "zonal_zonal_wind_stress",
    "global_sea_surface_height", "global_lsg_ice_volume",
)
MAP_FIELDS = (
    "theta_layer_0_100m",
    "theta_layer_150_300m",
    "theta_layer_300_600m",
    "barotropic_streamfunction",
    "u_layer_0_100m", "v_layer_0_100m",
    "u_layer_150_300m", "v_layer_150_300m",
    "u_layer_300_600m", "v_layer_300_600m",
)
SURFACE_FIELDS = (
    "surface_temperature", "sea_ice_concentration", "sea_ice_thickness",
    "zonal_evaporation", "zonal_snowfall",
)
PLA_MAP_FIELDS = ("rst", "rlut", "rsut", "rss", "rls", "hfss", "hfls", "as", "snd")
CURRENT_BANDS = ("0_100m", "150_300m")
SOURCE_COLUMNS = (
    "source_first_year", "source_last_year", "source_lsg_size", "source_pla_size",
    "source_lsg_mtime_ns", "source_pla_mtime_ns",
    "source_lsg_name_hash", "source_pla_name_hash",
)


def raw_map_root() -> Path:
    """Return the selected archive root, defaulting to repository data."""
    configured = os.environ.get("PLASIM_RAW_MAP_ROOT")
    if configured:
        return Path(configured).expanduser()
    from gsebm.paths import get_data_dir
    return get_data_dir(create=False) / "Plasim"


@dataclass(frozen=True)
class RawMapInventory:
    """Source coverage and paths for one experiment state."""

    experiment: str
    lsg_paths: tuple[Path, ...]
    pla_paths: tuple[Path, ...]
    first_year: int
    last_year: int

    @property
    def year_count(self) -> int:
        return self.last_year - self.first_year + 1


def _declared_years(path: Path) -> np.ndarray:
    match = re.search(r"_(?:LSG|PLA)\.(\d+)-(\d+)\.nc$", path.name)
    if match is None:
        raise ValueError(f"Unrecognized PlaSim annual filename: {path}")
    first, last = map(int, match.groups())
    if last < first:
        raise ValueError(f"Reversed year range in {path}")
    return np.arange(first, last + 1, dtype=np.int32)


def inventory_experiment(experiment_dir: Path, state: str = "spinup") -> RawMapInventory:
    """Match PLA and LSG files by declared years and reject gaps or overlaps."""
    lsg_paths, pla_paths = find_mechanism_field_files(experiment_dir, state)
    lsg_years = np.concatenate([_declared_years(path) for path in lsg_paths])
    pla_years = np.concatenate([_declared_years(path) for path in pla_paths])
    if not np.array_equal(lsg_years, pla_years):
        raise ValueError(f"PLA and LSG year coverage differs for {experiment_dir.name}")
    if len(lsg_paths) != len(pla_paths) or any(
        not np.array_equal(_declared_years(lsg), _declared_years(pla))
        for lsg, pla in zip(lsg_paths, pla_paths)
    ):
        raise ValueError(f"PLA and LSG file blocks differ for {experiment_dir.name}")
    if not np.array_equal(lsg_years, np.arange(lsg_years[0], lsg_years[-1] + 1)):
        raise ValueError(f"Missing or overlapping source years for {experiment_dir.name}")
    return RawMapInventory(
        experiment_dir.name,
        tuple(lsg_paths),
        tuple(pla_paths),
        int(lsg_years[0]),
        int(lsg_years[-1]),
    )


def existing_year_range(path: Path) -> tuple[int, int] | None:
    """Return archive coverage, or None when the archive is absent or outdated."""
    if not path.exists():
        return None
    with h5netcdf.File(path, "r") as file:
        if file.attrs.get("schema_version") != SCHEMA_VERSION:
            return None
        if file.attrs.get("mean_current_mask_applied") != 1:
            return None
        if "committed_last_year" not in file.attrs:
            return None
        completed = int(file.attrs["committed_blocks"])
        last = int(file.variables["source_last_year"][completed - 1]) if completed else int(file.attrs["first_year"]) - 1
        return int(file.attrs["first_year"]), last


@dataclass(frozen=True)
class ExtractionPlan:
    """Action for a source inventory and its selected destination."""

    action: str
    existing_years: tuple[int, int] | None
    completed_blocks: int


def _source_signature(lsg: Path, pla: Path) -> tuple[int, ...]:
    years = _declared_years(lsg)
    lsg_stat, pla_stat = lsg.stat(), pla.stat()
    name_hash = lambda path: int.from_bytes(
        hashlib.sha256(path.name.encode()).digest()[:8], "big", signed=True
    )
    return (
        int(years[0]), int(years[-1]), lsg_stat.st_size, pla_stat.st_size,
        lsg_stat.st_mtime_ns, pla_stat.st_mtime_ns, name_hash(lsg), name_hash(pla),
    )


def plan_extraction(inventory: RawMapInventory, archive: Path, mask: Path, *, refresh: bool = False) -> ExtractionPlan:
    """Choose a rebuild, append, or skip without reading the source arrays."""
    partial = archive.with_name(".raw-maps-partial.nc")
    building = partial.exists() and not refresh
    selected = partial if building else archive
    if refresh or not selected.exists() or (not building and not mask.exists()):
        return ExtractionPlan("rebuild", existing_year_range(archive), 0)
    if not building:
        with h5netcdf.File(mask, "r") as mask_file:
            if mask_file.attrs.get("schema_version") != SCHEMA_VERSION:
                return ExtractionPlan("rebuild", existing_year_range(archive), 0)
    with h5netcdf.File(selected, "r") as file:
        if file.attrs.get("schema_version") != SCHEMA_VERSION or file.attrs.get("mean_current_mask_applied") != 1:
            if building:
                raise ValueError(f"Partial archive has an incompatible schema: {partial}; use --refresh")
            return ExtractionPlan("rebuild", None, 0)
        if file.attrs.get("experiment_name") != inventory.experiment:
            raise ValueError(f"Archive experiment name differs from source: {selected}")
        first = int(file.attrs["first_year"])
        completed = int(file.attrs["committed_blocks"])
        if first != inventory.first_year or completed > len(inventory.lsg_paths):
            raise ValueError(f"Existing archive is not a prefix of the source run: {selected}")
        last = first - 1 if completed == 0 else int(file.variables["source_last_year"][completed - 1])
        for index in range(completed):
            actual = tuple(int(file.variables[name][index]) for name in SOURCE_COLUMNS)
            expected = _source_signature(inventory.lsg_paths[index], inventory.pla_paths[index])
            if actual != expected:
                raise ValueError(f"Previously extracted source block changed: {inventory.lsg_paths[index]}; use --refresh")
        if completed and last != int(_declared_years(inventory.lsg_paths[completed - 1])[-1]):
            raise ValueError(f"Committed year and source-block ledger disagree in {selected}")
        if completed < len(inventory.lsg_paths) and int(_declared_years(inventory.lsg_paths[completed])[0]) != last + 1:
            raise ValueError(f"New source blocks do not continue after year {last}")
        if completed and (
            int(file.variables["year"][0]) != first
            or int(file.variables["year"][last - first]) != last
        ):
            raise ValueError(f"Committed archive year coordinate is inconsistent in {selected}")
        dirty_tail = (
            file.dimensions["year"].size != last - first + 1
            or file.dimensions["source_block"].size != completed
            or int(file.attrs["committed_last_year"]) != last
        )
        action = "resume_build" if building else ("append" if completed < len(inventory.lsg_paths) else ("recover" if dirty_tail else "skip"))
        return ExtractionPlan(action, (first, last), completed)


def _static_geometry(lsg_path: Path, pla_path: Path) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    with xr.open_dataset(lsg_path) as source:
        lat, lon, depth, lower, upper, bathymetry, wet, area = _lsg_ocean_geometry(source, lsg_path)
        volume = _lsg_wet_cell_volumes(source, lsg_path)
        vector_lat = np.asarray(source["lat_2"], dtype=np.float64)
        vector_lon = np.asarray(source["lon_2"], dtype=np.float64)
        vector_bathymetry = np.asarray(source["depv"].isel(time=0, lev=0), dtype=np.float64)
        wetvec = np.asarray(source["wetvec"].isel(time=0), dtype=np.int8)
        upper_indices = np.flatnonzero(upper <= UPPER_DEPTH_LIMIT_M)
        if upper_indices.size != 13 or upper_indices[-1] != 12:
            raise ValueError(f"Unexpected upper LSG vertical grid in {lsg_path}")
        geometry: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "lsg_depth": (("lsg_depth",), depth.astype("float64")),
            "upper_depth": (("upper_depth",), depth[upper_indices].astype("float64")),
            "depth_bounds": (("lsg_depth", "bounds"), np.column_stack((lower, upper))),
            "lat": (("south_north", "west_east"), lat.astype("float64")),
            "lon": (("south_north", "west_east"), lon.astype("float64")),
            "lat_2": (("south_north", "west_east"), vector_lat),
            "lon_2": (("south_north", "west_east"), vector_lon),
            "vector_bathymetry": (("south_north", "west_east"), vector_bathymetry),
            "wet": (("lsg_depth", "south_north", "west_east"), wet.astype("int8")),
            "wetvec": (("lsg_depth", "south_north", "west_east"), wetvec),
            "wet_cell_volume": (("lsg_depth", "south_north", "west_east"), volume.astype("float64")),
            "lsg_horizontal_area": (("south_north", "west_east"), area.astype("float64")),
            "bathymetry": (("south_north", "west_east"), bathymetry.astype("float64")),
        }
        for label, start, stop in DEEP_BANDS_M:
            thickness = _layer_overlap_thickness(bathymetry, lower, upper, start, stop) * wet
            geometry[f"deep_wet_volume_{label}"] = (
                ("south_north", "west_east"),
                np.sum(thickness * area[None, :, :], axis=0).astype("float64"),
            )
    with xr.open_dataset(pla_path) as source:
        nodes, weights = np.polynomial.legendre.leggauss(32)
        t21_lat = np.asarray(source["lat"], dtype="float64")
        if not np.allclose(t21_lat, np.degrees(np.arcsin(nodes))[::-1], atol=1e-4):
            raise ValueError(f"Unexpected T21 Gaussian latitudes in {pla_path}")
        geometry.update(
            t21_lat=(("t21_lat",), t21_lat),
            t21_lon=(("t21_lon",), np.asarray(source["lon"], dtype="float64")),
            t21_gaussian_weight=(("t21_lat",), weights[::-1].astype("float64")),
            lsm=(("t21_lat", "t21_lon"), np.asarray(source["lsm"].isel(time=0), dtype="float64")),
        )
    return geometry


def _native_temperature(lsg_path: Path, geometry: dict[str, tuple[tuple[str, ...], np.ndarray]]) -> dict[str, np.ndarray]:
    with xr.open_dataset(lsg_path) as source:
        _, _, depth, _, _, bathymetry, wet_now, _ = _lsg_ocean_geometry(source, lsg_path)
        for name, current in (("lsg_depth", depth), ("bathymetry", bathymetry), ("wet", wet_now)):
            if not np.array_equal(current, geometry[name][1]):
                raise ValueError(f"LSG static {name} changes in {lsg_path}")
        temperature = np.asarray(source["t"], dtype="float64")
        wet = geometry["wet"][1].astype(bool)
        upper = np.where(wet[:13][None], temperature[:, :13], np.nan).astype("float32")
        bounds = geometry["depth_bounds"][1]
        lower_edges, upper_edges = bounds[:, 0], bounds[:, 1]
        bathymetry = geometry["bathymetry"][1]
        salinity = np.asarray(source["s"], dtype="float64")
        if salinity.shape != temperature.shape:
            raise ValueError(f"LSG salinity grid differs from temperature in {lsg_path}")
        deep: dict[str, np.ndarray] = {
            "temperature_upper": upper,
            "salinity_upper": np.where(wet[:13][None], salinity[:, :13], np.nan).astype("float32"),
        }
        for label, start, stop in DEEP_BANDS_M:
            thickness = _layer_overlap_thickness(bathymetry, lower_edges, upper_edges, start, stop) * wet
            denominator = thickness.sum(axis=0)
            mean = np.full((temperature.shape[0], *denominator.shape), np.nan, dtype="float64")
            np.divide(
                np.sum(temperature * thickness[None], axis=1),
                denominator[None],
                out=mean,
                where=denominator[None] > 0,
            )
            deep[f"temperature_{label}"] = mean.astype("float32")
        return deep


def _basin_meridional_transports(
    lsg_path: Path, geometry: dict[str, tuple[tuple[str, ...], np.ndarray]]
) -> dict[str, np.ndarray]:
    """Integrate annual mean v and v·theta across native wet vector faces."""
    with xr.open_dataset(lsg_path) as source:
        for name, values in (
            ("wetvec", np.asarray(source["wetvec"].isel(time=0), dtype="int8")),
            ("lat_2", np.asarray(source["lat_2"], dtype="float64")),
            ("lon_2", np.asarray(source["lon_2"], dtype="float64")),
            ("vector_bathymetry", np.asarray(source["depv"].isel(time=0, lev=0), dtype="float64")),
        ):
            if not np.array_equal(values, geometry[name][1]):
                raise ValueError(f"LSG vector geometry {name} changes in {lsg_path}")
        velocity = np.asarray(source["vtot"], dtype="float64")
        temperature = np.asarray(source["t"], dtype="float64")
        wet = geometry["wetvec"][1].astype(bool)
        bathymetry = geometry["vector_bathymetry"][1]
        bounds = geometry["depth_bounds"][1]
        thickness = (
            np.maximum(0.0, np.minimum(bathymetry[None], bounds[:, 1, None, None])
                       - bounds[:, 0, None, None]) * wet
        )
        latitude = geometry["lat_2"][1]
        width = LSG_EARTH_RADIUS_M * np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES) * np.cos(np.deg2rad(latitude))
        width = np.where(np.abs(latitude) <= 90.0, width, 0.0)
        face_area = thickness * width[None]
        face_temperature = np.full_like(temperature, np.nan)
        northern = np.concatenate((temperature[:, :, :1, :], temperature[:, :, :-2, :]), axis=2)
        face_temperature[:, :, :-1, :] = 0.5 * (
            np.roll(temperature[:, :, 1:, :], shift=-1, axis=-1) + northern
        )
        if not np.isfinite(velocity[:, wet]).all() or not np.isfinite(face_temperature[:, wet]).all():
            raise ValueError(f"Non-finite wet meridional transport inputs in {lsg_path}")
        longitude = (geometry["lon_2"][1] + 180.0) % 360.0 - 180.0
        atlantic = (longitude >= -65.0) & (longitude < 20.0)
        result: dict[str, np.ndarray] = {}
        for basin, sector in (("atlantic", atlantic), ("indo_pacific", ~atlantic)):
            weighted_v = np.where(wet[None] & sector[None, None], velocity, 0.0) * face_area[None]
            weighted_vtheta = weighted_v * np.where(wet[None], face_temperature, 0.0)
            for suffix, values in (("volume_transport", weighted_v), ("temperature_transport_proxy", weighted_vtheta)):
                rows = []
                for row in geometry["lsg_vector_lat"][1]:
                    row_mask = np.isclose(latitude, row, atol=1e-6) & np.any(wet, axis=0)
                    rows.append(np.sum(values[..., row_mask], axis=-1))
                result[f"{basin}_{suffix}"] = np.stack(rows, axis=1).astype("float32")
        return result


def _annual_block(
    lsg_path: Path,
    pla_path: Path,
    geometry: dict[str, tuple[tuple[str, ...], np.ndarray]],
) -> tuple[dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]], dict[str, np.ndarray]]:
    maps = lsg_ocean_layer_maps_for_file(lsg_path)
    zonal = lsg_zonal_fields_for_file(lsg_path)
    surface = plasim_surface_maps_for_file(pla_path)
    years = _declared_years(lsg_path)
    if not np.array_equal(np.asarray(maps["year"]), years) or not np.array_equal(np.asarray(zonal["year"]), years) or not np.array_equal(np.asarray(surface["year"]), years):
        raise ValueError(f"Decoded year mismatch for {lsg_path} and {pla_path}")
    block: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    for dataset, names in ((maps, MAP_FIELDS), (zonal, S2_LSG_FIELDS), (surface, SURFACE_FIELDS)):
        for name in names:
            if name not in dataset:
                raise ValueError(f"Required annual field {name} is absent from {lsg_path if dataset is not surface else pla_path}")
            variable = dataset[name]
            block[name] = (variable.dims, np.asarray(variable.values), dict(variable.attrs))
    upper = _native_temperature(lsg_path, geometry)
    block["temperature_upper"] = (
        ("year", "upper_depth", "south_north", "west_east"), upper["temperature_upper"],
        {"units": "K", "source_variable": "t", "description": "native annual LSG temperature at 13 levels through 950 m"},
    )
    block["salinity_upper"] = (
        ("year", "upper_depth", "south_north", "west_east"), upper["salinity_upper"],
        {"units": "0/00", "source_variable": "s", "description": "native annual LSG salinity at 13 levels through 950 m"},
    )
    for label, _, _ in DEEP_BANDS_M:
        block[f"temperature_{label}"] = (
            ("year", "south_north", "west_east"), upper[f"temperature_{label}"],
            {"units": "K", "source_variable": "t", "weighting": "wet cell volume including partial bottom cells"},
        )
    with xr.open_dataset(pla_path) as source:
        if not np.array_equal(np.asarray(source["lat"]), geometry["t21_lat"][1]) or not np.array_equal(np.asarray(source["lon"]), geometry["t21_lon"][1]) or not np.array_equal(np.asarray(source["lsm"].isel(time=0)), geometry["lsm"][1]):
            raise ValueError(f"T21 static grid or land mask changes in {pla_path}")
        block["zonal_toa_energy_imbalance"] = (
            ("year", "t21_lat"),
            np.asarray((source["rst"] + source["rlut"]).mean("lon"), dtype="float32"),
            {"units": "W m-2", "source_variables": "rst, rlut", "calculation": "longitude mean of rst + rlut"},
        )
        block["zonal_surface_albedo"] = (
            ("year", "t21_lat"), np.asarray(source["as"].mean("lon"), dtype="float32"),
            {"units": "1", "source_variable": "as", "calculation": "longitude mean"},
        )
        for name in PLA_MAP_FIELDS:
            if source[name].dims != ("time", "lat", "lon"):
                raise ValueError(f"Unexpected dimensions for PLA {name} in {pla_path}")
            block[name] = (
                ("year", "t21_lat", "t21_lon"), np.asarray(source[name], dtype="float32"),
                {**dict(source[name].attrs), "source_variable": name, "sign_convention": "native PlaSim net-downward sign"},
            )
        for kind, words in (("shortwave", ("solar", "shortwave")), ("longwave", ("thermal", "longwave"))):
            matches = [
                name for name, variable in source.data_vars.items()
                if variable.dims == ("time", "lat", "lon")
                and "clear" in str(variable.attrs.get("long_name", "")).lower()
                and "sky" in str(variable.attrs.get("long_name", "")).lower()
                and any(word in str(variable.attrs.get("long_name", "")).lower() for word in words)
                and any(word in str(variable.attrs.get("long_name", "")).lower() for word in ("top", "toa"))
            ]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous clear-sky {kind} variables in {pla_path}: {matches}")
            if matches:
                source_name = matches[0]
                block[f"toa_clear_sky_{kind}"] = (
                    ("year", "t21_lat", "t21_lon"),
                    np.asarray(source[source_name], dtype="float32"),
                    {**dict(source[source_name].attrs), "source_variable": source_name, "sign_convention": "native PlaSim sign"},
                )
    for name in ("wet_surface_area", "wet_volume", "wet_vector_cross_section_area"):
        variable = zonal[name]
        values = np.asarray(variable.values)
        if name in geometry and not np.array_equal(values, geometry[name][1]):
            raise ValueError(f"LSG static {name} changes in {lsg_path}")
        geometry[name] = (variable.dims, values)
    for name in ("lsg_lat", "lsg_vector_lat", "lsg_depth_interface"):
        variable = zonal[name]
        values = np.asarray(variable.values)
        if name in geometry and not np.array_equal(values, geometry[name][1]):
            raise ValueError(f"LSG static {name} changes in {lsg_path}")
        geometry[name] = (variable.dims, values)
    basin = _basin_meridional_transports(lsg_path, geometry)
    for name, values in basin.items():
        block[name] = (
            ("year", "lsg_vector_lat", "lsg_depth"), values,
            {"units": "K m3 s-1" if "temperature" in name else "m3 s-1",
             "source_variables": "vtot, t" if "temperature" in name else "vtot",
             "interpretation": "annual-mean-flow vbar*thetabar proxy; no sub-annual covariance" if "temperature" in name else "native-sign meridional volume transport",
             "basin_definition": "65W-20E on native vector longitude" if name.startswith("atlantic") else "wet vector faces outside the Atlantic sector"},
        )
    velocity_sums = {
        name: np.asarray(maps[name].values, dtype="float64")
        for name in maps.data_vars if name.startswith("_u_sum_") or name.startswith("_v_sum_")
    }
    return block, velocity_sums


def _initialize_worker(geometry: dict[str, tuple[tuple[str, ...], np.ndarray]]) -> None:
    global _WORKER_GEOMETRY
    _WORKER_GEOMETRY = geometry


def _worker_block(paths: tuple[Path, Path]) -> tuple[dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]], dict[str, np.ndarray]]:
    if _WORKER_GEOMETRY is None:
        raise RuntimeError("PlaSim map worker lacks static geometry")
    return _annual_block(paths[0], paths[1], _WORKER_GEOMETRY)


def _write_array(file: h5netcdf.File, name: str, dims: tuple[str, ...], values: np.ndarray, attrs: dict[str, object] | None = None) -> None:
    array = np.asarray(values)
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True} if array.size > 4096 else {}
    chunks = None
    if dims and dims[0] == "year" and array.ndim > 0:
        chunks = (min(10, file.dimensions["year"].size), *array.shape[1:])
    variable = file.create_variable(name, dims, dtype=array.dtype, chunks=chunks, **compression)
    variable[:] = array
    for key, value in (attrs or {}).items():
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            variable.attrs[key] = value


def _make_basin_masks(
    path: Path,
    geometry: dict[str, tuple[tuple[str, ...], np.ndarray]],
    experiment: str,
) -> None:
    wrap = lambda lon: (np.asarray(lon) + 180.0) % 360.0 - 180.0
    masks = {}
    for label, lat_name, lon_name, wet_name in (
        ("lsg_scalar", "lat", "lon", "wet"),
        ("lsg_vector", "lat_2", "lon_2", "wetvec"),
    ):
        lat = geometry[lat_name][1]
        lon = wrap(geometry[lon_name][1])
        wet = np.any(geometry[wet_name][1] > 0, axis=0)
        masks[f"{label}_south_atlantic"] = ((lat < 0) & (lon >= -65) & (lon < 20) & wet).astype("int8")
    lat, lon = np.meshgrid(geometry["t21_lat"][1], wrap(geometry["t21_lon"][1]), indexing="ij")
    masks["t21_south_atlantic"] = ((lat < 0) & (lon >= -65) & (lon < 20) & (geometry["lsm"][1] < 0.5)).astype("int8")
    dataset = xr.Dataset(
        {
            "lsg_scalar_south_atlantic": (("south_north", "west_east"), masks["lsg_scalar_south_atlantic"]),
            "lsg_vector_south_atlantic": (("south_north", "west_east"), masks["lsg_vector_south_atlantic"]),
            "t21_south_atlantic": (("t21_lat", "t21_lon"), masks["t21_south_atlantic"]),
            "lat": (geometry["lat"][0], geometry["lat"][1]),
            "lon": (geometry["lon"][0], geometry["lon"][1]),
            "lat_2": (geometry["lat_2"][0], geometry["lat_2"][1]),
            "lon_2": (geometry["lon_2"][0], geometry["lon_2"][1]),
            "wet": (geometry["wet"][0], geometry["wet"][1]),
            "wetvec": (geometry["wetvec"][0], geometry["wetvec"][1]),
            "lsm": (geometry["lsm"][0], geometry["lsm"][1]),
        },
        coords={"t21_lat": geometry["t21_lat"][1], "t21_lon": geometry["t21_lon"][1]},
        attrs={
            "schema_version": SCHEMA_VERSION,
            "experiment_name": experiment,
            "mask_definition": "wet South Atlantic sector: latitude < 0, -65 <= wrapped longitude < 20 degrees",
            "interpretation": "static geographic selection; refine in this file without re-extracting annual fields",
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def extract_raw_maps(
    inventory: RawMapInventory,
    destination: Path,
    *,
    workers: int = 4,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path]:
    """Create schema-v2 archives or append only new committed source blocks."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{inventory.experiment}_spinup"
    archive_path = destination / f"{stem}_raw_maps.nc"
    mask_path = destination / f"{stem}_basin_masks.nc"
    plan = plan_extraction(inventory, archive_path, mask_path, refresh=refresh)
    if plan.action == "skip":
        return archive_path, mask_path
    if plan.action == "recover":
        assert plan.existing_years is not None
        with h5netcdf.File(archive_path, "a") as output:
            committed_years = plan.existing_years[1] - plan.existing_years[0] + 1
            output.resize_dimension("year", committed_years)
            output.resize_dimension("source_block", plan.completed_blocks)
            _update_mean_currents(output, committed_years, recover=True)
            output.attrs["committed_last_year"] = plan.existing_years[1]
            output.attrs["last_year"] = plan.existing_years[1]
            output.flush()
        return archive_path, mask_path
    start_block = plan.completed_blocks if plan.action in ("append", "resume_build") else 0
    file_pairs = list(zip(inventory.lsg_paths, inventory.pla_paths, strict=True))
    first_lsg, first_pla = file_pairs[min(start_block, len(file_pairs) - 1)]
    geometry = _static_geometry(first_lsg, first_pla)
    first_block, first_velocity_sums = _annual_block(
        first_lsg, first_pla, geometry
    )
    del first_velocity_sums
    dimensions = {
        "year": None,
        "source_block": None,
        "lsg_depth": len(geometry["lsg_depth"][1]),
        "upper_depth": len(geometry["upper_depth"][1]),
        "south_north": geometry["lat"][1].shape[0],
        "west_east": geometry["lat"][1].shape[1],
        "t21_lat": len(geometry["t21_lat"][1]),
        "t21_lon": len(geometry["t21_lon"][1]),
        "lsg_lat": len(geometry["lsg_lat"][1]),
        "lsg_vector_lat": len(geometry["lsg_vector_lat"][1]),
        "lsg_depth_interface": len(geometry["lsg_depth_interface"][1]),
        "bounds": 2,
    }
    pool_context = (
        ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(geometry,))
        if workers > 1 else nullcontext(None)
    )
    partial_path = archive_path.with_name(".raw-maps-partial.nc")
    building = plan.action in ("rebuild", "resume_build")
    target = partial_path if building else archive_path
    with pool_context as pool, h5netcdf.File(target, "w" if plan.action == "rebuild" else "a") as output:
            if plan.action == "rebuild":
                for name, size in dimensions.items():
                    output.dimensions[name] = size
                output.attrs.update(
                    schema_version=SCHEMA_VERSION,
                    experiment_name=inventory.experiment,
                    experiment_state="spinup",
                    first_year=inventory.first_year,
                    committed_last_year=inventory.first_year - 1,
                    committed_blocks=0,
                    mean_current_mask_applied=1,
                    dynamic_fields=json.dumps(sorted(first_block)),
                    missing_optional_pla_maps=json.dumps([
                        name for name in ("toa_clear_sky_shortwave", "toa_clear_sky_longwave")
                        if name not in first_block
                    ]),
                    temporal_processing="raw annual source values; no detrending, smoothing, or compositing",
                )
                for name, (dims, values) in geometry.items():
                    _write_array(output, name, dims, values)
                output.create_variable("year", ("year",), dtype="int32", chunks=(10,))
                for name in SOURCE_COLUMNS:
                    output.create_variable(name, ("source_block",), dtype="int64", chunks=(100,))
                for name, (dims, values, attrs) in first_block.items():
                    var = output.create_variable(name, dims, dtype=values.dtype, chunks=(min(10, values.shape[0]), *values.shape[1:]), compression="gzip", compression_opts=4, shuffle=True)
                    for key, value in attrs.items():
                        if isinstance(value, (str, int, float, np.integer, np.floating)):
                            var.attrs[key] = value
                for label in CURRENT_BANDS:
                    for component in ("u", "v"):
                        name = f"{component}_mean_{label}"
                        shape = geometry["lat_2"][1].shape
                        mean = output.create_variable(name, ("south_north", "west_east"), dtype="float32")
                        mean[:] = np.full(shape, np.nan, dtype="float32")
                        mean.attrs["units"] = "m s-1"
                        mean.attrs["time_weighting"] = "mean of annual samples"
                        running_sum = output.create_variable(f"{name}_running_sum", ("south_north", "west_east"), dtype="float64")
                        running_sum[:] = np.zeros(shape, dtype="float64")
            else:
                if json.loads(output.attrs["dynamic_fields"]) != sorted(first_block):
                    raise ValueError(f"Source field set differs from existing archive: {first_lsg}; use --refresh")
                for name, (_, values) in geometry.items():
                    if name not in output.variables or not np.array_equal(np.asarray(output.variables[name][:]), values):
                        raise ValueError(f"Static geometry {name} changed in {first_lsg}; use --refresh")
                assert plan.existing_years is not None
                committed_years = plan.existing_years[1] - inventory.first_year + 1
                dirty_tail = (
                    output.dimensions["year"].size != committed_years
                    or output.dimensions["source_block"].size != start_block
                    or int(output.attrs["committed_last_year"]) != plan.existing_years[1]
                )
                if dirty_tail:
                    output.resize_dimension("year", committed_years)
                    output.resize_dimension("source_block", start_block)
                    _update_mean_currents(output, committed_years, recover=True)
                    output.attrs["committed_last_year"] = plan.existing_years[1]
                    output.attrs["last_year"] = plan.existing_years[1]
            offset = output.dimensions["year"].size
            pending_pairs = file_pairs[start_block + 1:]
            remaining = (
                chain.from_iterable(
                    pool.map(_worker_block, pending_pairs[index:index + 2 * workers])
                    for index in range(0, len(pending_pairs), 2 * workers)
                ) if pool is not None else
                (_annual_block(lsg, pla, geometry) for lsg, pla in pending_pairs)
            )
            first_result = ((first_block, {}),) if start_block < len(file_pairs) else ()
            for index, (block, _) in enumerate(chain(first_result, remaining), start=start_block):
                if sorted(block) != json.loads(output.attrs["dynamic_fields"]):
                    raise ValueError(f"Source field set changes at {file_pairs[index][0]}")
                n_years = block["surface_temperature"][1].shape[0]
                output.resize_dimension("year", offset + n_years)
                output.variables["year"][offset:offset + n_years] = _declared_years(file_pairs[index][0])
                for name, (_, values, _) in block.items():
                    output.variables[name][offset:offset + n_years] = values
                _update_mean_currents(output, offset + n_years, block=block)
                output.resize_dimension("source_block", index + 1)
                for name, value in zip(SOURCE_COLUMNS, _source_signature(*file_pairs[index]), strict=True):
                    output.variables[name][index] = value
                output.flush()
                output.attrs["committed_last_year"] = int(_declared_years(file_pairs[index][0])[-1])
                output.attrs["last_year"] = int(output.attrs["committed_last_year"])
                output.attrs["lsg_file_count"] = index + 1
                output.attrs["pla_file_count"] = index + 1
                output.attrs["committed_blocks"] = index + 1
                output.flush()
                offset += n_years
                if progress is not None and ((index + 1) % 100 == 0 or index + 1 == len(file_pairs)):
                    progress(index + 1, len(file_pairs))
            if offset != inventory.year_count:
                raise ValueError(f"Expected {inventory.year_count} years, received {offset}")
    if building:
        with NamedTemporaryFile(prefix=".basin-masks-", suffix=".nc", dir=destination, delete=False) as mask_tmp:
            temporary_mask = Path(mask_tmp.name)
        try:
            _make_basin_masks(temporary_mask, geometry, inventory.experiment)
            partial_path.replace(archive_path)
            temporary_mask.replace(mask_path)
        finally:
            temporary_mask.unlink(missing_ok=True)
    return archive_path, mask_path


def _update_mean_currents(
    output: h5netcdf.File, year_count: int, *,
    block: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] | None = None,
    recover: bool = False,
) -> None:
    """Refresh the retained full-archive mean currents from annual maps."""
    for label in CURRENT_BANDS:
        for component in ("u", "v"):
            name = f"{component}_mean_{label}"
            annual_name = f"{component}_layer_{label}"
            if recover:
                total = np.zeros(output.variables[name].shape, dtype="float64")
                for start in range(0, year_count, 100):
                    total += np.nansum(output.variables[annual_name][start:min(start + 100, year_count)], axis=0)
            else:
                assert block is not None
                total = np.asarray(output.variables[f"{name}_running_sum"][:], dtype="float64")
                total += np.nansum(block[annual_name][1], axis=0)
            valid = np.isfinite(output.variables[annual_name][0]) if recover else np.isfinite(block[annual_name][1][0])
            output.variables[f"{name}_running_sum"][:] = total
            mean = np.full(total.shape, np.nan, dtype="float32")
            if year_count:
                mean[valid] = (total[valid] / year_count).astype("float32")
            output.variables[name][:] = mean


def repair_mean_current_masks(path: Path) -> None:
    """Mask dry vector points in archives written before the mask correction."""
    with h5netcdf.File(path, "a") as file:
        wetvec = np.asarray(file.variables["wetvec"][:], dtype=bool)
        bounds = np.asarray(file.variables["depth_bounds"][:], dtype=float)
        latitude = np.asarray(file.variables["lat_2"][:], dtype=float)
        for label, start, stop in (("0_100m", 0.0, 100.0), ("150_300m", 150.0, 300.0)):
            overlap = np.maximum(0.0, np.minimum(bounds[:, 1], stop) - np.maximum(bounds[:, 0], start))
            valid = np.any(wetvec & (overlap[:, None, None] > 0), axis=0)
            valid &= np.abs(latitude) <= 90.0
            for component in ("u", "v"):
                variable = file.variables[f"{component}_mean_{label}"]
                values = np.asarray(variable[:])
                values[~valid] = np.nan
                variable[:] = values
        file.attrs["mean_current_mask_applied"] = 1
