"""Extract zonal climate diagnostics from PLASIM monthly NetCDF output.

The historical filename is retained for existing cluster workflows. This
script writes the richer ``zonal_climate_*.nc`` product; existing
``zonal_T_*.nc`` files are left unchanged.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import typer
import xarray as xr

app = typer.Typer(add_completion=False, no_args_is_help=True)
FILE_PATTERN = re.compile(r"CONTROL_360ppm_PLA\.(?P<year>\d{4})\.(?P<month>\d{2})\.nc$")
REQUIRED_FIELDS = ("tas", "ts", "lsm", "as", "sic", "sit", "rst", "rsut", "rlut")
REQUIRED_DIMENSIONS = {"time", "lat", "lon"}


@dataclass(frozen=True)
class InputFile:
    """One dated PLASIM monthly file selected for processing."""

    path: Path
    year: int
    month: int
    is_reference: bool = False


@dataclass(frozen=True)
class ZonalClimateResult:
    """Small, eagerly loaded result returned by one worker process."""

    filename: str
    year: int
    month: int
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    land_sea_mask: np.ndarray
    fields: dict[str, np.ndarray]
    dataset_attrs: dict[str, object] | None = None
    variable_attrs: dict[str, dict[str, object]] | None = None
    time_attrs: dict[str, object] | None = None
    latitude_attrs: dict[str, object] | None = None


def _filename_value(value: float) -> str:
    """Format a numeric experiment setting compactly for a filename."""

    return f"{value:g}"


def _require_source_fields(source_dataset: xr.Dataset, path: Path) -> None:
    """Validate fields and their common PLASIM source-grid dimensions."""

    for name in REQUIRED_FIELDS:
        if name not in source_dataset:
            raise ValueError(f"{path.name}: missing required variable {name!r}.")
        dimensions = source_dataset[name].dims
        missing = REQUIRED_DIMENSIONS.difference(dimensions)
        extra = set(dimensions).difference(REQUIRED_DIMENSIONS)
        if missing or extra:
            raise ValueError(
                f"{path.name}: expected exactly the {name} dimensions "
                f"{sorted(REQUIRED_DIMENSIONS)}, got {dimensions}."
            )


def _extract_zonal_climate(input_file: InputFile) -> ZonalClimateResult:
    """Read one monthly file and return its zonal climate diagnostics."""

    path = input_file.path
    with xr.open_dataset(path, engine="h5netcdf") as source_dataset:
        _require_source_fields(source_dataset, path)
        lsm = source_dataset["lsm"].transpose("time", "lat", "lon")
        mask = np.asarray(lsm.values)
        if not np.all(mask == mask[0]):
            raise ValueError(f"{path.name}: lsm must be time-invariant.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{path.name}: lsm must be a binary land-sea mask.")
        ocean = mask[0] == 0

        def zonal_mean(name: str) -> np.ndarray:
            return np.asarray(source_dataset[name].mean(dim="lon").transpose("time", "lat").values)

        def ocean_zonal_mean(name: str) -> np.ndarray:
            values = np.asarray(source_dataset[name].transpose("time", "lat", "lon").values)
            count = ocean.sum(axis=1)
            summed = np.where(ocean[None, :, :], values, 0.0).sum(axis=2)
            return np.divide(
                summed, count[None, :], out=np.full(summed.shape, np.nan, dtype=float),
                where=count[None, :] != 0,
            )

        fields = {
            "zonal_T": zonal_mean("tas"),
            "zonal_surface_temperature": zonal_mean("ts"),
            "ocean_zonal_T": ocean_zonal_mean("tas"),
            "ocean_zonal_sea_ice_cover": ocean_zonal_mean("sic"),
            "ocean_zonal_sea_ice_thickness": ocean_zonal_mean("sit"),
            "zonal_surface_albedo": zonal_mean("as"),
            "zonal_toa_net_shortwave": zonal_mean("rst") + zonal_mean("rsut"),
            "zonal_toa_outgoing_longwave": -zonal_mean("rlut"),
            "zonal_toa_net_radiation": zonal_mean("rst") + zonal_mean("rsut") + zonal_mean("rlut"),
        }
        source_for_output = {
            "zonal_T": "tas", "zonal_surface_temperature": "ts", "ocean_zonal_T": "tas",
            "ocean_zonal_sea_ice_cover": "sic", "ocean_zonal_sea_ice_thickness": "sit",
            "zonal_surface_albedo": "as", "zonal_toa_net_shortwave": "rst",
            "zonal_toa_outgoing_longwave": "rlut", "zonal_toa_net_radiation": "rst",
        }
        return ZonalClimateResult(
            filename=path.name, year=input_file.year, month=input_file.month,
            time=np.asarray(source_dataset["time"].values), latitude=np.asarray(source_dataset["lat"].values),
            longitude=np.asarray(source_dataset["lon"].values), land_sea_mask=mask[0], fields=fields,
            dataset_attrs=source_dataset.attrs.copy() if input_file.is_reference else None,
            variable_attrs=({output: source_dataset[source].attrs.copy() for output, source in source_for_output.items()} if input_file.is_reference else None),
            time_attrs=source_dataset["time"].attrs.copy() if input_file.is_reference else None,
            latitude_attrs=source_dataset["lat"].attrs.copy() if input_file.is_reference else None,
        )


def _validate_common_grid(results: list[ZonalClimateResult]) -> None:
    """Ensure each diagnostic was evaluated on one spatial grid and mask."""

    reference = results[0]
    for result in results[1:]:
        if not np.array_equal(result.latitude, reference.latitude):
            raise ValueError(f"{result.filename}: latitude grid differs from the reference file.")
        if not np.array_equal(result.longitude, reference.longitude):
            raise ValueError(f"{result.filename}: longitude grid differs from the reference file.")
        if not np.array_equal(result.land_sea_mask, reference.land_sea_mask):
            raise ValueError(f"{result.filename}: lsm differs from the reference file.")


def _output_dataset(results: list[ZonalClimateResult], co2_ppm: float, solar_irradiance_w_m2: float) -> xr.Dataset:
    """Assemble worker results into the aligned diagnostic dataset."""

    _validate_common_grid(results)
    reference = results[0]
    time = np.concatenate([result.time for result in results])
    coords = {
        "time": ("time", time), "lat": ("lat", reference.latitude),
        "source_year": ("time", np.concatenate([np.full(r.time.size, r.year, dtype=int) for r in results])),
        "source_month": ("time", np.concatenate([np.full(r.time.size, r.month, dtype=int) for r in results])),
        "source_file": ("time", np.concatenate([np.full(r.time.size, r.filename) for r in results])),
    }
    variables = {
        name: xr.DataArray(np.concatenate([result.fields[name] for result in results], axis=0), dims=("time", "lat"), coords=coords, name=name, attrs=(reference.variable_attrs or {}).get(name, {}))
        for name in reference.fields
    }
    variables["ocean_fraction"] = xr.DataArray(
        (reference.land_sea_mask == 0).mean(axis=1), dims=("lat",), coords={"lat": reference.latitude},
        attrs={"long_name": "ocean fraction of longitude cells", "units": "1", "calculation": "fraction of cells where lsm == 0"},
    )
    output = xr.Dataset(variables).sortby("time")
    output["time"].attrs = reference.time_attrs or {}
    output["lat"].attrs = reference.latitude_attrs or {}
    averaging = "unweighted mean over longitude"
    for name in ("zonal_T", "zonal_surface_temperature", "zonal_surface_albedo"):
        output[name].attrs.update(averaging=averaging)
    for name in ("ocean_zonal_T", "ocean_zonal_sea_ice_cover", "ocean_zonal_sea_ice_thickness"):
        output[name].attrs.update(averaging=averaging, masking="mean over cells where lsm == 0; NaN where a latitude belt has no ocean cells")
    output["zonal_T"].attrs.update(long_name="zonal-mean 2 m air temperature")
    output["zonal_surface_temperature"].attrs.update(long_name="zonal-mean surface temperature")
    output["zonal_surface_albedo"].attrs.update(long_name="zonal-mean surface albedo")
    for name, long_name, convention in (
        ("zonal_toa_net_shortwave", "zonal-mean net top-of-atmosphere shortwave radiation", "rst + rsut; positive downward"),
        ("zonal_toa_outgoing_longwave", "zonal-mean outgoing top-of-atmosphere longwave radiation", "-rlut; positive outward"),
        ("zonal_toa_net_radiation", "zonal-mean net top-of-atmosphere radiation", "rst + rsut + rlut; positive downward"),
    ):
        output[name].attrs.update(long_name=long_name, averaging=averaging, flux_sign_convention=convention)
    output.attrs = reference.dataset_attrs or {}
    output.attrs.update(
        zonal_mean_reference_file=reference.filename, zonal_mean_file_count=len(results),
        zonal_mean_processing="longitude means and ocean-only means derived from time-invariant binary lsm",
        experiment_co2_concentration_ppm=co2_ppm, experiment_solar_irradiance_w_m2=solar_irradiance_w_m2,
    )
    return output


@app.command()
def main(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Read-only directory containing CONTROL_360ppm_PLA.YYYY.MM.nc files."),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True, help="Directory in which to create the derived zonal-climate NetCDF file."),
    co2_ppm: float = typer.Option(..., min=0.0, help="Experiment CO2 concentration [ppm]."),
    solar_irradiance_w_m2: float = typer.Option(..., min=0.0, help="Experiment solar irradiance [W m^-2]."),
    workers: int = typer.Option(8, min=1, help="Number of monthly files to process concurrently."),
) -> None:
    """Create one aligned zonal climate-diagnostics dataset from PLASIM output."""

    files: list[InputFile] = []
    for path in sorted(input_dir.glob("CONTROL_360ppm_PLA.*.*.nc")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            if path.stat().st_size == 0:
                typer.echo(f"Skipping empty file: {path.name}")
                continue
            files.append(InputFile(path, int(match["year"]), int(match["month"])))
    if not files:
        raise typer.BadParameter("No non-empty files matching CONTROL_360ppm_PLA.YYYY.MM.nc were found.", param_hint="--input-dir")
    output_path = output_dir / f"zonal_climate_CO2_{_filename_value(co2_ppm)}_mu_{_filename_value(solar_irradiance_w_m2)}.nc"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    files[0] = InputFile(files[0].path, files[0].year, files[0].month, is_reference=True)
    results: list[ZonalClimateResult] = []
    typer.echo(f"Processing {len(files)} files for zonal climate diagnostics with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(executor.map(_extract_zonal_climate, files, chunksize=1), start=1):
            results.append(result)
            typer.echo(f"Processed climate diagnostics: {result.filename} ({completed}/{len(files)})")
    output_dataset = _output_dataset(results, co2_ppm, solar_irradiance_w_m2)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset.to_netcdf(output_path, engine="h5netcdf")
    typer.echo(f"Wrote zonal climate diagnostics to {output_path}")


if __name__ == "__main__":
    app()
