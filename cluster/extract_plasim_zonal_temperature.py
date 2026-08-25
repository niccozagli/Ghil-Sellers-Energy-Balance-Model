"""Extract zonal-mean 2 m temperature from PLASIM monthly NetCDF output."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import typer
import xarray as xr

app = typer.Typer(add_completion=False, no_args_is_help=True)

FILE_PATTERN = re.compile(
    r"CONTROL_360ppm_PLA\.(?P<year>\d{4})\.(?P<month>\d{2})\.nc$"
)


@dataclass(frozen=True)
class InputFile:
    """One dated PLASIM monthly file selected for processing."""

    path: Path
    year: int
    month: int
    is_reference: bool = False


@dataclass(frozen=True)
class ZonalTemperatureResult:
    """Small, eagerly loaded result returned by one worker process."""

    filename: str
    year: int
    month: int
    time: np.ndarray
    latitude: np.ndarray
    temperature: np.ndarray
    dataset_attrs: dict[str, object] | None = None
    tas_attrs: dict[str, object] | None = None
    time_attrs: dict[str, object] | None = None
    latitude_attrs: dict[str, object] | None = None


def _filename_value(value: float) -> str:
    """Format a numeric experiment setting compactly for a filename."""

    return f"{value:g}"


def _extract_zonal_temperature(input_file: InputFile) -> ZonalTemperatureResult:
    """Read one monthly file and return its zonal-mean temperature."""

    path = input_file.path
    # xarray.open_dataset opens the source read-only. Every source handle is
    # closed in its worker before the compact result is sent to the parent.
    with xr.open_dataset(path, engine="h5netcdf") as source_dataset:
        tas = source_dataset["tas"]
        required_dimensions = {"time", "lat", "lon"}
        missing_dimensions = required_dimensions.difference(tas.dims)
        extra_dimensions = set(tas.dims).difference(required_dimensions)
        if missing_dimensions or extra_dimensions:
            raise ValueError(
                f"{path.name}: expected exactly the tas dimensions "
                f"{sorted(required_dimensions)}, got {tas.dims}."
            )

        # Longitude cells have equal area at a fixed latitude, so the ordinary
        # longitude mean is the zonal-mean temperature.
        zonal_temperature = tas.mean(dim="lon").transpose("time", "lat").load()
        return ZonalTemperatureResult(
            filename=path.name,
            year=input_file.year,
            month=input_file.month,
            time=np.asarray(zonal_temperature["time"].values),
            latitude=np.asarray(zonal_temperature["lat"].values),
            temperature=np.asarray(zonal_temperature.values),
            dataset_attrs=source_dataset.attrs.copy() if input_file.is_reference else None,
            tas_attrs=tas.attrs.copy() if input_file.is_reference else None,
            time_attrs=tas["time"].attrs.copy() if input_file.is_reference else None,
            latitude_attrs=tas["lat"].attrs.copy() if input_file.is_reference else None,
        )


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Read-only directory containing CONTROL_360ppm_PLA.YYYY.MM.nc files.",
    ),
    output_dir: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
        help="Directory in which to create the derived zonal-temperature NetCDF file.",
    ),
    co2_ppm: float = typer.Option(..., min=0.0, help="Experiment CO2 concentration [ppm]."),
    solar_irradiance_w_m2: float = typer.Option(
        ..., min=0.0, help="Experiment solar irradiance [W m^-2]."
    ),
    workers: int = typer.Option(
        8,
        min=1,
        help="Number of monthly files to process concurrently.",
    ),
) -> None:
    """Create one zonal_T(time, lat) dataset from PLASIM monthly output."""

    # Keep only files whose name records a valid year and month.
    files: list[InputFile] = []
    for path in sorted(input_dir.glob("CONTROL_360ppm_PLA.*.*.nc")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            if path.stat().st_size == 0:
                typer.echo(f"Skipping empty file: {path.name}")
                continue
            files.append(
                InputFile(
                    path=path,
                    year=int(match["year"]),
                    month=int(match["month"]),
                )
            )

    if not files:
        raise typer.BadParameter(
            "No non-empty files matching CONTROL_360ppm_PLA.YYYY.MM.nc were found.",
            param_hint="--input-dir",
        )

    # The experiment settings make derived files from different runs distinct.
    output_name = (
        f"zonal_T_CO2_{_filename_value(co2_ppm)}_"
        f"mu_{_filename_value(solar_irradiance_w_m2)}.nc"
    )
    output_path = output_dir / output_name
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    # The first valid file supplies the output metadata. Input order remains
    # deterministic because ProcessPoolExecutor.map returns results in order.
    files[0] = InputFile(
        path=files[0].path,
        year=files[0].year,
        month=files[0].month,
        is_reference=True,
    )
    results: list[ZonalTemperatureResult] = []
    typer.echo(f"Processing {len(files)} files with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(
            executor.map(_extract_zonal_temperature, files, chunksize=1),
            start=1,
        ):
            results.append(result)
            typer.echo(f"Processed {result.filename} ({completed}/{len(files)})")

    reference = results[0]
    for result in results[1:]:
        if not np.array_equal(result.latitude, reference.latitude):
            raise ValueError(f"{result.filename}: latitude grid differs from the reference file.")

    # Construct one compact xarray object after all worker files are closed.
    time = np.concatenate([result.time for result in results])
    temperature = np.concatenate([result.temperature for result in results], axis=0)
    source_year = np.concatenate(
        [np.full(result.time.size, result.year, dtype=int) for result in results]
    )
    source_month = np.concatenate(
        [np.full(result.time.size, result.month, dtype=int) for result in results]
    )
    source_file = np.concatenate(
        [np.full(result.time.size, result.filename) for result in results]
    )
    zonal_T = xr.DataArray(
        temperature,
        dims=("time", "lat"),
        coords={
            "time": ("time", time),
            "lat": ("lat", reference.latitude),
            "source_year": ("time", source_year),
            "source_month": ("time", source_month),
            "source_file": ("time", source_file),
        },
        name="zonal_T",
        attrs=reference.tas_attrs or {},
    ).sortby("time")
    zonal_T["time"].attrs = reference.time_attrs or {}
    zonal_T["lat"].attrs = reference.latitude_attrs or {}
    zonal_T.attrs.update(
        long_name="zonal-mean 2 m air temperature",
        averaging="unweighted mean over longitude",
    )

    # Dataset-level attributes belong on the output Dataset, not on zonal_T.
    output_dataset = zonal_T.to_dataset()
    output_dataset.attrs = reference.dataset_attrs or {}
    output_dataset.attrs.update(
        zonal_mean_reference_file=reference.filename,
        zonal_mean_file_count=len(files),
        zonal_mean_processing="tas averaged uniformly over lon",
        experiment_co2_concentration_ppm=co2_ppm,
        experiment_solar_irradiance_w_m2=solar_irradiance_w_m2,
    )

    # This is the only write operation in the script.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset.to_netcdf(output_path, engine="h5netcdf")
    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
