"""Extract zonal-mean 2 m temperature from PLASIM monthly NetCDF output."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import typer
import xarray as xr

app = typer.Typer(add_completion=False, no_args_is_help=True)

FILE_PATTERN = re.compile(
    r"CONTROL_360ppm_PLA\.(?P<year>\d{4})\.(?P<month>\d{2})\.nc$"
)


def _filename_value(value: float) -> str:
    """Format a numeric experiment setting compactly for a filename."""

    return f"{value:g}"


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
) -> None:
    """Create one zonal_T(time, lat) dataset from PLASIM monthly output."""

    # Keep only files whose name records a valid year and month.
    files: list[tuple[Path, int, int]] = []
    for path in sorted(input_dir.glob("CONTROL_360ppm_PLA.*.*.nc")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            if path.stat().st_size == 0:
                typer.echo(f"Skipping empty file: {path.name}")
                continue
            files.append((path, int(match["year"]), int(match["month"])))

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

    # Each item held here is small: time by latitude, after longitude reduction.
    zonal_temperatures = []
    reference_dataset_attrs: dict[str, object] | None = None
    reference_tas_attrs: dict[str, object] | None = None

    for path, year, month in files:
        typer.echo(f"Reading {path.name}")
        # xarray.open_dataset opens input datasets read-only. This script only
        # calls to_netcdf below for output_path, never for files in input_dir.
        with xr.open_dataset(path, engine="h5netcdf") as source_dataset:
            tas = source_dataset["tas"]
            required_dimensions = {"time", "lat", "lon"}
            missing_dimensions = required_dimensions.difference(tas.dims)
            if missing_dimensions:
                raise ValueError(
                    f"{path.name}: expected tas dimensions including "
                    f"{sorted(required_dimensions)}, got {tas.dims}."
                )

            if reference_dataset_attrs is None:
                # Use the first monthly file as the metadata reference.
                reference_dataset_attrs = source_dataset.attrs.copy()
                reference_tas_attrs = tas.attrs.copy()

            # Longitude cells have equal area at a fixed latitude, so the
            # ordinary longitude mean is the zonal-mean temperature.
            zonal_temperature = tas.mean(dim="lon").assign_coords(
                source_year=("time", np.full(tas.sizes["time"], year, dtype=int)),
                source_month=("time", np.full(tas.sizes["time"], month, dtype=int)),
                source_file=("time", np.full(tas.sizes["time"], path.name)),
            )
            zonal_temperature.attrs = tas.attrs.copy()
            # Load only the reduced field before closing this source file.
            zonal_temperatures.append(zonal_temperature.load())

    # Combine every monthly field into one time series and retain source dates.
    zonal_T = xr.concat(zonal_temperatures, dim="time").sortby("time")
    zonal_T.name = "zonal_T"
    zonal_T.attrs = reference_tas_attrs or {}
    zonal_T.attrs.update(
        long_name="zonal-mean 2 m air temperature",
        averaging="unweighted mean over longitude",
    )

    # Dataset-level attributes belong on the output Dataset, not on zonal_T.
    output_dataset = zonal_T.to_dataset()
    output_dataset.attrs = reference_dataset_attrs or {}
    output_dataset.attrs.update(
        zonal_mean_reference_file=files[0][0].name,
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
