import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from gsebm.paths import get_data_dir

    return Path, get_data_dir, mo, np, plt, re, xr


@app.cell
def _(Path, get_data_dir, re):
    """Find PLASIM monthly output files and record their date from the name."""

    plasim_file_pattern = re.compile(
        r"CONTROL_360ppm_PLA\.(?P<year>\d{4})\.(?P<month>\d{2})\.nc$"
    )
    plasim_data_dir = get_data_dir(create=False)

    plasim_files: list[tuple[Path, int, int]] = []
    for candidate in sorted(plasim_data_dir.glob("CONTROL_360ppm_PLA.*.*.nc")):
        match = plasim_file_pattern.fullmatch(candidate.name)
        if match is not None:
            plasim_files.append(
                (candidate, int(match["year"]), int(match["month"]))
            )

    if not plasim_files:
        raise FileNotFoundError(
            "No PLASIM files matching CONTROL_360ppm_PLA.YYYY.MM.nc found in "
            f"{plasim_data_dir}."
        )
    return (plasim_files,)


@app.cell
def _(plasim_files: "list[tuple[Path, int, int]]"):
    plasim_files
    return


@app.cell
def _():
    # Keep these values synchronized with the PLASIM experiment configuration.
    # The NetCDF output records neither value as global metadata.
    experiment_co2_concentration_ppm = 360.0
    experiment_solar_irradiance_w_m2 = 1365.0
    return experiment_co2_concentration_ppm, experiment_solar_irradiance_w_m2


@app.cell
def _(
    experiment_co2_concentration_ppm,
    experiment_solar_irradiance_w_m2,
    np,
    plasim_files: "list[tuple[Path, int, int]]",
    xr,
):
    """Extract the 2 m zonal-mean temperature from every monthly file.

    ``source_year`` and ``source_month`` identify the file from which every
    timestep originated. Each source file is closed once its reduced
    time-latitude field has been loaded, so full longitude fields are not kept
    in memory when many files are processed.
    """

    zonal_temperature_by_file = []
    reference_dataset_attrs = None
    reference_tas_attrs = None
    for file_path, year, month in plasim_files:
        with xr.open_dataset(file_path) as source_dataset:
            tas = source_dataset["tas"]
            required_dimensions = {"time", "lat", "lon"}
            missing_dimensions = required_dimensions.difference(tas.dims)
            if missing_dimensions:
                raise ValueError(
                    f"{file_path.name}: expected tas dimensions including "
                    f"{sorted(required_dimensions)}, got {tas.dims}."
                )

            if reference_dataset_attrs is None:
                reference_dataset_attrs = source_dataset.attrs.copy()
                reference_tas_attrs = tas.attrs.copy()

            zonal_temperature = tas.mean(dim="lon").assign_coords(
                source_year=("time", np.full(tas.sizes["time"], year, dtype=int)),
                source_month=("time", np.full(tas.sizes["time"], month, dtype=int)),
                source_file=("time", np.full(tas.sizes["time"], file_path.name)),
            )
            zonal_temperature.attrs = tas.attrs.copy()
            zonal_temperature_by_file.append(zonal_temperature.load())

    zonal_T = xr.concat(zonal_temperature_by_file, dim="time").sortby("time")
    zonal_T.name = "zonal_T"
    zonal_T.attrs = reference_tas_attrs
    zonal_T.attrs.update(
        long_name="zonal-mean 2 m air temperature",
        averaging="unweighted mean over longitude",
    )

    zonal_temperature_dataset = zonal_T.to_dataset()
    zonal_temperature_dataset.attrs = reference_dataset_attrs
    zonal_temperature_dataset.attrs.update(
        zonal_mean_reference_file=plasim_files[0][0].name,
        zonal_mean_file_count=len(plasim_files),
        zonal_mean_processing="tas averaged uniformly over lon",
        experiment_co2_concentration_ppm=experiment_co2_concentration_ppm,
        experiment_solar_irradiance_w_m2=experiment_solar_irradiance_w_m2,
    )
    return zonal_T, zonal_temperature_dataset


@app.cell
def _(zonal_temperature_dataset):
    zonal_temperature_dataset
    return


@app.cell
def _(mo, plasim_files: "list[tuple[Path, int, int]]"):
    mo.md(
        f"""
        Loaded **{len(plasim_files)}** PLASIM monthly file(s), producing
        `zonal_T(time, lat)`. The `source_year`, `source_month`, and
        `source_file` coordinates retain the origin of each timestep.
        """
    )
    return


@app.cell
def _(plt, zonal_T):
    _fig, _ax = plt.subplots()
    zonal_T.isel(time=0).plot(ax=_ax)
    _ax.set_title("Zonal-mean 2 m air temperature: first available timestep")
    _ax.set_xlabel("Latitude [degrees north]")
    _ax.set_ylabel("Temperature [K]")
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell
def _(np, plt, zonal_T):
    lat_weights = np.cos(np.deg2rad(zonal_T["lat"]))
    global_T = zonal_T.weighted(lat_weights).mean(dim="lat")

    _fig, _ax = plt.subplots()
    global_T.plot(ax=_ax)
    _ax.set_title("Area-weighted global 2 m air temperature")
    _ax.set_xlabel("Time")
    _ax.set_ylabel("Temperature [K]")
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
