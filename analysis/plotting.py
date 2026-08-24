import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    from gsebm import (
        YEAR,
        build_ivp_operator_from_dataset,
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        meridional_heat_transfer_rate_watts_per_square_meter,
        surface_albedo,
    )

    return (
        YEAR,
        build_ivp_operator_from_dataset,
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        meridional_heat_transfer_rate_watts_per_square_meter,
        mo,
        np,
        plt,
        surface_albedo,
        xr,
    )


@app.cell
def _(get_data_dir):
    data_dir = get_data_dir()
    return (data_dir,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting stationary profiles
    """)
    return


@app.cell
def _():
    warm_filename = "new_stochastic_warm_mu1.nc"
    cold_filename = "new_stochastic_cold_mu1.nc"
    edge_filename = "edge_state_mu1.nc"

    transient_years = 500
    stop_years = 1e10
    return (
        cold_filename,
        edge_filename,
        stop_years,
        transient_years,
        warm_filename,
    )


@app.cell
def _(cold_filename, data_dir, edge_filename, warm_filename, xr):
    def open_loaded_dataset(filename: str):
        with xr.open_dataset(data_dir / filename, engine="scipy") as dataset:
            return dataset.load()

    warm_stochastic_ds = open_loaded_dataset(warm_filename)
    cold_stochastic_ds = open_loaded_dataset(cold_filename)
    deterministic_edge_ds = open_loaded_dataset(edge_filename)
    return cold_stochastic_ds, deterministic_edge_ds, warm_stochastic_ds


@app.cell
def _(
    YEAR,
    cold_stochastic_ds,
    stop_years,
    transient_years,
    warm_stochastic_ds,
):
    warm_final_year = float(warm_stochastic_ds["time"].max() / YEAR)
    cold_final_year = float(cold_stochastic_ds["time"].max() / YEAR)
    selected_stop_year = min(float(stop_years), warm_final_year, cold_final_year)
    selected_transient_year = float(transient_years)
    return selected_stop_year, selected_transient_year


@app.cell
def _(YEAR, selected_stop_year, selected_transient_year):
    def post_transient_window(dataset):
        return dataset.where(
            (dataset["time"] >= selected_transient_year * YEAR)
            & (dataset["time"] <= selected_stop_year * YEAR),
            drop=True,
        )

    return (post_transient_window,)


@app.cell
def _(cold_stochastic_ds, post_transient_window, warm_stochastic_ds):
    warm_post_transient_ds = post_transient_window(warm_stochastic_ds)
    cold_post_transient_ds = post_transient_window(cold_stochastic_ds)
    return cold_post_transient_ds, warm_post_transient_ds


@app.cell
def _(
    build_ivp_operator_from_dataset,
    meridional_heat_transfer_rate_watts_per_square_meter,
    np,
    surface_albedo,
    xr,
):
    def stochastic_profile_diagnostics(dataset):
        operator = build_ivp_operator_from_dataset(dataset)
        latitude = dataset["latitude"].values
        temperature = dataset["temperature"]

        albedo = xr.DataArray(
            surface_albedo(
                temperature.values,
                operator.empirical_fields.b_parameter[None, :],
                operator.empirical_fields.surface_height_offset[None, :],
                operator.params,
            ),
            dims=("time", "latitude"),
            coords={"time": dataset["time"], "latitude": dataset["latitude"]},
            name="albedo",
            attrs={"units": "1", "long_name": "post-transient albedo"},
        )
        temperature_x = np.gradient(temperature.values, latitude, axis=1)
        heat_flux = xr.DataArray(
            meridional_heat_transfer_rate_watts_per_square_meter(
                latitude[None, :],
                temperature.values,
                temperature_x,
                operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
                operator.empirical_fields.latent_heat_flux_coefficient[None, :],
                operator.params,
            ),
            dims=("time", "latitude"),
            coords={"time": dataset["time"], "latitude": dataset["latitude"]},
            name="heat_flux",
            attrs={
                "units": "W m^-2",
                "long_name": "post-transient meridional heat-transfer rate",
            },
        )

        return xr.Dataset(
            data_vars={
                "temperature": temperature,
                "albedo": albedo,
                "heat_flux": heat_flux,
            },
            coords={"time": dataset["time"], "latitude": dataset["latitude"]},
        )

    def stationary_profile_summary(diagnostics):
        return xr.Dataset(
            data_vars={
                "temperature_mean": diagnostics["temperature"].mean(dim="time"),
                "temperature_std": diagnostics["temperature"].std(dim="time"),
                "albedo_mean": diagnostics["albedo"].mean(dim="time"),
                "albedo_std": diagnostics["albedo"].std(dim="time"),
                "heat_flux_mean": diagnostics["heat_flux"].mean(dim="time"),
                "heat_flux_std": diagnostics["heat_flux"].std(dim="time"),
            }
        )

    return stationary_profile_summary, stochastic_profile_diagnostics


@app.cell
def _(
    cold_post_transient_ds,
    stationary_profile_summary,
    stochastic_profile_diagnostics,
    warm_post_transient_ds,
):
    warm_diagnostics = stochastic_profile_diagnostics(warm_post_transient_ds)
    cold_diagnostics = stochastic_profile_diagnostics(cold_post_transient_ds)
    warm_profiles = stationary_profile_summary(warm_diagnostics)
    cold_profiles = stationary_profile_summary(cold_diagnostics)
    return cold_profiles, warm_profiles


@app.cell
def _(
    deterministic_edge_ds,
    edge_state_albedo_from_dataset,
    edge_state_heat_transfer_from_dataset,
):
    edge_albedo = edge_state_albedo_from_dataset(deterministic_edge_ds)
    edge_heat_flux = edge_state_heat_transfer_from_dataset(deterministic_edge_ds)
    return edge_albedo, edge_heat_flux


@app.cell
def _(
    cold_profiles,
    deterministic_edge_ds,
    edge_albedo,
    edge_heat_flux,
    plt,
    warm_profiles,
):
    latitude = warm_profiles["latitude"]
    edge_latitude = deterministic_edge_ds["latitude"]
    profile_fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
    temperature_ax, albedo_ax, heat_flux_ax = axes.ravel()

    temperature_ax.plot(
        latitude,
        warm_profiles["temperature_mean"],
        color="red",
    )
    temperature_ax.fill_between(
        latitude,
        warm_profiles["temperature_mean"] - warm_profiles["temperature_std"],
        warm_profiles["temperature_mean"] + warm_profiles["temperature_std"],
        color="red",
        alpha=0.15,
        linewidth=0.0,
    )
    temperature_ax.plot(
        latitude,
        cold_profiles["temperature_mean"],
        color="blue",
    )
    temperature_ax.fill_between(
        latitude,
        cold_profiles["temperature_mean"] - cold_profiles["temperature_std"],
        cold_profiles["temperature_mean"] + cold_profiles["temperature_std"],
        color="blue",
        alpha=0.15,
        linewidth=0.0,
    )
    temperature_ax.plot(
        edge_latitude,
        deterministic_edge_ds["edge_state_temperature"],
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    temperature_ax.set_ylabel(r"$T \quad [\mathrm{K}]$",size=15)
    temperature_ax.grid(linestyle="--", alpha=0.4)


    albedo_ax.plot(
        latitude,
        warm_profiles["albedo_mean"],
        color="red",
    )
    albedo_ax.fill_between(
        latitude,
        warm_profiles["albedo_mean"] - warm_profiles["albedo_std"],
        warm_profiles["albedo_mean"] + warm_profiles["albedo_std"],
        color="red",
        alpha=0.15,
        linewidth=0.0,
    )
    albedo_ax.plot(
        latitude,
        cold_profiles["albedo_mean"],
        color="blue",
    )
    albedo_ax.fill_between(
        latitude,
        cold_profiles["albedo_mean"] - cold_profiles["albedo_std"],
        cold_profiles["albedo_mean"] + cold_profiles["albedo_std"],
        color="blue",
        alpha=0.15,
        linewidth=0.0,
    )
    albedo_ax.plot(
        edge_latitude,
        edge_albedo,
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    albedo_ax.set_ylabel(r"$\text{Albedo }$")
    albedo_ax.grid(linestyle="--", alpha=0.4)

    heat_flux_ax.plot(
        latitude,
        warm_profiles["heat_flux_mean"],
        color="red",
    )
    heat_flux_ax.fill_between(
        latitude,
        warm_profiles["heat_flux_mean"] - warm_profiles["heat_flux_std"],
        warm_profiles["heat_flux_mean"] + warm_profiles["heat_flux_std"],
        color="red",
        alpha=0.15,
        linewidth=0.0,
    )
    heat_flux_ax.plot(
        latitude,
        cold_profiles["heat_flux_mean"],
        color="blue",
    )
    heat_flux_ax.fill_between(
        latitude,
        cold_profiles["heat_flux_mean"] - cold_profiles["heat_flux_std"],
        cold_profiles["heat_flux_mean"] + cold_profiles["heat_flux_std"],
        color="blue",
        alpha=0.15,
        linewidth=0.0,
    )
    heat_flux_ax.plot(
        edge_latitude,
        edge_heat_flux,
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    heat_flux_ax.set_xlim(left=0.0, right=1.0)
    heat_flux_ax.set_xlabel(r"$x$",size=15)
    heat_flux_ax.set_ylabel(r"$ j \quad [\mathrm{W}\,\mathrm{m}^{-2}]$",size=15)
    heat_flux_ax.grid(linestyle="--", alpha=0.4)
    heat_flux_ax.set_ylim(bottom=-0.5)

    profile_fig.tight_layout()
    profile_fig
    # profile_fig.savefig("figures/stochastic_profiles_mu1.png",dpi=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sensitivity profiles
    """)
    return


@app.cell
def _(data_dir, np, xr):
    sensitivity_filenames = (
        "sensitivity_co2_mu1.nc",
        "sensitivity_co2_mu0p972.nc",
    )

    sensitivity_datasets = []
    for _filename in sensitivity_filenames:
        with xr.open_dataset(data_dir / _filename, engine="scipy") as _dataset:
            _loaded_dataset = _dataset.load()

        if "latitude" not in _loaded_dataset:
            raise ValueError(f"{_filename} is missing the latitude coordinate.")
        if "sensitivity" not in _loaded_dataset:
            raise ValueError(f"{_filename} is missing the sensitivity variable.")
        if "sensitivity_kdmd" not in _loaded_dataset:
            raise ValueError(
                f"{_filename} is missing the sensitivity_kdmd variable. "
                "Rerun the sensitivity export in analysis/analysis.py."
            )

        _latitude = np.asarray(_loaded_dataset["latitude"].values, dtype=float)
        _sensitivity = np.asarray(_loaded_dataset["sensitivity"].values, dtype=float)
        _sensitivity_kdmd = np.asarray(
            _loaded_dataset["sensitivity_kdmd"].values,
            dtype=float,
        )
        if _latitude.ndim != 1 or _sensitivity.ndim != 1 or _sensitivity_kdmd.ndim != 1:
            raise ValueError(f"{_filename} must contain 1D sensitivity arrays.")
        if _latitude.shape != _sensitivity.shape or _latitude.shape != _sensitivity_kdmd.shape:
            raise ValueError(f"{_filename} latitude and sensitivity shapes do not match.")
        if (
            not np.isfinite(_latitude).all()
            or not np.isfinite(_sensitivity).all()
            or not np.isfinite(_sensitivity_kdmd).all()
        ):
            raise ValueError(f"{_filename} contains non-finite sensitivity values.")

        sensitivity_datasets.append(_loaded_dataset)

    sensitivity_datasets = tuple(sensitivity_datasets)
    return sensitivity_datasets, sensitivity_filenames


@app.cell
def _(np, plt, sensitivity_datasets, sensitivity_filenames):
    sensitivity_fig, sensitivity_ax = plt.subplots(figsize=(8, 5))
    stride = 2
    for _dataset, _filename in zip(sensitivity_datasets, sensitivity_filenames):
        _latitude = np.asarray(_dataset["latitude"].values, dtype=float)
        _sensitivity = np.asarray(_dataset["sensitivity"].values, dtype=float) 
        _sensitivity_kdmd = np.asarray(_dataset["sensitivity_kdmd"].values, dtype=float) 
        _positive_hemisphere = _latitude >= 0
        _mu = _dataset.attrs.get("param_mu")
        _label = rf"$\mu={float(_mu):g}$" if _mu is not None else _filename.removesuffix(".nc")

        (_line,) = sensitivity_ax.plot(
            _latitude[_positive_hemisphere],
            _sensitivity[_positive_hemisphere] / np.max(_sensitivity[_positive_hemisphere]),
            lw=2,
            label=rf"{_label}",
        )
        sensitivity_ax.scatter(
            _latitude[_positive_hemisphere][::stride],
            _sensitivity_kdmd[_positive_hemisphere][::stride] / np.max(_sensitivity_kdmd[_positive_hemisphere]),
            color=_line.get_color(),
            marker=".",
            lw=2,
        )

        if "ice_line_latitude" in _dataset:
            _ice_line_latitudes = np.asarray(
                _dataset["ice_line_latitude"].values,
                dtype=float,
            )
            _positive_ice_lines = _ice_line_latitudes[
                np.isfinite(_ice_line_latitudes) & (_ice_line_latitudes >= 0)
            ]
            for _ice_line_latitude in _positive_ice_lines:
                sensitivity_ax.axvline(
                    _ice_line_latitude,
                    color=_line.get_color(),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=1,
                )

    sensitivity_ax.set_xlabel(r"$x$", size=15)
    sensitivity_ax.set_ylabel(
        r"$\varepsilon \int G_T(x,t)\,dt \; [\mathrm{K}]$",
        size=15,
    )
    sensitivity_ax.set_xlim(left=0.0)
    sensitivity_ax.grid(linestyle="--", alpha=0.3)
    sensitivity_ax.legend(frameon=False)
    sensitivity_fig.tight_layout()
    sensitivity_fig
    sensitivity_fig.savefig("../figures/sensitivity_profiles_co2.png", dpi=400)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
