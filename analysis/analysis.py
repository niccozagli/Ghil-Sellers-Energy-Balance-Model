import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np

    from gsebm import (
        build_ivp_operator_from_dataset,
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        latitude_weighted_mean,
        meridional_heat_transfer_rate_watts_per_square_meter,
        plot_asymptotic_state_diagnostics,
        surface_albedo,
        warm_cold_state_albedo_from_dataset,
        warm_cold_state_heat_transfer_from_dataset,
    )

    from gsebm.time import DAY, YEAR

    return (
        YEAR,
        build_ivp_operator_from_dataset,
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        latitude_weighted_mean,
        meridional_heat_transfer_rate_watts_per_square_meter,
        mo,
        np,
        plot_asymptotic_state_diagnostics,
        plt,
        surface_albedo,
        warm_cold_state_albedo_from_dataset,
        warm_cold_state_heat_transfer_from_dataset,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # GSEBM Analysis

    This app reads the saved NetCDF outputs and reconstructs albedo and
    meridional heat-transfer diagnostics from the dataset metadata.
    """)
    return


@app.cell
def _(get_data_dir, xr):
    data_dir = get_data_dir()
    warm_cold_dataset = xr.open_dataset(data_dir / "warm_cold_state.nc", engine="scipy")
    edge_dataset = xr.open_dataset(data_dir / "edge_state.nc", engine="scipy")
    return data_dir, edge_dataset, warm_cold_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Global temperature as a function of time
    """)
    return


@app.cell
def _(latitude_weighted_mean, plt, warm_cold_dataset):
    global_temperature = latitude_weighted_mean(warm_cold_dataset)
    mean_temperature_fig, mean_temperature_ax = plt.subplots(figsize=(8, 4))
    mean_temperature_ax.plot(
        global_temperature["time"],
        global_temperature["warm_state_temperature"],
        color="red",
        label="Warm State",
    )
    mean_temperature_ax.plot(
        global_temperature["time"],
        global_temperature["cold_state_temperature"],
        color="blue",
        label="Cold State",
    )
    mean_temperature_ax.set_xlabel("Time [s]")
    mean_temperature_ax.set_ylabel("Mean temperature [K]")
    mean_temperature_ax.grid(True, alpha=0.3)
    mean_temperature_ax.legend()
    mean_temperature_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Asymptotic state diagnostics
    """)
    return


@app.cell
def _(
    edge_dataset,
    edge_state_albedo_from_dataset,
    edge_state_heat_transfer_from_dataset,
    plot_asymptotic_state_diagnostics,
    warm_cold_dataset,
    warm_cold_state_albedo_from_dataset,
    warm_cold_state_heat_transfer_from_dataset,
):
    warm_cold_albedo = warm_cold_state_albedo_from_dataset(dataset=warm_cold_dataset)
    edge_albedo = edge_state_albedo_from_dataset(dataset=edge_dataset)
    warm_cold_heat_transfer = warm_cold_state_heat_transfer_from_dataset(dataset=warm_cold_dataset)
    edge_heat_transfer = edge_state_heat_transfer_from_dataset(dataset=edge_dataset)

    asymptotic_states = warm_cold_dataset.isel(time=-1)
    asymptotic_albedo = warm_cold_albedo.isel(time=-1)
    asymptotic_heat_transfer = warm_cold_heat_transfer.isel(time=-1)

    asymptotic_fig = plot_asymptotic_state_diagnostics(
        latitude=asymptotic_states["latitude"],
        warm_temperature=asymptotic_states["warm_state_temperature"],
        cold_temperature=asymptotic_states["cold_state_temperature"],
        edge_latitude=edge_dataset["latitude"],
        edge_temperature=edge_dataset["edge_state_temperature"],
        warm_albedo=asymptotic_albedo["warm_state_albedo"],
        cold_albedo=asymptotic_albedo["cold_state_albedo"],
        edge_albedo=edge_albedo,
        warm_heat_transfer=asymptotic_heat_transfer["warm_state_heat_transfer"],
        cold_heat_transfer=asymptotic_heat_transfer["cold_state_heat_transfer"],
        edge_heat_transfer=edge_heat_transfer,
    ) 
    asymptotic_fig
    return


@app.cell
def _(edge_dataset, warm_cold_dataset):
    warm_cold_dataset.close()
    edge_dataset.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bifurcation analysis
    """)
    return


@app.cell
def _(data_dir, latitude_weighted_mean, xr):
    bif_wc_ds = xr.open_dataset(filename_or_obj=data_dir / "warm_cold_mu_bifurcation.nc")
    bif_edge_ds = xr.open_dataset(filename_or_obj=data_dir / "edge_mu_bifurcation.nc")
    global_asymp_temperature_wc = latitude_weighted_mean(bif_wc_ds).isel(time=-1)
    global_asymp_temperature_edge = latitude_weighted_mean(bif_edge_ds)
    return global_asymp_temperature_edge, global_asymp_temperature_wc


@app.cell
def _(global_asymp_temperature_edge, global_asymp_temperature_wc, plt):
    _fig, _ax = plt.subplots()
    _ax.scatter(global_asymp_temperature_wc["mu"],global_asymp_temperature_wc["cold_state_temperature"])
    _ax.scatter(global_asymp_temperature_wc["mu"],global_asymp_temperature_wc["warm_state_temperature"])
    _ax.scatter(global_asymp_temperature_edge["mu"],global_asymp_temperature_edge["edge_state_temperature"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stochastic Runs
    """)
    return


@app.cell
def _(data_dir, xr):
    warm_ds = xr.open_dataset(data_dir / "stochastic_warm_state.nc")
    return (warm_ds,)


@app.cell
def _(YEAR, latitude_weighted_mean, warm_ds):
    transient = 100* YEAR
    avg_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1)
    eq_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1/3)
    pole_T = latitude_weighted_mean(warm_ds, xmin=1/3, xmax=1)
    Delta_T = eq_T - pole_T
    return Delta_T, avg_T, transient


@app.cell
def _(YEAR, avg_T, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(avg_T["time"].values / YEAR,avg_T["temperature"].values)
    _ax.set_xlabel(xlabel=r"$t$ (years)",size=16)
    _ax.set_ylabel(ylabel=r"$\overline{T} $ (K) ",size=16)
    _ax.grid(alpha=0.4,linestyle='--')
    _ax.set_xlim(left=-5,right=500)
    plt.show()
    return


@app.cell
def _(transient, warm_ds):
    asymptotic_warm_ds = warm_ds.where(
        warm_ds["time"] > transient,
        drop=True,
    )
    warm_asymptotic_temperature = asymptotic_warm_ds["temperature"].mean(dim="time")
    warm_asymptotic_temperature_std = asymptotic_warm_ds["temperature"].std(dim="time")
    return (
        asymptotic_warm_ds,
        warm_asymptotic_temperature,
        warm_asymptotic_temperature_std,
    )


@app.cell
def _(
    asymptotic_warm_ds,
    build_ivp_operator_from_dataset,
    meridional_heat_transfer_rate_watts_per_square_meter,
    np,
    surface_albedo,
    warm_ds,
    xr,
):
    warm_operator = build_ivp_operator_from_dataset(warm_ds)
    warm_albedo = xr.DataArray(
        surface_albedo(
            asymptotic_warm_ds["temperature"].values,
            warm_operator.empirical_fields.b_parameter[None, :],
            warm_operator.empirical_fields.surface_height_offset[None, :],
            warm_operator.params,
        ),
        dims=("time", "latitude"),
        coords={
            "time": asymptotic_warm_ds["time"],
            "latitude": warm_ds["latitude"],
        },
        name="temperature_albedo",
        attrs={"units": "1", "long_name": "post-transient albedo"},
    )
    warm_temperature_x = np.gradient(
        asymptotic_warm_ds["temperature"].values,
        warm_ds["latitude"].values,
        axis=1,
    )
    warm_heat_flux = xr.DataArray(
        meridional_heat_transfer_rate_watts_per_square_meter(
            warm_ds["latitude"].values[None, :],
            asymptotic_warm_ds["temperature"].values,
            warm_temperature_x,
            warm_operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
            warm_operator.empirical_fields.latent_heat_flux_coefficient[None, :],
            warm_operator.params,
        ),
        dims=("time", "latitude"),
        coords={
            "time": asymptotic_warm_ds["time"],
            "latitude": warm_ds["latitude"],
        },
        name="temperature_heat_flux",
        attrs={"units": "W m^-2", "long_name": "post-transient meridional heat-transfer rate"},
    )
    warm_albedo_profile = warm_albedo.mean(dim="time")
    warm_albedo_profile_std = warm_albedo.std(dim="time")
    warm_heat_flux_profile = warm_heat_flux.mean(dim="time")
    warm_heat_flux_profile_std = warm_heat_flux.std(dim="time")
    return (
        warm_albedo_profile,
        warm_albedo_profile_std,
        warm_heat_flux_profile,
        warm_heat_flux_profile_std,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Asymptotic Stochastic Profiles
    """)
    return


@app.cell
def _(
    YEAR,
    plt,
    transient,
    warm_albedo_profile,
    warm_albedo_profile_std,
    warm_asymptotic_temperature,
    warm_asymptotic_temperature_std,
    warm_ds,
    warm_heat_flux_profile,
    warm_heat_flux_profile_std,
):
    _transient_years = transient / YEAR
    _latitude = warm_ds["latitude"].values
    _temperature_mean = warm_asymptotic_temperature.values
    _temperature_std = warm_asymptotic_temperature_std.values
    _albedo_mean = warm_albedo_profile.values
    _albedo_std = warm_albedo_profile_std.values
    _flux_mean = warm_heat_flux_profile.values
    _flux_std = warm_heat_flux_profile_std.values
    _profile_fig, (_temperature_ax, _albedo_ax, _flux_ax) = plt.subplots(
        3,
        1,
        figsize=(8, 9),
        sharex=True,
    )
    _temperature_ax.plot(
        _latitude,
        _temperature_mean,
        color="red",
        label="Warm Stochastic Run",
    )
    _temperature_ax.fill_between(
        _latitude,
        _temperature_mean - _temperature_std,
        _temperature_mean + _temperature_std,
        color="red",
        alpha=0.15,
    )
    _temperature_ax.set_ylabel("Temperature [K]")
    _temperature_ax.set_title(
        f"Mean temperature profile for t > {_transient_years:.0f} years"
    )
    _temperature_ax.legend()
    _temperature_ax.grid(True, alpha=0.3)

    _albedo_ax.plot(
        _latitude,
        _albedo_mean,
        color="red",
        label="Warm Stochastic Run",
    )
    _albedo_ax.fill_between(
        _latitude,
        _albedo_mean - _albedo_std,
        _albedo_mean + _albedo_std,
        color="red",
        alpha=0.15,
    )
    _albedo_ax.set_ylabel("Albedo [-]")
    _albedo_ax.set_title(
        f"Mean albedo profile for t > {_transient_years:.0f} years"
    )
    _albedo_ax.legend()
    _albedo_ax.grid(True, alpha=0.3)

    _flux_ax.plot(
        _latitude,
        _flux_mean,
        color="red",
        label="Warm Stochastic Run",
    )
    _flux_ax.fill_between(
        _latitude,
        _flux_mean - _flux_std,
        _flux_mean + _flux_std,
        color="red",
        alpha=0.15,
    )
    _flux_ax.set_xlabel("Normalized latitude x [-]")
    _flux_ax.set_ylabel(r"Heat flux $j$ [W m$^{-2}$]")
    _flux_ax.set_title(
        f"Mean meridional heat-transfer rate for t > {_transient_years:.0f} years"
    )
    _flux_ax.legend()
    _flux_ax.grid(True, alpha=0.3)
    _flux_ax.set_xlim(left=0, right=1)
    _flux_ax.set_ylim(bottom=0)
    _profile_fig.tight_layout()
    plt.show()
    return


@app.cell
def _(Delta_T, avg_T, np, plt, transient):
    from matplotlib.colors import LogNorm
    from scipy.stats import gaussian_kde

    _fig, _ax = plt.subplots(figsize=(8, 6))

    asymptotic_Delta_T = Delta_T.where(Delta_T["time"] > transient, drop=True)
    asymptotic_avg_T = avg_T.where(avg_T["time"] > transient, drop=True)

    _x = asymptotic_avg_T["temperature"].values.ravel()
    _y = asymptotic_Delta_T["temperature"].values.ravel()
    _mask = np.isfinite(_x) & np.isfinite(_y)
    _x = _x[_mask]
    _y = _y[_mask]

    _x_pad = 0.1 * (_x.max() - _x.min())
    _y_pad = 0.1 * (_y.max() - _y.min())

    _x_grid = np.linspace(_x.min() - _x_pad, _x.max() + _x_pad, 300)
    _y_grid = np.linspace(_y.min() - _y_pad, _y.max() + _y_pad, 300)


    _X, _Y = np.meshgrid(_x_grid, _y_grid)

    _samples = np.vstack([_x, _y])
    _kde = gaussian_kde(_samples, bw_method=0.15)
    _Z = _kde(np.vstack([_X.ravel(), _Y.ravel()])).reshape(_X.shape)

    _levels = np.array([1e-6,5e-6,1e-5,5e-5,1e-4, 5e-4, 1e-3, 5e-3, 1e-2,5e-2,1e-1])
    _Z = np.ma.masked_less(_Z, _levels[0])

    _contour = _ax.contourf(
        _X,
        _Y,
        _Z,
        levels=_levels,
        cmap="coolwarm",
        norm=LogNorm(vmin=_levels[0], vmax=_levels[-1]),
        extend="max",
    )
    _ax.contour(
        _X,
        _Y,
        _Z,
        levels=_levels,
        colors="white",
        linewidths=0.6,
        alpha=0.4,
    )

    _colorbar = _fig.colorbar(_contour, ax=_ax, ticks=_levels)
    _colorbar.set_ticklabels([f"{_level:.0e}" for _level in _levels])
    _ax.set_xlabel(r"$\overline{T} [K]$",size=16)
    _ax.set_ylabel(r"$\Delta T [K]$",size=16)
    # _ax.set_xlim(xmin=290,right=305)
    # _ax.set_ylim(bottom=5,top=11)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
