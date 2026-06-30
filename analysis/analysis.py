import marimo

__generated_with = "0.23.9"
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
        get_repo_root,
        latitude_weighted_mean,
        meridional_heat_transfer_rate_watts_per_square_meter,
        plot_asymptotic_state_diagnostics,
        surface_albedo,
        warm_cold_state_albedo_from_dataset,
        warm_cold_state_heat_transfer_from_dataset,
    )

    from gsebm.time import DAY, YEAR

    return (
        DAY,
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
    warm_cold_dataset = xr.open_dataset(data_dir / "warm_cold_state_mu1.nc", engine="scipy")
    edge_dataset = xr.open_dataset(data_dir / "edge_state_mu1.nc", engine="scipy")
    return data_dir, edge_dataset, warm_cold_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Global temperature as a function of time
    """)
    return


@app.cell
def _(YEAR, latitude_weighted_mean, plt, warm_cold_dataset):
    global_temperature = latitude_weighted_mean(warm_cold_dataset)
    mean_temperature_fig, mean_temperature_ax = plt.subplots(figsize=(8, 4))
    mean_temperature_ax.plot(
        global_temperature["time"] / YEAR,
        global_temperature["warm_state_temperature"],
        color="red",
        label="Warm State",
    )
    mean_temperature_ax.plot(
        global_temperature["time"] / YEAR,
        global_temperature["cold_state_temperature"],
        color="blue",
        label="Cold State",
    )
    mean_temperature_ax.set_xlabel("Time [year]")
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
    # asymptotic_fig.savefig(get_repo_root() / "figures"/ "deterministic_profiles.png",dpi=300)
    asymptotic_fig
    return (edge_albedo,)


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
    return (
        bif_edge_ds,
        bif_wc_ds,
        global_asymp_temperature_edge,
        global_asymp_temperature_wc,
    )


@app.cell
def _(global_asymp_temperature_edge, global_asymp_temperature_wc, plt):
    _fig, _ax = plt.subplots()
    _ax.scatter(global_asymp_temperature_wc["mu"],global_asymp_temperature_wc["cold_state_temperature"])
    _ax.scatter(global_asymp_temperature_wc["mu"],global_asymp_temperature_wc["warm_state_temperature"])
    _ax.scatter(global_asymp_temperature_edge["mu"],global_asymp_temperature_edge["edge_state_temperature"])
    _ax.grid(alpha=0.4,linestyle='--')
    plt.show()
    return


@app.cell
def _(bif_edge_ds, bif_wc_ds):
    bif_wc_ds.close()
    bif_edge_ds.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stochastic Runs
    """)
    return


@app.cell
def _(data_dir, xr):
    warm_input_filename ="new_stochastic_warm_mu1.nc" #"stochastic_warm_state_mu0p97_{5}.nc"
    warm_ds = xr.open_dataset(data_dir / warm_input_filename)
    edge_state_ds = xr.open_dataset(data_dir / "edge_state_mu1.nc", engine="scipy")
    return edge_state_ds, warm_ds, warm_input_filename


@app.cell
def _(latitude_weighted_mean, warm_ds):
    avg_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1)
    eq_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1/3)
    pole_T = latitude_weighted_mean(warm_ds, xmin=1/3, xmax=1)
    Delta_T = eq_T - pole_T
    return Delta_T, avg_T


@app.cell
def _(YEAR, avg_T, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(avg_T["time"].values / YEAR,avg_T["temperature"].values)
    _ax.set_xlabel(xlabel=r"$t$ (years)",size=16)
    _ax.set_ylabel(ylabel=r"$\overline{T} $ (K) ",size=16)
    _ax.grid(alpha=0.4,linestyle='--')
    _ax.set_xlim(left=0,right=5000)
    # _ax.set_ylim(bottom=268,top=285)
    # _fig.savefig(get_repo_root() / "figures" /"warm_stochastic_trajectory_mu0p97.png",dpi=400)
    plt.show()
    return


@app.cell
def _(YEAR, warm_ds):
    transient = 500
    stop = 1e10

    condition = (
        ( warm_ds["time"] > transient * YEAR) &
        (warm_ds["time"] < stop * YEAR)

    )
    asymptotic_warm_ds = warm_ds.where(
        cond=condition,
        drop=True,
    )
    warm_asymptotic_temperature = asymptotic_warm_ds["temperature"].mean(dim="time")
    warm_asymptotic_temperature_std = asymptotic_warm_ds["temperature"].std(dim="time")
    return (
        asymptotic_warm_ds,
        condition,
        stop,
        transient,
        warm_asymptotic_temperature,
        warm_asymptotic_temperature_std,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation functions
    """)
    return


@app.cell
def _(Delta_T, avg_T, condition):
    from koopman_response.utils.signal import cross_correlation

    avg_T_asymp = avg_T.where(cond=condition,drop=True)
    Delta_T_asymp = Delta_T.where(cond=condition,drop=True)

    dt = avg_T_asymp.attrs["stochastic_dt"] # in seconds

    obs_avgT = avg_T_asymp["temperature"].values
    lags_avgT, corr_avgT = cross_correlation(x=obs_avgT,y=obs_avgT,dt=dt,normalization="biased")


    obs_DeltaT = Delta_T_asymp["temperature"].values
    lags_DeltaT, corr_DeltaT = cross_correlation(x=obs_DeltaT,y=obs_DeltaT,dt=dt,normalization="biased")
    return corr_DeltaT, corr_avgT, lags_DeltaT, lags_avgT


@app.cell
def _(YEAR, corr_DeltaT, corr_avgT, lags_DeltaT, lags_avgT, plt):
    def limits_with_zero_fraction(ax, zero_fraction=0.15):
          ymin, ymax = ax.get_ylim()

          ymax = max(ymax, 0)
          ymin = min(ymin, 0)

          above = ymax
          below = -ymin

          required_below = zero_fraction / (1 - zero_fraction) * above
          required_above = (1 - zero_fraction) / zero_fraction * below

          if below < required_below:
              ymin = -required_below
          if above < required_above:
              ymax = required_above

          return ymin, ymax


    def align_zero_yaxis(ax1, ax2, zero_fraction=0.15):
          ax1.set_ylim(*limits_with_zero_fraction(ax1, zero_fraction))
          ax2.set_ylim(*limits_with_zero_fraction(ax2, zero_fraction))


    _fig, _ax1 = plt.subplots()
    _ax2 = _ax1.twinx()

    _ax1.plot(lags_avgT / YEAR * 12, corr_avgT, color="tab:blue", label=r"$\overline{T}$")
    _ax2.plot(lags_DeltaT / YEAR * 12, corr_DeltaT, color="tab:red", label=r"$\Delta T$")

    _ax1.set_xlim(left=-1, right=12)

    align_zero_yaxis(_ax1, _ax2)


    _ax1.set_xlabel(r"$\mathrm{Lag}\;[\mathrm{months}]$")
    _ax1.set_ylabel(r"$C_{\overline{T}}$", color="tab:blue")
    _ax2.set_ylabel(r"$C_{\Delta T}$", color="tab:red")

    _ax1.tick_params(axis="y", labelcolor="tab:blue")
    _ax2.tick_params(axis="y", labelcolor="tab:red")

    _ax1.grid(alpha=0.3, linestyle="--")

    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2)

    _fig.tight_layout()
    _fig
    return


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
        warm_albedo,
        warm_albedo_profile,
        warm_albedo_profile_std,
        warm_heat_flux,
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
    )
    _temperature_ax.fill_between(
        _latitude,
        _temperature_mean - _temperature_std,
        _temperature_mean + _temperature_std,
        color="red",
        alpha=0.15,
    )
    _temperature_ax.set_ylabel("Temperature [K]")
    _temperature_ax.grid(True, alpha=0.3)

    _albedo_ax.plot(
        _latitude,
        _albedo_mean,
        color="red",
    )
    _albedo_ax.fill_between(
        _latitude,
        _albedo_mean - _albedo_std,
        _albedo_mean + _albedo_std,
        color="red",
        alpha=0.15,
    )
    _albedo_ax.set_ylabel("Albedo")
    _albedo_ax.grid(True, alpha=0.3)

    _flux_ax.plot(
        _latitude,
        _flux_mean,
        color="red",
    )
    _flux_ax.fill_between(
        _latitude,
        _flux_mean - _flux_std,
        _flux_mean + _flux_std,
        color="red",
        alpha=0.15,
    )
    _flux_ax.set_xlabel("Normalized latitude x")
    _flux_ax.set_ylabel(r"Heat flux $j$ [W m$^{-2}$]")

    _flux_ax.grid(True, alpha=0.3)
    _flux_ax.set_xlim(left=0, right=1)
    _flux_ax.set_ylim(bottom=0)
    _profile_fig.tight_layout()
    # _profile_fig.savefig(get_repo_root() / "figures" / "stochastic_avg_profiles_near.png", dpi=400)
    plt.show()
    return


@app.cell
def _(
    Delta_T,
    avg_T,
    condition,
    edge_state_ds,
    latitude_weighted_mean,
    np,
    plt,
):
    from matplotlib.colors import LogNorm
    from scipy.stats import gaussian_kde

    _fig, _ax = plt.subplots(figsize=(8, 6))

    asymptotic_Delta_T = Delta_T.where(cond=condition, drop=True)
    asymptotic_avg_T = avg_T.where(cond=condition, drop=True)

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



    edge_avg_T = latitude_weighted_mean(edge_state_ds, xmin=0, xmax=1)
    edge_eq_T = latitude_weighted_mean(edge_state_ds, xmin=0, xmax=1/3)
    edge_pole_T = latitude_weighted_mean(edge_state_ds, xmin=1/3, xmax=1)
    edge_Delta_T = edge_eq_T - edge_pole_T

    # _ax.scatter(x=edge_avg_T["edge_state_temperature"].item(), 
    #             y=edge_Delta_T["edge_state_temperature"].item(),
    #             marker="^",
    #             s=45,
    #             color="green"
    # )
    # _ax.set_xlim(xmin=290,right=305)
    # _ax.set_ylim(bottom=5,top=11)
    # _fig.savefig(get_repo_root() / "figures"/ "statistics_warm_near.png", dpi=400)
    plt.show()
    return asymptotic_Delta_T, asymptotic_avg_T, edge_Delta_T, edge_avg_T


@app.cell
def _(asymptotic_Delta_T, asymptotic_avg_T, edge_Delta_T, edge_avg_T, np, plt):

    _fig, _ax = plt.subplots(figsize=(8, 6))


    _x = asymptotic_avg_T["temperature"].values.ravel()
    _y = asymptotic_Delta_T["temperature"].values.ravel()
    _mask = np.isfinite(_x) & np.isfinite(_y)
    _x = _x[_mask]
    _y = _y[_mask]

    _ax.scatter(x=_x,y=_y,marker='.',s=25,color="blue")

    _ax.scatter(x=edge_avg_T["edge_state_temperature"].item(), 
                y=edge_Delta_T["edge_state_temperature"].item(),
                marker="^",
                s=50,
                color="red"
    )
    # _ax.set_xlim(xmin=290,right=305)
    # _ax.set_ylim(bottom=5,top=11)
    # _fig.savefig(get_repo_root() / "figures"/ "statistics_warm_near.png", dpi=400)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving the data
    """)
    return


@app.cell
def _(
    DAY,
    asymptotic_Delta_T,
    asymptotic_avg_T,
    asymptotic_warm_ds,
    data_dir,
    edge_albedo,
    edge_dataset,
    stop,
    transient,
    warm_albedo,
    warm_ds,
    warm_heat_flux,
    warm_input_filename,
    xr,
):
    dt = (asymptotic_warm_ds["time"].values[1] - asymptotic_warm_ds["time"].values[0] ) / DAY
    export_attrs = dict(warm_ds.attrs)
    export_attrs.update(
        {
            "title": "Derived asymptotic stochastic diagnostics",
            "source_dataset_filename": warm_input_filename,
            "source_dataset_path": str(data_dir / warm_input_filename),
            "output_dataset_filename": warm_input_filename,
            "output_dataset_path": str(data_dir / warm_input_filename),
            "analysis_transient_years": float(transient),
            "analysis_stop_years": float(stop),
            "tau [days]": dt,
            "latitude slicing" : "Northern Emisphere"
        }
    )
    export_dataset = xr.Dataset(
        data_vars={
            "edge_state_temperature" : edge_dataset.where(edge_dataset["latitude"] > 0,drop=True)["edge_state_temperature"],
            "edge_state_albedo" : edge_albedo.where(edge_albedo["latitude"]>0,drop=True),
            "asymptotic_temperature": asymptotic_warm_ds.where(asymptotic_warm_ds["latitude"]>0,drop=True)["temperature"],
            "warm_albedo": warm_albedo.where(warm_albedo["latitude"]>0,drop=True),
            "warm_heat_flux": warm_heat_flux.where(warm_heat_flux["latitude"] > 0,drop=True),
            "asymptotic_Delta_T": asymptotic_Delta_T["temperature"],
            "asymptotic_avg_T": asymptotic_avg_T["temperature"],
        },
        attrs=export_attrs,
    )

    output_path = data_dir / "koopman_data" / warm_input_filename
    export_dataset.to_netcdf(output_path, engine="scipy")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
