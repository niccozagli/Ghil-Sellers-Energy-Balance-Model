import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import xarray as xr

    from gsebm import (
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        plot_asymptotic_state_diagnostics,
        warm_cold_state_albedo_from_dataset,
        warm_cold_state_heat_transfer_from_dataset,
    )

    return (
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        mo,
        plot_asymptotic_state_diagnostics,
        plt,
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
def _(plt, warm_cold_dataset):
    global_temperature = warm_cold_dataset.mean(dim="latitude")
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
def _(data_dir, xr):
    bif_wc_ds = xr.open_dataset(filename_or_obj=data_dir / "warm_cold_mu_bifurcation.nc")
    bif_edge_ds = xr.open_dataset(filename_or_obj=data_dir / "edge_mu_bifurcation.nc")
    global_asymp_temperature_wc = bif_wc_ds.mean(dim="latitude").isel(time=-1)
    global_asymp_temperature_edge = bif_edge_ds.mean(dim="latitude")
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
    return


@app.cell
def _(bif_wc_ds):
    asymp = bif_wc_ds.isel(time=-1)
    return (asymp,)


@app.cell
def _(asymp, bif_edge_ds, plt):
    _ds = asymp.where( abs(asymp["mu"] < 1.1) < 1e-3,drop=True)

    _ds = _ds.isel(mu=2)
    _asymp = asymp.isel(mu=0)

    mu_far = _asymp["mu"].item()
    mu_close = _ds["mu"].item()


    _closest_edge_state= bif_edge_ds.where(abs( bif_edge_ds["mu"]-mu_close ) < 1e-3 ,drop=True).isel(mu=0)
    _fig, _ax = plt.subplots(nrows=2,sharex=True,sharey=True)

    _ax[0].plot(_ds["latitude"],_asymp["cold_state_temperature"],color="blue")
    _ax[1].plot(_ds["latitude"],_ds["cold_state_temperature"],color="blue")
    _ax[1].plot(_closest_edge_state["latitude"],_closest_edge_state["edge_state_temperature"],color="green",linestyle='--')

    _ax[0].set_title("Far from bifurcation")
    _ax[1].set_title("Close to bifurcation")
    _ax[1].set_xlim(left=0,right=1)



    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
