import marimo

__generated_with = "0.22.0"
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
    return edge_dataset, warm_cold_dataset


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


if __name__ == "__main__":
    app.run()
