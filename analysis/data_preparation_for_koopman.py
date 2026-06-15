import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    from matplotlib import pyplot as plt

    from gsebm import (
        get_data_dir,
    )

    from gsebm.time import DAY, YEAR

    return DAY, YEAR, get_data_dir, mo, plt, xr


@app.cell
def _(mo):
    mo.md(r"""
    # GSEBM data preparation
    """)
    return


@app.cell
def _(get_data_dir, xr):
    data_dir = get_data_dir()
    fname = "stochastic_warm_state_{5}.nc"
    ds = xr.open_dataset(filename_or_obj=data_dir / fname)
    return data_dir, ds, fname


@app.cell
def _(YEAR, ds, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(ds["time"]/YEAR,ds.mean(dim="latitude")["temperature"])
    plt.show()
    return


@app.cell
def _(YEAR, ds):
    # Select only north emisphere
    _ds = ds.where(ds["latitude"] > 0, drop=True)
    # Only asymptotic values
    transient_year = 500
    stop_before = 1e10
    condition = (
        (_ds["time"] > transient_year * YEAR) &
        (_ds["time"] < stop_before * YEAR)
    )
    ds_asymptotic = _ds.where( condition ,drop=True)
    return ds_asymptotic, stop_before, transient_year


@app.cell
def _(DAY, ds_asymptotic):
    t = ds_asymptotic["time"].values
    Delta_t  = ( t[2] - t[1]  ) / DAY
    return (Delta_t,)


@app.cell
def _(Delta_t, ds_asymptotic, stop_before, transient_year):
    ds_asymptotic.attrs["transient [years]"] = transient_year
    ds_asymptotic.attrs["tau [days]"] = Delta_t
    ds_asymptotic.attrs["latitude slicing"] = "Northern Emisphere"
    ds_asymptotic.attrs["transient [years]"] = transient_year
    ds_asymptotic.attrs["stop [years]"] = stop_before
    return


@app.cell
def _(data_dir, ds_asymptotic, fname):
    ds_asymptotic.to_netcdf(data_dir / "koopman_data" / fname)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
