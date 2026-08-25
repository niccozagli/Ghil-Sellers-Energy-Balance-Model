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

    return get_data_dir, mo, np, plt, xr


@app.cell
def _(get_data_dir, np, xr):
    data_dir = get_data_dir()
    fname = "zonal_T_CO2_360_mu_1365.nc"

    zonal_T = xr.open_dataset(data_dir / fname)["zonal_T"]

    # PLASIM's ``lat`` coordinate is a Gaussian grid in degrees north.  Do
    # not use gsebm.latitude_weighted_mean here: it expects the model's
    # normalized ``latitude`` coordinate on [-1, 1].
    gaussian_nodes, gaussian_weights = np.polynomial.legendre.leggauss(zonal_T.sizes["lat"])
    latitudes_from_weights = np.degrees(np.arcsin(gaussian_nodes))
    if not np.allclose(zonal_T["lat"].values, latitudes_from_weights[::-1]):
        raise ValueError(
            "PLASIM latitude coordinate is not the expected descending Gaussian grid."
        )

    latitude_weights = xr.DataArray(
        gaussian_weights[::-1],
        dims=("lat",),
        coords={"lat": zonal_T["lat"]},
        name="latitude_weights",
    )
    return latitude_weights, zonal_T


@app.cell
def _(zonal_T):
    # Collapse the 30 daily samples in each monthly source file. A MultiIndex
    # grouping retains the provenance year and month, then unstacking exposes
    # the missing September 1007 as a NaN month rather than joining its
    # neighbouring months.
    monthly_groups = zonal_T.assign_coords(
        year=zonal_T["source_year"],
        month=zonal_T["source_month"],
    ).set_index(time=("year", "month"))

    monthly_temperature = (
        monthly_groups.groupby("time")
        .mean()
        .unstack("time")
        .transpose("year", "month", "lat")
    )
    monthly_temperature.attrs = {
        **zonal_T.attrs,
        "long_name": "monthly mean zonal-mean 2 m air temperature",
        "averaging": "mean over 30 daily samples, then unweighted mean over longitude",
    }
    return (monthly_temperature,)


@app.cell
def _(latitude_weights, monthly_temperature):
    global_monthly_temperature = monthly_temperature.weighted(latitude_weights).mean(
        dim="lat"
    )
    global_monthly_temperature.attrs = {
        **monthly_temperature.attrs,
        "long_name": "monthly global mean 2 m air temperature",
        "averaging": "mean over 30 daily samples and area-weighted mean over Gaussian latitudes",
    }
    return (global_monthly_temperature,)


@app.cell
def _(monthly_temperature):
    # Remove the deterministic seasonal cycle separately at every latitude.
    # The mean skips September 1007, whose monthly field is entirely missing.
    monthly_temperature_float = monthly_temperature.astype("float64")
    monthly_temperature_climatology = monthly_temperature_float.mean(
        dim="year", skipna=True
    )
    monthly_temperature_climatology.attrs = {
        **monthly_temperature.attrs,
        "long_name": "calendar-month climatology of zonal-mean 2 m air temperature",
        "averaging": "mean over available source years for each calendar month",
    }

    monthly_temperature_anomaly = (
        monthly_temperature_float.groupby("month") - monthly_temperature_climatology
    )
    monthly_temperature_anomaly.attrs = {
        **monthly_temperature.attrs,
        "long_name": "calendar-month anomaly of zonal-mean 2 m air temperature",
        "description": "monthly temperature minus the climatology for the same calendar month",
    }
    return monthly_temperature_anomaly, monthly_temperature_climatology


@app.cell
def _(global_monthly_temperature, np, plt):
    year_grid, month_grid = np.meshgrid(
        global_monthly_temperature["year"].values,
        global_monthly_temperature["month"].values,
        indexing="ij",
    )
    time_years = year_grid + (month_grid - 0.5) / 12.0

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(
        time_years.ravel(),
        global_monthly_temperature.values.ravel(),
        color="tab:blue",
        linewidth=0.7,
    )
    _ax.set_xlabel("Source year")
    _ax.set_ylabel("Monthly global mean temperature [K]")
    _ax.grid(alpha=0.3, linestyle="--")
    _ax.set_xlim(left=1000,right=1050)
    _fig
    return


@app.cell
def _(monthly_temperature_climatology, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))

    for _month in monthly_temperature_climatology["month"].values:
        _ax.plot(
            monthly_temperature_climatology["lat"],
            monthly_temperature_climatology.sel(month=_month),
            label=f"Month {_month}",
        )

    _ax.set_xlabel("Latitude [degrees north]")
    _ax.set_ylabel("Monthly climatological temperature [K]")
    _ax.grid(alpha=0.3, linestyle="--")
    _ax.legend(ncol=2, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Koopman analysis

    We analyze the full monthly zonal 2 m-temperature (`tas`) anomaly field,
    retaining PLASIM's month-conditioned anomaly construction. The initial fit
    is an effective two-month Koopman map averaged over calendar phase; pairs
    crossing missing source months are excluded and Gaussian latitude weights
    are normalized over the full sphere.
    """)
    return


@app.cell
def _(monthly_temperature_anomaly):
    monthly_temperature_anomaly_trajectory = (
        monthly_temperature_anomaly.stack(sample=("year", "month"))
        .transpose("sample", "lat")
        .dropna(dim="sample", how="any")
    )
    monthly_step = (
        (monthly_temperature_anomaly_trajectory["year"]
        - monthly_temperature_anomaly_trajectory["year"].min())
        * 12
        + (monthly_temperature_anomaly_trajectory["month"] - 1)
    )
    monthly_temperature_anomaly_trajectory.attrs = {
        **monthly_temperature_anomaly.attrs,
        "long_name": "full-field monthly zonal temperature anomaly trajectory",
        "sampling_interval": "1 model month",
    }
    return monthly_step, monthly_temperature_anomaly_trajectory


@app.cell
def _(monthly_step, monthly_temperature_anomaly_trajectory, np):
    snapshot_lag_months = 2
    valid_monthly_transition = (
        monthly_step.values[snapshot_lag_months:]
        - monthly_step.values[:-snapshot_lag_months]
        == snapshot_lag_months
    )
    monthly_snapshot_origins = np.flatnonzero(valid_monthly_transition)
    X_snap = monthly_temperature_anomaly_trajectory.values[monthly_snapshot_origins]
    Y_snap = monthly_temperature_anomaly_trajectory.values[
        monthly_snapshot_origins + snapshot_lag_months
    ]
    snapshot_interval_days = 30.0 * snapshot_lag_months
    if X_snap.shape != Y_snap.shape or X_snap.shape[0] == 0:
        raise ValueError("Monthly Koopman snapshot pairs are empty or misaligned.")
    return X_snap, Y_snap, snapshot_interval_days


@app.cell
def _():
    from scipy.spatial.distance import pdist

    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.algorithms import KernelDMD, WeightedGaussianKernel
    from koopman_response.algorithms.regularization import TSVDRegularizer

    return (
        KernelDMD,
        KoopmanSpectrumKDMD,
        TSVDRegularizer,
        WeightedGaussianKernel,
        pdist,
    )


@app.cell
def _(X_snap, Y_snap, latitude_weights, np):
    n_snapshots_training = 10_000
    rng = np.random.default_rng(seed=0)
    n_train = min(n_snapshots_training, X_snap.shape[0])
    training_indices = rng.choice(X_snap.shape[0], size=n_train, replace=False)
    X_train = X_snap[training_indices]
    Y_train = Y_snap[training_indices]

    # These normalized Gauss--Legendre weights make the kernel distance an
    # area-weighted mean-square difference on the full-sphere latitude state.
    kernel_weight = np.array(latitude_weights.values, dtype=float, copy=True)
    kernel_weight /= kernel_weight.sum()
    return X_train, Y_train, kernel_weight


@app.cell
def _(WeightedGaussianKernel, X_train, kernel_weight, np, pdist):
    weighted_training_data = X_train * np.sqrt(kernel_weight)[None, :]
    kernel_bandwidth = float(
        np.median(pdist(weighted_training_data, metric="euclidean"))
    )
    if not np.isfinite(kernel_bandwidth) or kernel_bandwidth <= 0.0:
        raise ValueError("The weighted Gaussian-kernel bandwidth must be finite and positive.")

    kernel = WeightedGaussianKernel(
        sigma=kernel_bandwidth,
        weights=kernel_weight,
    )
    return (kernel,)


@app.cell
def _(KernelDMD, TSVDRegularizer, X_train, Y_train, kernel):
    rel_threshold_svd_temporary = 1e-4

    kdmd = KernelDMD(kernel=kernel)
    kdmd.fit_snapshots(X=X_train, Y=Y_train)

    tsvd = TSVDRegularizer()
    # The Gaussian Gram matrix is symmetric. Avoiding symmetrization prevents
    # an additional dense copy while retaining the factorization used below.
    tsvd.factorize(
        kdmd.G,
        method="eigsh",
        symmetrize=False,
        rel_threshold=rel_threshold_svd_temporary,
    )
    # kdmd.G = None
    return kdmd, tsvd


@app.cell
def _(plt, tsvd):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(tsvd.S / tsvd.S[0], ".")
    _ax.set_yscale("log")
    _ax.set_xlabel("Singular-value index")
    _ax.set_ylabel(r"$\sigma_i^2 / \sigma_1^2$")
    _ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(KoopmanSpectrumKDMD, kdmd, snapshot_interval_days, tsvd):
    rel_threshold_svd = 6e-3
    koopman_matrix, U_r, S_r = tsvd.solve_from_factorization(
        kdmd.A,
        rel_threshold=rel_threshold_svd,
    )
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        koopman_matrix,
        kernel=kdmd.kernel,
        reference_data=kdmd.reference_data,
        U_r=U_r,
        S_r=S_r,
    )
    eigs_ct = spectrum.continuous_time_eigenvalues(snapshot_interval_days)
    # kdmd.A = None
    return S_r, U_r, eigs_ct, spectrum


@app.cell
def _(eigs_ct, plt, snapshot_interval_days):
    eigs_per_month = eigs_ct * snapshot_interval_days

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.plot(eigs_per_month.real, eigs_per_month.imag, ".", markersize=5)
    _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [model month$^{-1}$]")
    _ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [model month$^{-1}$]")
    _ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(monthly_temperature_anomaly_trajectory, spectrum):
    phi_vals = spectrum.evaluate_eigenfunctions(
        monthly_temperature_anomaly_trajectory.values,
        batch_size=5_000,
    )
    return (phi_vals,)


@app.cell
def _(
    latitude_weights,
    monthly_temperature_anomaly_trajectory,
    np,
):
    temperature_anomaly = monthly_temperature_anomaly_trajectory.values
    _latitude = monthly_temperature_anomaly_trajectory["lat"].values
    area_weights = np.asarray(latitude_weights.values, dtype=float)

    def regional_mean(latitude_mask):
        regional_weights = area_weights * latitude_mask
        if np.count_nonzero(regional_weights) < 2:
            raise ValueError("A temperature region must contain at least two latitude points.")
        return np.average(temperature_anomaly, axis=1, weights=regional_weights)

    global_temperature = regional_mean(np.ones(_latitude.size, dtype=bool))
    north_tropical_temperature = regional_mean((_latitude > 0.0) & (_latitude < 30.0))
    north_extratropical_temperature = regional_mean(_latitude >= 30.0)
    south_tropical_temperature = regional_mean((_latitude < 0.0) & (_latitude > -30.0))
    south_extratropical_temperature = regional_mean(_latitude <= -30.0)
    delta_temperature_symmetric = 0.5 * (
        (north_tropical_temperature - north_extratropical_temperature)
        + (south_tropical_temperature - south_extratropical_temperature)
    )
    return delta_temperature_symmetric, global_temperature


@app.cell
def _(
    delta_temperature_symmetric,
    global_temperature,
    np,
    phi_vals,
    plt,
):
    from scipy.stats import binned_statistic_2d

    eigenfunction_indices = range(1, 7)
    if phi_vals.shape[1] <= max(eigenfunction_indices):
        raise ValueError("The Koopman spectrum does not contain six non-stationary eigenfunctions.")

    n_bins = 70
    min_count = 4

    bin_count, global_edges, delta_edges, _ = binned_statistic_2d(
        global_temperature,
        delta_temperature_symmetric,
        None,
        statistic="count",
        bins=n_bins,
    )
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for _eigenfunction_index, _ax in zip(eigenfunction_indices, _axes.ravel()):
        mean_eigenfunction, _, _, _ = binned_statistic_2d(
            global_temperature,
            delta_temperature_symmetric,
            phi_vals[:, _eigenfunction_index].real,
            statistic="mean",
            bins=[global_edges, delta_edges],
        )
        mean_eigenfunction = np.ma.masked_where(
            bin_count < min_count,
            mean_eigenfunction,
        )
        color_limit = float(np.nanmax(np.abs(mean_eigenfunction)))
        if not np.isfinite(color_limit) or color_limit == 0.0:
            raise ValueError(
                f"Koopman eigenfunction {_eigenfunction_index} has no plottable variation."
            )

        _image = _ax.pcolormesh(
            global_edges,
            delta_edges,
            mean_eigenfunction.T,
            shading="auto",
            cmap="seismic",
            vmin=-color_limit,
            vmax=color_limit,
        )
        _ax.set_title(rf"$\Re\,\phi_{{{_eigenfunction_index}}}$")
        _fig.colorbar(_image, ax=_ax, label=rf"$\Re\,\phi_{{{_eigenfunction_index}}}$")

    for _ax in _axes[-1, :]:
        _ax.set_xlabel(r"$\overline{T}'$ [K]")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(r"$\Delta T'_{\mathrm{sym}}$ [K]")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Koopman modes

    Koopman modes are the latitude-dependent spatial reconstruction patterns
    associated with the eigenfunctions above. They are the appropriate
    Koopman objects to compare with EOF patterns.
    """)
    return


@app.cell
def _(S_r, U_r, X_train, np, spectrum):
    _mode_indices = range(1, 7)
    koopman_mode_matrix = np.column_stack(
        [
            spectrum.koopman_modes(
                X_train[:, _latitude_index],
                U_r=U_r,
                S_r=S_r,
            )
            for _latitude_index in range(X_train.shape[1])
        ]
    ).T
    if koopman_mode_matrix.shape[1] <= max(_mode_indices):
        raise ValueError("The Koopman spectrum does not contain six non-stationary modes.")
    return (koopman_mode_matrix,)


@app.cell
def _(
    eigs_ct,
    koopman_mode_matrix,
    monthly_temperature_anomaly_trajectory,
    plt,
    snapshot_interval_days,
):
    _mode_indices = range(1, 7)
    eigenvalues_per_month = eigs_ct * snapshot_interval_days
    _latitude = monthly_temperature_anomaly_trajectory["lat"]

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for _mode_index, _ax in zip(_mode_indices, _axes.ravel()):
        _ax.plot(_latitude, koopman_mode_matrix[:, _mode_index].real)
        _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        _ax.set_title(
            rf"$\Re\,v_{{{_mode_index}}}$, "
            rf"$\Re\,\lambda={eigenvalues_per_month[_mode_index].real:.3g}$ month$^{{-1}}$"
        )
        _ax.grid(alpha=0.3, linestyle="--")

    for _ax in _axes[-1, :]:
        _ax.set_xlabel("Latitude [degrees north]")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(r"$\Re\,v$ [K]")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EOF analysis

    EOFs provide a variance-based reference for the same full-field monthly
    anomaly trajectory and Gaussian area metric used by KDMD. They are not a
    replacement for Koopman eigenfunctions, which are selected by dynamical
    evolution rather than explained variance.
    """)
    return


@app.cell
def _(
    latitude_weights,
    monthly_temperature_anomaly_trajectory,
    np,
    xr,
):
    anomaly_data = monthly_temperature_anomaly_trajectory.values
    spatial_weights = np.array(
        latitude_weights.values,
        dtype=float,
        copy=True,
    )
    spatial_weights /= spatial_weights.sum()

    weighted_anomaly_data = anomaly_data * np.sqrt(spatial_weights)[None, :]
    U, singular_values, Vt = np.linalg.svd(weighted_anomaly_data, full_matrices=False)
    eof_indices = np.arange(1, singular_values.size + 1)

    eof_patterns = xr.DataArray(
        Vt / np.sqrt(spatial_weights)[None, :],
        dims=("eof", "lat"),
        coords={"eof": eof_indices, "lat": monthly_temperature_anomaly_trajectory["lat"]},
        name="eof_patterns",
        attrs={
            "long_name": "Gaussian-area-weighted EOF patterns",
            "normalization": "unit norm under normalized Gaussian latitude weights",
        },
    )
    eof_principal_components = xr.DataArray(
        U * singular_values[None, :],
        dims=("sample", "eof"),
        coords={
            "sample": monthly_temperature_anomaly_trajectory["sample"],
            "eof": eof_indices,
        },
        name="eof_principal_components",
        attrs={"long_name": "EOF principal-component time series", "units": "K"},
    )
    eof_explained_variance_fraction = xr.DataArray(
        singular_values**2 / np.sum(singular_values**2),
        dims=("eof",),
        coords={"eof": eof_indices},
        name="eof_explained_variance_fraction",
        attrs={"long_name": "fraction of area-weighted anomaly variance explained"},
    )
    return eof_explained_variance_fraction, eof_patterns


@app.cell
def _(eof_explained_variance_fraction, eof_patterns, plt):
    _fig, (_pattern_ax, _variance_ax) = plt.subplots(1, 2, figsize=(12, 4))
    for _eof_index in (1, 2):
        _pattern_ax.plot(
            eof_patterns["lat"],
            eof_patterns.sel(eof=_eof_index),
            label=rf"EOF {_eof_index}",
        )
    _pattern_ax.set_xlabel("Latitude [degrees north]")
    _pattern_ax.set_ylabel("Weighted-normalized EOF pattern")
    _pattern_ax.grid(alpha=0.3, linestyle="--")
    _pattern_ax.legend()

    _variance_ax.plot(
        eof_explained_variance_fraction["eof"].isel(eof=slice(0, 10)),
        eof_explained_variance_fraction.isel(eof=slice(0, 10)),
        ".-",
    )
    _variance_ax.set_yscale("log")
    _variance_ax.set_xlabel("EOF index")
    _variance_ax.set_ylabel("Explained variance fraction")
    _variance_ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
