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
    return data_dir, latitude_weights, zonal_T


@app.cell
def _(data_dir, plt, xr):
    fnamee = "BENCHMARK_360ppm_T21L10_LSG_MPI4_20Y_PLA.2000-2009.nc"
    fnamee2 = "BENCHMARK_360ppm_T21L10_LSG_MPI4_20Y_PLA.2010-2019.nc"

    ds = xr.open_dataset(data_dir / fnamee)
    zonal = ds["tas"].mean(dim="lon")

    ds2 = xr.open_dataset(data_dir / fnamee2)
    zonal2 = ds["tas"].mean(dim="lon")

    z = xr.concat([zonal,zonal2],dim="time")
    res = z - z.mean(dim="time")
    _fig , _ax = plt.subplots()
    for i in range(len( zonal["time"] )):
        res.isel(time=i).plot(ax=_ax)
    _fig
    return


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
    # Use only complete model years: September 1007 is absent from the source
    # data and must not produce an 11-month annual state.
    monthly_count = monthly_temperature.notnull().sum(dim="month")
    annual_temperature = monthly_temperature.mean(dim="month", skipna=True).where(
        monthly_count == 12,
        drop=True,
    )
    annual_temperature.attrs = {
        **monthly_temperature.attrs,
        "long_name": "annual mean zonal-mean 2 m air temperature",
        "averaging": "mean over 12 monthly means",
    }
    annual_temperature_anomaly = annual_temperature - annual_temperature.mean(
        dim="year"
    )
    annual_temperature_anomaly.attrs = {
        **annual_temperature.attrs,
        "long_name": "annual-mean zonal 2 m air temperature anomaly",
        "description": "annual zonal temperature minus the available-year annual mean",
    }
    return annual_temperature, annual_temperature_anomaly


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
    return (monthly_temperature_climatology,)


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
    ## Annual global-mean temperature

    Before fitting the Koopman model, inspect the low-frequency global signal.
    Each annual value is the mean of its 12 monthly global means. The incomplete
    year containing the missing September 1007 is excluded, so every retained
    annual mean has the same temporal support.
    """)
    return


@app.cell
def _(annual_temperature, latitude_weights, np, xr):
    from koopman_response.utils.signal import cross_correlation

    annual_global_temperature = annual_temperature.weighted(latitude_weights).mean(dim="lat")
    annual_global_temperature.attrs = {
        **annual_temperature.attrs,
        "long_name": "annual global mean 2 m air temperature",
        "averaging": "mean over 12 monthly global means",
    }

    # The missing September 1007 creates a gap in the otherwise annual grid.
    # ``cross_correlation`` requires regularly sampled data, so retain the
    # longest contiguous annual segment (1008--2000) for this diagnostic.
    annual_years = annual_global_temperature["year"].values
    segment_starts = np.concatenate(([0], np.flatnonzero(np.diff(annual_years) != 1) + 1))
    segment_stops = np.concatenate((segment_starts[1:], [annual_years.size]))
    segment_lengths = segment_stops - segment_starts
    longest_segment = int(np.argmax(segment_lengths))
    correlation_temperature = annual_global_temperature.isel(
        year=slice(segment_starts[longest_segment], segment_stops[longest_segment])
    )
    correlation_dt_years = float(
        np.median(np.diff(correlation_temperature["year"].values))
    )
    if not np.allclose(
        np.diff(correlation_temperature["year"].values), correlation_dt_years
    ):
        raise ValueError("Annual temperatures for correlation are not regularly sampled.")

    # Use the same helper and biased normalization as the global-temperature
    # correlation in analysis_new.py. The helper removes the temporal mean.
    max_correlation_lag_years = min(100, correlation_temperature.size - 1)
    correlation_lags, annual_temperature_correlation_values = cross_correlation(
        x=correlation_temperature.values,
        y=correlation_temperature.values,
        dt=correlation_dt_years,
        max_lag=max_correlation_lag_years,
        normalization="biased",
    )
    annual_temperature_autocorrelation = xr.DataArray(
        annual_temperature_correlation_values / annual_temperature_correlation_values[0],
        dims=("lag_years",),
        coords={"lag_years": correlation_lags},
        name="annual_temperature_autocorrelation",
        attrs={
            "long_name": "normalized autocorrelation of annual global mean 2 m air temperature",
            "description": (
                "mean-removed biased autocovariance normalized by its zero-lag value; "
                f"computed from contiguous source years "
                f"{int(correlation_temperature.year.min())}--"
                f"{int(correlation_temperature.year.max())}"
            ),
        },
    )
    return annual_global_temperature, annual_temperature_autocorrelation


@app.cell
def _(annual_global_temperature, annual_temperature_autocorrelation, plt):
    _fig, (_temperature_ax, _correlation_ax) = plt.subplots(1, 2, figsize=(13, 4))

    _temperature_ax.plot(
        annual_global_temperature["year"],
        annual_global_temperature,
        color="tab:blue",
        linewidth=0.8,
    )
    _temperature_ax.set_xlabel("Source year")
    _temperature_ax.set_ylabel("Annual global mean temperature [K]")
    _temperature_ax.grid(alpha=0.3, linestyle="--")

    _correlation_ax.plot(
        annual_temperature_autocorrelation["lag_years"],
        annual_temperature_autocorrelation,
        ".-",
        color="tab:orange",
    )
    _correlation_ax.axhline(0.0, color="black", linewidth=0.8)
    _correlation_ax.set_xlabel("Lag [years]")
    _correlation_ax.set_ylabel("Autocorrelation")
    _correlation_ax.set_ylim(-0.2, 1.05)
    _correlation_ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Koopman analysis

    We analyze the full annual-mean zonal 2 m-temperature (`tas`) anomaly
    field. Annual averaging removes the deterministic seasonal cycle; the
    anomaly is then formed relative to the available-year annual mean at every
    latitude. The initial fit is a one-year Koopman map. The missing 1007 year
    is excluded, and pairs across its resulting gap are not used.
    """)
    return


@app.cell
def _(annual_temperature_anomaly):
    annual_temperature_anomaly_trajectory = annual_temperature_anomaly.rename(year="sample")
    annual_temperature_anomaly_trajectory.attrs = {
        **annual_temperature_anomaly.attrs,
        "long_name": "full-field annual zonal temperature anomaly trajectory",
        "sampling_interval": "1 model year",
    }
    return (annual_temperature_anomaly_trajectory,)


@app.cell
def _(annual_temperature_anomaly_trajectory, np):
    snapshot_lag_years = 1
    source_years = annual_temperature_anomaly_trajectory["sample"].values
    valid_annual_transition = (
        source_years[snapshot_lag_years:]
        - source_years[:-snapshot_lag_years]
        == snapshot_lag_years
    )
    annual_snapshot_origins = np.flatnonzero(valid_annual_transition)
    X_snap = annual_temperature_anomaly_trajectory.values[annual_snapshot_origins]
    Y_snap = annual_temperature_anomaly_trajectory.values[
        annual_snapshot_origins + snapshot_lag_years
    ]
    snapshot_interval_days = 360.0 * snapshot_lag_years
    if X_snap.shape != Y_snap.shape or X_snap.shape[0] == 0:
        raise ValueError("Annual Koopman snapshot pairs are empty or misaligned.")
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
    rel_threshold_svd = 5e-3
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
    eigs_per_year = eigs_ct * snapshot_interval_days

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.plot(eigs_per_year.real, eigs_per_year.imag, ".", markersize=5)
    _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [model year$^{-1}$]")
    _ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [model year$^{-1}$]")
    _ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(annual_temperature_anomaly_trajectory, spectrum):
    phi_vals = spectrum.evaluate_eigenfunctions(
        annual_temperature_anomaly_trajectory.values,
        batch_size=5_000,
    )
    return (phi_vals,)


@app.cell
def _(annual_temperature_anomaly_trajectory, latitude_weights, np):
    temperature_anomaly = annual_temperature_anomaly_trajectory.values
    _latitude = annual_temperature_anomaly_trajectory["lat"].values
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
def _(delta_temperature_symmetric, global_temperature, np, phi_vals, plt):
    from scipy.stats import binned_statistic_2d

    eigenfunction_indices = range(1, 7)
    if phi_vals.shape[1] <= max(eigenfunction_indices):
        raise ValueError("The Koopman spectrum does not contain six non-stationary eigenfunctions.")

    n_bins = 30
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
    annual_temperature_anomaly_trajectory,
    eigs_ct,
    koopman_mode_matrix,
    plt,
    snapshot_interval_days,
):
    _mode_indices = range(1, 7)
    eigenvalues_per_year = eigs_ct * snapshot_interval_days
    _latitude = annual_temperature_anomaly_trajectory["lat"]

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for _mode_index, _ax in zip(_mode_indices, _axes.ravel()):
        _ax.plot(_latitude, koopman_mode_matrix[:, _mode_index].real)
        _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        _ax.set_title(
            rf"$\Re\,v_{{{_mode_index}}}$, "
            rf"$\Re\,\lambda={eigenvalues_per_year[_mode_index].real:.3g}$ year$^{{-1}}$"
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

    EOFs provide a variance-based reference for the same full-field annual
    anomaly trajectory and Gaussian area metric used by KDMD. They are not a
    replacement for Koopman eigenfunctions, which are selected by dynamical
    evolution rather than explained variance.
    """)
    return


@app.cell
def _(annual_temperature_anomaly_trajectory, latitude_weights, np, xr):
    anomaly_data = annual_temperature_anomaly_trajectory.values
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
        coords={"eof": eof_indices, "lat": annual_temperature_anomaly_trajectory["lat"]},
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
            "sample": annual_temperature_anomaly_trajectory["sample"],
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
