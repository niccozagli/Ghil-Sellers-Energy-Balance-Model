import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

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
def _():
    # global_temperature = latitude_weighted_mean(warm_cold_dataset)
    # mean_temperature_fig, mean_temperature_ax = plt.subplots(figsize=(8, 4))
    # mean_temperature_ax.plot(
    #     global_temperature["time"] / YEAR,
    #     global_temperature["warm_state_temperature"],
    #     color="red",
    #     label="Warm State",
    # )
    # mean_temperature_ax.plot(
    #     global_temperature["time"] / YEAR,
    #     global_temperature["cold_state_temperature"],
    #     color="blue",
    #     label="Cold State",
    # )
    # mean_temperature_ax.set_xlabel("Time [year]")
    # mean_temperature_ax.set_ylabel("Mean temperature [K]")
    # mean_temperature_ax.grid(True, alpha=0.3)
    # mean_temperature_ax.legend()
    # mean_temperature_fig
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
    _ax.set_xlim(left=0.96,right=0.97)
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
    warm_input_filename ="new_stochastic_warm_mu1" #"stochastic_warm_state_mu0p97_{5}.nc"
    warm_ds = xr.open_dataset(data_dir / (warm_input_filename + ".nc") )
    return warm_ds, warm_input_filename


@app.cell
def _(data_dir, warm_input_filename, xr):
    correlation_input_filenames = tuple(
        warm_input_filename+f"_{{{_index}}}.nc"
        for _index in range(1, 6)
    )
    correlation_warm_datasets = tuple(
        xr.open_dataset(data_dir / _filename)
        for _filename in correlation_input_filenames
    )
    return (correlation_warm_datasets,)


@app.cell
def _(latitude_weighted_mean, warm_ds):
    avg_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1)
    eq_T = latitude_weighted_mean(warm_ds, xmin=0, xmax=1/3)
    pole_T = latitude_weighted_mean(warm_ds, xmin=1/3, xmax=1)
    Delta_T = eq_T - pole_T
    return (avg_T,)


@app.cell
def _(YEAR, avg_T, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(avg_T["time"].values / YEAR,avg_T["temperature"].values)
    _ax.set_xlabel(xlabel=r"$t$ (years)",size=16)
    _ax.set_ylabel(ylabel=r"$\overline{T} $ (K) ",size=16)
    _ax.grid(alpha=0.4,linestyle='--')
    _ax.set_xlim(left=0,right=2000)
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
    return (asymptotic_warm_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation functions
    """)
    return


@app.cell
def _(YEAR, correlation_warm_datasets, latitude_weighted_mean, np):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    _transient = 500
    _stop = 1e10

    _corr_avgT_replicates = []
    _corr_DeltaT_replicates = []
    dt = None
    lags_avgT = None
    lags_DeltaT = None

    for _dataset in correlation_warm_datasets:
        _condition = (
            (_dataset["time"] > _transient * YEAR) &
            (_dataset["time"] < _stop * YEAR)
        )
        _avg_T = latitude_weighted_mean(_dataset, xmin=0, xmax=1)
        _eq_T = latitude_weighted_mean(_dataset, xmin=0, xmax=1 / 3)
        _pole_T = latitude_weighted_mean(_dataset, xmin=1 / 3, xmax=1)
        _Delta_T = _eq_T - _pole_T

        _avg_T_asymp = _avg_T.where(cond=_condition, drop=True)
        _Delta_T_asymp = _Delta_T.where(cond=_condition, drop=True)

        # Sampling interval of the SAVED series is stochastic_dt * save_every, not
        # the raw solver step. Derive it from the time coordinate so save_every is
        # always accounted for.
        _dt = float(np.median(np.diff(_avg_T_asymp["time"].values)))  # in seconds
        _expected_dt = (
            _avg_T_asymp.attrs["stochastic_dt"] *
            _avg_T_asymp.attrs["stochastic_save_every"]
        )
        assert np.isclose(_dt, _expected_dt, rtol=1e-6), (_dt, _expected_dt)

        _obs_avgT = _avg_T_asymp["temperature"].values
        _lags_avgT, _corr_avgT = _cross_correlation(
            x=_obs_avgT,
            y=_obs_avgT,
            dt=_dt,
            normalization="biased",
        )

        _obs_DeltaT = _Delta_T_asymp["temperature"].values
        _lags_DeltaT, _corr_DeltaT = _cross_correlation(
            x=_obs_DeltaT,
            y=_obs_DeltaT,
            dt=_dt,
            normalization="unbiased",
        )

        if dt is None:
            dt = _dt
            lags_avgT = _lags_avgT
            lags_DeltaT = _lags_DeltaT
        elif not np.isclose(_dt, dt, rtol=1e-12, atol=0.0):
            raise ValueError("Correlation replicate sampling intervals do not match.")
        elif not np.array_equal(_lags_avgT, lags_avgT):
            raise ValueError("avgT correlation replicate lag grids do not match.")
        elif not np.array_equal(_lags_DeltaT, lags_DeltaT):
            raise ValueError("Delta_T correlation replicate lag grids do not match.")

        _corr_avgT_replicates.append(_corr_avgT)
        _corr_DeltaT_replicates.append(_corr_DeltaT)

    corr_avgT = np.mean(np.stack(_corr_avgT_replicates, axis=0), axis=0)
    corr_DeltaT = np.mean(np.stack(_corr_DeltaT_replicates, axis=0), axis=0)
    return corr_DeltaT, corr_avgT, dt, lags_DeltaT, lags_avgT


@app.cell
def _(YEAR, correlation_warm_datasets, dt, np):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation
    _transient = 500
    _stop = 1e10
    _local_corr_lag_window_years = 20.0

    _local_corr_replicates = []
    local_corr_lags = None
    local_corr_latitude = None

    for _dataset in correlation_warm_datasets:
        _condition = (
            (_dataset["time"] > _transient * YEAR) &
            (_dataset["time"] < _stop * YEAR)
        )
        _asymptotic_dataset = _dataset.where(cond=_condition, drop=True)
        _local_corr_latitude_condition = _asymptotic_dataset["latitude"] >= 0
        local_corr_ds = _asymptotic_dataset.where(
            cond=_local_corr_latitude_condition,
            drop=True,
        )
        _local_corr_latitude = local_corr_ds["latitude"].values
        local_temperature = local_corr_ds["temperature"].values

        if local_temperature.ndim != 2:
            raise ValueError("Expected temperature data with dimensions (time, latitude).")
        if local_temperature.shape[0] == 0 or local_temperature.shape[1] == 0:
            raise ValueError("Cannot compute local correlations from an empty time-latitude slice.")
        if not np.isfinite(local_temperature).all():
            raise ValueError("Local temperature data contain non-finite values.")

        local_corr_max_lag = min(
            int(round(_local_corr_lag_window_years * YEAR / dt)),
            local_temperature.shape[0] - 1,
        )
        local_corr_columns = []
        _local_corr_lags = None

        for _latitude_index in range(local_temperature.shape[1]):
            _lags, _corr = _cross_correlation(
                x=local_temperature[:, _latitude_index],
                y=local_temperature[:, _latitude_index],
                dt=dt,
                max_lag=local_corr_max_lag,
                normalization="unbiased",
            )
            if _local_corr_lags is None:
                _local_corr_lags = _lags
            local_corr_columns.append(_corr)

        _local_corr = np.stack(local_corr_columns, axis=1)
        if local_corr_lags is None:
            local_corr_lags = _local_corr_lags
            local_corr_latitude = _local_corr_latitude
        elif not np.array_equal(_local_corr_lags, local_corr_lags):
            raise ValueError("Local correlation replicate lag grids do not match.")
        elif not np.allclose(_local_corr_latitude, local_corr_latitude):
            raise ValueError("Local correlation replicate latitude grids do not match.")
        elif _local_corr.shape != _local_corr_replicates[0].shape:
            raise ValueError("Local correlation replicate shapes do not match.")

        _local_corr_replicates.append(_local_corr)

    local_corr = np.mean(
        np.stack(_local_corr_replicates, axis=0),
        axis=0,
    )
    return local_corr, local_corr_lags, local_corr_latitude


@app.cell
def _(YEAR, local_corr, local_corr_lags, local_corr_latitude, np, plt):
    _lag_years = local_corr_lags / YEAR
    _plot_stride = max(1, local_corr.shape[0] // 300)
    _latitude_grid, _lag_grid = np.meshgrid(
        local_corr_latitude,
        _lag_years[::_plot_stride],
    )
    _surface_values = local_corr[::_plot_stride, :]

    _fig = plt.figure(figsize=(9, 6))
    _ax = _fig.add_subplot(111, projection="3d")
    _surface = _ax.plot_surface(
        _latitude_grid,
        _lag_grid,
        _surface_values,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        rcount=_surface_values.shape[0],
        ccount=_surface_values.shape[1],
    )

    _ax.set_xlabel(r"$x$", fontsize=14)
    _ax.set_ylabel(r"$t \; [\mathrm{years}]$", fontsize=14)
    _ax.set_zlabel(r"$C_T(x,t)$", fontsize=14)
    _ax.set_xlim(local_corr_latitude.min(), local_corr_latitude.max())
    _ax.set_ylim(_lag_years.min(), _lag_years.max())
    _ax.view_init(elev=28, azim=45)
    _fig.colorbar(_surface, ax=_ax, shrink=0.65, pad=0.12, label=r"$C_T(x,t)$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Koopman Analysis
    """)
    return


@app.cell
def _():
    from scipy.spatial.distance import pdist

    from koopman_response.algorithms import KernelDMD, WeightedGaussianKernel
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.utils import cosine_trapezoid_weights
    from koopman_response.utils.preprocessing import make_snapshots
    from koopman_response.algorithms.regularization import TSVDRegularizer

    return (
        KernelDMD,
        KoopmanSpectrumKDMD,
        TSVDRegularizer,
        WeightedGaussianKernel,
        cosine_trapezoid_weights,
        make_snapshots,
        pdist,
    )


@app.cell
def _(asymptotic_warm_ds, np):
    n_snapshots_training = 30_000
    rng = np.random.default_rng()

    # Select only the positive latitudes, the model is symmetric
    latitude_condition = asymptotic_warm_ds["latitude"] >= 0
    sliced_asymptotic_warm_ds = asymptotic_warm_ds.where(cond=latitude_condition,drop=True)

    # Final data
    space_coord = sliced_asymptotic_warm_ds["latitude"].values
    X = sliced_asymptotic_warm_ds["temperature"].values
    return X, n_snapshots_training, rng, space_coord


@app.cell
def _(
    X,
    cosine_trapezoid_weights,
    dt,
    make_snapshots,
    n_snapshots_training,
    rng,
    space_coord,
):
    # Pre-processing
    # 1. We target fluctuations: we remove the temporal mean at each point
    # 2. We use a weighted kernel

    # Remove the temporal mean at each grid point
    mean_field = X.mean(axis=0)
    centered_data = X - mean_field[None, :]

    # Trapezoidal quadrature weights for the weighted kernel metric.
    kernel_weight = cosine_trapezoid_weights(space_coord, n_features=X.shape[1])

    # Snapshot data and sub-sampling
    X_snap, Y_snap, dt_eff = make_snapshots(centered_data, dt=dt,lag=2)
    n_train = min(n_snapshots_training, X_snap.shape[0])
    idx = rng.choice(X_snap.shape[0], size=n_train, replace=False)
    X_snap = X_snap[idx]
    Y_snap = Y_snap[idx]
    return X_snap, Y_snap, centered_data, dt_eff, idx, kernel_weight


@app.cell
def _(
    KernelDMD,
    TSVDRegularizer,
    WeightedGaussianKernel,
    X_snap,
    Y_snap,
    centered_data,
    idx,
    kernel_weight,
    np,
    pdist,
):
    # Kernel DMD algorithm
    rel_threshold_svd_temporary = 1e-3

    weighted_for_bandwidth = centered_data[idx] * np.sqrt(kernel_weight)[None, :]
    sigma_median = float(np.median(pdist(weighted_for_bandwidth, metric="euclidean")))

    kdmd = KernelDMD(
        kernel=WeightedGaussianKernel(
            sigma=sigma_median,
            weights=kernel_weight,
        )
    )
    kdmd.fit_snapshots(X=X_snap, Y=Y_snap)

    tsvd = TSVDRegularizer()
    # Kxx from a symmetric kernel is already symmetric, so symmetrize=False avoids
    # a full-size 0.5*(G + G.T) copy (~7 GB at N=30k). Free G right after
    # factorizing: it is not referenced anywhere else.
    tsvd.factorize(
        kdmd.G,
        method="eigsh",
        symmetrize=False,
        rel_threshold=rel_threshold_svd_temporary,
    )
    kdmd.G = None
    return kdmd, tsvd


@app.cell
def _(plt, tsvd):
    fig_sv, ax_sv = plt.subplots()
    ax_sv.plot(tsvd.S / tsvd.S[0], ".")
    ax_sv.set_yscale(value="log")
    ax_sv.set_xlabel(xlabel="$i$", size=16)
    ax_sv.set_ylabel(ylabel=r"$\sigma^2_i / \sigma^2_1$", size=16)
    ax_sv.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(KoopmanSpectrumKDMD, dt_eff, kdmd, np, tsvd):
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

    _tsvd_thresholds = np.asarray((1e-2, 5e-3, 1e-3), dtype=float)
    _tsvd_n_modes = 6
    _tsvd_raw_spectra = []
    _tsvd_tracked_spectra = []
    _tsvd_ranks = []
    spectrum = None
    eigs_ct = None

    # Reuse the same Gram factorization and lagged cross-kernel matrix at every
    # final threshold. Only the retained TSVD basis changes.
    for _threshold_index, _threshold in enumerate(_tsvd_thresholds):
        _kr, _ur, _sr = tsvd.solve_from_factorization(
            kdmd.A,
            rel_threshold=float(_threshold),
        )
        _threshold_spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
            _kr,
            kernel=kdmd.kernel,
            reference_data=kdmd.reference_data,
            U_r=_ur,
            S_r=_sr,
        )
        _threshold_eigs_ct = _threshold_spectrum.continuous_time_eigenvalues(
            dt_eff
        )
        if _threshold_eigs_ct.size < _tsvd_n_modes + 1:
            raise ValueError(
                f"TSVD threshold {_threshold:g} retained fewer than "
                f"{_tsvd_n_modes} non-stationary eigenvalues."
            )

        _tsvd_raw_spectra.append(_threshold_eigs_ct.copy())
        _tsvd_ranks.append(_sr.size)
        if _threshold_index == 0:
            _tracked_at_threshold = _threshold_eigs_ct[
                1 : _tsvd_n_modes + 1
            ].copy()
        else:
            _candidate_eigs = _threshold_eigs_ct[1:]
            _matching_cost = np.abs(
                _tsvd_tracked_spectra[-1][:, None]
                - _candidate_eigs[None, :]
            )
            _matched_rows, _matched_columns = _linear_sum_assignment(
                _matching_cost
            )
            _tracked_at_threshold = np.empty(_tsvd_n_modes, dtype=complex)
            _tracked_at_threshold[_matched_rows] = _candidate_eigs[
                _matched_columns
            ]
        _tsvd_tracked_spectra.append(_tracked_at_threshold)

        # Keep 5e-3 as the canonical spectrum for all existing downstream cells.
        if _threshold_index == 1:
            spectrum = _threshold_spectrum
            eigs_ct = _threshold_eigs_ct

    if spectrum is None or eigs_ct is None:
        raise AssertionError("The canonical 5e-3 TSVD spectrum was not constructed.")

    tsvd_thresholds = _tsvd_thresholds.copy()
    tsvd_ranks = np.asarray(_tsvd_ranks, dtype=int)
    tsvd_eigenvalues_raw = tuple(_tsvd_raw_spectra)
    tsvd_eigenvalues_tracked = np.vstack(_tsvd_tracked_spectra)
    kdmd.A = None
    return (
        eigs_ct,
        spectrum,
        tsvd_eigenvalues_tracked,
        tsvd_ranks,
        tsvd_thresholds,
    )


@app.cell
def _(YEAR, eigs_ct, plt):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eigs_ct.real * YEAR , eigs_ct.imag * YEAR    , ".", ms=4)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [year$^{-1}$]",size=16)
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [year$^{-1}$]",size=16)
    ax.grid(alpha=0.3)
    # ax.set_xlim(left=-2, right=0.01)
    # fig.savefig("figures/eigenvalues_near.png",dpi=400)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TSVD truncation sensitivity

    Compare the six leading non-stationary Koopman eigenvalues while varying
    the final TSVD truncation threshold. The snapshot pairs, kernel matrices,
    and preliminary Gram factorization are shared across all thresholds.
    """)
    return


@app.cell
def _(YEAR, plt, tsvd_eigenvalues_tracked, tsvd_ranks, tsvd_thresholds):
    tsvd_sensitivity_fig, (
        _tsvd_real_ax,
        _tsvd_imag_ax,
        _tsvd_rank_ax,
    ) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 9),
        sharex=True,
    )

    for _mode_index in range(tsvd_eigenvalues_tracked.shape[1]):
        _mode_values_per_year = (
            tsvd_eigenvalues_tracked[:, _mode_index] * YEAR
        )
        _mode_label = rf"$\lambda_{{{_mode_index + 1}}}$"
        _tsvd_real_ax.plot(
            tsvd_thresholds,
            _mode_values_per_year.real,
            marker="o",
            label=_mode_label,
        )
        _tsvd_imag_ax.plot(
            tsvd_thresholds,
            _mode_values_per_year.imag,
            marker="o",
            label=_mode_label,
        )

    _tsvd_rank_ax.plot(
        tsvd_thresholds,
        tsvd_ranks,
        marker="o",
        color="black",
    )
    for _ax in (_tsvd_real_ax, _tsvd_imag_ax, _tsvd_rank_ax):
        _ax.set_xscale("log")
        _ax.invert_xaxis()
        _ax.grid(alpha=0.3, linestyle="--")
    for _ax in (_tsvd_real_ax, _tsvd_imag_ax):
        _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)

    _tsvd_real_ax.set_ylabel(
        r"$\mathrm{Re}\,\lambda$ [year$^{-1}$]",
        size=16,
    )
    _tsvd_imag_ax.set_ylabel(
        r"$\mathrm{Im}\,\lambda$ [year$^{-1}$]",
        size=16,
    )
    _tsvd_rank_ax.set_ylabel("Retained rank", size=16)
    _tsvd_rank_ax.set_xlabel("TSVD relative threshold", size=16)
    _tsvd_rank_ax.set_xticks(tsvd_thresholds)
    # _tsvd_rank_ax.set_xticklabels(
    #     (r"$10^{-2}$", r"$5\times10^{-3}$", r"$10^{-3}$", r"$5\times10^{-4}$")
    # )
    _tsvd_real_ax.legend(ncols=2)
    tsvd_sensitivity_fig.tight_layout()
    tsvd_sensitivity_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Snapshot-lag sensitivity

    Compare the leading non-stationary Koopman eigenvalues obtained from
    snapshot pairs separated by 1, 2, 6, 12, and 24 months. The reference
    snapshots, kernel, and reduced basis are held fixed so that only the
    snapshot separation changes.
    """)
    return


@app.cell
def _(
    DAY,
    KernelDMD,
    KoopmanSpectrumKDMD,
    TSVDRegularizer,
    WeightedGaussianKernel,
    asymptotic_warm_ds,
    centered_data,
    dt,
    kernel_weight,
    n_snapshots_training,
    np,
    pdist,
):
    import gc as _gc
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

    _lag_months = np.asarray((1, 2, 6, 12, 24), dtype=int)
    _lag_rng_seed = 0
    _lag_kernel_batch_size = 1_000
    _lag_rel_threshold_svd_temporary = 1e-3
    _lag_rel_threshold_svd = 5e-3
    _lag_n_modes = 6

    # The requested lags are integer numbers of saved 30-day snapshots.
    _trajectory_time = np.asarray(asymptotic_warm_ds["time"].values, dtype=float)
    _trajectory_steps = np.diff(_trajectory_time)
    if _trajectory_steps.size == 0:
        raise ValueError("Lag sensitivity requires at least two saved snapshots.")
    if not np.allclose(_trajectory_steps, _trajectory_steps[0], rtol=1e-12, atol=0.0):
        raise ValueError("Lag sensitivity requires a regularly sampled trajectory.")
    _trajectory_dt = float(_trajectory_steps[0])
    if not np.isclose(_trajectory_dt, dt, rtol=1e-12, atol=0.0):
        raise ValueError("The Koopman trajectory and correlation data have different sampling intervals.")

    _model_month = 30.0 * DAY
    _lag_steps_float = _lag_months * _model_month / _trajectory_dt
    _lag_steps = np.rint(_lag_steps_float).astype(int)
    if not np.allclose(_lag_steps_float, _lag_steps, rtol=0.0, atol=1e-12):
        raise ValueError("Requested month lags do not map to integer saved-snapshot lags.")
    if np.any(_lag_steps < 1):
        raise ValueError("Every snapshot lag must be at least one saved time step.")

    # Restrict all fits to origins that have a target at the largest lag, then
    # use the same deterministic subsample for every lag.
    _max_lag = int(_lag_steps.max())
    _n_common_origins = centered_data.shape[0] - _max_lag
    if _n_common_origins < 1:
        raise ValueError("The trajectory is too short for the requested maximum lag.")
    _n_lag_train = min(int(n_snapshots_training), _n_common_origins)
    if _n_lag_train < 2:
        raise ValueError("Lag sensitivity requires at least two training snapshots.")
    _lag_rng = np.random.default_rng(_lag_rng_seed)
    _lag_origins = _lag_rng.choice(
        _n_common_origins,
        size=_n_lag_train,
        replace=False,
    )
    _lag_reference_data = centered_data[_lag_origins]
    if np.any(_lag_origins[:, None] + _lag_steps[None, :] >= centered_data.shape[0]):
        raise AssertionError("At least one lagged target falls outside the trajectory.")

    _lag_weighted_reference = (
        _lag_reference_data * np.sqrt(kernel_weight)[None, :]
    )
    _lag_sigma = float(
        np.median(pdist(_lag_weighted_reference, metric="euclidean"))
    )
    if not np.isfinite(_lag_sigma) or _lag_sigma <= 0.0:
        raise ValueError("Lag-sensitivity kernel bandwidth must be positive and finite.")
    _lag_kernel = WeightedGaussianKernel(
        sigma=_lag_sigma,
        weights=kernel_weight,
    )

    # The first fit supplies the shared Gram factorization. Later lagged
    # cross-kernel matrices are projected into this basis block by block, so
    # no additional full N-by-N matrices are allocated.
    _first_targets = centered_data[_lag_origins + int(_lag_steps[0])]
    _lag_kdmd = KernelDMD(kernel=_lag_kernel)
    _lag_kdmd.fit_snapshots(X=_lag_reference_data, Y=_first_targets)
    _lag_tsvd = TSVDRegularizer()
    _lag_tsvd.factorize(
        _lag_kdmd.G,
        method="eigsh",
        symmetrize=False,
        rel_threshold=_lag_rel_threshold_svd_temporary,
    )
    _first_kr, _lag_ur, _lag_sr = _lag_tsvd.solve_from_factorization(
        _lag_kdmd.A,
        _lag_rel_threshold_svd,
    )
    _lag_kdmd.G = None
    _lag_kdmd.A = None

    _lag_sr_inv_sqrt = 1.0 / np.sqrt(_lag_sr)

    def _reduced_koopman_for_targets(_targets):
        _reduced_cross_kernel = np.zeros(
            (_lag_ur.shape[1], _lag_ur.shape[1]),
            dtype=np.result_type(_lag_ur.dtype, np.float64),
        )
        for _start in range(0, _targets.shape[0], _lag_kernel_batch_size):
            _end = min(_start + _lag_kernel_batch_size, _targets.shape[0])
            _cross_kernel_block = _lag_kernel(
                _targets[_start:_end],
                _lag_reference_data,
            )
            _reduced_cross_kernel += (
                _lag_ur[_start:_end].conj().T
                @ (_cross_kernel_block @ _lag_ur)
            )
            del _cross_kernel_block
        return (
            _lag_sr_inv_sqrt[:, None]
            * _reduced_cross_kernel
            * _lag_sr_inv_sqrt[None, :]
        )

    # Check the blockwise projection against the direct expression on a small
    # row subset before using it for the expensive lag sweep.
    _probe_size = min(32, _n_lag_train)
    _probe_targets = _first_targets[:_probe_size]
    _probe_kernel = _lag_kernel(_probe_targets, _lag_reference_data)
    _probe_direct = (
        _lag_ur[:_probe_size].conj().T @ (_probe_kernel @ _lag_ur)
    )
    _probe_batched = np.zeros_like(_probe_direct)
    _probe_batch_size = max(1, _probe_size // 3)
    for _start in range(0, _probe_size, _probe_batch_size):
        _end = min(_start + _probe_batch_size, _probe_size)
        _probe_block = _lag_kernel(
            _probe_targets[_start:_end],
            _lag_reference_data,
        )
        _probe_batched += (
            _lag_ur[_start:_end].conj().T @ (_probe_block @ _lag_ur)
        )
    if not np.allclose(_probe_batched, _probe_direct, rtol=1e-11, atol=1e-11):
        raise AssertionError("Batched reduced cross-kernel projection is inconsistent.")
    del _probe_kernel, _probe_direct, _probe_batched, _probe_block

    _lag_raw_spectra = []
    _lag_tracked_spectra = []
    for _lag_index, (_lag_month, _lag_step) in enumerate(
        zip(_lag_months, _lag_steps)
    ):
        print(_lag_month)
        if _lag_index == 0:
            _lag_kr = _first_kr
        else:
            _lag_targets = centered_data[_lag_origins + int(_lag_step)]
            _lag_kr = _reduced_koopman_for_targets(_lag_targets)

        _lag_spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(_lag_kr)
        _lag_dt_eff = float(_lag_step) * _trajectory_dt
        _lag_eigs_ct = _lag_spectrum.continuous_time_eigenvalues(_lag_dt_eff)
        if _lag_eigs_ct.size < _lag_n_modes + 1:
            raise ValueError(
                f"Lag {_lag_month} months produced fewer than "
                f"{_lag_n_modes} non-stationary eigenvalues."
            )
        _lag_raw_spectra.append(_lag_eigs_ct.copy())

        if _lag_index == 0:
            _tracked_at_lag = _lag_eigs_ct[1 : _lag_n_modes + 1].copy()
        else:
            _candidate_eigs = _lag_eigs_ct[1:]
            _matching_cost = np.abs(
                _lag_tracked_spectra[-1][:, None] - _candidate_eigs[None, :]
            )
            _matched_rows, _matched_columns = _linear_sum_assignment(_matching_cost)
            _tracked_at_lag = np.empty(_lag_n_modes, dtype=complex)
            _tracked_at_lag[_matched_rows] = _candidate_eigs[_matched_columns]
        _lag_tracked_spectra.append(_tracked_at_lag)

        if _lag_index > 0:
            del _lag_targets
        del _lag_kr, _lag_spectrum, _lag_eigs_ct
        _gc.collect()

    lag_months = _lag_months.copy()
    lag_eigenvalues_raw = np.vstack(_lag_raw_spectra)
    lag_eigenvalues_tracked = np.vstack(_lag_tracked_spectra)
    return lag_eigenvalues_tracked, lag_months


@app.cell
def _(YEAR, lag_eigenvalues_tracked, lag_months, plt):
    lag_spectrum_fig, (_lag_real_ax, _lag_imag_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 7),
        sharex=True,
    )
    for _mode_index in range(lag_eigenvalues_tracked.shape[1]):
        _mode_values_per_year = lag_eigenvalues_tracked[:, _mode_index] * YEAR
        _mode_label = rf"$\lambda_{{{_mode_index + 1}}}$"
        _lag_real_ax.plot(
            lag_months,
            _mode_values_per_year.real,
            marker="o",
            label=_mode_label,
        )
        _lag_imag_ax.plot(
            lag_months,
            _mode_values_per_year.imag,
            marker="o",
            label=_mode_label,
        )

    for _lag_ax in (_lag_real_ax, _lag_imag_ax):
        _lag_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        _lag_ax.grid(alpha=0.3, linestyle="--")
        _lag_ax.set_xticks(lag_months)

    _lag_real_ax.set_ylabel(r"$\mathrm{Re}\,\lambda$ [year$^{-1}$]", size=16)
    _lag_imag_ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [year$^{-1}$]", size=16)
    _lag_imag_ax.set_xlabel("Snapshot lag [months]", size=16)
    _lag_real_ax.legend(ncols=2)
    lag_spectrum_fig.tight_layout()
    lag_spectrum_fig
    return


@app.cell
def _(centered_data, spectrum):
    # Get the eigenfunctions evaluated on the data
    phi_vals = spectrum.evaluate_eigenfunctions(centered_data,batch_size=5_000)
    return (phi_vals,)


@app.cell
def _(X, centered_data, dt_eff, np, space_coord):
    x_grid = np.asarray(space_coord, dtype=float)
    if x_grid.ndim != 1 or x_grid.shape[0] != X.shape[1]:
        x_grid = np.linspace(0.0, 1.0, X.shape[1])
    else:
        x_min = x_grid.min()
        x_max = x_grid.max()
        if np.isclose(x_max, x_min):
            x_grid = np.linspace(0.0, 1.0, X.shape[1])
        else:
            x_grid = (x_grid - x_min) / (x_max - x_min)

    def clipped_cell_widths(xmin, xmax):
        edges = np.empty(x_grid.size + 1, dtype=float)
        if x_grid.size == 1:
            edges[0] = xmin
            edges[1] = xmax
        else:
            edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
            edges[0] = x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])
            edges[-1] = x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])

        clipped_left = np.maximum(edges[:-1], xmin)
        clipped_right = np.minimum(edges[1:], xmax)
        return np.maximum(clipped_right - clipped_left, 0.0)

    def weighted_mean(xmin, xmax, data):
        weights = np.cos(0.5 * np.pi * x_grid) * clipped_cell_widths(xmin, xmax)
        if np.count_nonzero(weights) < 2:
            raise ValueError(f"Not enough grid points in interval [{xmin}, {xmax}]")
        return np.average(data, axis=1, weights=weights)

    global_temperature = weighted_mean(0.0, 1.0,data=centered_data)
    delta_temperature = weighted_mean(0.0, 1.0 / 3.0,data=centered_data) - weighted_mean(
        1.0 / 3.0,
        1.0,data=centered_data
    )
    time_days = dt_eff * np.arange(X.shape[0])
    return delta_temperature, global_temperature


@app.cell
def _(YEAR, delta_temperature, eigs_ct, global_temperature, np, phi_vals, plt):
    from matplotlib.cm import ScalarMappable
    from scipy.stats import binned_statistic_2d
    grid4_bins = 80
    grid4_min_count = 4
    grid4_eigfunc_indices =  (1, 2, 3, 5,6)

    grid4_T = global_temperature
    grid4_dT = delta_temperature

    grid4_count, grid4_T_edges, grid4_dT_edges, _ = binned_statistic_2d(
        grid4_T,
        grid4_dT,
        None,
        statistic="count",
        bins=grid4_bins,
    )

    grid4_T_centers = 0.5 * (grid4_T_edges[:-1] + grid4_T_edges[1:])
    grid4_dT_centers = 0.5 * (grid4_dT_edges[:-1] + grid4_dT_edges[1:])

    grid4_fig, grid4_axes = plt.subplots(3, 2, figsize=(12, 9))
    grid4_axes = grid4_axes.ravel()

    grid4_eigs_ax = grid4_axes[0]
    grid4_eigs_ax.plot(eigs_ct.real * YEAR, eigs_ct.imag * YEAR, ".", ms=4)

    grid4_eigs_ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [year$^{-1}$]", size=16)
    grid4_eigs_ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [year$^{-1}$]", size=16)
    grid4_eigs_ax.set_xlim(left=-2.5, right=0.01)
    grid4_eigs_ax.set_ylim(bottom=-0.001, top=0.001)
    grid4_eigs_ax.grid(alpha=0.3)
    grid4_eigs_cbar = grid4_fig.colorbar(ScalarMappable(), ax=grid4_eigs_ax)
    grid4_eigs_cbar.ax.set_visible(False)

    for grid4_eig_idx, grid4_ax in zip(grid4_eigfunc_indices, grid4_axes[1:]):
        grid4_phi = phi_vals[:, grid4_eig_idx].real
        grid4_mean_phi, _, _, _ = binned_statistic_2d(
            grid4_T,
            grid4_dT,
            grid4_phi,
            statistic="mean",
            bins=[grid4_T_edges, grid4_dT_edges],
        )
        grid4_mean_phi_masked = np.ma.masked_where(
            grid4_count < grid4_min_count,
            grid4_mean_phi,
        )

        grid4_absmax = 0.75*np.nanmax(np.abs(grid4_mean_phi_masked))
        grid4_im = grid4_ax.pcolormesh(
            grid4_T_edges,
            grid4_dT_edges,
            grid4_mean_phi_masked.T,
            shading="auto",
            cmap="seismic",
            vmin=-grid4_absmax,
            vmax=grid4_absmax
        )

        # if np.ma.count(grid4_mean_phi_masked) > 0:
        #     grid4_levels = np.linspace(
        #         grid4_mean_phi_masked.min(),
        #         grid4_mean_phi_masked.max(),
        #         8,
        #     )
        #     if np.unique(grid4_levels).size > 1:
        #         grid4_ax.contour(
        #             grid4_T_centers,
        #             grid4_dT_centers,
        #             grid4_mean_phi_masked.T,
        #             levels=grid4_levels,
        #             colors="k",
        #             linewidths=0.8,
        #             alpha=0.9,
        #         )

        grid4_ax.set_title(rf"$\phi_{{{grid4_eig_idx}}}$", size=16)
        grid4_ax.set_xlim(left=-5, right=5)
        grid4_ax.set_ylim(bottom=-3, top=3)
        if grid4_eig_idx in (1, 2, 5):
            grid4_ax.set_ylabel(r"$\Delta T \quad  [K]$", size=16)
        grid4_fig.colorbar(
            grid4_im,
            ax=grid4_ax,
        )

    for grid4_ax in grid4_axes[4:]:
        grid4_ax.set_xlabel(r"$\overline{T} \quad [K]$", size=16)



    plt.tight_layout()
    plt.show()

    #grid4_fig.savefig("../figures/Koopman_Eigenfunctions_mu1.png",dpi=400)
    return


@app.cell
def _(
    X_snap,
    asymptotic_warm_ds,
    build_ivp_operator_from_dataset,
    centered_data,
    idx,
    meridional_heat_transfer_rate_watts_per_square_meter,
    np,
    plt,
    space_coord,
    spectrum,
    surface_albedo,
    tsvd,
    warm_ds,
):
    modes_to_show = (1, 5)

    _mode_observables_train = centered_data[:-1, :][idx]
    if _mode_observables_train.shape[0] != X_snap.shape[0]:
        raise ValueError("Local observables are not aligned with the KDMD training snapshots.")

    koopman_mode_matrix = np.column_stack(
        [
            spectrum.koopman_modes(
                _mode_observables_train[:, _latitude_index],
                U_r=tsvd.Ur,
                S_r=tsvd.Sr,
            )
            for _latitude_index in range(_mode_observables_train.shape[1])
        ]
    ).T

    _mode_coord = np.asarray(space_coord, dtype=float)
    if _mode_coord.ndim != 1 or _mode_coord.shape[0] != koopman_mode_matrix.shape[0]:
        raise ValueError("Mode coordinate does not match Koopman mode shape.")

    _warm_operator = build_ivp_operator_from_dataset(warm_ds)
    _warm_albedo = surface_albedo(
        asymptotic_warm_ds["temperature"].values,
        _warm_operator.empirical_fields.b_parameter[None, :],
        _warm_operator.empirical_fields.surface_height_offset[None, :],
        _warm_operator.params,
    )
    _positive_albedo_mask = np.asarray(warm_ds["latitude"] >= 0, dtype=bool)
    _albedo_coord = np.asarray(warm_ds["latitude"].values[_positive_albedo_mask], dtype=float)
    _albedo_mean = np.asarray(_warm_albedo[:, _positive_albedo_mask].mean(axis=0), dtype=float)

    _sorter = np.argsort(_albedo_coord)
    _albedo_coord = _albedo_coord[_sorter]
    _albedo_mean = _albedo_mean[_sorter]

    if _albedo_coord.shape != _mode_coord.shape or not np.allclose(_albedo_coord, _mode_coord):
        _albedo_mean_for_modes = np.interp(_mode_coord, _albedo_coord, _albedo_mean)
    else:
        _albedo_mean_for_modes = _albedo_mean

    _latitude = np.asarray(warm_ds["latitude"].values, dtype=float)
    _temperature = np.asarray(asymptotic_warm_ds["temperature"].values, dtype=float)
    _temperature_x = np.gradient(_temperature, _latitude, axis=1)
    _heat_flux = meridional_heat_transfer_rate_watts_per_square_meter(
        _latitude[None, :],
        _temperature,
        _temperature_x,
        _warm_operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
        _warm_operator.empirical_fields.latent_heat_flux_coefficient[None, :],
        _warm_operator.params,
    )
    _mean_heat_flux = _heat_flux.mean(axis=0)
    _positive_flux_mask = _latitude >= 0
    _flux_coord = _latitude[_positive_flux_mask]
    _mean_heat_flux_positive = _mean_heat_flux[_positive_flux_mask]
    _flux_sorter = np.argsort(_flux_coord)
    _flux_coord = _flux_coord[_flux_sorter]
    _mean_heat_flux_positive = _mean_heat_flux_positive[_flux_sorter]
    _mean_flux_derivative = np.gradient(_mean_heat_flux_positive, _flux_coord)
    if _flux_coord.shape != _mode_coord.shape or not np.allclose(_flux_coord, _mode_coord):
        _mean_flux_derivative_for_modes = np.interp(
            _mode_coord,
            _flux_coord,
            _mean_flux_derivative,
        )
    else:
        _mean_flux_derivative_for_modes = _mean_flux_derivative

    _flux_derivative_finite = _mean_flux_derivative_for_modes[
        np.isfinite(_mean_flux_derivative_for_modes)
    ]
    if _flux_derivative_finite.size:
        _flux_derivative_absmax = float(np.nanmax(np.abs(_flux_derivative_finite)))
    else:
        _flux_derivative_absmax = 1.0
    if _flux_derivative_absmax == 0.0:
        _flux_derivative_absmax = 1.0

    ice_line_latitude = np.nan
    _valid = np.isfinite(_mode_coord) & np.isfinite(_albedo_mean_for_modes)
    if np.count_nonzero(_valid) >= 2:
        _ice_x = _mode_coord[_valid]
        _ice_albedo = _albedo_mean_for_modes[_valid]
        _ice_sorter = np.argsort(_ice_x)
        _ice_x = _ice_x[_ice_sorter]
        _ice_delta = _ice_albedo[_ice_sorter] - 0.5
        _exact_crossings = np.flatnonzero(np.isclose(_ice_delta, 0.0, atol=1e-12))
        if _exact_crossings.size:
            ice_line_latitude = float(_ice_x[_exact_crossings[0]])
        else:
            _sign_crossings = np.flatnonzero(_ice_delta[:-1] * _ice_delta[1:] < 0.0)
            if _sign_crossings.size:
                _crossing_index = _sign_crossings[0]
                _x0 = _ice_x[_crossing_index]
                _x1 = _ice_x[_crossing_index + 1]
                _y0 = _ice_delta[_crossing_index]
                _y1 = _ice_delta[_crossing_index + 1]
                ice_line_latitude = float(_x0 - _y0 * (_x1 - _x0) / (_y1 - _y0))

    _mode_indices = tuple(
        _mode_index
        for _mode_index in modes_to_show
        if _mode_index < koopman_mode_matrix.shape[1]
    )
    if not _mode_indices:
        raise ValueError("No Koopman mode indices are available to plot.")

    _mode_values = np.concatenate(
        [
            koopman_mode_matrix[:, _mode_index].real
            for _mode_index in _mode_indices
        ]
    )
    _mode_values_finite = _mode_values[np.isfinite(_mode_values)]
    if _mode_values_finite.size:
        _mode_absmax = float(np.nanmax(np.abs(_mode_values_finite)))
    else:
        _mode_absmax = 1.0
    if _mode_absmax == 0.0:
        _mode_absmax = 1.0

    koopman_mode_fig, _mode_axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
    )
    _mode_axes = np.atleast_1d(_mode_axes).ravel()

    _flux_derivative_axes = []
    for _mode_ax, _mode_index in zip(_mode_axes, _mode_indices):

        _mode_profile = -koopman_mode_matrix[:, _mode_index]

        _mode_ax.plot(
            _mode_coord,
            _mode_profile.real,
            color="blue",
            label="Mode",
            linewidth=2
        )
        if np.isfinite(ice_line_latitude):
            _mode_ax.axvline(
                ice_line_latitude,
                color="tab:red",
                linestyle="--",
                lw=1.2,
            )
            _mode_ax.annotate(
                r"$x_{\alpha=0.5}$",
                xy=(ice_line_latitude, 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(4, 0),
                textcoords="offset points",
                color="tab:red",
                rotation=90,
                va="center",
                ha="left",
                fontsize= 16
            )

        _mode_ax.set_title(rf"Koopman mode for $\phi_{{{_mode_index}}}$", size=14)
        _mode_ax.grid(alpha=0.3)

        _flux_derivative_ax = _mode_ax.twinx()
        _flux_derivative_ax.plot(
            _mode_coord,
            _mean_flux_derivative_for_modes,
            color="0.45",
            linestyle="--",
            lw=1.2,
            alpha=0.75,
            label=r"$\partial_x j$",
        )
        _flux_derivative_ax.set_ylim(
            -1.05 * _flux_derivative_absmax,
            1.05 * _flux_derivative_absmax,
        )
        _flux_derivative_ax.tick_params(axis="y", labelcolor="0.35")
        _flux_derivative_axes.append(_flux_derivative_ax)

    for _unused_ax in _mode_axes[len(_mode_indices):]:
        _unused_ax.axis("off")

    for _mode_ax in _mode_axes:
        _mode_ax.set_xlabel(r"$x$", size=16)
    for _mode_ax in _mode_axes:
        _mode_ax.set_ylabel("Mode amplitude")
        _mode_ax.tick_params(axis="y")
    for _mode_ax in _mode_axes:
        _mode_ax.set_xlim(left=float(_mode_coord.min()), right=float(_mode_coord.max()))
        _mode_ax.set_ylim(-1.05 * _mode_absmax, 1.05 * _mode_absmax)
    for _flux_derivative_ax in _flux_derivative_axes[:-1]:
        _flux_derivative_ax.set_ylabel("")
        _flux_derivative_ax.tick_params(axis="y", labelright=False)
    if _flux_derivative_axes:
        _flux_derivative_axes[-1].set_ylabel(
            r"$\partial_x j \; [\mathrm{W}\,\mathrm{m}^{-2}]$",
            color="0.35",
        )

    koopman_mode_fig.tight_layout()
    koopman_mode_fig
    #koopman_mode_fig.savefig("../figures/Koopman_modes_mu0p972.png", dpi=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reconstructing Correlation Functions
    """)
    return


@app.cell
def _(X_snap, delta_temperature, global_temperature, idx, np):
    observable_avgT = global_temperature - np.mean(global_temperature)
    observable_DeltaT = delta_temperature - np.mean(delta_temperature)

    observable_avgT_train = observable_avgT[:-1][idx]
    observable_DeltaT_train = observable_DeltaT[:-1][idx]

    assert observable_avgT_train.shape[0] == X_snap.shape[0]
    assert observable_DeltaT_train.shape[0] == X_snap.shape[0]
    return observable_DeltaT_train, observable_avgT_train


@app.cell
def _(
    X_snap,
    eigs_ct,
    observable_DeltaT_train,
    observable_avgT_train,
    spectrum,
    tsvd,
):
    koopman_modes_avgT = spectrum.koopman_modes(
        observable_avgT_train,
        U_r=tsvd.Ur,
        S_r=tsvd.Sr,
    )

    koopman_modes_DeltaT = spectrum.koopman_modes(
        observable_DeltaT_train,
        U_r=tsvd.Ur,
        S_r=tsvd.Sr,
    )

    G_phi = spectrum.eigenfunction_gram(
        S_r=tsvd.Sr,
        n_samples=X_snap.shape[0],
        normalize=True,
    )

    corr_avgT_kdmd = spectrum.correlation_function_continuous(
        G_phi=G_phi,
        coeff_f=koopman_modes_avgT,
        coeff_g=koopman_modes_avgT,
        eigenvalues=eigs_ct,
    )

    corr_DeltaT_kdmd = spectrum.correlation_function_continuous(
        G_phi=G_phi,
        coeff_f=koopman_modes_DeltaT,
        coeff_g=koopman_modes_DeltaT,
        eigenvalues=eigs_ct,
    )
    return G_phi, corr_DeltaT_kdmd, corr_avgT_kdmd


@app.cell
def _(
    G_phi,
    X_snap,
    centered_data,
    eigs_ct,
    idx,
    local_corr,
    local_corr_lags,
    np,
    spectrum,
    tsvd,
):
    local_observables_train = centered_data[:-1, :][idx]

    if local_observables_train.shape[0] != X_snap.shape[0]:
        raise ValueError("Local observables are not aligned with the KDMD training snapshots.")
    if local_observables_train.shape[1] != local_corr.shape[1]:
        raise ValueError("Local observable count does not match the empirical local correlations.")

    local_corr_kdmd_columns = []
    for _latitude_index in range(local_observables_train.shape[1]):
        _koopman_modes = spectrum.koopman_modes(
            local_observables_train[:, _latitude_index],
            U_r=tsvd.Ur,
            S_r=tsvd.Sr,
        )
        _corr_kdmd = spectrum.correlation_function_continuous(
            G_phi=G_phi,
            coeff_f=_koopman_modes,
            coeff_g=_koopman_modes,
            eigenvalues=eigs_ct,
        )
        local_corr_kdmd_columns.append(_corr_kdmd(local_corr_lags))

    local_corr_kdmd = np.stack(local_corr_kdmd_columns, axis=1)
    if local_corr_kdmd.shape != local_corr.shape:
        raise ValueError("KDMD local correlations do not match the empirical correlation shape.")
    if not np.isfinite(local_corr_kdmd).all():
        raise ValueError("KDMD local correlations contain non-finite values.")
    return (local_corr_kdmd,)


@app.cell
def _(
    YEAR,
    local_corr,
    local_corr_kdmd,
    local_corr_lags,
    local_corr_latitude,
    np,
    plt,
):
    _lag_months = local_corr_lags / YEAR 
    _plot_stride = max(1, local_corr.shape[0] // 300)

    _latitude_grid, _lag_grid = np.meshgrid(
        local_corr_latitude,
        _lag_months[::_plot_stride],
    )

    _empirical_surface = local_corr[::_plot_stride, :]
    _kdmd_surface = local_corr_kdmd.real[::_plot_stride, :]
    _error_surface = _kdmd_surface - _empirical_surface

    _corr_absmax = np.nanmax(np.abs(np.concatenate([
        _empirical_surface.ravel(),
        _kdmd_surface.ravel(),
    ])))
    _error_absmax = np.nanmax(np.abs(_error_surface))

    _fig = plt.figure(figsize=(16, 5))
    _surface_specs = (
        (r"$C_T(x,t)$ empirical", _empirical_surface, "coolwarm", 0, _corr_absmax),
        (r"$C_T(x,t)$ KDMD", _kdmd_surface, "coolwarm", 0, _corr_absmax),
        (r"Difference", _error_surface, "coolwarm", -_error_absmax, _error_absmax),
    )

    for _panel_index, (_title, _values, _cmap, _vmin, _vmax) in enumerate(_surface_specs, start=1):
        _ax = _fig.add_subplot(1, 3, _panel_index, projection="3d")
        _surface = _ax.plot_surface(
            _latitude_grid,
            _lag_grid,
            _values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            linewidth=0,
            antialiased=True,
            rcount=_values.shape[0],
            ccount=_values.shape[1],
        )
        _ax.set_title(_title, fontsize=14)
        _ax.set_xlabel(r"$x$", fontsize=11)
        _ax.set_ylabel(r"$t \; [\mathrm{year}]$", fontsize=11)
        _ax.set_xlim(local_corr_latitude.min(), local_corr_latitude.max())
        _ax.set_ylim(_lag_months.min(), _lag_months.max())
        _ax.view_init(elev=28, azim=45)
        _fig.colorbar(_surface, ax=_ax, shrink=0.58, pad=0.12)

    _fig.tight_layout()
    _fig
    #_fig.savefig("../figures/spatial_correlation_functions.png",dpi=400)
    return


@app.cell
def _(
    YEAR,
    corr_DeltaT,
    corr_DeltaT_kdmd,
    corr_avgT,
    corr_avgT_kdmd,
    lags_DeltaT,
    lags_avgT,
    plt,
):
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
    _marker_stride = 10

    _ax1.plot(lags_avgT / YEAR , corr_avgT, color="blue", label=r"$\overline{T}$")
    _ax1.plot(
        lags_avgT[::_marker_stride] / YEAR ,
        corr_avgT_kdmd(lags_avgT[::_marker_stride]).real,
        color="blue",
        linestyle="none",
        marker=".",
        markersize=7,
        label=r"$\overline{T}$ KDMD",
    )
    _ax2.plot(lags_DeltaT / YEAR , corr_DeltaT, color="red", label=r"$\Delta T$")
    _ax2.plot(
        lags_DeltaT[::_marker_stride] / YEAR,
        corr_DeltaT_kdmd(lags_DeltaT[::_marker_stride]).real,
        color="red",
        linestyle="none",
        marker=".",
        markersize=7,
        label=r"$\Delta T$ KDMD",
    )

    _ax1.set_xlim(left=-1, right=40)

    align_zero_yaxis(_ax1, _ax2)


    _ax1.set_xlabel(r"$t \; [\mathrm{year}]$",fontsize=16)
    _ax1.set_ylabel(r"$C_{\overline{T}}$", color="blue",fontsize=16)
    _ax2.set_ylabel(r"$C_{\Delta T}$", color="red",fontsize=16)

    _ax1.tick_params(axis="y", labelcolor="blue",labelsize=12)
    _ax2.tick_params(axis="y", labelcolor="red",labelsize=12)
    _ax1.tick_params(axis="x",labelsize=12)

    _ax1.grid(alpha=0.3, linestyle="--")

    _fig.tight_layout()
    _fig
    #_fig.savefig("../figures/correlation_function_reconstruction_mu1.png",dpi=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Response Functions
    """)
    return


@app.cell
def _(data_dir, np, xr):
    response_input_filenames = tuple(
        f"response_co2_mu1_{{{_index}}}.nc"
        for _index in range(1, 3)
    )
    _response_datasets = tuple(
        xr.open_dataset(data_dir / _filename)
        for _filename in response_input_filenames
    )
    try:
        _reference_response = _response_datasets[0]
        _reference_time = np.asarray(_reference_response["time"].values, dtype=float)
        _reference_latitude = np.asarray(_reference_response["latitude"].values, dtype=float)

        _response_data_vars = {}
        for _variable_name in _reference_response.data_vars:
            _reference_variable = _reference_response[_variable_name]
            _variable_values = []

            for _filename, _dataset in zip(response_input_filenames, _response_datasets):
                if not np.array_equal(
                    np.asarray(_dataset["time"].values, dtype=float),
                    _reference_time,
                ):
                    raise ValueError(f"{_filename} has a mismatched response time grid.")
                if not np.array_equal(
                    np.asarray(_dataset["latitude"].values, dtype=float),
                    _reference_latitude,
                ):
                    raise ValueError(f"{_filename} has a mismatched response latitude grid.")
                if _variable_name not in _dataset:
                    raise ValueError(f"{_filename} is missing {_variable_name!r}.")
                if _dataset[_variable_name].dims != _reference_variable.dims:
                    raise ValueError(f"{_filename} has mismatched {_variable_name!r} dimensions.")

                _values = np.asarray(_dataset[_variable_name].values, dtype=float)
                if _values.shape != _reference_variable.shape:
                    raise ValueError(f"{_filename} has a mismatched {_variable_name!r} shape.")
                if not np.isfinite(_values).all():
                    raise ValueError(f"{_filename} contains non-finite {_variable_name!r} values.")
                _variable_values.append(_values)

            _response_data_vars[_variable_name] = (
                _reference_variable.dims,
                np.mean(np.stack(_variable_values, axis=0), axis=0),
                dict(_reference_variable.attrs),
            )

        for _attribute_name in ("stochastic_dt", "epsilon"):
            _reference_value = _reference_response.attrs[_attribute_name]
            for _filename, _dataset in zip(response_input_filenames, _response_datasets):
                if not np.isclose(_dataset.attrs[_attribute_name], _reference_value):
                    raise ValueError(f"{_filename} has a mismatched {_attribute_name!r} attribute.")

        response_numerical = xr.Dataset(
            data_vars=_response_data_vars,
            coords={
                "time": (
                    "time",
                    _reference_time,
                    dict(_reference_response["time"].attrs),
                ),
                "latitude": (
                    "latitude",
                    _reference_latitude,
                    dict(_reference_response["latitude"].attrs),
                ),
            },
            attrs=dict(_reference_response.attrs),
        )
        response_numerical.attrs["source_response_datasets"] = ", ".join(
            response_input_filenames
        )
        response_numerical.attrs["response_average_n_runs"] = len(response_input_filenames)
    finally:
        for _dataset in _response_datasets:
            _dataset.close()

    response_numerical
    return (response_numerical,)


@app.cell
def _(DAY, YEAR, np, plt, response_numerical):
    response_latitude = response_numerical["latitude"].values
    response_time_years = response_numerical["time"].values / YEAR
    response_green_function_per_year = response_numerical["green_function_temperature"] * DAY

    if response_green_function_per_year.ndim != 2:
        raise ValueError("Expected green_function_temperature with dimensions (time, latitude).")
    if response_green_function_per_year.shape != (response_time_years.size, response_latitude.size):
        raise ValueError("Response Green's function shape does not match time and latitude coordinates.")
    if not np.isfinite(response_green_function_per_year).all():
        raise ValueError("Response Green's function contains non-finite values.")

    _plot_stride = max(1, response_green_function_per_year.shape[0] // 300)
    _latitude_grid, _time_grid = np.meshgrid(
        response_latitude,
        response_time_years[::_plot_stride],
    )
    _surface_values = response_green_function_per_year[::_plot_stride, :]
    _response_min = np.nanmin(_surface_values)
    _response_max = np.nanmax(_surface_values)

    _fig = plt.figure(figsize=(9, 6))
    _ax = _fig.add_subplot(111, projection="3d")
    _surface = _ax.plot_surface(
        _latitude_grid,
        _time_grid,
        _surface_values,
        cmap="coolwarm",
        vmin=_response_min,
        vmax=_response_max,
        linewidth=0,
        antialiased=True,
        rcount=_surface_values.shape[0],
        ccount=_surface_values.shape[1],
    )

    _ax.set_xlabel(r"$x$", fontsize=14)
    _ax.set_ylabel(r"$t \; [\mathrm{year}]$", fontsize=14)
    _ax.set_zlabel(r"$G_T(x,t) \; [\mathrm{K}\,\mathrm{day}^{-1}]$", fontsize=14)
    _ax.set_xlim(response_latitude.min(), response_latitude.max())
    _ax.set_ylim(response_time_years.min(), response_time_years.max())
    _ax.view_init(elev=28, azim=45)
    _fig.colorbar(
        _surface,
        ax=_ax,
        shrink=0.65,
        pad=0.12,
        label=r"$G_T(x,t) \; [\mathrm{K}\,\mathrm{day}^{-1}]$",
    )
    _fig.tight_layout()
    #_fig.savefig("../figures/response_green_function_co2_mu1.png", dpi=400)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reconstructing the Response Function

    We reconstruct the numerical CO2 Green's function from the unperturbed
    KDMD spectrum via the fluctuation-dissipation route. The $m_1$ perturbation
    enters only the outgoing-longwave term
    $\mathrm{OLR} = \sigma T^4\,(1 - m_1 \tanh(c_3 T^6))$ and lives in the
    explicit reaction operator, so the phase-space perturbation direction is
    diagonal (local per grid point):
    $$
    X(T)_i = \frac{\partial (\text{reaction}_i)}{\partial m_1}
           = \frac{\sigma\,T_i^4\,\tanh(c_3 T_i^6)}{C(x_i)} ,
    $$
    with $C(x_i)$ the local heat capacity. Because the KDMD data are only
    centered (not scaled), no rescaling is needed and the reconstruction is in
    physical units.
    """)
    return


@app.cell
def _(X, build_ivp_operator_from_dataset, np, space_coord, warm_ds):
    # Diagonal m1-perturbation direction X(T)_i = sig T_i^4 tanh(c3 T_i^6) / C(x_i).
    # Evaluated on the RAW (physical) temperature X, since X(T) is nonlinear in T.
    _response_operator = build_ivp_operator_from_dataset(warm_ds)
    _response_params = _response_operator.params

    # Align the heat capacity C(x) onto the positive-latitude KDMD grid.
    _hc_x = np.asarray(_response_operator.empirical_fields.x, dtype=float)
    _hc = np.asarray(_response_operator.empirical_fields.heat_capacity, dtype=float)
    _hc_positive_mask = _hc_x >= 0
    _hc_x_positive = _hc_x[_hc_positive_mask]
    _hc_positive = _hc[_hc_positive_mask]
    _hc_sorter = np.argsort(_hc_x_positive)
    _hc_x_positive = _hc_x_positive[_hc_sorter]
    _hc_positive = _hc_positive[_hc_sorter]
    if _hc_x_positive.shape == space_coord.shape and np.allclose(_hc_x_positive, space_coord):
        heat_capacity_on_grid = _hc_positive
    else:
        heat_capacity_on_grid = np.interp(space_coord, _hc_x_positive, _hc_positive)

    perturbation_field = (
        _response_params.sig
        * X**4
        * np.tanh(_response_params.c3 * X**6)
        / heat_capacity_on_grid[None, :]
    )
    if perturbation_field.shape != X.shape:
        raise ValueError("Perturbation field shape does not match the temperature data.")
    if not np.isfinite(perturbation_field).all():
        raise ValueError("Perturbation field contains non-finite values.")
    return (perturbation_field,)


@app.cell
def _(G_phi, centered_data, np, perturbation_field, spectrum):
    # Response coefficients Gamma = G_phi^+ Delta with the full spatial perturbation
    # vector field. The kernel gradient is translation-invariant, so the centered
    # trajectory coordinates are used with the perturbation evaluated at raw T.
    _response_target_samples = 30_000
    _response_stride = max(1, centered_data.shape[0] // _response_target_samples)
    _response_trajectory = centered_data[::_response_stride]
    _response_perturbation = perturbation_field[::_response_stride]
    if _response_trajectory.shape != _response_perturbation.shape:
        raise ValueError("Response trajectory and perturbation field are misaligned.")

    Gamma = spectrum.response_coefficients_from_trajectory(
        trajectories=_response_trajectory,
        X_values=_response_perturbation,
        G_phi=G_phi,
        # delta_from_trajectory now contracts X·∇k via kernel.grad_x_dot, so the
        # per-batch cost is a (batch, n_ref) matrix (~1 GB here) rather than the
        # old (batch, n_ref, dim) gradient tensor. batch_size is no longer the
        # memory bottleneck.
        batch_size=5_000,
        show_progress=False,
    )
    if not np.isfinite(Gamma).all():
        raise ValueError("Response coefficients contain non-finite values.")
    return (Gamma,)


@app.cell
def _(
    G_phi,
    Gamma,
    centered_data,
    eigs_ct,
    idx,
    np,
    response_numerical,
    spectrum,
    tsvd,
):
    response_lags = np.asarray(response_numerical["time"].values, dtype=float) 
    response_local_observables_train = centered_data[:-1, :][idx]

    _response_kdmd_columns = []
    for _response_latitude_index in range(response_local_observables_train.shape[1]):
        _response_modes = spectrum.koopman_modes(
            response_local_observables_train[:, _response_latitude_index],
            U_r=tsvd.Ur,
            S_r=tsvd.Sr,
        )
        _response_greens = spectrum.correlation_function_continuous(
            G_phi=G_phi,
            coeff_f=_response_modes,
            coeff_g=Gamma,
            eigenvalues=eigs_ct,
        )
        _response_kdmd_columns.append(_response_greens(response_lags))

    G_kdmd = np.stack(_response_kdmd_columns, axis=1).real
    if not np.isfinite(G_kdmd).all():
        raise ValueError("KDMD Green's function contains non-finite values.")
    return (G_kdmd,)


@app.cell
def _(DAY, G_kdmd, YEAR, np, plt, response_numerical, space_coord):
    # Compare the KDMD reconstruction against the numerical Green's function on the
    # full globe. Each panel is rendered exactly like the standalone numerical plot
    # above (full mesh, coolwarm, own min/max) so the "numerical" panel is identical
    # to it. The KDMD analysis lives on the positive latitudes; the model is
    # north/south symmetric, so G(x) = G(|x|) mirrors it onto the southern hemisphere.

    # Numerical field on its native full-globe grid, sorted by latitude.
    _num_latitude = np.asarray(response_numerical["latitude"].values, dtype=float)
    _num_greens = np.asarray(
        response_numerical["green_function_temperature"].values, dtype=float
    )
    _num_sorter = np.argsort(_num_latitude)
    full_latitude = _num_latitude[_num_sorter]
    G_numerical = _num_greens[:, _num_sorter]

    # Positive-latitude KDMD field, sorted so np.interp sees increasing x.
    _pos_sorter = np.argsort(space_coord)
    _pos_latitude = space_coord[_pos_sorter]
    _G_kdmd_pos = G_kdmd[:, _pos_sorter]

    # Mirror onto the full latitude grid: the value at latitude y is the
    # reconstruction evaluated at |y|. Result shares G_numerical's exact grid.
    G_kdmd_full = np.vstack(
        [
            np.interp(np.abs(full_latitude), _pos_latitude, _G_kdmd_pos[_t])
            for _t in range(_G_kdmd_pos.shape[0])
        ]
    )
    if G_numerical.shape != G_kdmd_full.shape:
        raise ValueError("Numerical and KDMD Green's functions have mismatched shapes.")

    # Use the true response lag tau = t - dt: the numerical impulse is applied during
    # the first integrator step, so the response saved at time t is really R(t - dt),
    # and the KDMD reconstruction was evaluated on the same shifted grid. The first
    # sample (t=0, tau=-dt) is the shared unperturbed initial condition -- exactly 0 by
    # construction -- so it is dropped; the comparison starts at the first real step.
    _dt_response = float(response_numerical.attrs["stochastic_dt"])
    full_time_years = (
        np.asarray(response_numerical["time"].values, dtype=float) 
    )[1:] / YEAR

    G_numerical = G_numerical[1:]
    G_kdmd_full = G_kdmd_full[1:]

    # Same mesh, stride and units (K/day) as the standalone numerical plot.
    _plot_stride = max(1, G_numerical.shape[0] // 300)
    _latitude_grid, _time_grid = np.meshgrid(
        full_latitude,
        full_time_years[::_plot_stride],
    )

    _numerical_surface = G_numerical[::_plot_stride, :] * DAY
    _kdmd_surface = G_kdmd_full[::_plot_stride, :] * DAY
    _difference_surface = _kdmd_surface - _numerical_surface

    # Global least-squares amplitude ratio (validation only; surfaces not rescaled).
    _global_ratio = float(
        np.sum(G_kdmd_full * G_numerical) / np.sum(G_kdmd_full**2)
    )

    # Numerical and KDMD panels: own min/max, so colorbar and z-axis span the same
    # interval and the numerical panel matches the standalone plot. Difference panel:
    # symmetric about zero, with a robust limit (99th percentile of |difference|) so
    # the localized early-time spike at the poles -- a sharp feature the smooth KDMD
    # reconstruction cannot resolve -- does not stretch the axis past the typical,
    # much smaller difference.
    _diff_lim = float(np.percentile(np.abs(_difference_surface), 99.0))
    if _diff_lim <= 0.0:
        _diff_lim = float(np.nanmax(np.abs(_difference_surface))) or 1.0

    _response_fig = plt.figure(figsize=(16, 5))
    _response_surface_specs = (
        (
            r"$G_T(x,t)$ numerical",
            _numerical_surface,
            float(np.nanmin(_numerical_surface)),
            float(np.nanmax(_numerical_surface)),
        ),
        (
            r"$G_T(x,t)$ KDMD",
            _kdmd_surface,
            float(np.nanmin(_kdmd_surface)),
            float(np.nanmax(_kdmd_surface)),
        ),
        # Clip the plotted values to the robust limit so the pole spike caps flat at
        # the top of the box instead of shooting far above it as a saturated red wall.
        (r"Difference", np.clip(_difference_surface, -_diff_lim, _diff_lim), -_diff_lim, _diff_lim),
    )

    for _panel_index, (_title, _values, _vmin, _vmax) in enumerate(
        _response_surface_specs, start=1
    ):
        print(_panel_index)
        _ax = _response_fig.add_subplot(1, 3, _panel_index, projection="3d")
        _surface = _ax.plot_surface(
            _latitude_grid,
            _time_grid,
            _values,
            cmap="coolwarm",
            vmin=_vmin,
            vmax=_vmax,
            linewidth=0,
            antialiased=True,
            rcount=_values.shape[0],
            ccount=_values.shape[1],
        )
        _ax.set_title(_title, fontsize=14)
        _ax.set_xlabel(r"$x$", fontsize=11)
        _ax.set_ylabel(r"$t \; [\mathrm{year}]$", fontsize=11)
        if _panel_index == 1:
            _ax.set_zlabel(r"$G_T \; [\mathrm{K}\,\mathrm{day}^{-1}]$", fontsize=11)

        _ax.set_xlim(full_latitude.min(), full_latitude.max())
        _ax.set_ylim(full_time_years.min(), full_time_years.max())
        _ax.set_zlim(_vmin, _vmax)
        _ax.view_init(elev=28, azim=45)
        _response_fig.colorbar(_surface, ax=_ax, shrink=0.58, pad=0.12)


    _response_fig.tight_layout()
    _response_fig
    #_response_fig.savefig("../figures/response_reconstruction_co2_mu1.png", dpi=400)
    return G_kdmd_full, G_numerical, full_latitude, full_time_years


@app.cell
def _(DAY, G_kdmd_full, G_numerical, full_latitude, full_time_years, plt):
    # Spatial profiles G(x) of the numerical response and the KDMD reconstruction at
    # a single time.
    # Bump _profile_t_index for a later slice.
    _profile_t_index = 150
    _profile_time = float(full_time_years[_profile_t_index])

    _num_profile = G_numerical[_profile_t_index, :] * DAY
    _kdmd_profile = G_kdmd_full[_profile_t_index, :] * DAY

    _profile_fig, _profile_ax = plt.subplots(figsize=(8, 5))
    _profile_ax.plot(full_latitude, _num_profile, color="black", lw=2, label="numerical")
    _profile_ax.plot(
        full_latitude, _kdmd_profile, color="crimson", lw=2, ls="--", label="KDMD"
    )
    _profile_ax.set_xlabel(r"$x$", fontsize=14)
    _profile_ax.set_ylabel(r"$G_T(x,t) \; [\mathrm{K}\,\mathrm{day}^{-1}]$", fontsize=14)
    _profile_ax.set_title(
        rf"Response profile at $t = {_profile_time:.2f}$ year", fontsize=14
    )
    _profile_ax.set_xlim(full_latitude.min(), full_latitude.max())
    _profile_ax.set_ylim(bottom = 0, top=0.4)
    _profile_ax.legend(frameon=False, fontsize=12)
    _profile_ax.grid(True, alpha=0.3)
    _profile_fig.tight_layout()
    _profile_fig
    return


@app.cell
def _(
    G_kdmd_full,
    G_numerical,
    YEAR,
    asymptotic_warm_ds,
    build_ivp_operator_from_dataset,
    full_latitude,
    full_time_years,
    np,
    plt,
    response_numerical,
    surface_albedo,
    warm_ds,
):
    # Sensitivity profile S_T(x) = integral G_T(x, t) dt. G_T is in K/s, so
    # integrating over seconds gives S_T in K. The initial zero sample was already
    # dropped in the comparison cell that produced G_numerical and G_kdmd_full.
    full_time_seconds = full_time_years * YEAR

    if G_numerical.shape != G_kdmd_full.shape:
        raise ValueError("Numerical and KDMD Green's functions have mismatched shapes.")
    if G_numerical.shape != (full_time_seconds.size, full_latitude.size):
        raise ValueError("Green's function shape does not match time and latitude coordinates.")
    if not np.isfinite(G_numerical).all() or not np.isfinite(G_kdmd_full).all():
        raise ValueError("Green's function contains non-finite values.")

    sensitivity_numerical = np.trapezoid(G_numerical, x=full_time_seconds, axis=0) * response_numerical.attrs["epsilon"]
    sensitivity_kdmd = np.trapezoid(G_kdmd_full, x=full_time_seconds, axis=0) * response_numerical.attrs["epsilon"]

    if sensitivity_numerical.shape != full_latitude.shape:
        raise ValueError("Numerical sensitivity shape does not match latitude coordinates.")
    if sensitivity_kdmd.shape != full_latitude.shape:
        raise ValueError("KDMD sensitivity shape does not match latitude coordinates.")
    if not np.isfinite(sensitivity_numerical).all() or not np.isfinite(sensitivity_kdmd).all():
        raise ValueError("Sensitivity profile contains non-finite values.")

    _warm_operator = build_ivp_operator_from_dataset(warm_ds)
    _warm_albedo = surface_albedo(
        asymptotic_warm_ds["temperature"].values,
        _warm_operator.empirical_fields.b_parameter[None, :],
        _warm_operator.empirical_fields.surface_height_offset[None, :],
        _warm_operator.params,
    )
    _albedo_coord = np.asarray(warm_ds["latitude"].values, dtype=float)
    _albedo_mean = np.asarray(_warm_albedo.mean(axis=0), dtype=float)
    _albedo_sorter = np.argsort(_albedo_coord)
    _albedo_coord = _albedo_coord[_albedo_sorter]
    _albedo_delta = _albedo_mean[_albedo_sorter] - 0.5

    _ice_lines = []
    _valid_albedo = np.isfinite(_albedo_coord) & np.isfinite(_albedo_delta)
    if np.count_nonzero(_valid_albedo) >= 2:
        _ice_x = _albedo_coord[_valid_albedo]
        _ice_delta = _albedo_delta[_valid_albedo]
        _exact_crossings = np.flatnonzero(np.isclose(_ice_delta, 0.0, atol=1e-12))
        _ice_lines.extend(float(_ice_x[_index]) for _index in _exact_crossings)
        _sign_crossings = np.flatnonzero(_ice_delta[:-1] * _ice_delta[1:] < 0.0)
        for _crossing_index in _sign_crossings:
            _x0 = _ice_x[_crossing_index]
            _x1 = _ice_x[_crossing_index + 1]
            _y0 = _ice_delta[_crossing_index]
            _y1 = _ice_delta[_crossing_index + 1]
            _ice_lines.append(float(_x0 - _y0 * (_x1 - _x0) / (_y1 - _y0)))
    ice_line_latitudes = np.asarray(sorted(set(np.round(_ice_lines, 12))), dtype=float)

    _sensitivity_fig, _sensitivity_ax = plt.subplots(figsize=(8, 5))
    _sensitivity_ax.plot(
        full_latitude,
        sensitivity_numerical,
        color="black",
        lw=2,
        label="numerical",
    )
    _sensitivity_ax.plot(
        full_latitude,
        sensitivity_kdmd,
        color="crimson",
        lw=2,
        ls="--",
        label="KDMD",
    )
    for _ice_line_index, _ice_line_latitude in enumerate(ice_line_latitudes):
        _sensitivity_ax.axvline(
            _ice_line_latitude,
            color="tab:red",
            linestyle="--",
            lw=1.2,
        )
        _sensitivity_ax.annotate(
            r"$x_{\alpha=0.5}$",
            xy=(_ice_line_latitude, 0.2),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            color="tab:red",
            rotation=90,
            va="center",
            ha="left",
            fontsize=16,
        )
    _sensitivity_ax.set_xlabel(r"$x$", fontsize=14)
    _sensitivity_ax.set_ylabel(r"$ \varepsilon \int G_T(x,t)\,dt \; [\mathrm{K}]$", fontsize=14)
    _sensitivity_ax.set_title("Sensitivity profile", fontsize=14)
    _sensitivity_ax.set_xlim(full_latitude.min(), full_latitude.max())
    _sensitivity_ax.legend(frameon=False, fontsize=12)
    _sensitivity_ax.grid(True, alpha=0.3)
    _sensitivity_fig.tight_layout()
    _sensitivity_ax.set_xlim(left=0,right=1)
    _sensitivity_fig
    return ice_line_latitudes, sensitivity_kdmd, sensitivity_numerical


@app.cell
def _(
    data_dir,
    full_latitude,
    ice_line_latitudes,
    response_numerical,
    sensitivity_kdmd,
    sensitivity_numerical,
    xr,
):
    sensitivity_dataset = xr.Dataset(
        data_vars={
            "sensitivity": (
                ("latitude",),
                sensitivity_numerical,
                {
                    "units": "K",
                    "long_name": "epsilon-scaled time-integrated temperature Green's function",
                    "description": "epsilon * integral G_T(x, t) dt using the first-sample-dropped response grid",
                },
            ),
            "sensitivity_kdmd": (
                ("latitude",),
                sensitivity_kdmd,
                {
                    "units": "K",
                    "long_name": "epsilon-scaled time-integrated KDMD temperature Green's function",
                    "description": "epsilon * integral G_T^KDMD(x, t) dt using the first-sample-dropped response grid",
                },
            ),
            "ice_line_latitude": (
                ("ice_line",),
                ice_line_latitudes,
                {
                    "units": "1",
                    "long_name": "latitude where mean albedo equals 0.5",
                },
            ),
        },
        coords={
            "latitude": (
                "latitude",
                full_latitude,
                {"units": "1", "long_name": "normalized latitude"},
            ),
            "ice_line": (
                "ice_line",
                range(ice_line_latitudes.size),
                {"long_name": "ice-line crossing index"},
            ),
        },
        attrs=dict(response_numerical.attrs),
    )
    sensitivity_dataset.attrs["source_response_dataset"] = "response_co2_mu1.nc"
    sensitivity_dataset.attrs["sensitivity_definition"] = (
        "epsilon * integral G_T(x, t) dt; G_T has units K s^-1 and time is integrated in seconds"
    )

    sensitivity_output_path = data_dir / "sensitivity_co2_mu1.nc"
    sensitivity_dataset.to_netcdf(sensitivity_output_path, engine="scipy")
    sensitivity_output_path
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tipping Points

    This sweep repeats the KDMD calculation for several stochastic runs and is
    memory intensive. The single-run Koopman objects from the previous section
    (`X_snap`, `Y_snap`, `centered_data`, `kernel_weight`, `kdmd`, `tsvd`, `spectrum`, and
    `phi_vals`) are not needed below and can be safely removed before running
    the sweep.
    """)
    return


@app.cell
def _():
    import gc as _gc

    # The tipping sweep below rebuilds KDMD objects for five datasets. Drop the
    # large single-run Koopman objects before starting the sweep.
    for _name in (
        "X_snap",
        "Y_snap",
        "centered_data",
        "kernel_weight",
        "kdmd",
        "tsvd",
        "spectrum",
        "phi_vals",
    ):
        globals().pop(_name, None)
    _gc.collect()
    return


@app.cell
def _(
    KernelDMD,
    KoopmanSpectrumKDMD,
    TSVDRegularizer,
    WeightedGaussianKernel,
    YEAR,
    cosine_trapezoid_weights,
    data_dir,
    make_snapshots,
    n_snapshots_training,
    np,
    pdist,
    xr,
):
    import gc as _gc

    _tipping_dataset_specs = (
        (1.00, "new_stochastic_warm_mu1_{1}.nc"),
        (0.99, "new_stochastic_warm_mu0p99_{1}.nc"),
        (0.98, "new_stochastic_warm_mu0p98_{1}.nc"),
        (0.975, "new_stochastic_warm_mu0p975_{1}.nc"),
        (0.972, "new_stochastic_warm_mu0p972_{1}.nc"),
    )
    _tipping_rng_seed = 0
    _tipping_rel_threshold_svd_temporary = 1e-3
    _tipping_rel_threshold_svd = 5e-3
    _tipping_n_eigenvalues = 10
    _tipping_transient_years = 500.0

    def _tipping_centered_positive_temperature_and_weights(_dataset):
        # Mirror the single-dataset Koopman preprocessing: use the northern
        # hemisphere, remove the temporal mean, and put quadrature weights in
        # the kernel metric.
        _latitude_condition = _dataset["latitude"] >= 0
        _sliced_dataset = _dataset.where(cond=_latitude_condition, drop=True)
        _space_coord = np.asarray(_sliced_dataset["latitude"].values, dtype=float)
        _temperature = np.asarray(_sliced_dataset["temperature"].values, dtype=float)

        _mean_field = _temperature.mean(axis=0)
        _centered_data = _temperature - _mean_field[None, :]
        _kernel_weight = cosine_trapezoid_weights(
            _space_coord,
            n_features=_temperature.shape[1],
        )
        return _centered_data, _kernel_weight

    def _tipping_koopman_eigenvalues(_dataset_path, _dataset_index):
        # Load one dataset at a time, discard the same initial transient used in
        # the single-dataset KDMD analysis, and keep only centered coordinates
        # with their spatial kernel weights.
        # Closing the xarray dataset early helps keep memory bounded.
        _dataset = xr.open_dataset(_dataset_path)
        try:
            _mu_value = float(_dataset.attrs["param_mu"])
            _asymptotic_dataset = _dataset.where(
                cond=_dataset["time"] > _tipping_transient_years * YEAR,
                drop=True,
            )
            # Saved-snapshot spacing is stochastic_dt * save_every; derive it from
            # the time coordinate so continuous-time eigenvalues use the correct
            # timescale (raw stochastic_dt would be too fast by save_every).
            _dt_value = float(np.median(np.diff(_asymptotic_dataset["time"].values)))
            _expected_dt_value = (
                _dataset.attrs["stochastic_dt"] * _dataset.attrs["stochastic_save_every"]
            )
            assert np.isclose(_dt_value, _expected_dt_value, rtol=1e-6), (
                _dt_value,
                _expected_dt_value,
            )
            _centered_data, _kernel_weight = (
                _tipping_centered_positive_temperature_and_weights(_asymptotic_dataset)
            )
        finally:
            _dataset.close()

        _x_snap, _y_snap, _dt_eff_value = make_snapshots(_centered_data, dt=_dt_value)
        _n_train = min(int(n_snapshots_training), _x_snap.shape[0])

        # Use a deterministic but distinct subsample for each mu value. The
        # kernel matrices scale quadratically with _n_train, so this is the
        # main memory control knob for the sweep.
        _rng = np.random.default_rng(_tipping_rng_seed + _dataset_index)
        _idx = _rng.choice(_x_snap.shape[0], size=_n_train, replace=False)
        _x_snap = _x_snap[_idx]
        _y_snap = _y_snap[_idx]

        # Fit KDMD using the same median-distance kernel bandwidth and TSVD
        # thresholds as the single-dataset analysis above.
        _weighted_for_bandwidth = (
            _centered_data[_idx] * np.sqrt(_kernel_weight)[None, :]
        )
        _sigma_median = float(
            np.median(pdist(_weighted_for_bandwidth, metric="euclidean"))
        )
        _kdmd = KernelDMD(
            kernel=WeightedGaussianKernel(
                sigma=_sigma_median,
                weights=_kernel_weight,
            )
        )
        _kdmd.fit_snapshots(X=_x_snap, Y=_y_snap)

        _tsvd = TSVDRegularizer()
        # symmetrize=False: Kxx is already symmetric, avoids a full-size copy.
        # Free each Gram as soon as it is consumed to cap the per-mu peak.
        _tsvd.factorize(
            _kdmd.G,
            method="eigsh",
            symmetrize=False,
            rel_threshold=_tipping_rel_threshold_svd_temporary,
        )
        _kdmd.G = None
        _kr, _ur, _sr = _tsvd.solve_from_factorization(
            _kdmd.A,
            _tipping_rel_threshold_svd,
        )
        _kdmd.A = None
        _spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
            _kr,
            kernel=_kdmd.kernel,
            reference_data=_kdmd.reference_data,
            U_r=_ur,
            S_r=_sr,
        )
        _eigs_ct = _spectrum.continuous_time_eigenvalues(_dt_eff_value)

        # Keep the static mode in the evaluation output. Plotting cells can
        # choose to skip it without forcing this expensive sweep to rerun.
        _leading_eigs = _eigs_ct[:_tipping_n_eigenvalues]
        if _leading_eigs.size < _tipping_n_eigenvalues:
            _leading_eigs = np.pad(
                _leading_eigs,
                (0, _tipping_n_eigenvalues - _leading_eigs.size),
                constant_values=np.nan + 0j,
            )
        _leading_eigs = _leading_eigs.copy()

        del (
            _centered_data,
            _kernel_weight,
            _weighted_for_bandwidth,
            _x_snap,
            _y_snap,
            _idx,
            _kdmd,
            _tsvd,
            _kr,
            _ur,
            _sr,
            _spectrum,
            _eigs_ct,
        )
        _gc.collect()
        return _mu_value, _leading_eigs

    # Process datasets sequentially so each KDMD fit can release memory before
    # the next mu value is loaded.
    _tipping_records = []
    for _dataset_index, (_expected_mu, _filename) in enumerate(_tipping_dataset_specs):
        _tipping_records.append(
            _tipping_koopman_eigenvalues(data_dir / _filename, _dataset_index)
        )
        _gc.collect()
    _tipping_records = sorted(_tipping_records, key=lambda _record: _record[0])

    tipping_mu_values = np.asarray([_record[0] for _record in _tipping_records], dtype=float)
    tipping_eigenvalues = np.vstack([_record[1] for _record in _tipping_records])
    return tipping_eigenvalues, tipping_mu_values


@app.cell
def _(DAY, YEAR, np, plt, tipping_eigenvalues, tipping_mu_values):
    # Keep the raw KDMD ordering available, but plot a tracked copy. Manual
    # inspection shows that phi_5 at mu=0.975 continues as raw phi_6 at
    # mu=0.972, so relabel only that endpoint in the displayed branches.
    tipping_tracked_eigenvalues = tipping_eigenvalues.copy()
    _mode_swap_row_indices = np.flatnonzero(np.isclose(tipping_mu_values, 0.972))
    for _mode_swap_row in _mode_swap_row_indices:
        tipping_tracked_eigenvalues[_mode_swap_row, 5] = tipping_eigenvalues[
            _mode_swap_row,
            6,
        ]
        tipping_tracked_eigenvalues[_mode_swap_row, 6] = tipping_eigenvalues[
            _mode_swap_row,
            5,
        ]

    # Column 0 is the static mode. Plot the first six non-static modes.
    _selected_eigenvalue_indices = (1, 2, 3, 4, 5)
    tipping_selected_eigenvalues = tipping_tracked_eigenvalues[
        :,
        _selected_eigenvalue_indices,
    ]
    tipping_selected_eigenvalues_per_day = tipping_selected_eigenvalues.real * DAY
    with np.errstate(divide="ignore", invalid="ignore"):
        tipping_selected_timescales_months = np.abs(
            1.0 / tipping_selected_eigenvalues.real
        ) / YEAR 
    tipping_selected_timescales_months = np.where(
        np.isfinite(tipping_selected_timescales_months),
        tipping_selected_timescales_months,
        np.nan,
    )

    # The first panel shows decay rates, while the second shows the equivalent
    # timescales in months.
    tipping_spectrum_fig,  _tipping_timescale_ax = plt.subplots(
        nrows=1,
        ncols=1,
    )
    for _plot_column, _eigenvalue_index in enumerate(_selected_eigenvalue_indices):
        _label = rf"$\lambda_{{{_eigenvalue_index}}}$"

        _tipping_timescale_ax.plot(
            tipping_mu_values,
            tipping_selected_timescales_months[:, _plot_column],
            marker="o",
            label=_label,
        )

    for _ax in [_tipping_timescale_ax]:
        _ax.set_xlabel(r"$\mu$", size=16)
        _ax.set_xlim(float(tipping_mu_values.max()), float(tipping_mu_values.min()))
        _ax.grid(alpha=0.3, linestyle="--")


    _tipping_handles, _tipping_labels = _tipping_timescale_ax.get_legend_handles_labels()
    tipping_spectrum_fig.legend(
        _tipping_handles,
        _tipping_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=16
    )


    _tipping_timescale_ax.set_ylabel(r"$\tau$ [year]", size=16)
    tipping_spectrum_fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    #tipping_spectrum_fig.savefig("../figures/tipping_eigenvalues.png",dpi=400)
    tipping_spectrum_fig.savefig("../figures/tipping_timescale.png",dpi=400)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
