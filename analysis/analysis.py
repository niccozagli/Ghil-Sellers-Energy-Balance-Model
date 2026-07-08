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
    return (warm_ds,)


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
    return asymptotic_warm_ds, condition


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation functions
    """)
    return


@app.cell
def _(Delta_T, avg_T, condition, np):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    avg_T_asymp = avg_T.where(cond=condition,drop=True)
    Delta_T_asymp = Delta_T.where(cond=condition,drop=True)

    # Sampling interval of the SAVED series is stochastic_dt * save_every, not the
    # raw solver step. Derive it from the time coordinate so save_every is always
    # accounted for (otherwise correlation/Koopman timescales are too fast by
    # save_every, e.g. 30x -> months instead of years).
    dt = float(np.median(np.diff(avg_T_asymp["time"].values)))  # in seconds
    _expected_dt = (
        avg_T_asymp.attrs["stochastic_dt"] * avg_T_asymp.attrs["stochastic_save_every"]
    )
    assert np.isclose(dt, _expected_dt, rtol=1e-6), (dt, _expected_dt)

    obs_avgT = avg_T_asymp["temperature"].values
    lags_avgT, corr_avgT = _cross_correlation(x=obs_avgT,y=obs_avgT,dt=dt,normalization="biased")


    obs_DeltaT = Delta_T_asymp["temperature"].values
    lags_DeltaT, corr_DeltaT = _cross_correlation(x=obs_DeltaT,y=obs_DeltaT,dt=dt,normalization="biased")
    return corr_DeltaT, corr_avgT, dt, lags_DeltaT, lags_avgT


@app.cell
def _(YEAR, asymptotic_warm_ds, dt, np):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation
    _local_corr_lag_window_years = 25.0


    _local_corr_latitude_condition = asymptotic_warm_ds["latitude"] >= 0
    local_corr_ds = asymptotic_warm_ds.where(cond=_local_corr_latitude_condition, drop=True)
    local_corr_latitude = local_corr_ds["latitude"].values
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
    local_corr_lags = None

    for _latitude_index in range(local_temperature.shape[1]):
        _lags, _corr = _cross_correlation(
            x=local_temperature[:, _latitude_index],
            y=local_temperature[:, _latitude_index],
            dt=dt,
            max_lag=local_corr_max_lag,
            normalization="biased",
        )
        if local_corr_lags is None:
            local_corr_lags = _lags
        local_corr_columns.append(_corr)

    local_corr = np.stack(local_corr_columns, axis=1)
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


@app.cell
def _():
    # def limits_with_zero_fraction(ax, zero_fraction=0.15):
    #       ymin, ymax = ax.get_ylim()

    #       ymax = max(ymax, 0)
    #       ymin = min(ymin, 0)

    #       above = ymax
    #       below = -ymin

    #       required_below = zero_fraction / (1 - zero_fraction) * above
    #       required_above = (1 - zero_fraction) / zero_fraction * below

    #       if below < required_below:
    #           ymin = -required_below
    #       if above < required_above:
    #           ymax = required_above

    #       return ymin, ymax


    # def align_zero_yaxis(ax1, ax2, zero_fraction=0.15):
    #       ax1.set_ylim(*limits_with_zero_fraction(ax1, zero_fraction))
    #       ax2.set_ylim(*limits_with_zero_fraction(ax2, zero_fraction))


    # _fig, _ax1 = plt.subplots()
    # _ax2 = _ax1.twinx()

    # _ax1.plot(lags_avgT / YEAR * 12, corr_avgT, color="tab:blue", label=r"$\overline{T}$")
    # _ax2.plot(lags_DeltaT / YEAR * 12, corr_DeltaT, color="tab:red", label=r"$\Delta T$")

    # _ax1.set_xlim(left=-1, right=12)

    # align_zero_yaxis(_ax1, _ax2)


    # _ax1.set_xlabel(r"$\mathrm{Lag}\;[\mathrm{months}]$")
    # _ax1.set_ylabel(r"$C_{\overline{T}}$", color="tab:blue")
    # _ax2.set_ylabel(r"$C_{\Delta T}$", color="tab:red")

    # _ax1.tick_params(axis="y", labelcolor="tab:blue")
    # _ax2.tick_params(axis="y", labelcolor="tab:red")

    # _ax1.grid(alpha=0.3, linestyle="--")

    # _lines1, _labels1 = _ax1.get_legend_handles_labels()
    # _lines2, _labels2 = _ax2.get_legend_handles_labels()
    # _ax1.legend(_lines1 + _lines2, _labels1 + _labels2)

    # _fig.tight_layout()
    # _fig
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

    from koopman_response.algorithms import GaussianKernel, KernelDMD
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.utils.preprocessing import make_snapshots
    from koopman_response.algorithms.regularization import TSVDRegularizer

    return (
        GaussianKernel,
        KernelDMD,
        KoopmanSpectrumKDMD,
        TSVDRegularizer,
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
def _(X, dt, make_snapshots, n_snapshots_training, np, rng, space_coord):
    # Pre-processing
    # 1. We target fluctuations: we remove the temporal mean at each point
    # 2. We apply a quadrature-aware rescaling, the scalar product has cos(pi/2 x)

    ######## QUADRATURE-AWARE RESCALING #########
    # Build a normalized spatial grid in [0, 1]
    _x_grid = np.asarray(space_coord, dtype=float)

    _x_min = _x_grid.min()
    _x_max = _x_grid.max()
    if np.isclose(_x_max, _x_min):
        _x_grid = np.linspace(0.0, 1.0, X.shape[1])
    else:
        _x_grid = (_x_grid - _x_min) / (_x_max - _x_min)

    # Remove the temporal mean at each grid point
    mean_field = X.mean(axis=0)
    centered_data = X - mean_field[None, :]

    # Use trapezoidal quadrature weights so the kernel metric approximates
    # the weighted L2 norm with weight cos(pi x / 2).
    dx_weight = np.empty_like(_x_grid)
    if _x_grid.size < 2:
        dx_weight[...] = 1.0
    else:
        dx_weight[0] = 0.5 * (_x_grid[1] - _x_grid[0])
        dx_weight[-1] = 0.5 * (_x_grid[-1] - _x_grid[-2])
        if _x_grid.size > 2:
            dx_weight[1:-1] = 0.5 * (_x_grid[2:] - _x_grid[:-2])

    cos_weight = np.cos(0.5 * np.pi * _x_grid)
    kernel_weight = np.clip(cos_weight * dx_weight, a_min=0.0, a_max=None)
    scaled_data = centered_data * np.sqrt(kernel_weight)[None, :]

    #########################################
    # Snapshot data and sub-sampling
    X_snap, Y_snap, dt_eff = make_snapshots(scaled_data, dt=dt)
    n_train = min(n_snapshots_training, X_snap.shape[0])
    idx = rng.choice(X_snap.shape[0], size=n_train, replace=False)
    X_snap = X_snap[idx]
    Y_snap = Y_snap[idx]
    return X_snap, Y_snap, centered_data, dt_eff, idx, scaled_data


@app.cell
def _(
    GaussianKernel,
    KernelDMD,
    TSVDRegularizer,
    X_snap,
    Y_snap,
    idx,
    np,
    pdist,
    scaled_data,
):
    # Kernel DMD algorithm
    rel_threshold_svd_temporary = 1e-3

    sigma_median = float(np.median(pdist(scaled_data[idx], metric="euclidean")))

    kdmd = KernelDMD(kernel=GaussianKernel(sigma=sigma_median))
    kdmd.fit_snapshots(X=X_snap, Y=Y_snap)

    tsvd = TSVDRegularizer()
    tsvd.factorize(kdmd.G, method="eigsh", rel_threshold=rel_threshold_svd_temporary)
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
def _(KoopmanSpectrumKDMD, dt_eff, kdmd, tsvd):
    rel_threshold_svd = 5e-3
    Kr, Ur, Sr = tsvd.solve_from_factorization(
        kdmd.A,
        rel_threshold_svd
    )
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        Kr,
        kernel=kdmd.kernel,
        reference_data=kdmd.reference_data,
        U_r=Ur,
        S_r=Sr,
    )
    eigs_ct = spectrum.continuous_time_eigenvalues(dt_eff)
    return eigs_ct, spectrum


@app.cell
def _(YEAR, eigs_ct, plt):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eigs_ct.real * YEAR , eigs_ct.imag * YEAR    , ".", ms=4)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [year$^{-1}$]",size=16)
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [year$^{-1}$]",size=16)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-2, right=0.01)
    ax.set_ylim(bottom=-0.001, top=0.001)
    # fig.savefig("figures/eigenvalues_near.png",dpi=400)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(scaled_data, spectrum):
    # Get the eigenfunctions evaluated on the data
    phi_vals = spectrum.evaluate_eigenfunctions(scaled_data,batch_size=5_000)
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
        for _mode_index in (1, 6)
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

    _ax1.set_xlim(left=-1, right=25)

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
def _(data_dir, xr):
    response_numerical = xr.open_dataset(data_dir / "response_co2_mu1.nc")
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
    _fig
    _fig.savefig("../figures/response_green_function_mu1.png", dpi=400)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tipping Points

    This sweep repeats the KDMD calculation for several stochastic runs and is
    memory intensive. The single-run Koopman objects from the previous section
    (`X_snap`, `Y_snap`, `scaled_data`, `kdmd`, `tsvd`, `spectrum`, and
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
        "scaled_data",
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
    GaussianKernel,
    KernelDMD,
    KoopmanSpectrumKDMD,
    TSVDRegularizer,
    YEAR,
    data_dir,
    make_snapshots,
    n_snapshots_training,
    np,
    pdist,
    xr,
):
    import gc as _gc

    _tipping_dataset_specs = (
        (1.00, "new_stochastic_warm_mu1.nc"),
        (0.99, "new_stochastic_warm_mu0p99.nc"),
        (0.98, "new_stochastic_warm_mu0p98.nc"),
        (0.975, "new_stochastic_warm_mu0p975.nc"),
        (0.972, "new_stochastic_warm_mu0p972.nc"),
    )
    _tipping_rng_seed = 0
    _tipping_rel_threshold_svd_temporary = 1e-3
    _tipping_rel_threshold_svd = 5e-3
    _tipping_n_eigenvalues = 10
    _tipping_transient_years = 500.0

    def _tipping_scaled_positive_temperature(_dataset):
        # Mirror the single-dataset Koopman preprocessing: use the northern
        # hemisphere, remove the temporal mean, and apply quadrature weights so
        # Euclidean distances approximate the weighted latitude norm.
        _latitude_condition = _dataset["latitude"] >= 0
        _sliced_dataset = _dataset.where(cond=_latitude_condition, drop=True)
        _space_coord = np.asarray(_sliced_dataset["latitude"].values, dtype=float)
        _temperature = np.asarray(_sliced_dataset["temperature"].values, dtype=float)

        _x_grid = np.asarray(_space_coord, dtype=float)
        _x_min = _x_grid.min()
        _x_max = _x_grid.max()
        if np.isclose(_x_max, _x_min):
            _x_grid = np.linspace(0.0, 1.0, _temperature.shape[1])
        else:
            _x_grid = (_x_grid - _x_min) / (_x_max - _x_min)

        _mean_field = _temperature.mean(axis=0)
        _centered_data = _temperature - _mean_field[None, :]

        _dx_weight = np.empty_like(_x_grid)
        if _x_grid.size < 2:
            _dx_weight[...] = 1.0
        else:
            _dx_weight[0] = 0.5 * (_x_grid[1] - _x_grid[0])
            _dx_weight[-1] = 0.5 * (_x_grid[-1] - _x_grid[-2])
            if _x_grid.size > 2:
                _dx_weight[1:-1] = 0.5 * (_x_grid[2:] - _x_grid[:-2])

        _cos_weight = np.cos(0.5 * np.pi * _x_grid)
        _kernel_weight = np.clip(_cos_weight * _dx_weight, a_min=0.0, a_max=None)
        return _centered_data * np.sqrt(_kernel_weight)[None, :]

    def _tipping_koopman_eigenvalues(_dataset_path, _dataset_index):
        # Load one dataset at a time, discard the same initial transient used in
        # the single-dataset KDMD analysis, and keep only the scaled trajectory.
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
            _scaled_data = _tipping_scaled_positive_temperature(_asymptotic_dataset)
        finally:
            _dataset.close()

        _x_snap, _y_snap, _dt_eff_value = make_snapshots(_scaled_data, dt=_dt_value)
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
        _sigma_median = float(np.median(pdist(_scaled_data[_idx], metric="euclidean")))
        _kdmd = KernelDMD(kernel=GaussianKernel(sigma=_sigma_median))
        _kdmd.fit_snapshots(X=_x_snap, Y=_y_snap)

        _tsvd = TSVDRegularizer()
        _tsvd.factorize(
            _kdmd.G,
            method="eigsh",
            rel_threshold=_tipping_rel_threshold_svd_temporary,
        )
        _kr, _ur, _sr = _tsvd.solve_from_factorization(
            _kdmd.A,
            _tipping_rel_threshold_svd,
        )
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
            _scaled_data,
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
        ) / YEAR * 12.0
    tipping_selected_timescales_months = np.where(
        np.isfinite(tipping_selected_timescales_months),
        tipping_selected_timescales_months,
        np.nan,
    )

    # The first panel shows decay rates, while the second shows the equivalent
    # timescales in months.
    tipping_spectrum_fig, (_tipping_eigs_ax, _tipping_timescale_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 7),
        sharex=True,
    )
    for _plot_column, _eigenvalue_index in enumerate(_selected_eigenvalue_indices):
        _label = rf"$\lambda_{{{_eigenvalue_index}}}$"
        _tipping_eigs_ax.plot(
            tipping_mu_values,
            tipping_selected_eigenvalues_per_day[:, _plot_column],
            marker="o",
            label=_label,
        )
        _tipping_timescale_ax.plot(
            tipping_mu_values,
            tipping_selected_timescales_months[:, _plot_column],
            marker="o",
            label=_label,
        )

    for _ax in (_tipping_eigs_ax, _tipping_timescale_ax):
        _ax.set_xlabel(r"$\mu$", size=16)
        _ax.set_xlim(float(tipping_mu_values.max()), float(tipping_mu_values.min()))
        _ax.grid(alpha=0.3, linestyle="--")


    _tipping_handles, _tipping_labels = _tipping_eigs_ax.get_legend_handles_labels()
    tipping_spectrum_fig.legend(
        _tipping_handles,
        _tipping_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=16
    )

    _tipping_eigs_ax.set_ylabel(r"$\mathrm{Re}\,\lambda$ [day$^{-1}$]", size=16)
    _tipping_timescale_ax.set_ylabel(r"$\tau$ [months]", size=16)
    tipping_spectrum_fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    tipping_spectrum_fig.savefig("../figures/tipping_eigenvalues.png",dpi=400)
    return


if __name__ == "__main__":
    app.run()
