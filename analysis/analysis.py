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
        edge_state_albedo_from_dataset,
        edge_state_heat_transfer_from_dataset,
        get_data_dir,
        latitude_weighted_mean,
        mo,
        np,
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
    edge_state_ds = xr.open_dataset(data_dir / "edge_state_mu1.nc", engine="scipy")
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
    return asymptotic_warm_ds, condition


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation functions
    """)
    return


@app.cell
def _(Delta_T, avg_T, condition):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    avg_T_asymp = avg_T.where(cond=condition,drop=True)
    Delta_T_asymp = Delta_T.where(cond=condition,drop=True)

    dt = avg_T_asymp.attrs["stochastic_dt"] # in seconds

    obs_avgT = avg_T_asymp["temperature"].values
    lags_avgT, corr_avgT = _cross_correlation(x=obs_avgT,y=obs_avgT,dt=dt,normalization="biased")


    obs_DeltaT = Delta_T_asymp["temperature"].values
    lags_DeltaT, corr_DeltaT = _cross_correlation(x=obs_DeltaT,y=obs_DeltaT,dt=dt,normalization="biased")
    return corr_DeltaT, corr_avgT, dt, lags_DeltaT, lags_avgT


@app.cell
def _(YEAR, asymptotic_warm_ds, dt, np):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation
    _local_corr_lag_window_months = 12.0


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
        int(round(_local_corr_lag_window_months / 12.0 * YEAR / dt)),
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
    _lag_months = local_corr_lags / YEAR * 12.0
    _plot_stride = max(1, local_corr.shape[0] // 300)
    _latitude_grid, _lag_grid = np.meshgrid(
        local_corr_latitude,
        _lag_months[::_plot_stride],
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
    _ax.set_ylabel(r"$t \; [\mathrm{months}]$", fontsize=14)
    _ax.set_zlabel(r"$C_T(x,t)$", fontsize=14)
    _ax.set_xlim(local_corr_latitude.min(), local_corr_latitude.max())
    _ax.set_ylim(_lag_months.min(), _lag_months.max())
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
def _(DAY, eigs_ct, plt):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eigs_ct.real * DAY  , eigs_ct.imag * DAY    , ".", ms=4)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [day$^{-1}$]",size=16)
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [day$^{-1}$]",size=16)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.2, right=0.01)
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
def _(DAY, delta_temperature, eigs_ct, global_temperature, np, phi_vals, plt):
    from matplotlib.cm import ScalarMappable
    from scipy.stats import binned_statistic_2d
    grid4_bins = 80
    grid4_min_count = 5
    grid4_eigfunc_indices = (1, 2, 3, 5, 6)

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
    grid4_eigs_ax.plot(eigs_ct.real * DAY, eigs_ct.imag * DAY, ".", ms=4)

    grid4_eigs_ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [day$^{-1}$]", size=16)
    grid4_eigs_ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [day$^{-1}$]", size=16)
    grid4_eigs_ax.set_xlim(left=-0.2, right=0.01)
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

    #grid4_fig.savefig("../figures/Koopman_Eigenfunctions.png",dpi=400)
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
    _lag_months = local_corr_lags / YEAR * 12.0
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
        (r"$C_T(x,t)$ empirical", _empirical_surface, "coolwarm", -_corr_absmax, _corr_absmax),
        (r"$C_T(x,t)$ KDMD", _kdmd_surface, "coolwarm", -_corr_absmax, _corr_absmax),
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
        _ax.set_ylabel(r"$t \; [\mathrm{months}]$", fontsize=11)
        _ax.set_xlim(local_corr_latitude.min(), local_corr_latitude.max())
        _ax.set_ylim(_lag_months.min(), _lag_months.max())
        _ax.view_init(elev=28, azim=45)
        _fig.colorbar(_surface, ax=_ax, shrink=0.58, pad=0.12)

    _fig.tight_layout()
    _fig
    # _fig.savefig("../figures/spatial_correlation_functions.png",dpi=400)
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

    _ax1.plot(lags_avgT / YEAR * 12, corr_avgT, color="blue", label=r"$\overline{T}$")
    _ax1.plot(
        lags_avgT[::_marker_stride] / YEAR * 12,
        corr_avgT_kdmd(lags_avgT[::_marker_stride]).real,
        color="blue",
        linestyle="none",
        marker=".",
        markersize=7,
        label=r"$\overline{T}$ KDMD",
    )
    _ax2.plot(lags_DeltaT / YEAR * 12, corr_DeltaT, color="red", label=r"$\Delta T$")
    _ax2.plot(
        lags_DeltaT[::_marker_stride] / YEAR * 12,
        corr_DeltaT_kdmd(lags_DeltaT[::_marker_stride]).real,
        color="red",
        linestyle="none",
        marker=".",
        markersize=7,
        label=r"$\Delta T$ KDMD",
    )

    _ax1.set_xlim(left=-1, right=12)

    align_zero_yaxis(_ax1, _ax2)


    _ax1.set_xlabel(r"$t \; [\mathrm{months}]$",fontsize=16)
    _ax1.set_ylabel(r"$C_{\overline{T}}$", color="blue",fontsize=16)
    _ax2.set_ylabel(r"$C_{\Delta T}$", color="red",fontsize=16)

    _ax1.tick_params(axis="y", labelcolor="blue",labelsize=12)
    _ax2.tick_params(axis="y", labelcolor="red",labelsize=12)
    _ax1.tick_params(axis="x",labelsize=12)

    _ax1.grid(alpha=0.3, linestyle="--")

    _fig.tight_layout()
    _fig
    # _fig.savefig("../figures/correlation_function_reconstruction.png",dpi=400)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # warm_operator = build_ivp_operator_from_dataset(warm_ds)
    # warm_albedo = xr.DataArray(
    #     surface_albedo(
    #         asymptotic_warm_ds["temperature"].values,
    #         warm_operator.empirical_fields.b_parameter[None, :],
    #         warm_operator.empirical_fields.surface_height_offset[None, :],
    #         warm_operator.params,
    #     ),
    #     dims=("time", "latitude"),
    #     coords={
    #         "time": asymptotic_warm_ds["time"],
    #         "latitude": warm_ds["latitude"],
    #     },
    #     name="temperature_albedo",
    #     attrs={"units": "1", "long_name": "post-transient albedo"},
    # )
    # warm_temperature_x = np.gradient(
    #     asymptotic_warm_ds["temperature"].values,
    #     warm_ds["latitude"].values,
    #     axis=1,
    # )
    # warm_heat_flux = xr.DataArray(
    #     meridional_heat_transfer_rate_watts_per_square_meter(
    #         warm_ds["latitude"].values[None, :],
    #         asymptotic_warm_ds["temperature"].values,
    #         warm_temperature_x,
    #         warm_operator.empirical_fields.sensible_heat_flux_coefficient[None, :],
    #         warm_operator.empirical_fields.latent_heat_flux_coefficient[None, :],
    #         warm_operator.params,
    #     ),
    #     dims=("time", "latitude"),
    #     coords={
    #         "time": asymptotic_warm_ds["time"],
    #         "latitude": warm_ds["latitude"],
    #     },
    #     name="temperature_heat_flux",
    #     attrs={"units": "W m^-2", "long_name": "post-transient meridional heat-transfer rate"},
    # )
    # warm_albedo_profile = warm_albedo.mean(dim="time")
    # warm_albedo_profile_std = warm_albedo.std(dim="time")
    # warm_heat_flux_profile = warm_heat_flux.mean(dim="time")
    # warm_heat_flux_profile_std = warm_heat_flux.std(dim="time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Asymptotic Stochastic Profiles
    """)
    return


@app.cell
def _():
    # _transient_years = transient / YEAR
    # _latitude = warm_ds["latitude"].values
    # _temperature_mean = warm_asymptotic_temperature.values
    # # _temperature_std = warm_asymptotic_temperature_std.values
    # _albedo_mean = warm_albedo_profile.values
    # _albedo_std = warm_albedo_profile_std.values
    # _flux_mean = warm_heat_flux_profile.values
    # _flux_std = warm_heat_flux_profile_std.values
    # _profile_fig, (_temperature_ax, _albedo_ax, _flux_ax) = plt.subplots(
    #     3,
    #     1,
    #     figsize=(8, 9),
    #     sharex=True,
    # )
    # _temperature_ax.plot(
    #     _latitude,
    #     _temperature_mean,
    #     color="red",
    # )
    # _temperature_ax.fill_between(
    #     _latitude,
    #     _temperature_mean - _temperature_std,
    #     _temperature_mean + _temperature_std,
    #     color="red",
    #     alpha=0.15,
    # )
    # _temperature_ax.set_ylabel("Temperature [K]")
    # _temperature_ax.grid(True, alpha=0.3)

    # _albedo_ax.plot(
    #     _latitude,
    #     _albedo_mean,
    #     color="red",
    # )
    # _albedo_ax.fill_between(
    #     _latitude,
    #     _albedo_mean - _albedo_std,
    #     _albedo_mean + _albedo_std,
    #     color="red",
    #     alpha=0.15,
    # )
    # _albedo_ax.set_ylabel("Albedo")
    # _albedo_ax.grid(True, alpha=0.3)

    # _flux_ax.plot(
    #     _latitude,
    #     _flux_mean,
    #     color="red",
    # )
    # _flux_ax.fill_between(
    #     _latitude,
    #     _flux_mean - _flux_std,
    #     _flux_mean + _flux_std,
    #     color="red",
    #     alpha=0.15,
    # )
    # _flux_ax.set_xlabel("Normalized latitude x")
    # _flux_ax.set_ylabel(r"Heat flux $j$ [W m$^{-2}$]")

    # _flux_ax.grid(True, alpha=0.3)
    # _flux_ax.set_xlim(left=0, right=1)
    # _flux_ax.set_ylim(bottom=0)
    # _profile_fig.tight_layout()
    # # _profile_fig.savefig(get_repo_root() / "figures" / "stochastic_avg_profiles_near.png", dpi=400)
    # plt.show()
    return


@app.cell
def _():
    # from matplotlib.colors import LogNorm
    # from scipy.stats import gaussian_kde

    # _fig, _ax = plt.subplots(figsize=(8, 6))

    # asymptotic_Delta_T = Delta_T.where(cond=condition, drop=True)
    # asymptotic_avg_T = avg_T.where(cond=condition, drop=True)

    # _x = asymptotic_avg_T["temperature"].values.ravel()
    # _y = asymptotic_Delta_T["temperature"].values.ravel()
    # _mask = np.isfinite(_x) & np.isfinite(_y)
    # _x = _x[_mask]
    # _y = _y[_mask]

    # _x_pad = 0.1 * (_x.max() - _x.min())
    # _y_pad = 0.1 * (_y.max() - _y.min())

    # _x_grid = np.linspace(_x.min() - _x_pad, _x.max() + _x_pad, 300)
    # _y_grid = np.linspace(_y.min() - _y_pad, _y.max() + _y_pad, 300)


    # _X, _Y = np.meshgrid(_x_grid, _y_grid)

    # _samples = np.vstack([_x, _y])
    # _kde = gaussian_kde(_samples, bw_method=0.15)
    # _Z = _kde(np.vstack([_X.ravel(), _Y.ravel()])).reshape(_X.shape)

    # _levels = np.array([1e-6,5e-6,1e-5,5e-5,1e-4, 5e-4, 1e-3, 5e-3, 1e-2,5e-2,1e-1])
    # _Z = np.ma.masked_less(_Z, _levels[0])

    # _contour = _ax.contourf(
    #     _X,
    #     _Y,
    #     _Z,
    #     levels=_levels,
    #     cmap="coolwarm",
    #     norm=LogNorm(vmin=_levels[0], vmax=_levels[-1]),
    #     extend="max",
    # )
    # _ax.contour(
    #     _X,
    #     _Y,
    #     _Z,
    #     levels=_levels,
    #     colors="white",
    #     linewidths=0.6,
    #     alpha=0.4,
    # )

    # _colorbar = _fig.colorbar(_contour, ax=_ax, ticks=_levels)
    # _colorbar.set_ticklabels([f"{_level:.0e}" for _level in _levels])
    # _ax.set_xlabel(r"$\overline{T} [K]$",size=16)
    # _ax.set_ylabel(r"$\Delta T [K]$",size=16)



    # edge_avg_T = latitude_weighted_mean(edge_state_ds, xmin=0, xmax=1)
    # edge_eq_T = latitude_weighted_mean(edge_state_ds, xmin=0, xmax=1/3)
    # edge_pole_T = latitude_weighted_mean(edge_state_ds, xmin=1/3, xmax=1)
    # edge_Delta_T = edge_eq_T - edge_pole_T

    # # _ax.scatter(x=edge_avg_T["edge_state_temperature"].item(), 
    # #             y=edge_Delta_T["edge_state_temperature"].item(),
    # #             marker="^",
    # #             s=45,
    # #             color="green"
    # # )
    # # _ax.set_xlim(xmin=290,right=305)
    # # _ax.set_ylim(bottom=5,top=11)
    # # _fig.savefig(get_repo_root() / "figures"/ "statistics_warm_near.png", dpi=400)
    # plt.show()
    return


@app.cell
def _():

    # _fig, _ax = plt.subplots(figsize=(8, 6))


    # _x = asymptotic_avg_T["temperature"].values.ravel()
    # _y = asymptotic_Delta_T["temperature"].values.ravel()
    # _mask = np.isfinite(_x) & np.isfinite(_y)
    # _x = _x[_mask]
    # _y = _y[_mask]

    # _ax.scatter(x=_x,y=_y,marker='.',s=25,color="blue")

    # _ax.scatter(x=edge_avg_T["edge_state_temperature"].item(), 
    #             y=edge_Delta_T["edge_state_temperature"].item(),
    #             marker="^",
    #             s=50,
    #             color="red"
    # )
    # # _ax.set_xlim(xmin=290,right=305)
    # # _ax.set_ylim(bottom=5,top=11)
    # # _fig.savefig(get_repo_root() / "figures"/ "statistics_warm_near.png", dpi=400)
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving the data
    """)
    return


@app.cell
def _():
    # dt = (asymptotic_warm_ds["time"].values[1] - asymptotic_warm_ds["time"].values[0] ) / DAY
    # export_attrs = dict(warm_ds.attrs)
    # export_attrs.update(
    #     {
    #         "title": "Derived asymptotic stochastic diagnostics",
    #         "source_dataset_filename": warm_input_filename,
    #         "source_dataset_path": str(data_dir / warm_input_filename),
    #         "output_dataset_filename": warm_input_filename,
    #         "output_dataset_path": str(data_dir / warm_input_filename),
    #         "analysis_transient_years": float(transient),
    #         "analysis_stop_years": float(stop),
    #         "tau [days]": dt,
    #         "latitude slicing" : "Northern Emisphere"
    #     }
    # )
    # export_dataset = xr.Dataset(
    #     data_vars={
    #         "edge_state_temperature" : edge_dataset.where(edge_dataset["latitude"] > 0,drop=True)["edge_state_temperature"],
    #         "edge_state_albedo" : edge_albedo.where(edge_albedo["latitude"]>0,drop=True),
    #         "asymptotic_temperature": asymptotic_warm_ds.where(asymptotic_warm_ds["latitude"]>0,drop=True)["temperature"],
    #         "warm_albedo": warm_albedo.where(warm_albedo["latitude"]>0,drop=True),
    #         "warm_heat_flux": warm_heat_flux.where(warm_heat_flux["latitude"] > 0,drop=True),
    #         "asymptotic_Delta_T": asymptotic_Delta_T["temperature"],
    #         "asymptotic_avg_T": asymptotic_avg_T["temperature"],
    #     },
    #     attrs=export_attrs,
    # )

    # output_path = data_dir / "koopman_data" / warm_input_filename
    # export_dataset.to_netcdf(output_path, engine="scipy")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
