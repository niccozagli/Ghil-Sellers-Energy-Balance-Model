import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from gsebm.paths import get_data_dir

    return get_data_dir, np, plt, xr


@app.cell
def _(get_data_dir):
    analysis_mu = "1240"
    fname = f"CONTROL_360ppm_T21L10_10000Y_MU_{analysis_mu}"
    state = "spinup"
    output_dir = get_data_dir() / "Plasim" / fname
    diagnostic_output_path = output_dir / f"{fname}_{state}_diagnostics.nc"
    return analysis_mu, diagnostic_output_path, fname


@app.cell
def _(diagnostic_output_path, xr):
    if not diagnostic_output_path.exists():
        raise FileNotFoundError(
            f"Run scripts/extract_plasim_diagnostics.py first: {diagnostic_output_path}"
        )
    with xr.open_dataset(diagnostic_output_path) as source:
        diagnostics = source.load()
    return (diagnostics,)


@app.cell
def _():
    diagnostic_colors = {
        "global": "tab:blue",
        "northern": "tab:orange",
        "southern": "tab:green",
    }
    return (diagnostic_colors,)


@app.cell
def _(diagnostic_colors, diagnostics, fname, plt):
    _fig, _ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(11, 9))
    _ax[0, 0].plot(
        diagnostics["year"],
        diagnostics["global_temperature"],
        color=diagnostic_colors["global"],
    )
    _ax[0, 1].plot(diagnostics["year"], diagnostics["amoc_strength"])
    _ax[1, 0].plot(
        diagnostics["year"],
        diagnostics["northern_hemisphere_surface_temperature"],
        color=diagnostic_colors["northern"],
    )
    _ax[1, 1].plot(
        diagnostics["year"],
        diagnostics["northern_polar_temperature_gradient"],
        color=diagnostic_colors["northern"],
    )
    _ax[2, 0].plot(
        diagnostics["year"],
        diagnostics["southern_hemisphere_surface_temperature"],
        color=diagnostic_colors["southern"],
    )
    _ax[2, 1].plot(
        diagnostics["year"],
        diagnostics["southern_polar_temperature_gradient"],
        color=diagnostic_colors["southern"],
    )

    _ax[0, 0].set(ylabel=r"global $\langle T \rangle$ (K)")
    _ax[0, 1].set(ylabel="AMOC (Sv)")
    _ax[1, 0].set(ylabel=r"Northern $\langle T \rangle$ (K)")
    _ax[1, 1].set(ylabel=r"$\Delta T_N$ (K)")
    _ax[2, 0].set(xlabel="year", ylabel=r"Southern $\langle T \rangle$ (K)")
    _ax[2, 1].set(xlabel="year", ylabel=r"$\Delta T_S$ (K)")
    for _axis in _ax.flat:
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    _mu = diagnostics.attrs.get("mu", fname.rsplit("_MU_", maxsplit=1)[-1])
    _fig.suptitle(fr"Surface temperature diagnostics ($\mu = {_mu}$)")
    _fig.tight_layout(rect=(0, 0, 1, 0.96))
    _fig
    return


@app.cell
def _(diagnostic_colors, diagnostics, plt):
    _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=(9, 9))

    _ax[0].plot(
        diagnostics["year"],
        diagnostics["global_toa_absorbed_shortwave"],
        label="absorbed shortwave",
    )
    _ax[0].plot(
        diagnostics["year"],
        diagnostics["global_toa_outgoing_longwave"],
        label="outgoing longwave",
    )
    _ax[0].set(ylabel=r"TOA flux (W m$^{-2}$)")
    _ax[0].legend()

    _ax[1].plot(
        diagnostics["year"],
        diagnostics["global_toa_energy_imbalance"],
        color=diagnostic_colors["global"],
    )
    _ax[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _ax[1].set(ylabel=r"TOA imbalance (W m$^{-2}$)")

    _ax[2].plot(
        diagnostics["year"],
        diagnostics["global_planetary_albedo"],
        color=diagnostic_colors["global"],
    )
    _ax[2].set(xlabel="year", ylabel="planetary albedo")
    _ax[2].ticklabel_format(axis="x", style="plain", useOffset=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(diagnostic_colors, diagnostics, plt):
    _fig, _ax = plt.subplots(nrows=4, sharex=True, figsize=(9, 11))

    _ax[0].plot(
        diagnostics["year"],
        diagnostics["northern_sea_ice_margin_latitude"],
        label="Northern margin",
        color=diagnostic_colors["northern"],
    )
    _ax[0].plot(
        diagnostics["year"],
        diagnostics["southern_sea_ice_margin_latitude"],
        label="Southern margin",
        color=diagnostic_colors["southern"],
    )
    _ax[0].set(ylabel="margin latitude (degrees N)")
    _ax[0].legend()

    for _name, _label, _color in (
        ("lsg_sea_ice_area", "total", diagnostic_colors["global"]),
        (
            "lsg_northern_sea_ice_area",
            "Northern Hemisphere",
            diagnostic_colors["northern"],
        ),
        (
            "lsg_southern_sea_ice_area",
            "Southern Hemisphere",
            diagnostic_colors["southern"],
        ),
    ):
        _ax[1].plot(
            diagnostics["year"],
            diagnostics[_name] / 1.0e12,
            label=_label,
            color=_color,
        )
    _ax[1].set(ylabel=r"ice area ($10^{12}$ m$^2$)")
    _ax[1].legend()

    for _name, _label, _color in (
        ("lsg_sea_ice_volume", "total", diagnostic_colors["global"]),
        (
            "lsg_northern_sea_ice_volume",
            "Northern Hemisphere",
            diagnostic_colors["northern"],
        ),
        (
            "lsg_southern_sea_ice_volume",
            "Southern Hemisphere",
            diagnostic_colors["southern"],
        ),
    ):
        _ax[2].plot(
            diagnostics["year"],
            diagnostics[_name] / 1.0e12,
            label=_label,
            color=_color,
        )
    _ax[2].set(ylabel=r"ice volume ($10^{12}$ m$^3$)")
    _ax[2].legend()

    _ax[3].plot(
        diagnostics["year"],
        diagnostics["lsg_sea_ice_mean_thickness"],
        color=diagnostic_colors["global"],
    )
    _ax[3].set(xlabel="year", ylabel="mean ice thickness (m)")
    _ax[3].ticklabel_format(axis="x", style="plain", useOffset=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Correlation analysis
    """)
    return


@app.cell
def _():
    transients = {
        "1367" : 6000,
        "1312" : 4000,
        "1288" : 4000,
        "1265" : 4000,
        "1240" : 6000
    }
    return (transients,)


@app.cell
def _(diagnostic_colors, get_data_dir, plt, transients, xr):
    _fig, _axes = plt.subplots(
        nrows=len(transients), ncols=1, sharex=True, figsize=(10, 10)
    )

    for _axis, (_mu, _transient_year) in zip(_axes, transients.items()):
        _experiment_name = f"CONTROL_360ppm_T21L10_10000Y_MU_{_mu}"
        _diagnostic_paths = sorted(
            (get_data_dir() / "Plasim" / _experiment_name).glob(
                f"{_experiment_name}_*_diagnostics.nc"
            )
        )
        if len(_diagnostic_paths) != 1:
            raise ValueError(
                f"Expected one saved diagnostics dataset for mu={_mu}, found "
                f"{len(_diagnostic_paths)}: {_diagnostic_paths}"
            )

        with xr.open_dataset(_diagnostic_paths[0]) as _source:
            _dataset = _source.load()
        _axis.plot(
            _dataset["year"],
            _dataset["northern_hemisphere_surface_temperature"],
            color=diagnostic_colors["northern"],
        )
        _axis.axvline(
            _transient_year,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="transient threshold",
        )
        _axis.set(ylabel=fr"$\mu={_mu}$\n$\langle T_N \rangle$ (K)")

    _axes[0].legend(loc="best")
    _axes[-1].set(xlabel="year")
    for _axis in _axes:
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    _fig.suptitle("Northern Hemisphere surface temperature")
    _fig.tight_layout(rect=(0, 0, 1, 0.97))
    _fig
    return


@app.cell
def _(analysis_mu, get_data_dir, transients, xr):
    _mu = analysis_mu
    _experiment_name = f"CONTROL_360ppm_T21L10_10000Y_MU_{_mu}"
    _zonal_paths = sorted(
        (get_data_dir() / "Plasim" / _experiment_name).glob(
            f"{_experiment_name}_*_zonal_temperatures.nc"
        )
    )
    if len(_zonal_paths) != 1:
        raise ValueError(
            f"Expected one saved zonal-temperature dataset for mu={_mu}, found "
            f"{len(_zonal_paths)}: {_zonal_paths}"
        )

    with xr.open_dataset(_zonal_paths[0]) as _source:
        stationary_zonal_temperature = _source[
            ["zonal_surface_temperature", "zonal_2m_temperature"]
        ].where(
            _source["year"] > transients[_mu],
            drop=True,
        ).load()
    return (stationary_zonal_temperature,)


@app.cell
def _(analysis_mu, plt, stationary_zonal_temperature):
    _mean_temperature = stationary_zonal_temperature.mean("time").sortby("lat")
    _temperature_std = stationary_zonal_temperature.std("time").sortby("lat")
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _surface_line, = _ax.plot(
        _mean_temperature["lat"],
        _mean_temperature["zonal_surface_temperature"],
        linewidth=2,
        label="surface",
    )
    _ax.fill_between(
        _mean_temperature["lat"],
        _mean_temperature["zonal_surface_temperature"]
        - _temperature_std["zonal_surface_temperature"],
        _mean_temperature["zonal_surface_temperature"]
        + _temperature_std["zonal_surface_temperature"],
        color=_surface_line.get_color(),
        alpha=0.2,
    )
    _two_metre_line, = _ax.plot(
        _mean_temperature["lat"],
        _mean_temperature["zonal_2m_temperature"],
        linewidth=2,
        linestyle="--",
        label="2 m",
    )
    _ax.fill_between(
        _mean_temperature["lat"],
        _mean_temperature["zonal_2m_temperature"]
        - _temperature_std["zonal_2m_temperature"],
        _mean_temperature["zonal_2m_temperature"]
        + _temperature_std["zonal_2m_temperature"],
        color=_two_metre_line.get_color(),
        alpha=0.2,
    )
    _ax.set(
        xlabel="latitude (degrees N)",
        ylabel="temperature (K)",
        title=fr"Stationary zonal-mean temperature ($\mu={analysis_mu}$)",
    )
    _ax.legend()
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Koopman analysis of Northern Hemisphere zonal surface temperature

    The state is the 16-latitude Northern Hemisphere annual zonal-mean
    surface-temperature anomaly for $\mu={analysis_mu}$ after the selected
    transient period. Consecutive
    annual states define a one-year Koopman map. Kernel distances use Gaussian
    area weights renormalized over the Northern Hemisphere, and a weighted
    Gaussian kernel is regularized with a truncated SVD. Eigenfunctions are
    shown in the reduced phase space formed by Northern Hemisphere mean surface
    temperature and $\Delta T_N$, where $\Delta T_N$ is the 0°--30°N mean
    minus the 30°N--90°N mean. The
    plotted field is a Gaussian-kernel conditional mean of each eigenfunction
    on a dense reduced-space grid; points without adequate trajectory support
    are masked rather than extrapolated.
    """)
    return


@app.cell
def _(np, stationary_zonal_temperature, xr):
    _latitude = stationary_zonal_temperature["lat"]
    _gaussian_nodes, _gaussian_weights = np.polynomial.legendre.leggauss(
        stationary_zonal_temperature.sizes["lat"]
    )
    _expected_latitudes = np.degrees(np.arcsin(_gaussian_nodes))[::-1]
    if not np.allclose(
        _latitude.values,
        _expected_latitudes,
    ):
        raise ValueError(
            "PLASIM latitude coordinate is not the expected descending Gaussian grid."
        )

    _northern_mask = _latitude > 0.0
    _northern_latitude = _latitude.where(_northern_mask, drop=True)
    _northern_weights = _gaussian_weights[::-1][_northern_mask.values]
    _northern_weights /= _northern_weights.sum()
    koopman_latitude_weights = xr.DataArray(
        _northern_weights,
        dims=("lat",),
        coords={"lat": _northern_latitude},
    )
    _surface_temperature = stationary_zonal_temperature[
        "zonal_surface_temperature"
    ].where(_northern_mask, drop=True).astype("float64")
    koopman_temperature_anomaly = _surface_temperature - _surface_temperature.mean(
        "time"
    )
    return koopman_latitude_weights, koopman_temperature_anomaly


@app.cell
def _(koopman_temperature_anomaly, np):
    snapshot_lag_years = 1
    _years = koopman_temperature_anomaly["year"].values
    koopman_snapshot_indices = np.flatnonzero(
        _years[snapshot_lag_years:] - _years[:-snapshot_lag_years]
        == snapshot_lag_years
    )
    X_snap = koopman_temperature_anomaly.values[koopman_snapshot_indices]
    Y_snap = koopman_temperature_anomaly.values[
        koopman_snapshot_indices + snapshot_lag_years
    ]
    snapshot_interval_days = 360.0 * snapshot_lag_years
    if X_snap.shape != Y_snap.shape or X_snap.shape[0] == 0:
        raise ValueError("Annual Koopman snapshot pairs are empty or misaligned.")
    return X_snap, Y_snap, koopman_snapshot_indices, snapshot_interval_days


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
def _(
    WeightedGaussianKernel,
    X_snap,
    Y_snap,
    koopman_latitude_weights,
    np,
    pdist,
):
    _maximum_training_snapshots = 10_000
    _rng = np.random.default_rng(seed=0)
    _training_count = min(_maximum_training_snapshots, X_snap.shape[0])
    koopman_training_indices = _rng.choice(
        X_snap.shape[0],
        size=_training_count,
        replace=False,
    )
    X_train = X_snap[koopman_training_indices]
    Y_train = Y_snap[koopman_training_indices]

    _kernel_weight = np.asarray(koopman_latitude_weights.values, dtype=float)
    _weighted_training_data = X_train * np.sqrt(_kernel_weight)[None, :]
    koopman_kernel_bandwidth = float(
        np.median(pdist(_weighted_training_data, metric="euclidean"))
    )
    if not np.isfinite(koopman_kernel_bandwidth) or koopman_kernel_bandwidth <= 0.0:
        raise ValueError(
            "The weighted Gaussian-kernel bandwidth must be finite and positive."
        )
    koopman_kernel = WeightedGaussianKernel(
        sigma=koopman_kernel_bandwidth,
        weights=_kernel_weight,
    )
    return X_train, Y_train, koopman_kernel, koopman_training_indices


@app.cell
def _(KernelDMD, TSVDRegularizer, X_train, Y_train, koopman_kernel):
    kdmd = KernelDMD(kernel=koopman_kernel)
    kdmd.fit_snapshots(X=X_train, Y=Y_train)

    tsvd = TSVDRegularizer()
    tsvd.factorize(
        kdmd.G,
        method="eigsh",
        symmetrize=False,
        rel_threshold=1.0e-4,
        max_rank=512,
    )
    return kdmd, tsvd


@app.cell
def _(plt, tsvd):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(tsvd.S / tsvd.S[0], ".")
    _ax.axhline(5.0e-3, color="black", linestyle="--", linewidth=0.8)
    _ax.set(
        xlabel="singular-value index",
        ylabel=r"$\sigma_i^2 / \sigma_1^2$",
        title="KDMD kernel singular spectrum",
        yscale="log",
    )
    _ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(KoopmanSpectrumKDMD, kdmd, snapshot_interval_days, tsvd):
    koopman_matrix, U_r, S_r = tsvd.solve_from_factorization(
        kdmd.A,
        rel_threshold=1e-2,
    )
    koopman_spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        koopman_matrix,
        kernel=kdmd.kernel,
        reference_data=kdmd.reference_data,
        U_r=U_r,
        S_r=S_r,
    )
    koopman_eigenvalues_per_year = (
        koopman_spectrum.continuous_time_eigenvalues(snapshot_interval_days)
        * snapshot_interval_days
    )
    kdmd.G = None
    kdmd.A = None
    return koopman_eigenvalues_per_year, koopman_spectrum


@app.cell
def _(koopman_eigenvalues_per_year, plt):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.plot(
        koopman_eigenvalues_per_year.real,
        koopman_eigenvalues_per_year.imag,
        ".",
        markersize=7,
    )
    _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _ax.set(
        xlabel=r"$\mathrm{Re}\,\lambda$ (year$^{-1}$)",
        ylabel=r"$\mathrm{Im}\,\lambda$ (year$^{-1}$)",
        title="Northern Hemisphere surface-temperature Koopman spectrum",
    )
    _ax.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Direct multi-lag KDMD consistency

    Each operator below is fitted directly at its stated lag, rather than
    obtained by taking powers of the one-year operator.  Stable rates that
    change with fitting lag indicate memory unresolved by the observed
    Northern Hemisphere temperature field.
    """)
    return


@app.cell
def _():
    lag_consistency_lags_years = (1, 2, 3, 5, 10,50)
    lag_consistency_mode_count = 4
    lag_consistency_max_training_snapshots = 10_000
    return (
        lag_consistency_lags_years,
        lag_consistency_max_training_snapshots,
        lag_consistency_mode_count,
    )


@app.cell
def _(
    KernelDMD,
    KoopmanSpectrumKDMD,
    TSVDRegularizer,
    koopman_kernel,
    koopman_temperature_anomaly,
    lag_consistency_lags_years,
    lag_consistency_max_training_snapshots,
    lag_consistency_mode_count,
    np,
):
    from gsebm.reduced_dynamics import match_rates as _match_rates

    if lag_consistency_mode_count < 1:
        raise ValueError("lag_consistency_mode_count must be at least one.")

    _years = koopman_temperature_anomaly["year"].values
    _temperature_anomaly = koopman_temperature_anomaly.values
    _rng = np.random.default_rng(seed=1)
    _reference_rates = None
    _matched_rates = []
    lag_consistency_retained_ranks = []

    for _lag_years in lag_consistency_lags_years:
        _snapshot_indices = np.flatnonzero(
            _years[_lag_years:] - _years[:-_lag_years] == _lag_years
        )
        _training_count = min(
            lag_consistency_max_training_snapshots,
            _snapshot_indices.size,
        )
        if _training_count == 0:
            raise ValueError(f"No valid snapshot pairs for lag {_lag_years} years.")
        _training_indices = _rng.choice(
            _snapshot_indices.size,
            size=_training_count,
            replace=False,
        )
        _origins = _snapshot_indices[_training_indices]
        _kdmd = KernelDMD(kernel=koopman_kernel)
        _kdmd.fit_snapshots(
            X=_temperature_anomaly[_origins],
            Y=_temperature_anomaly[_origins + _lag_years],
            fit_kernel=False,
            show_progress=False,
        )
        _tsvd = TSVDRegularizer()
        _tsvd.factorize(
            _kdmd.G,
            method="eigsh",
            symmetrize=False,
            rel_threshold=1.0e-4,
            max_rank=512,
        )
        _koopman_matrix, _U_r, _S_r = _tsvd.solve_from_factorization(
            _kdmd.A,
            rel_threshold=1.0e-2,
        )
        _spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
            _koopman_matrix,
            kernel=koopman_kernel,
            reference_data=_kdmd.reference_data,
            U_r=_U_r,
            S_r=_S_r,
        )
        _rates_per_year = (
            _spectrum.continuous_time_eigenvalues(360.0 * _lag_years) * 360.0
        )
        _stationary_index = int(np.argmin(np.abs(_spectrum.eigenvalues - 1.0)))
        _candidate_indices = np.flatnonzero(
            (np.arange(_rates_per_year.size) != _stationary_index)
            & (np.abs(_spectrum.eigenvalues) < 1.0 - 1.0e-10)
            & np.isfinite(_rates_per_year.real)
            & np.isfinite(_rates_per_year.imag)
        )
        _candidate_indices = _candidate_indices[
            np.argsort(_rates_per_year[_candidate_indices].real)[::-1]
        ]
        if _candidate_indices.size < lag_consistency_mode_count:
            raise ValueError(
                f"Only {_candidate_indices.size} stable non-stationary modes are "
                f"available at lag {_lag_years} years."
            )
        _candidate_rates = _rates_per_year[_candidate_indices]
        if _reference_rates is None:
            _reference_rates = _candidate_rates[:lag_consistency_mode_count]
            _matched_rates.append(_reference_rates)
        else:
            _matched_rates.append(
                _match_rates(_reference_rates, _candidate_rates)
            )
        lag_consistency_retained_ranks.append(_S_r.size)

    direct_lag_koopman_rates = np.stack(_matched_rates, axis=0)
    lag_consistency_retained_ranks = np.asarray(lag_consistency_retained_ranks)
    return direct_lag_koopman_rates, lag_consistency_retained_ranks


@app.cell
def _(
    direct_lag_koopman_rates,
    lag_consistency_lags_years,
    lag_consistency_retained_ranks,
    plt,
):
    _fig, (_rate_axis, _rank_axis) = plt.subplots(1, 2, figsize=(12, 4))
    for _mode_index in range(direct_lag_koopman_rates.shape[1]):
        _rate_axis.plot(
            lag_consistency_lags_years,
            direct_lag_koopman_rates[:, _mode_index].real,
            "o-",
            label=fr"matched mode {_mode_index + 1}",
        )
    _rate_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _rate_axis.set(
        xlabel="direct fitting lag (years)",
        ylabel=r"$\mathrm{Re}\,\lambda$ (year$^{-1}$)",
        title="Direct-lag KDMD rate consistency",
    )
    _rate_axis.legend()
    _rate_axis.grid(alpha=0.3, linestyle="--")

    _rank_axis.plot(
        lag_consistency_lags_years,
        lag_consistency_retained_ranks,
        "o-",
        color="tab:purple",
    )
    _rank_axis.set(
        xlabel="direct fitting lag (years)",
        ylabel="retained KDMD rank",
        title="Regularized rank by fitting lag",
    )
    _rank_axis.grid(alpha=0.3, linestyle="--")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(koopman_spectrum, koopman_temperature_anomaly):
    koopman_eigenfunctions = koopman_spectrum.evaluate_eigenfunctions(
        koopman_temperature_anomaly.values,
        batch_size=5_000,
    )
    if koopman_eigenfunctions.shape[1] < 7:
        raise ValueError(
            "The Koopman spectrum does not contain six non-stationary eigenfunctions."
        )
    return (koopman_eigenfunctions,)


@app.cell
def _(np, stationary_zonal_temperature):
    _latitude = stationary_zonal_temperature["lat"].values
    _surface_temperature = stationary_zonal_temperature[
        "zonal_surface_temperature"
    ].values
    _, _gaussian_weights = np.polynomial.legendre.leggauss(_latitude.size)
    _gaussian_weights = _gaussian_weights[::-1]

    def _regional_mean(_mask):
        _weights = _gaussian_weights * _mask
        return np.average(_surface_temperature, axis=1, weights=_weights)

    northern_mean_surface_temperature = _regional_mean(_latitude > 0.0)
    _tropical_surface_temperature = _regional_mean(
        (_latitude > 0.0) & (_latitude <= 30.0)
    )
    _northern_polar_surface_temperature = _regional_mean(_latitude >= 30.0)
    northern_temperature_gradient = (
        _tropical_surface_temperature - _northern_polar_surface_temperature
    )
    return northern_mean_surface_temperature, northern_temperature_gradient


@app.cell
def _(
    koopman_eigenfunctions,
    northern_mean_surface_temperature,
    northern_temperature_gradient,
    np,
):
    from scipy.spatial import cKDTree

    _phase_points = np.column_stack(
        (northern_mean_surface_temperature, northern_temperature_gradient)
    )
    _phase_center = _phase_points.mean(axis=0)
    _phase_scale = _phase_points.std(axis=0)
    if np.any(~np.isfinite(_phase_scale)) or np.any(_phase_scale == 0.0):
        raise ValueError("Reduced phase-space coordinates must have finite variance.")
    _scaled_phase_points = (_phase_points - _phase_center) / _phase_scale

    _grid_size = 120
    phase_temperature_grid = np.linspace(
        northern_mean_surface_temperature.min(),
        northern_mean_surface_temperature.max(),
        _grid_size,
    )
    phase_gradient_grid = np.linspace(
        northern_temperature_gradient.min(),
        northern_temperature_gradient.max(),
        _grid_size,
    )
    _temperature_mesh, _gradient_mesh = np.meshgrid(
        phase_temperature_grid,
        phase_gradient_grid,
        indexing="xy",
    )
    _query_points = np.column_stack(
        (_temperature_mesh.ravel(), _gradient_mesh.ravel())
    )
    _scaled_query_points = (_query_points - _phase_center) / _phase_scale

    # Scott's rule for a two-dimensional Gaussian kernel. Both coordinates
    # are standardized above, so a single isotropic bandwidth is appropriate.
    phase_kernel_bandwidth = _phase_points.shape[0] ** (-1.0 / 6.0)
    _nearest_distance = cKDTree(_scaled_phase_points).query(
        _scaled_query_points,
        k=1,
    )[0]
    _smoothed_values = np.empty(
        (_query_points.shape[0], 6),
        dtype=koopman_eigenfunctions.dtype,
    )
    _effective_sample_size = np.empty(_query_points.shape[0])
    _chunk_size = 250
    for _start in range(0, _query_points.shape[0], _chunk_size):
        _stop = min(_start + _chunk_size, _query_points.shape[0])
        _squared_distance = np.sum(
            (
                _scaled_query_points[_start:_stop, None, :]
                - _scaled_phase_points[None, :, :]
            )
            ** 2,
            axis=2,
        )
        _kernel_weight = np.exp(
            -0.5 * _squared_distance / phase_kernel_bandwidth**2
        )
        _weight_sum = _kernel_weight.sum(axis=1)
        _smoothed_values[_start:_stop] = (
            _kernel_weight @ koopman_eigenfunctions[:, 1:7]
        ) / _weight_sum[:, None]
        _effective_sample_size[_start:_stop] = _weight_sum**2 / np.sum(
            _kernel_weight**2,
            axis=1,
        )
    phase_support_mask = (
        (_effective_sample_size >= 20.0)
        & (_nearest_distance <= 2.0 * phase_kernel_bandwidth)
    ).reshape(_grid_size, _grid_size)
    smoothed_koopman_eigenfunctions = _smoothed_values.reshape(
        _grid_size,
        _grid_size,
        6,
    )
    return (
        phase_gradient_grid,
        phase_support_mask,
        phase_temperature_grid,
        smoothed_koopman_eigenfunctions,
    )


@app.cell
def _(
    koopman_eigenvalues_per_year,
    np,
    phase_gradient_grid,
    phase_support_mask,
    phase_temperature_grid,
    plt,
    smoothed_koopman_eigenfunctions,
):
    _eigenfunction_indices = range(1, 7)

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for _eigenfunction_index, _ax in zip(
        _eigenfunction_indices,
        _axes.ravel(),
    ):
        _mean_eigenfunction = np.ma.masked_where(
            ~phase_support_mask,
            smoothed_koopman_eigenfunctions[
                :, :, _eigenfunction_index - 1
            ].real,
        )
        _color_limit = float(np.max(np.abs(_mean_eigenfunction.compressed())))
        if not np.isfinite(_color_limit) or _color_limit == 0.0:
            raise ValueError(
                f"Koopman eigenfunction {_eigenfunction_index} has no "
                "plottable variation."
            )

        _image = _ax.contourf(
            phase_temperature_grid,
            phase_gradient_grid,
            _mean_eigenfunction,
            levels=np.linspace(-_color_limit, _color_limit, 31),
            cmap="seismic",
            extend="both",
        )
        _eigenvalue = koopman_eigenvalues_per_year[_eigenfunction_index]
        _relaxation_time = (
            -1.0 / _eigenvalue.real if _eigenvalue.real < 0.0 else np.inf
        )
        _ax.set_title(
            rf"$\Re\,\phi_{{{_eigenfunction_index}}}$, "
            rf"$\tau={_relaxation_time:.2g}$ years"
        )
        _fig.colorbar(
            _image,
            ax=_ax,
            label=rf"$\Re\,\phi_{{{_eigenfunction_index}}}$",
        )

    for _ax in _axes[-1, :]:
        _ax.set_xlabel(r"Northern Hemisphere $\langle T_s\rangle$ (K)")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(r"$\Delta T_N$ (K)")
    _fig.suptitle("Koopman eigenfunctions in reduced temperature phase space")
    _fig.tight_layout(rect=(0, 0, 1, 0.96))
    _fig
    return


@app.cell
def _():
    from scipy.signal import welch
    from scipy.stats import binned_statistic_2d

    from gsebm.reduced_dynamics import (
        fit_edmd_from_basis,
        fit_variable_bandwidth_diffusion_map,
        leading_nonstationary_indices,
        match_rates,
        modal_covariance,
        modal_psd,
        project_observable,
    )

    return


@app.cell
def _(
    X_train,
    koopman_eigenvalues_per_year,
    koopman_snapshot_indices,
    koopman_spectrum,
    koopman_training_indices,
    northern_mean_surface_temperature,
    np,
    tsvd,
):
    koopman_correlation_mode_count = 50
    if koopman_correlation_mode_count < 1:
        raise ValueError("koopman_correlation_mode_count must be at least one.")

    _observable_at_snapshots = northern_mean_surface_temperature[
        koopman_snapshot_indices
    ]
    _observable_train = _observable_at_snapshots[koopman_training_indices]
    _observable_train = _observable_train - _observable_train.mean()
    if _observable_train.shape != (X_train.shape[0],):
        raise ValueError("Correlation observable is not aligned with KDMD training data.")

    _koopman_modes = koopman_spectrum.koopman_modes(
        _observable_train,
        U_r=tsvd.Ur,
        S_r=tsvd.Sr,
    )
    _eigenfunction_gram = koopman_spectrum.eigenfunction_gram(
        S_r=tsvd.Sr,
        n_samples=X_train.shape[0],
        normalize=True,
    )
    retained_koopman_mode_count = min(
        koopman_correlation_mode_count,
        koopman_eigenvalues_per_year.size - 1,
    )
    _retained_indices = np.arange(retained_koopman_mode_count + 1)
    koopman_correlation = koopman_spectrum.correlation_function_continuous(
        G_phi=_eigenfunction_gram[np.ix_(_retained_indices, _retained_indices)],
        coeff_f=_koopman_modes[_retained_indices],
        coeff_g=_koopman_modes[_retained_indices],
        eigenvalues=koopman_eigenvalues_per_year[_retained_indices],
    )
    return koopman_correlation, retained_koopman_mode_count


@app.cell
def _(
    analysis_mu,
    koopman_correlation,
    northern_mean_surface_temperature,
    np,
    plt,
    retained_koopman_mode_count,
    stationary_zonal_temperature,
):
    from koopman_response.utils.signal import cross_correlation

    _fig, _ax = plt.subplots(figsize=(9, 5))
    _maximum_lag_years = 5000

    _years = stationary_zonal_temperature["year"].values
    _lag_interval_years = float(np.median(np.diff(_years)))
    if not np.allclose(
        np.diff(_years), _lag_interval_years
    ):
        raise ValueError("Correlation input years must be regularly sampled.")
    _lags, _correlation = cross_correlation(
        x=northern_mean_surface_temperature,
        y=northern_mean_surface_temperature,
        dt=_lag_interval_years,
        max_lag=min(_maximum_lag_years, northern_mean_surface_temperature.size - 1),
        normalization="unbiased",
    )
    _koopman_correlation = np.asarray(koopman_correlation(_lags)).real
    if not np.isfinite(_koopman_correlation).all():
        raise ValueError("Koopman correlation reconstruction contains non-finite values.")
    _ax.plot(_lags, _correlation, label="empirical")
    _ax.plot(
        _lags,
        _koopman_correlation,
        color="tab:orange",
        label=fr"Koopman ({retained_koopman_mode_count} modes)",
    )

    _ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _ax.set(
        xlabel="lag (years)",
        ylabel=r"$\langle f(t + \tau) f(t) \rangle$ (K$^2$)",
        title=(
            "Stationary Northern Hemisphere surface-temperature correlation "
            fr"($\mu={analysis_mu}$)"
        ),
    )
    _ax.legend(title="transient excluded")
    _ax.set_xlim(left=-1, right=150)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    analysis_mu,
    diagnostic_colors,
    diagnostics,
    np,
    plt,
    stationary_zonal_temperature,
):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    _maximum_lag_years = 5000
    _stationary_years = stationary_zonal_temperature["year"].values
    _stationary_mask = np.isin(diagnostics["year"].values, _stationary_years)
    _ice_years = diagnostics["year"].values[_stationary_mask]
    if not np.array_equal(_ice_years, _stationary_years):
        raise ValueError(
            "Sea-ice diagnostics and stationary temperature years are not aligned."
        )

    _lag_interval_years = float(np.median(np.diff(_ice_years)))
    if not np.allclose(np.diff(_ice_years), _lag_interval_years):
        raise ValueError("Sea-ice correlation input years must be regularly sampled.")

    _ice_variables = (
        (
            "lsg_northern_sea_ice_area",
            r"$C_{A_N}(\tau)$ (m$^4$)",
            "area",
        ),
        (
            "lsg_northern_sea_ice_volume",
            r"$C_{V_N}(\tau)$ (m$^6$)",
            "volume",
        ),
    )
    _fig, _axes = plt.subplots(
        nrows=len(_ice_variables),
        ncols=1,
        sharex=True,
        figsize=(9, 8),
    )
    for _axis, (_variable_name, _ylabel, _label) in zip(_axes, _ice_variables):
        _values = np.asarray(
            diagnostics[_variable_name].values[_stationary_mask],
            dtype=float,
        )
        if not np.isfinite(_values).all():
            raise ValueError(
                f"Northern Hemisphere sea-ice {_label} contains non-finite values."
            )
        _lags, _correlation = _cross_correlation(
            x=_values,
            y=_values,
            dt=_lag_interval_years,
            max_lag=min(_maximum_lag_years, _values.size - 1),
            normalization="unbiased",
        )
        _axis.plot(
            _lags,
            _correlation,
            color=diagnostic_colors["northern"],
        )
        _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        _axis.set(ylabel=_ylabel, title=f"Northern Hemisphere sea-ice {_label}")
        _axis.set_xlim(left=-1, right=250)

    _axes[-1].set(xlabel="lag (years)")
    _fig.suptitle(
        "Stationary Northern Hemisphere sea-ice correlation functions "
        fr"($\mu={analysis_mu}$)"
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.96))
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
