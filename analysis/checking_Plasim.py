import marimo

__generated_with = "0.24.2"
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
    ocean_diagnostic_output_path = (
        output_dir / f"{fname}_{state}_ocean_diagnostics.nc"
    )
    return (
        analysis_mu,
        diagnostic_output_path,
        fname,
        ocean_diagnostic_output_path,
    )


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
def _(fname, ocean_diagnostic_output_path, xr):
    if not ocean_diagnostic_output_path.exists():
        raise FileNotFoundError(
            "Run PYTHONPATH=src uv run python "
            "scripts/extract_plasim_ocean_diagnostics.py "
            "--experiment-dir "
            f"/Volumes/Nicco/Plasim/experiments/{fname}"
        )
    with xr.open_dataset(ocean_diagnostic_output_path) as _source:
        ocean_diagnostics = _source.load()
    return (ocean_diagnostics,)


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
def _():
    # _diagnostic_years = np.asarray(diagnostics["year"].values, dtype=int)
    # _ocean_years = np.asarray(ocean_diagnostics["year"].values, dtype=int)
    # if not np.array_equal(_ocean_years, _diagnostic_years):
    #     raise ValueError(
    #         "Ocean and existing PLASIM diagnostics do not have identical years."
    #     )

    # _regions = (
    #     ("global", "Global ocean", diagnostic_colors["global"]),
    #     (
    #         "northern_midlatitudes",
    #         r"Northern midlatitudes (20--60$^\circ$N)",
    #         diagnostic_colors["northern"],
    #     ),
    #     (
    #         "southern_midlatitudes",
    #         r"Southern midlatitudes (30--60$^\circ$S)",
    #         diagnostic_colors["southern"],
    #     ),
    # )
    # _depth_bands = (
    #     ("0_200m", "0--200 m"),
    #     ("200_750m", "200--750 m"),
    #     ("750_2000m", "750--2000 m"),
    # )
    # _stationary = ocean_diagnostics["year"] > transients[analysis_mu]
    # _heat_content = ocean_diagnostics["ocean_heat_content"]
    # _stationary_mean = _heat_content.where(_stationary, drop=True).mean("year")
    # _heat_content_anomaly = (_heat_content - _stationary_mean) / 1.0e21

    # _fig, _axes = plt.subplots(
    #     nrows=len(_regions),
    #     ncols=len(_depth_bands),
    #     sharex=True,
    #     figsize=(13, 9),
    # )
    # for _row, (_region, _region_label, _color) in enumerate(_regions):
    #     for _column, (_depth_band, _depth_label) in enumerate(_depth_bands):
    #         _axis = _axes[_row, _column]
    #         _is_primary_candidate = (
    #             _region == "northern_midlatitudes"
    #             and _depth_band == "200_750m"
    #         )
    #         _axis.plot(
    #             _ocean_years,
    #             _heat_content_anomaly.sel(
    #                 ocean_region=_region,
    #                 depth_band=_depth_band,
    #             ),
    #             color=_color,
    #             linewidth=1.5 if _is_primary_candidate else 1.0,
    #         )
    #         _axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    #         _axis.axvline(
    #             transients[analysis_mu],
    #             color="black",
    #             linestyle="--",
    #             linewidth=0.8,
    #             alpha=0.7,
    #         )
    #         if _row == 0:
    #             _axis.set_title(_depth_label)
    #         if _column == 0:
    #             _axis.set_ylabel(_region_label + "\n" + r"$\Delta H$ (ZJ)")
    #         if _row == len(_regions) - 1:
    #             _axis.set_xlabel("year")
    #         if _is_primary_candidate:
    #             _axis.text(
    #                 0.98,
    #                 0.95,
    #                 "primary candidate",
    #                 color=_color,
    #                 ha="right",
    #                 va="top",
    #                 transform=_axis.transAxes,
    #             )
    #             for _spine in _axis.spines.values():
    #                 _spine.set_color(_color)
    #                 _spine.set_linewidth(1.5)
    #         _axis.ticklabel_format(axis="x", style="plain", useOffset=False)

    # _fig.suptitle(
    #     "LSG regional ocean heat-content anomalies "
    #     fr"($\mu={analysis_mu}$; reference: stationary mean)"
    # )
    # _fig.tight_layout(rect=(0, 0, 1, 0.96))
    # _fig
    return


@app.cell
def _(diagnostic_colors, diagnostics, plt):
    _fig, _ax = plt.subplots(nrows=4, sharex=True, figsize=(9, 11))

    _ax[0].plot(
        diagnostics["year"],
        diagnostics["northern_persistent_sea_ice_edge_latitude"],
        label="Northern persistent-ice edge",
        color=diagnostic_colors["northern"],
    )
    _ax[0].plot(
        diagnostics["year"],
        diagnostics["southern_persistent_sea_ice_edge_latitude"],
        label="Southern persistent-ice edge",
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
        "1240" : 7000,
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
            _path
            for _path in (get_data_dir() / "Plasim" / _experiment_name).glob(
                f"{_experiment_name}_*_diagnostics.nc"
            )
            if not _path.name.endswith("_ocean_diagnostics.nc")
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

    _zonal_variables = [
        "zonal_surface_temperature",
        "zonal_2m_temperature",
        "zonal_sea_ice_concentration",
        "zonal_persistent_sea_ice_fraction",
        "zonal_sea_ice_thickness",
        "zonal_surface_albedo",
        "zonal_toa_energy_imbalance",
    ]
    with xr.open_dataset(_zonal_paths[0]) as _source:
        _missing_variables = sorted(set(_zonal_variables) - set(_source.data_vars))
        if _missing_variables:
            raise ValueError(
                "The saved zonal diagnostics predate the ice-field extraction; "
                "rerun scripts/extract_plasim_diagnostics.py. Missing variables: "
                f"{_missing_variables}"
            )
        stationary_zonal_temperature = _source[_zonal_variables].where(
            _source["year"] > transients[_mu],
            drop=True,
        ).load()
    return (stationary_zonal_temperature,)


@app.cell
def _(analysis_mu, plt, stationary_zonal_temperature):
    _temperature_fields = stationary_zonal_temperature[
        ["zonal_surface_temperature", "zonal_2m_temperature"]
    ]
    _mean_temperature = _temperature_fields.mean("time").sortby("lat")
    _temperature_std = _temperature_fields.std("time").sortby("lat")
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
def _(analysis_mu, mo, transients):
    mo.md(rf"""
    ### Stationary ocean-memory diagnostics

    The following diagnostics use years after the selected transient cutoff
    (year {transients[analysis_mu]} for $\mu={analysis_mu}$). A linear trend is removed from each
    series before estimating correlations so that residual deep-ocean spin-up
    is not mistaken for stationary memory.
    """)
    return


@app.cell
def _(np, ocean_diagnostics, stationary_zonal_temperature):
    ocean_analysis_years = np.asarray(
        stationary_zonal_temperature["year"].values,
        dtype=int,
    )
    _selected_ocean = ocean_diagnostics.sel(year=ocean_analysis_years)
    ocean_analysis_heat_content = np.asarray(
        _selected_ocean["ocean_heat_content"].values,
        dtype=float,
    ) / 1.0e21
    _time = ocean_analysis_years - ocean_analysis_years.mean()

    def _detrend(_values):
        _flattened = np.asarray(_values, dtype=float).reshape(_time.size, -1)
        _design = np.column_stack((np.ones(_time.size), _time))
        _coefficients = np.linalg.lstsq(_design, _flattened, rcond=None)[0]
        return (_flattened - _design @ _coefficients).reshape(_values.shape)

    detrended_ocean_heat_content = _detrend(ocean_analysis_heat_content)

    _latitude = np.asarray(stationary_zonal_temperature["lat"].values)
    _northern_mask = _latitude > 0.0
    ocean_analysis_northern_surface_field = np.asarray(
        stationary_zonal_temperature["zonal_surface_temperature"]
        .isel(lat=_northern_mask)
        .values,
        dtype=float,
    )
    _gaussian_nodes, _gaussian_weights = np.polynomial.legendre.leggauss(
        _latitude.size
    )
    _expected_latitude = np.degrees(np.arcsin(_gaussian_nodes))[::-1]
    if not np.allclose(_latitude, _expected_latitude):
        raise ValueError("Unexpected latitude grid in ocean-memory diagnostics.")
    _northern_weights = _gaussian_weights[::-1][_northern_mask]
    _northern_weights = _northern_weights / _northern_weights.sum()
    _northern_temperature = (
        ocean_analysis_northern_surface_field @ _northern_weights
    )
    ocean_analysis_northern_temperature = _detrend(
        _northern_temperature[:, None]
    )[:, 0]
    return (
        detrended_ocean_heat_content,
        ocean_analysis_heat_content,
        ocean_analysis_northern_surface_field,
        ocean_analysis_northern_temperature,
        ocean_analysis_years,
    )


@app.cell
def _(
    analysis_mu,
    detrended_ocean_heat_content,
    diagnostic_colors,
    np,
    ocean_diagnostics,
    plt,
    transients,
):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    _maximum_lag = 250
    _regions = (
        ("global", "global", diagnostic_colors["global"]),
        ("northern_midlatitudes", "Northern", diagnostic_colors["northern"]),
        ("southern_midlatitudes", "Southern", diagnostic_colors["southern"]),
    )
    _depth_bands = (
        ("0_200m", "0--200 m"),
        ("200_750m", "200--750 m"),
        ("750_2000m", "750--2000 m"),
    )

    _fig, _axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5))
    for _axis, (_depth_band, _depth_label) in zip(_axes, _depth_bands):
        _band_index = int(
            np.flatnonzero(ocean_diagnostics["depth_band"].values == _depth_band)[0]
        )
        for _region, _label, _color in _regions:
            _region_index = int(
                np.flatnonzero(
                    ocean_diagnostics["ocean_region"].values == _region
                )[0]
            )
            _values = detrended_ocean_heat_content[:, _region_index, _band_index]
            _lags, _autocovariance = _cross_correlation(
                x=_values,
                y=_values,
                dt=1.0,
                max_lag=_maximum_lag,
                normalization="unbiased",
            )
            _autocorrelation = _autocovariance / _autocovariance[0]
            _axis.plot(_lags, _autocorrelation, color=_color, label=_label)
        _axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        _axis.set(title=_depth_label, xlabel="lag (years)")
        _axis.grid(alpha=0.2)
    _axes[0].set_ylabel("detrended autocorrelation")
    _axes[0].legend()
    _fig.suptitle(
        "LSG ocean heat-content memory after year "
        f"{transients[analysis_mu]}"
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.94))
    _fig
    return


@app.cell
def _(
    detrended_ocean_heat_content,
    diagnostic_colors,
    np,
    ocean_analysis_northern_temperature,
    ocean_diagnostics,
    plt,
):
    from koopman_response.utils.signal import cross_correlation as _cross_correlation

    _maximum_lag = 250
    _depth_bands = (
        ("0_200m", "0--200 m", "tab:blue"),
        ("200_750m", "200--750 m", diagnostic_colors["northern"]),
        ("750_2000m", "750--2000 m", "tab:red"),
    )
    _region_index = int(
        np.flatnonzero(
            ocean_diagnostics["ocean_region"].values
            == "northern_midlatitudes"
        )[0]
    )

    _fig, _axes = plt.subplots(1, 2, sharey=True, figsize=(12, 4.5))
    for _depth_band, _label, _color in _depth_bands:
        _band_index = int(
            np.flatnonzero(ocean_diagnostics["depth_band"].values == _depth_band)[0]
        )
        _ocean = detrended_ocean_heat_content[:, _region_index, _band_index]
        _surface = ocean_analysis_northern_temperature
        _normalization = _ocean.std() * _surface.std()
        _lags, _ocean_leads = _cross_correlation(
            x=_surface,
            y=_ocean,
            dt=1.0,
            max_lag=_maximum_lag,
            normalization="unbiased",
        )
        _, _surface_leads = _cross_correlation(
            x=_ocean,
            y=_surface,
            dt=1.0,
            max_lag=_maximum_lag,
            normalization="unbiased",
        )
        _ocean_leads = _ocean_leads / _normalization
        _surface_leads = _surface_leads / _normalization
        _axes[0].plot(_lags, _ocean_leads, color=_color, label=_label)
        _axes[1].plot(_lags, _surface_leads, color=_color, label=_label)

    _axes[0].set_title(r"Ocean $H(t)$ leads $T_N(t+\tau)$")
    _axes[1].set_title(r"Surface $T_N(t)$ leads $H(t+\tau)$")
    for _axis in _axes:
        _axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        _axis.set(xlabel="lag (years)", ylabel="Pearson correlation")
        _axis.grid(alpha=0.2)
        _axis.legend()
    _fig.suptitle("Detrended Northern midlatitude ocean--surface lead correlations")
    _fig.tight_layout(rect=(0, 0, 1, 0.94))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The conditional test predicts future Northern Hemisphere mean surface
    temperature from the current 16-latitude Northern surface-temperature
    field, with and without the scalar 20--60°N, 200--750 m heat content.
    It uses the first 75% of the stationary record for training and the final
    25% for testing. Linear trends and standardization are fitted from the
    training data only; ridge regularization is selected on the final 20% of
    the training interval. Thus, $\Delta R^2$ measures genuinely held-out
    information added by the ocean variable.
    """)
    return


@app.cell
def _(
    np,
    ocean_analysis_heat_content,
    ocean_analysis_northern_surface_field,
    ocean_analysis_years,
    ocean_diagnostics,
    plt,
):
    _prediction_lags = np.asarray([10, 25, 50])
    _region_index = int(
        np.flatnonzero(
            ocean_diagnostics["ocean_region"].values
            == "northern_midlatitudes"
        )[0]
    )
    _band_index = int(
        np.flatnonzero(
            ocean_diagnostics["depth_band"].values == "200_750m"
        )[0]
    )
    _ocean_candidate = ocean_analysis_heat_content[:, _region_index, _band_index]
    _latitude_count = ocean_analysis_northern_surface_field.shape[1]
    _gaussian_nodes, _gaussian_weights = np.polynomial.legendre.leggauss(
        2 * _latitude_count
    )
    _northern_weights = _gaussian_weights[::-1][:_latitude_count]
    _northern_weights = _northern_weights / _northern_weights.sum()
    _target_series = ocean_analysis_northern_surface_field @ _northern_weights
    _ridge_alphas = np.logspace(-3.0, 8.0, 23)

    def _remove_training_trend(_values, _times, _training_count):
        _array = np.asarray(_values, dtype=float)
        _was_vector = _array.ndim == 1
        if _was_vector:
            _array = _array[:, None]
        _training_design = np.column_stack(
            (np.ones(_training_count), _times[:_training_count])
        )
        _coefficients = np.linalg.lstsq(
            _training_design,
            _array[:_training_count],
            rcond=None,
        )[0]
        _design = np.column_stack((np.ones(_times.size), _times))
        _result = _array - _design @ _coefficients
        return _result[:, 0] if _was_vector else _result

    def _ridge_fit_predict(_train_x, _train_y, _test_x, _alpha):
        _x_mean = _train_x.mean(axis=0)
        _x_scale = _train_x.std(axis=0)
        _x_scale[_x_scale == 0.0] = 1.0
        _scaled_train = (_train_x - _x_mean) / _x_scale
        _scaled_test = (_test_x - _x_mean) / _x_scale
        _y_mean = _train_y.mean()
        _coefficients = np.linalg.solve(
            _scaled_train.T @ _scaled_train
            + _alpha * np.eye(_scaled_train.shape[1]),
            _scaled_train.T @ (_train_y - _y_mean),
        )
        return _y_mean + _scaled_test @ _coefficients

    def _select_alpha(_train_x, _train_y):
        _validation_start = int(0.8 * _train_y.size)
        _mse = []
        for _alpha in _ridge_alphas:
            _prediction = _ridge_fit_predict(
                _train_x[:_validation_start],
                _train_y[:_validation_start],
                _train_x[_validation_start:],
                _alpha,
            )
            _mse.append(
                np.mean((_train_y[_validation_start:] - _prediction) ** 2)
            )
        return float(_ridge_alphas[int(np.argmin(_mse))])

    _baseline_scores = []
    _augmented_scores = []
    for _lag in _prediction_lags:
        _predictor_time = ocean_analysis_years[:-_lag].astype(float)
        _target_time = ocean_analysis_years[_lag:].astype(float)
        _baseline = ocean_analysis_northern_surface_field[:-_lag]
        _candidate = _ocean_candidate[:-_lag, None]
        _target = _target_series[_lag:]
        _training_count = int(0.75 * _target.size)

        _baseline = _remove_training_trend(
            _baseline, _predictor_time, _training_count
        )
        _candidate = _remove_training_trend(
            _candidate, _predictor_time, _training_count
        )
        _target = _remove_training_trend(_target, _target_time, _training_count)
        _augmented = np.column_stack((_baseline, _candidate))

        _baseline_alpha = _select_alpha(
            _baseline[:_training_count], _target[:_training_count]
        )
        _augmented_alpha = _select_alpha(
            _augmented[:_training_count], _target[:_training_count]
        )
        _baseline_prediction = _ridge_fit_predict(
            _baseline[:_training_count],
            _target[:_training_count],
            _baseline[_training_count:],
            _baseline_alpha,
        )
        _augmented_prediction = _ridge_fit_predict(
            _augmented[:_training_count],
            _target[:_training_count],
            _augmented[_training_count:],
            _augmented_alpha,
        )
        _test_target = _target[_training_count:]
        _denominator = np.sum(
            (_test_target - _target[:_training_count].mean()) ** 2
        )
        _baseline_scores.append(
            1.0 - np.sum((_test_target - _baseline_prediction) ** 2) / _denominator
        )
        _augmented_scores.append(
            1.0 - np.sum((_test_target - _augmented_prediction) ** 2) / _denominator
        )

    _baseline_scores = np.asarray(_baseline_scores)
    _augmented_scores = np.asarray(_augmented_scores)
    _x = np.arange(_prediction_lags.size)
    _width = 0.36
    _fig, _axis = plt.subplots(figsize=(8, 5))
    _axis.bar(
        _x - _width / 2,
        _baseline_scores,
        width=_width,
        label="surface-temperature field",
    )
    _axis.bar(
        _x + _width / 2,
        _augmented_scores,
        width=_width,
        label="surface field + NH 200--750 m heat content",
    )
    for _index, _improvement in enumerate(
        _augmented_scores - _baseline_scores
    ):
        _axis.text(
            _index,
            max(_baseline_scores[_index], _augmented_scores[_index]),
            fr"$\Delta R^2={_improvement:+.3f}$",
            ha="center",
            va="bottom",
        )
    _axis.axhline(0.0, color="black", linewidth=0.7)
    _axis.set(
        xticks=_x,
        xticklabels=[f"{_lag} years" for _lag in _prediction_lags],
        ylabel=r"chronological test $R^2$",
        title="Does thermocline heat content add predictive information?",
    )
    _axis.legend()
    _axis.grid(axis="y", alpha=0.2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Koopman analysis of Northern Hemisphere surface temperature

    The state is the 16-latitude Northern Hemisphere annual zonal-mean
    surface-temperature anomaly for $\mu={analysis_mu}$ after the selected
    transient period. Consecutive annual states define a one-year Koopman map.
    The Gaussian-kernel distance uses Gaussian latitude-area weights and is
    normalized by the median pairwise distance of the training states.

    Eigenfunctions are shown after projection into the reduced phase space
    formed by Northern Hemisphere mean surface temperature and $\Delta T_N$,
    where $\Delta T_N$ is the 0°--30°N mean minus the 30°N--90°N mean. The
    plotted field is a Gaussian-kernel conditional mean of each eigenfunction
    on a dense reduced-space grid; points without adequate trajectory support
    are masked rather than extrapolated. The reconstructed correlation remains
    that of Northern Hemisphere mean surface temperature.
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

    koopman_state_anomaly = np.asarray(
        koopman_temperature_anomaly.values,
        dtype=float,
    )
    _expected_feature_count = koopman_temperature_anomaly.sizes["lat"]
    if koopman_state_anomaly.shape != (
        koopman_temperature_anomaly.sizes["time"],
        _expected_feature_count,
    ):
        raise ValueError("The temperature-only Koopman state has an unexpected shape.")
    return (
        koopman_latitude_weights,
        koopman_state_anomaly,
        koopman_temperature_anomaly,
    )


@app.cell
def _(koopman_state_anomaly, koopman_temperature_anomaly, np):
    snapshot_lag_years = 1
    _years = koopman_temperature_anomaly["year"].values
    koopman_snapshot_indices = np.flatnonzero(
        _years[snapshot_lag_years:] - _years[:-snapshot_lag_years]
        == snapshot_lag_years
    )
    X_snap = koopman_state_anomaly[koopman_snapshot_indices]
    Y_snap = koopman_state_anomaly[
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

    _temperature_feature_count = koopman_latitude_weights.size
    if X_train.shape[1] != _temperature_feature_count:
        raise ValueError(
            "The Koopman state must contain only the Northern Hemisphere "
            "temperature field."
        )
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
        title="NH surface-temperature KDMD kernel singular spectrum",
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
        rel_threshold=2e-2,
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
    # kdmd.G = None
    # kdmd.A = None
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
        title="NH surface-temperature Koopman spectrum",
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
    Northern Hemisphere surface-temperature field.
    """)
    return


@app.cell
def _():
    # lag_consistency_lags_years = (1, 2, 3, 5, 10)
    # lag_consistency_mode_count = 6
    # lag_consistency_max_training_snapshots = 10_000
    return


@app.cell
def _():
    # from gsebm.reduced_dynamics import match_rates as _match_rates

    # if lag_consistency_mode_count < 1:
    #     raise ValueError("lag_consistency_mode_count must be at least one.")

    # _years = koopman_temperature_anomaly["year"].values
    # _state_anomaly = koopman_state_anomaly
    # _rng = np.random.default_rng(seed=1)
    # _reference_rates = None
    # _matched_rates = []
    # lag_consistency_retained_ranks = []

    # for _lag_years in lag_consistency_lags_years:
    #     _snapshot_indices = np.flatnonzero(
    #         _years[_lag_years:] - _years[:-_lag_years] == _lag_years
    #     )
    #     _training_count = min(
    #         lag_consistency_max_training_snapshots,
    #         _snapshot_indices.size,
    #     )
    #     if _training_count == 0:
    #         raise ValueError(f"No valid snapshot pairs for lag {_lag_years} years.")
    #     _training_indices = _rng.choice(
    #         _snapshot_indices.size,
    #         size=_training_count,
    #         replace=False,
    #     )
    #     _origins = _snapshot_indices[_training_indices]
    #     _kdmd = KernelDMD(kernel=koopman_kernel)
    #     _kdmd.fit_snapshots(
    #         X=_state_anomaly[_origins],
    #         Y=_state_anomaly[_origins + _lag_years],
    #         fit_kernel=False,
    #         show_progress=False,
    #     )
    #     _tsvd = TSVDRegularizer()
    #     _tsvd.factorize(
    #         _kdmd.G,
    #         method="eigsh",
    #         symmetrize=False,
    #         rel_threshold=1.0e-4,
    #         max_rank=512,
    #     )
    #     _koopman_matrix, _U_r, _S_r = _tsvd.solve_from_factorization(
    #         _kdmd.A,
    #         rel_threshold=1.0e-2,
    #     )
    #     _spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
    #         _koopman_matrix,
    #         kernel=koopman_kernel,
    #         reference_data=_kdmd.reference_data,
    #         U_r=_U_r,
    #         S_r=_S_r,
    #     )
    #     _rates_per_year = (
    #         _spectrum.continuous_time_eigenvalues(360.0 * _lag_years) * 360.0
    #     )
    #     _stationary_index = int(np.argmin(np.abs(_spectrum.eigenvalues - 1.0)))
    #     _candidate_indices = np.flatnonzero(
    #         (np.arange(_rates_per_year.size) != _stationary_index)
    #         & (np.abs(_spectrum.eigenvalues) < 1.0 - 1.0e-10)
    #         & np.isfinite(_rates_per_year.real)
    #         & np.isfinite(_rates_per_year.imag)
    #     )
    #     _candidate_indices = _candidate_indices[
    #         np.argsort(_rates_per_year[_candidate_indices].real)[::-1]
    #     ]
    #     if _candidate_indices.size < lag_consistency_mode_count:
    #         raise ValueError(
    #             f"Only {_candidate_indices.size} stable non-stationary modes are "
    #             f"available at lag {_lag_years} years."
    #         )
    #     _candidate_rates = _rates_per_year[_candidate_indices]
    #     if _reference_rates is None:
    #         _reference_rates = _candidate_rates[:lag_consistency_mode_count]
    #         _matched_rates.append(_reference_rates)
    #     else:
    #         _matched_rates.append(
    #             _match_rates(_reference_rates, _candidate_rates)
    #         )
    #     lag_consistency_retained_ranks.append(_S_r.size)

    # direct_lag_koopman_rates = np.stack(_matched_rates, axis=0)
    # lag_consistency_retained_ranks = np.asarray(lag_consistency_retained_ranks)
    return


@app.cell
def _():
    # _fig, (_rate_axis, _rank_axis) = plt.subplots(1, 2, figsize=(12, 4))
    # for _mode_index in range(direct_lag_koopman_rates.shape[1]):
    #     _rate_axis.plot(
    #         lag_consistency_lags_years,
    #         direct_lag_koopman_rates[:, _mode_index].real,
    #         "o-",
    #         label=fr"matched mode {_mode_index + 1}",
    #     )
    # _rate_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    # _rate_axis.set(
    #     xlabel="direct fitting lag (years)",
    #     ylabel=r"$\mathrm{Re}\,\lambda$ (year$^{-1}$)",
    #     title="Direct-lag KDMD rate consistency",
    # )
    # _rate_axis.legend()
    # _rate_axis.grid(alpha=0.3, linestyle="--")

    # _rank_axis.plot(
    #     lag_consistency_lags_years,
    #     lag_consistency_retained_ranks,
    #     "o-",
    #     color="tab:purple",
    # )
    # _rank_axis.set(
    #     xlabel="direct fitting lag (years)",
    #     ylabel="retained KDMD rank",
    #     title="Regularized rank by fitting lag",
    # )
    # _rank_axis.grid(alpha=0.3, linestyle="--")
    # _fig.tight_layout()
    # _fig
    return


@app.cell
def _(koopman_spectrum, koopman_state_anomaly):
    koopman_eigenfunctions = koopman_spectrum.evaluate_eigenfunctions(
        koopman_state_anomaly,
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
    _fig.suptitle(
        "Temperature-only Koopman eigenfunctions in reduced temperature phase space"
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.96))
    _fig
    return


@app.cell
def _():
    # Select the temperature Koopman mode to compare with the stationary
    # zonal-mean net TOA energy balance. Index 0 is the stationary mode.
    toa_comparison_mode_index = 2
    return (toa_comparison_mode_index,)


@app.cell
def _(
    X_train,
    koopman_snapshot_indices,
    koopman_spectrum,
    koopman_temperature_anomaly,
    koopman_training_indices,
    np,
    tsvd,
):
    _temperature_at_snapshots = np.asarray(
        koopman_temperature_anomaly.values[koopman_snapshot_indices],
        dtype=float,
    )
    _temperature_train = _temperature_at_snapshots[koopman_training_indices]
    if _temperature_train.shape != X_train.shape or not np.allclose(
        _temperature_train,
        X_train,
    ):
        raise ValueError(
            "Local temperature observables are not aligned with KDMD training data."
        )

    temperature_koopman_mode_matrix = np.stack(
        [
            koopman_spectrum.koopman_modes(
                _temperature_train[:, _latitude_index],
                U_r=tsvd.Ur,
                S_r=tsvd.Sr,
            )
            for _latitude_index in range(_temperature_train.shape[1])
        ],
        axis=0,
    )
    return (temperature_koopman_mode_matrix,)


@app.cell
def _(
    koopman_eigenvalues_per_year,
    koopman_latitude_weights,
    np,
    plt,
    stationary_zonal_temperature,
    temperature_koopman_mode_matrix,
    toa_comparison_mode_index,
):
    _mode_index = toa_comparison_mode_index
    if not 1 <= _mode_index < temperature_koopman_mode_matrix.shape[1]:
        raise ValueError(
            "toa_comparison_mode_index must select an available non-stationary mode."
        )

    _latitude = np.asarray(koopman_latitude_weights["lat"].values, dtype=float)
    _raw_mode = temperature_koopman_mode_matrix[:, _mode_index]
    _mode_scale = float(np.max(np.abs(_raw_mode)))
    if not np.isfinite(_mode_scale) or _mode_scale == 0.0:
        raise ValueError(f"Temperature Koopman mode {_mode_index} is degenerate.")

    _northern_toa = stationary_zonal_temperature[
        "zonal_toa_energy_imbalance"
    ].sel(lat=koopman_latitude_weights["lat"])
    _toa_latitude = np.asarray(_northern_toa["lat"].values, dtype=float)
    _toa_mean = np.asarray(_northern_toa.mean("time").values, dtype=float)
    if not np.allclose(_toa_latitude, _latitude):
        raise ValueError("The TOA and temperature-mode latitude grids are not aligned.")

    _toa_zero_crossings = []
    for _lower_index in range(_toa_mean.size - 1):
        _lower_value = _toa_mean[_lower_index]
        _upper_value = _toa_mean[_lower_index + 1]
        if not np.isfinite(_lower_value) or not np.isfinite(_upper_value):
            continue
        if np.isclose(_lower_value, 0.0):
            _toa_zero_crossings.append(float(_toa_latitude[_lower_index]))
        elif _lower_value * _upper_value < 0.0:
            _fraction = -_lower_value / (_upper_value - _lower_value)
            _toa_zero_crossings.append(
                float(
                    _toa_latitude[_lower_index]
                    + _fraction
                    * (_toa_latitude[_lower_index + 1] - _toa_latitude[_lower_index])
                )
            )
    if np.isclose(_toa_mean[-1], 0.0):
        _toa_zero_crossings.append(float(_toa_latitude[-1]))

    _fig, _mode_axis = plt.subplots(figsize=(9, 5.5))
    _mode_axis.plot(
        _latitude,
        _raw_mode.real,
        "o-",
        color="tab:blue",
        linewidth=2.0,
        label=fr"$\Re\,v_{{T,{_mode_index}}}(\phi)$",
    )
    if np.max(np.abs(_raw_mode.imag)) > 1.0e-6 * _mode_scale:
        _mode_axis.plot(
            _latitude,
            _raw_mode.imag,
            "o--",
            color="tab:orange",
            linewidth=1.6,
            label=fr"$\Im\,v_{{T,{_mode_index}}}(\phi)$",
        )
    _mode_axis.axhline(0.0, color="tab:blue", linewidth=0.8, alpha=0.6)
    _mode_axis.set(
        xlabel="latitude (degrees N)",
        ylabel="raw temperature Koopman-mode coefficient",
    )

    _toa_axis = _mode_axis.twinx()
    _toa_axis.plot(
        _toa_latitude,
        _toa_mean,
        "s-",
        color="0.35",
        linewidth=1.7,
        markersize=4.0,
        label=r"stationary mean $R_{\mathrm{TOA}}$",
    )
    _toa_axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
    _toa_axis.set_ylabel(
        r"zonal net TOA energy imbalance (W m$^{-2}$)",
        color="0.3",
    )
    _toa_axis.tick_params(axis="y", labelcolor="0.3")

    for _crossing_index, _crossing in enumerate(_toa_zero_crossings):
        _mode_axis.axvline(
            _crossing,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=r"$R_{\mathrm{TOA}}=0$" if _crossing_index == 0 else None,
        )

    _rate = koopman_eigenvalues_per_year[_mode_index]
    _mode_axis.set_title(
        rf"Temperature mode {_mode_index} and stationary net TOA balance"
        "\n"
        rf"$\lambda_{{{_mode_index}}}={_rate.real:.3g}{_rate.imag:+.3g}i$ "
        r"year$^{-1}$"
    )
    _mode_axis.set_xlim(0.0, 90.0)
    _mode_axis.grid(alpha=0.25)
    _mode_lines, _mode_labels = _mode_axis.get_legend_handles_labels()
    _toa_lines, _toa_labels = _toa_axis.get_legend_handles_labels()
    _mode_axis.legend(
        _mode_lines + _toa_lines,
        _mode_labels + _toa_labels,
        fontsize=9,
        loc="best",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    diagnostics,
    koopman_eigenvalues_per_year,
    koopman_latitude_weights,
    koopman_spectrum,
    mo,
    np,
    plt,
    stationary_zonal_temperature,
    temperature_koopman_mode_matrix,
):
    _mode_index = 1
    _stationary_index = int(
        np.argmin(np.abs(koopman_spectrum.eigenvalues - 1.0))
    )
    if _stationary_index != 0:
        raise ValueError(
            "The stationary Koopman mode is not at index 0; the configured "
            "mode index cannot be interpreted consistently."
        )
    if _mode_index >= koopman_spectrum.eigenvalues.size:
        raise ValueError(f"Temperature Koopman mode {_mode_index} is unavailable.")

    _raw_mode = temperature_koopman_mode_matrix[:, _mode_index]
    _latitude_weight = np.asarray(koopman_latitude_weights.values, dtype=float)
    _hemispheric_mean_mode = np.sum(_latitude_weight * _raw_mode)
    _mode_scale = float(np.max(np.abs(_raw_mode)))
    if not np.isfinite(_mode_scale) or _mode_scale == 0.0:
        raise ValueError(f"Temperature Koopman mode {_mode_index} is degenerate.")
    if np.abs(_hemispheric_mean_mode) <= 1.0e-8 * _mode_scale:
        raise ValueError(
            f"Temperature Koopman mode {_mode_index} has negligible NH-mean projection; "
            "area-mean normalization is not appropriate."
        )

    _phase = np.exp(-1j * np.angle(_hemispheric_mean_mode))
    _phase_aligned_mode = _raw_mode * _phase
    _normalizer = float(
        np.real(np.sum(_latitude_weight * _phase_aligned_mode))
    )
    _normalized_mode = _phase_aligned_mode / _normalizer
    if not np.isclose(
        np.sum(_latitude_weight * _normalized_mode.real),
        1.0,
        atol=1.0e-10,
    ):
        raise ValueError("The normalized temperature mode does not have NH mean one.")

    _stationary_years = stationary_zonal_temperature["year"].values
    _diagnostic_mask = np.isin(diagnostics["year"].values, _stationary_years)
    if not np.array_equal(
        diagnostics["year"].values[_diagnostic_mask],
        _stationary_years,
    ):
        raise ValueError(
            "Sea-ice-margin and stationary temperature years are not aligned."
        )
    _ice_margin = np.asarray(
        diagnostics["northern_persistent_sea_ice_edge_latitude"].values[
            _diagnostic_mask
        ],
        dtype=float,
    )
    _ice_margin = _ice_margin[np.isfinite(_ice_margin)]
    if _ice_margin.size == 0:
        raise ValueError("The stationary Northern Hemisphere ice margin is undefined.")
    _mean_ice_margin = float(_ice_margin.mean())
    _std_ice_margin = float(_ice_margin.std())
    _latitude = np.asarray(koopman_latitude_weights["lat"].values, dtype=float)
    _northern_ice = stationary_zonal_temperature.where(
        stationary_zonal_temperature["lat"] > 0.0,
        drop=True,
    ).sortby("lat")
    _diagnostic_latitude = np.asarray(_northern_ice["lat"].values, dtype=float)
    _sic_mean = np.asarray(
        _northern_ice["zonal_sea_ice_concentration"].mean("time").values,
        dtype=float,
    )
    _sic_std = np.asarray(
        _northern_ice["zonal_sea_ice_concentration"].std("time").values,
        dtype=float,
    )
    _persistent_ice_fraction_mean = np.asarray(
        _northern_ice["zonal_persistent_sea_ice_fraction"].mean("time").values,
        dtype=float,
    )
    _persistent_ice_fraction_std = np.asarray(
        _northern_ice["zonal_persistent_sea_ice_fraction"].std("time").values,
        dtype=float,
    )
    _sit_mean = np.asarray(
        _northern_ice["zonal_sea_ice_thickness"].mean("time").values,
        dtype=float,
    )
    _sit_std = np.asarray(
        _northern_ice["zonal_sea_ice_thickness"].std("time").values,
        dtype=float,
    )
    _albedo_mean = np.asarray(
        _northern_ice["zonal_surface_albedo"].mean("time").values,
        dtype=float,
    )
    _albedo_std = np.asarray(
        _northern_ice["zonal_surface_albedo"].std("time").values,
        dtype=float,
    )

    _sic_variability_latitude = float(
        _diagnostic_latitude[np.nanargmax(_sic_std)]
    )
    _sic_gradient = np.gradient(_sic_mean, _diagnostic_latitude)
    _sic_gradient_latitude = float(
        _diagnostic_latitude[np.nanargmax(np.abs(_sic_gradient))]
    )
    _albedo_gradient = np.gradient(_albedo_mean, _diagnostic_latitude)
    _albedo_gradient_latitude = float(
        _diagnostic_latitude[np.nanargmax(np.abs(_albedo_gradient))]
    )

    _rate = koopman_eigenvalues_per_year[_mode_index]
    _relaxation_time = -1.0 / _rate.real if _rate.real < 0.0 else np.inf
    _mode_peak_latitude = float(
        _latitude[np.nanargmax(np.abs(_normalized_mode.real))]
    )

    _mode_comparison_fig, (_raw_mode_axis, _normalized_mode_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        sharex=True,
    )
    _raw_mode_axis.plot(
        _latitude,
        _raw_mode.real,
        "o-",
        linewidth=2.0,
        label=fr"$\Re\,v_{{T,{_mode_index}}}(\phi)$",
    )
    _raw_scale = float(np.max(np.abs(_raw_mode.real)))
    if np.max(np.abs(_raw_mode.imag)) > 1.0e-6 * max(_raw_scale, _mode_scale):
        _raw_mode_axis.plot(
            _latitude,
            _raw_mode.imag,
            "o--",
            linewidth=1.5,
            label=fr"$\Im\,v_{{T,{_mode_index}}}(\phi)$",
        )
    _raw_mode_axis.set(
        xlabel="latitude (degrees N)",
        ylabel="raw Koopman-mode coefficient",
        title="Raw mode\n(arbitrary KDMD scale and phase)",
    )
    _raw_mode_axis.legend(fontsize=9)

    _normalized_mode_axis.plot(
        _latitude,
        _normalized_mode.real,
        "o-",
        linewidth=2.0,
        label=fr"$\Re\,\widetilde v_{{T,{_mode_index}}}(\phi)$",
    )
    _normalized_real_scale = float(np.max(np.abs(_normalized_mode.real)))
    if np.max(np.abs(_normalized_mode.imag)) > 1.0e-6 * _normalized_real_scale:
        _normalized_mode_axis.plot(
            _latitude,
            _normalized_mode.imag,
            "o--",
            linewidth=1.5,
            label=fr"$\Im\,\widetilde v_{{T,{_mode_index}}}(\phi)$",
        )
    _normalized_mode_axis.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="NH-area mean",
    )
    _normalized_mode_axis.set(
        xlabel="latitude (degrees N)",
        ylabel="mode response relative to NH mean",
        title="Phase-aligned normalized mode\n(NH-area mean = 1)",
    )
    _normalized_mode_axis.legend(fontsize=9)

    for _comparison_axis in (_raw_mode_axis, _normalized_mode_axis):
        _comparison_axis.axvspan(
            _mean_ice_margin - _std_ice_margin,
            _mean_ice_margin + _std_ice_margin,
            color="tab:cyan",
            alpha=0.15,
        )
        _comparison_axis.axvline(
            _mean_ice_margin,
            color="tab:cyan",
            linestyle="--",
            linewidth=1.3,
            label="persistent sea-ice edge",
        )
        _comparison_axis.set_xlim(0.0, 90.0)
        _comparison_axis.grid(alpha=0.25)
        _comparison_axis.legend(fontsize=9)
    _mode_comparison_fig.suptitle(
        f"Temperature Koopman mode {_mode_index}: raw and normalized"
    )
    _mode_comparison_fig.tight_layout(rect=(0, 0, 1, 0.93))

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    _mode_axis, _sic_axis, _sit_axis, _albedo_axis = _axes.ravel()

    _mode_axis.plot(
        _latitude,
        _normalized_mode.real,
        "o-",
        linewidth=2.0,
        label=fr"$\Re\,v_{{T,{_mode_index}}}(\phi) / "
        fr"\langle v_{{T,{_mode_index}}}\rangle_N$",
    )
    _real_scale = float(np.max(np.abs(_normalized_mode.real)))
    if np.max(np.abs(_normalized_mode.imag)) > 1.0e-6 * _real_scale:
        _mode_axis.plot(
            _latitude,
            _normalized_mode.imag,
            "o--",
            linewidth=1.5,
            label=fr"$\Im\,v_{{T,{_mode_index}}}(\phi) / "
            fr"\langle v_{{T,{_mode_index}}}\rangle_N$",
        )
    _mode_axis.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="spatially uniform warming",
    )
    _mode_axis.axvspan(
        _mean_ice_margin - _std_ice_margin,
        _mean_ice_margin + _std_ice_margin,
        color="tab:cyan",
        alpha=0.15,
        label=r"persistent sea-ice edge $\pm1\sigma$",
    )
    _mode_axis.axvline(
        _mean_ice_margin,
        color="tab:cyan",
        linestyle="--",
        linewidth=1.5,
    )
    _mode_axis.axvline(
        _sic_variability_latitude,
        color="tab:green",
        linestyle="-.",
        linewidth=1.3,
        label="maximum concentration variability",
    )
    _mode_axis.set(
        ylabel="temperature-mode response relative to NH mean",
        title="Temperature mode and persistent sea-ice edge",
    )
    _mode_axis.legend(fontsize=8)

    _sic_axis.plot(
        _diagnostic_latitude,
        _sic_mean,
        color="tab:blue",
        linewidth=2.0,
        label="stationary mean",
    )
    _sic_axis.fill_between(
        _diagnostic_latitude,
        np.clip(_sic_mean - _sic_std, 0.0, 1.0),
        np.clip(_sic_mean + _sic_std, 0.0, 1.0),
        color="tab:blue",
        alpha=0.2,
        label=r"$\pm1\sigma$",
    )
    _sic_axis.plot(
        _diagnostic_latitude,
        _persistent_ice_fraction_mean,
        color="tab:cyan",
        linewidth=2.0,
        label=r"fraction of longitudes with annual SIC $\geq 0.5$",
    )
    _sic_axis.fill_between(
        _diagnostic_latitude,
        np.clip(
            _persistent_ice_fraction_mean - _persistent_ice_fraction_std,
            0.0,
            1.0,
        ),
        np.clip(
            _persistent_ice_fraction_mean + _persistent_ice_fraction_std,
            0.0,
            1.0,
        ),
        color="tab:cyan",
        alpha=0.15,
    )
    _sic_axis.axhline(
        0.50,
        color="tab:cyan",
        linestyle="--",
        linewidth=1.0,
        label="zonal edge threshold",
    )
    _sic_axis.axvline(
        _sic_gradient_latitude,
        color="tab:orange",
        linestyle=":",
        label="maximum mean gradient",
    )
    _sic_axis.axvline(
        _sic_variability_latitude,
        color="tab:green",
        linestyle="-.",
        label="maximum variability",
    )
    _sic_axis.axvline(
        _mode_peak_latitude,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="temperature-mode maximum",
    )
    _sic_axis.set(
        ylabel="sea-ice concentration",
        ylim=(-0.03, 1.03),
        title="Zonal sea-ice concentration and persistent-ice fraction",
    )
    _sic_axis.legend(fontsize=8)

    _sit_axis.plot(
        _diagnostic_latitude,
        _sit_mean,
        color="tab:purple",
        linewidth=2.0,
        label="stationary mean",
    )
    _sit_axis.fill_between(
        _diagnostic_latitude,
        np.maximum(_sit_mean - _sit_std, 0.0),
        _sit_mean + _sit_std,
        color="tab:purple",
        alpha=0.2,
        label=r"$\pm1\sigma$",
    )
    _sit_axis.axvline(
        _mode_peak_latitude,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="temperature-mode maximum",
    )
    _sit_axis.set(
        xlabel="latitude (degrees N)",
        ylabel="sea-ice thickness (m)",
        title="Ocean-only zonal sea-ice thickness",
    )
    _sit_axis.legend(fontsize=8)

    _albedo_axis.plot(
        _diagnostic_latitude,
        _albedo_mean,
        color="tab:brown",
        linewidth=2.0,
        label="stationary mean",
    )
    _albedo_axis.fill_between(
        _diagnostic_latitude,
        _albedo_mean - _albedo_std,
        _albedo_mean + _albedo_std,
        color="tab:brown",
        alpha=0.2,
        label=r"$\pm1\sigma$",
    )
    _albedo_axis.axvline(
        _albedo_gradient_latitude,
        color="tab:brown",
        linestyle=":",
        label="maximum mean gradient",
    )
    _albedo_axis.axvline(
        _mode_peak_latitude,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="temperature-mode maximum",
    )
    _albedo_axis.set(
        xlabel="latitude (degrees N)",
        ylabel="surface albedo",
        title="Zonal surface albedo (land and ocean)",
    )
    _albedo_axis.legend(fontsize=8)

    for _axis in _axes.ravel():
        _axis.set_xlim(0.0, 90.0)
        _axis.grid(alpha=0.25)
    _fig.suptitle(
        f"Temperature Koopman mode {_mode_index} and ice diagnostics\n"
        rf"$\lambda_{{{_mode_index}}}={_rate.real:.3g}{_rate.imag:+.3g}i$ "
        r"year$^{-1}$, "
        rf"$\tau={_relaxation_time:.3g}$ years"
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.94))
    mo.vstack([_mode_comparison_fig, _fig])
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
    koopman_correlation_mode_count = 10
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
        label=fr"Koopman temperature ({retained_koopman_mode_count} modes)",
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
    _ax.set_xlim(left=-1, right=5)
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
