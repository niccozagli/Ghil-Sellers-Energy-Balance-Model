"""Interactive Southern Hemisphere and ocean-box PlaSim diagnostics."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import welch
    from gsebm.plasim_raw_maps import RAW_MAP_ROOTS
    from gsebm.plasim_composites import (
        compute_map_composite, phase_sampling_schedule, prepare_cycle_preview,
    )
    from gsebm.plasim_exploratory import compute_diagnostics
    from gsebm.plasim_vertical_timing import (
        LAYER_NAMES, REGIONS, compute_vertical_timing,
    )

    return (
        LAYER_NAMES,
        RAW_MAP_ROOTS,
        REGIONS,
        compute_diagnostics,
        compute_map_composite,
        compute_vertical_timing,
        io,
        mo,
        np,
        phase_sampling_schedule,
        plt,
        prepare_cycle_preview,
        welch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Southern Hemisphere PlaSim--LSG diagnostics

    The ocean panels average each entire wet box by native cell volume:
    **global ocean** and **South Atlantic, 0–60°S** (65°W–20°E).
    Depth bands follow native LSG layer boundaries through 1025 m, then the
    saved deep bands. This emphasizes basin-scale oscillations rather than one
    latitude or depth point.

    `sic` is annual mean local sea-ice cover from 0 to 1. The mean SIC and the
    fraction of ocean area with annual SIC ≥ 0.5 are distinct. Coupling flux
    retains its native LSG sign. South Atlantic surface temperature uses T21
    `ts` over ocean cells, including ice-covered cells. The additional panels
    below also read annual radiation, surface flux, albedo, snow-depth,
    salinity, and LSG ice-volume fields from the same archive.
    """)
    return


@app.cell
def _(RAW_MAP_ROOTS, mo):
    selected_root = mo.ui.dropdown(
        options=RAW_MAP_ROOTS, value="repo", label="Archive root",
    )
    selected_root
    return (selected_root,)


@app.cell
def _(mo, selected_root):
    _root = selected_root.value
    archives = {
        _path.parent.name.removeprefix("CONTROL_360ppm_T21L10_10000Y_MU_"): _path
        for _path in sorted(_root.glob("*/*_spinup_raw_maps.nc"))
    }
    mo.stop(not archives, mo.md(f"No raw-map archives in `{_root}`."))
    _options = sorted(
        archives,
        key=lambda _value: (
            float(_value.split("_", 1)[0].replace("p", ".")), _value,
        ),
    )
    selected_mu = mo.ui.dropdown(
        options=_options, value="1240" if "1240" in archives else _options[0],
        label="PlaSim μ",
    )
    mo.vstack([mo.md(f"**Archive root:** `{_root}`"), selected_mu])
    return archives, selected_mu


@app.cell
def _(archives, compute_diagnostics, selected_mu):
    archive_path = archives[selected_mu.value]
    diagnostics = compute_diagnostics(archive_path)
    return archive_path, diagnostics


@app.cell
def _(diagnostics, mo, np):
    years = np.asarray(diagnostics["year"].values, dtype=int)
    _default_window = (
        (int(years[0]), int(years[-1]))
    )
    visible_years = mo.ui.range_slider(
        start=int(years[0]), stop=int(years[-1]), step=1,
        value=_default_window, show_value=True,
        label="Analysis window",
    )
    temperature_view = mo.ui.dropdown(
        options=["Window anomaly", "Absolute"], value="Absolute",
        label="Surface and ocean temperature display",
    )
    field_view = mo.ui.dropdown(
        options=["Window anomaly", "Absolute"], value="Absolute",
        label="Radiation, salinity, and ice-volume display",
    )
    mo.vstack([
        mo.md("### Display controls"),
        mo.md(f"**Coverage:** {years[0]}–{years[-1]} ({years.size:,} annual records)"),
        mo.hstack([temperature_view, field_view]), visible_years,
    ])
    return field_view, temperature_view, visible_years, years


@app.cell
def _(visible_years, years):
    window = (years >= visible_years.value[0]) & (years <= visible_years.value[1])
    if window.sum() < 2:
        raise ValueError("Select at least two annual records in the time-series window.")
    return (window,)


@app.cell
def _(
    diagnostics,
    np,
    plt,
    selected_mu,
    temperature_view,
    visible_years,
    window,
    years,
):
    _absolute = temperature_view.value == "Absolute"
    _figure, _axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    _temperatures = (
        ("global_temperature", "Global surface T"),
        ("northern_temperature", "Northern Hemisphere surface T"),
        ("southern_temperature", "Southern Hemisphere surface T"),
    )
    for _axis, (_name, _label) in zip(_axes.flat[:3], _temperatures):
        _values = np.asarray(diagnostics[_name].values, dtype=float)
        _reference = 0 if _absolute else _values[window].mean()
        _axis.plot(years, _values - _reference, linewidth=0.75)
        _axis.set_ylabel(f"{_label} ({'K' if _absolute else 'K anomaly'})")
    _atlantic_surface = np.asarray(diagnostics["south_atlantic_surface_temperature"].values, dtype=float)
    _atlantic_reference = 0 if _absolute else _atlantic_surface[window].mean()
    _axes[1, 0].plot(
        years, _atlantic_surface - _atlantic_reference,
        linewidth=0.8, label="South Atlantic ocean-cell surface T",
    )
    _axes[1, 0].legend(fontsize="small")
    _axes[1, 1].plot(years, diagnostics["southern_mean_sic"], label="mean SIC")
    _axes[1, 1].plot(years, diagnostics["southern_ice_covered_fraction"], label="area with SIC ≥ 0.5")
    _axes[1, 1].set_ylabel("Southern ocean fraction")
    _axes[1, 1].legend(fontsize="small")
    _axes[2, 0].plot(years, diagnostics["southern_toa_imbalance"], label="SH")
    _axes[2, 0].plot(years, diagnostics["global_toa_imbalance"], alpha=0.65, label="global")
    _axes[2, 0].set_ylabel("TOA imbalance (W m⁻²)")
    _axes[2, 0].legend(fontsize="small")
    _axes[2, 1].plot(years, diagnostics["southern_coupling_flux"])
    _axes[2, 1].set_ylabel("60–0°S coupling flux (W m⁻²)")
    for _axis in _axes.flat:
        _axis.set_xlim(*visible_years.value)
        _axis.grid(alpha=0.2)
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    for _axis in _axes[-1]:
        _axis.set_xlabel("Model year")
    _figure.suptitle(f"PlaSim μ={selected_mu.value}: climate context")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Southern radiation, South Atlantic surface fluxes, and LSG ice volume

    T21 means use the stored Gaussian latitude weights. TOA fields cover the
    Southern Hemisphere; surface fields cover ocean cells in the South Atlantic
    between 60°S and the equator. Radiation and turbulent fluxes retain their
    archived source signs. `rst` is net TOA shortwave, so `rsut` is shown as a
    separate reflected-shortwave diagnostic rather than added to `rst`.
    LSG ice volume is global and includes snow in water-equivalent thickness.
    """)
    return


@app.cell
def _(
    diagnostics,
    field_view,
    np,
    plt,
    selected_mu,
    visible_years,
    window,
    years,
):
    _anomaly = field_view.value == "Window anomaly"
    _figure, _axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    _groups = (
        (_axes[0, 0], "Southern TOA", "W m⁻²", (
            ("southern_toa_shortwave", "net shortwave"),
            ("southern_toa_longwave", "net longwave"),
            ("southern_toa_reflected_shortwave", "reflected shortwave"),
            ("southern_toa_imbalance", "SW + LW"),
        ), 1.0),
        (_axes[0, 1], "South Atlantic surface radiation", "W m⁻²", (
            ("south_atlantic_surface_shortwave", "shortwave"),
            ("south_atlantic_surface_longwave", "longwave"),
        ), 1.0),
        (_axes[1, 0], "South Atlantic turbulent fluxes", "W m⁻²", (
            ("south_atlantic_sensible_heat_flux", "sensible"),
            ("south_atlantic_latent_heat_flux", "latent"),
        ), 1.0),
        (_axes[1, 1], "South Atlantic surface albedo", "1", (
            ("south_atlantic_surface_albedo", "albedo"),
        ), 1.0),
        (_axes[2, 0], "South Atlantic snow depth", "m", (
            ("south_atlantic_snow_depth", "snow depth"),
        ), 1.0),
        (_axes[2, 1], "Global LSG ice volume", "10¹² m³", (
            ("global_lsg_ice_volume", "ice and snow"),
        ), 1e12),
    )
    for _axis, _title, _unit, _series, _scale in _groups:
        for _name, _label in _series:
            _values = np.asarray(diagnostics[_name].values, dtype=float) / _scale
            _reference = np.nanmean(_values[window]) if _anomaly else 0.0
            _axis.plot(years, _values - _reference, linewidth=0.8, label=_label)
        _axis.set_title(_title)
        _axis.set_ylabel(f"{'Anomaly' if _anomaly else 'Value'} ({_unit})")
        if len(_series) > 1:
            _axis.legend(fontsize="small")
        _axis.set_xlim(*visible_years.value)
        _axis.grid(alpha=0.2)
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    for _axis in _axes[-1]:
        _axis.set_xlabel("Model year")
    _figure.suptitle(f"PlaSim μ={selected_mu.value}: additional annual fields")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Volume-weighted ocean box temperatures
    """)
    return


@app.cell
def _(diagnostics, np, plt, temperature_view, visible_years, window, years):
    _box = np.asarray(diagnostics["ocean_box_temperature"].values, dtype=float)
    _labels = (
        "0–100 m", "100–312.5 m", "312.5–700 m", "700–1025 m",
        "1025–2000 m", "2000–6000 m",
    )
    _colors = ("tab:blue", "tab:orange")
    _figure, _axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for _band, _axis in enumerate(_axes.flat):
        for _region, _label in enumerate(("Global ocean", "South Atlantic 0–60°S")):
            _values = _box[:, _region, _band]
            _reference = (
                np.nanmean(_values[window])
                if temperature_view.value == "Window anomaly" else 0
            )
            _axis.plot(
                years, _values - _reference, label=_label,
                color=_colors[_region], linewidth=0.8,
            )
        _axis.set_title(_labels[_band])
        _axis.set_ylabel("Ocean T anomaly (K)" if temperature_view.value == "Window anomaly" else "Ocean T (K)")
        _axis.set_xlim(*visible_years.value)
        _axis.grid(alpha=0.2)
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    _axes[0, 0].legend(fontsize="small")
    for _axis in _axes[-1]:
        _axis.set_xlabel("Model year")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upper-ocean salinity

    Native LSG salinity is averaged by wet cell volume in the global ocean and
    South Atlantic (0–60°S). Each band uses the same native layer boundaries
    as the temperature panels. This is a basin-scale view of salinity changes;
    it does not convert salinity to density.
    """)
    return


@app.cell
def _(diagnostics, field_view, np, plt, visible_years, window, years):
    _salinity = np.asarray(diagnostics["ocean_box_salinity"].values, dtype=float)
    _labels = ("0–100 m", "100–312.5 m", "312.5–700 m", "700–1025 m")
    _figure, _axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for _band, _axis in enumerate(_axes.flat):
        for _region, _label in enumerate(("Global ocean", "South Atlantic 0–60°S")):
            _values = _salinity[:, _region, _band]
            _reference = (
                np.nanmean(_values[window])
                if field_view.value == "Window anomaly" else 0.0
            )
            _axis.plot(
                years, _values - _reference, linewidth=0.8,
                color=("tab:blue", "tab:orange")[_region], label=_label,
            )
        _axis.set_title(_labels[_band])
        _axis.set_ylabel(
            "Salinity anomaly (‰)" if field_view.value == "Window anomaly"
            else "Salinity (‰)"
        )
        _axis.set_xlim(*visible_years.value)
        _axis.grid(alpha=0.2)
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    _axes[0, 0].legend(fontsize="small")
    for _axis in _axes[-1]:
        _axis.set_xlabel("Model year")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ocean-box power spectra

    Welch spectra use the selected time window, annual sampling, and a linear
    detrend within each segment. Peaks indicate candidate oscillation periods;
    compare them with the time series before interpreting a changing run.
    """)
    return


@app.cell
def _(diagnostics, mo, np, plt, welch, window, years):
    _selected_years = years[window]
    if _selected_years.size < 100:
        _output = mo.md("Select at least 100 years to display the power spectra.")
    else:
        _box = np.asarray(diagnostics["ocean_box_temperature"].values[window], dtype=float)
        _segment = min(2048, _selected_years.size)
        _longest_period = min(300.0, _segment / 2)
        _labels = (
            "0–100 m", "100–312.5 m", "312.5–700 m", "700–1025 m",
            "1025–2000 m", "2000–6000 m",
        )
        _figure, _axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for _band, _axis in enumerate(_axes.flat):
            for _region, _label in enumerate(("Global ocean", "South Atlantic 0–60°S")):
                _values = _box[:, _region, _band]
                _frequency, _power = welch(
                    _values, fs=1.0, window="hann", nperseg=_segment,
                    noverlap=_segment // 2, detrend="linear", scaling="density",
                )
                _positive = _frequency > 0
                _period = 1.0 / _frequency[_positive]
                _show = (_period >= 10.0) & (_period <= _longest_period)
                _axis.plot(_period[_show], _power[_positive][_show], label=_label, linewidth=1)
            _axis.set_title(_labels[_band])
            _axis.set_xscale("log")
            _axis.set_yscale("log")
            _axis.set_xlim(10, _longest_period)
            _axis.set_ylabel("Power (K² yr)")
            _axis.grid(alpha=0.2, which="both")
        _axes[0, 0].legend(fontsize="small")
        for _axis in _axes[-1]:
            _axis.set_xlabel("Period (years)")
        _figure.suptitle(
            f"{_selected_years[0]}–{_selected_years[-1]}: ocean temperature spectra "
            f"(Welch segments: {_segment} years)"
        )
        _figure.tight_layout()
        _output = _figure
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## South Atlantic surface-temperature spectrum
    """)
    return


@app.cell
def _(diagnostics, mo, np, plt, welch, window, years):
    _selected_years = years[window]
    if _selected_years.size < 100:
        _output = mo.md("Select at least 100 years to display the surface spectrum.")
    else:
        _values = np.asarray(diagnostics["south_atlantic_surface_temperature"].values[window], dtype=float)
        _segment = min(2048, _selected_years.size)
        _frequency, _power = welch(
            _values, fs=1.0, window="hann", nperseg=_segment,
            noverlap=_segment // 2, detrend="linear", scaling="density",
        )
        _positive = _frequency > 0
        _period = 1.0 / _frequency[_positive]
        _longest_period = min(300.0, _segment / 2)
        _show = (_period >= 10.0) & (_period <= _longest_period)
        _figure, _axis = plt.subplots(figsize=(9, 4))
        _axis.plot(_period[_show], _power[_positive][_show], color="tab:orange")
        _axis.set(
            xscale="log", yscale="log", xlim=(10, _longest_period),
            xlabel="Period (years)", ylabel="Power (K² yr)",
            title=f"South Atlantic ocean-cell surface T: {_selected_years[0]}–{_selected_years[-1]}",
        )
        _axis.grid(alpha=0.2, which="both")
        _figure.tight_layout()
        _output = _figure
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ice-cycle preview and map composites

    Compare the mean-centered and linearly detrended SH ice-area anomalies,
    their periods, and detected maxima before reading the annual map fields.
    Choose which strategy supplies the events and map preprocessing. Phase 0
    is an ice maximum; phases near ±0.5 are the surrounding ice minima. Each
    event uses its own neighboring maxima to map phase to model years. The
    original μ1240 S1 choices are the defaults.
    The cell markers use a band of 0.4–3 times the median cycle period,
    equivalent to about 20–150 years for μ1240.
    The S3 pathway panels show −40%, −20%, 0%, +20%, and +40% of each cycle.
    Current arrows can show phase composites of the annual 150–300 m current
    anomalies, processed with the selected detrending and running mean, or
    the full-archive mean currents for comparison.
    """)
    return


@app.cell
def _(mo):
    detrend_choice = mo.ui.dropdown(
        options={"Linear detrend": "linear", "No linear detrend": "none"},
        value="Linear detrend", label="Strategy for the composite",
    )
    running_mean_years = mo.ui.slider(
        start=1, stop=31, step=2, value=11, show_value=True,
        label="Centered running mean (years; 1 = none)",
    )
    peak_radius_fraction = mo.ui.slider(
        start=0.1, stop=0.49, step=0.01, value=0.4, show_value=True,
        label="Peak search radius (fraction of estimated period)",
    )
    period_override = mo.ui.number(
        start=0, stop=150, step=1, value=0,
        label="Period override (years; 0 = spectral estimate)",
    )
    max_phase_window = mo.ui.range_slider(
        start=-0.3, stop=0.3, step=0.01, value=(-0.02, 0.02),
        show_value=True, label="Ice-maximum phase window",
    )
    min_before_phase_window = mo.ui.range_slider(
        start=-1.0, stop=0.0, step=0.01, value=(-0.51, -0.47),
        show_value=True, label="Preceding ice-minimum phase window",
    )
    min_after_phase_window = mo.ui.range_slider(
        start=0.0, stop=1.0, step=0.01, value=(0.47, 0.51),
        show_value=True, label="Following ice-minimum phase window",
    )
    show_edge_cells = mo.ui.checkbox(
        value=True, label="Mark cells carrying 50% of the ice-area cycle",
    )
    current_view = mo.ui.dropdown(
        options={
            "Phase-current anomalies": "phase_anomaly",
            "Full-archive mean currents": "archive_mean",
        },
        value="Phase-current anomalies", label="S3 current arrows",
    )
    mo.vstack([
        mo.md("### Composite controls"),
        mo.hstack([detrend_choice, running_mean_years]),
        mo.hstack([peak_radius_fraction, period_override]),
        mo.hstack([max_phase_window, min_before_phase_window, min_after_phase_window]),
        mo.hstack([show_edge_cells, current_view]),
    ])
    return (
        current_view,
        detrend_choice,
        max_phase_window,
        min_after_phase_window,
        min_before_phase_window,
        peak_radius_fraction,
        period_override,
        running_mean_years,
        show_edge_cells,
    )


@app.cell
def _(
    detrend_choice,
    diagnostics,
    max_phase_window,
    min_after_phase_window,
    min_before_phase_window,
    mo,
    np,
    peak_radius_fraction,
    period_override,
    phase_sampling_schedule,
    plt,
    prepare_cycle_preview,
    running_mean_years,
    visible_years,
    years,
):
    cycle_preview = None
    composite_ready = False
    complete_event_count = 0
    try:
        _area = np.asarray(diagnostics["southern_ice_area"].values, dtype=float)
        _previews = {
            _method: prepare_cycle_preview(
                years, _area, *visible_years.value,
                detrend_method=_method,
                smooth_years=running_mean_years.value,
                radius_fraction=peak_radius_fraction.value,
                manual_period=period_override.value,
            )
            for _method in ("none", "linear")
        }
        cycle_preview = _previews[detrend_choice.value]
        _figure = plt.figure(figsize=(12, 7.5), constrained_layout=True)
        _grid = _figure.add_gridspec(2, 2, height_ratios=[1, 0.9])
        _normal_axis = _figure.add_subplot(_grid[0, 0])
        _detrended_axis = _figure.add_subplot(
            _grid[0, 1], sharex=_normal_axis, sharey=_normal_axis,
        )
        _spectrum_axis = _figure.add_subplot(_grid[1, :])
        _colors = {"none": "tab:blue", "linear": "tab:orange"}
        _labels = {
            "none": "Mean-centered anomaly",
            "linear": "Linearly detrended anomaly",
        }
        _event_counts = {}
        _event_errors = {}
        for _method, _axis in (("none", _normal_axis), ("linear", _detrended_axis)):
            _preview = _previews[_method]
            _color = _colors[_method]
            _axis.plot(
                _preview.years, _preview.smoothed_area,
                linewidth=0.8, color=_color,
            )
            _axis.scatter(
                _preview.years[_preview.peak_indices],
                _preview.smoothed_area[_preview.peak_indices],
                s=11, color="tab:red", label="Ice maxima",
            )
            _axis.axhline(0, color="0.6", linewidth=0.6)
            _axis.set(
                title=f"{_labels[_method]}: {len(_preview.peak_indices)} maxima",
                xlabel="Model year", ylabel="SH ice-area anomaly (10¹² m²)",
            )
            _axis.legend(fontsize="small")
            _axis.grid(alpha=0.2)
            _positive_power = _preview.spectral_power > 0
            _spectrum_axis.plot(
                _preview.spectral_periods[_positive_power],
                _preview.spectral_power[_positive_power],
                color=_color, linewidth=1.2,
                label=f"{_labels[_method]} ({_preview.spectral_period:.1f} yr peak)",
            )
            _spectrum_axis.axvline(
                _preview.spectral_period, color=_color,
                linestyle="--", linewidth=0.9,
            )
            try:
                _events, _positions = phase_sampling_schedule(
                    _preview.peak_indices, len(_preview.years),
                    _preview.smooth_years,
                    max_phase_window.value, min_before_phase_window.value,
                    min_after_phase_window.value,
                )
                _event_counts[_method] = len(_events)
            except ValueError as _error:
                _event_counts[_method] = 0
                _event_errors[_method] = str(_error)
        _spectrum_axis.set(
            xscale="log", yscale="log", xlim=(20, 150),
            xlabel="Period (years)", ylabel="Power ((10¹² m²)² yr)",
            title="Welch spectra of the smoothed ice-area anomalies",
        )
        _spectrum_axis.legend(fontsize="small")
        _spectrum_axis.grid(alpha=0.2, which="both")
        _normal_peaks = _previews["none"].years[_previews["none"].peak_indices]
        _linear_peaks = _previews["linear"].years[_previews["linear"].peak_indices]
        _match_tolerance = max(
            2, round(0.1 * min(
                _previews["none"].search_period,
                _previews["linear"].search_period,
            )),
        )
        _matched = (
            sum(np.min(np.abs(_linear_peaks - _year)) <= _match_tolerance
                for _year in _normal_peaks)
            if _linear_peaks.size else 0
        )
        _summary = (
            f"**Mean-centered:** spectral peak {_previews['none'].spectral_period:.1f} yr; "
            f"median peak spacing {np.median(_previews['none'].cycle_lengths):.1f} yr; "
            f"{_event_counts['none']} complete events.  "
            f"**Detrended:** spectral peak {_previews['linear'].spectral_period:.1f} yr; "
            f"median peak spacing {np.median(_previews['linear'].cycle_lengths):.1f} yr; "
            f"{_event_counts['linear']} complete events.  "
            f"**Agreement:** {_matched} maxima match within ±{_match_tolerance} yr "
            f"(out of {len(_normal_peaks)} mean-centered and {len(_linear_peaks)} detrended)."
        ) if _normal_peaks.size > 1 and _linear_peaks.size > 1 else (
            "Too few maxima to compare peak spacing and event agreement."
        )
        complete_event_count = _event_counts[detrend_choice.value]
        composite_ready = complete_event_count >= 3
        _selection = (
            f"**Selected for composite:** {_labels[detrend_choice.value]}; "
            f"search radius ±{cycle_preview.peak_radius_years} yr; "
            f"{complete_event_count} complete events."
            if composite_ready else
            f"**Composite unavailable:** {_event_errors.get(detrend_choice.value, 'too few complete events')}."
        )
        _output = mo.vstack([
            mo.md(f"{_summary}\n\n{_selection}"),
            _figure,
        ])
    except ValueError as _error:
        _output = mo.md(f"**Ice-cycle preview unavailable:** {_error}")
    _output
    return composite_ready, cycle_preview


@app.cell
def _(composite_ready, mo):
    run_composite = mo.ui.run_button(
        label="Run S1, S3, and vertical timing", disabled=not composite_ready,
    )
    mo.vstack([
        run_composite,
        mo.md("The map calculation starts only when this button is clicked. "
              "For live controls, open the notebook with `marimo run`."),
    ])
    return (run_composite,)


@app.cell
def _(
    archive_path,
    composite_ready,
    compute_map_composite,
    current_view,
    cycle_preview,
    io,
    max_phase_window,
    min_after_phase_window,
    min_before_phase_window,
    mo,
    np,
    plt,
    run_composite,
    selected_mu,
    show_edge_cells,
):
    mo.stop(not run_composite.value, mo.md("Review the ice-cycle preview, then run the composite."))
    mo.stop(not composite_ready or cycle_preview is None, mo.md("Choose a window with complete cycles."))
    _maps = compute_map_composite(
        archive_path, cycle_preview,
        max_phase=max_phase_window.value,
        min_before_phase=min_before_phase_window.value,
        min_after_phase=min_after_phase_window.value,
        show_edge_cells=show_edge_cells.value,
    )
    _wrap = lambda _longitude: ((np.asarray(_longitude) + 180) % 360) - 180
    _order = np.argsort(_wrap(_maps.t21_lon))
    _lon = _wrap(_maps.t21_lon)[_order]
    _lon_step = np.diff(_lon).mean()
    _lon_edges = np.r_[_lon - _lon_step / 2, _lon[-1] + _lon_step / 2]
    _lat = _maps.t21_lat
    _lat_edges = np.r_[
        _lat[0] - (_lat[1] - _lat[0]) / 2,
        (_lat[1:] + _lat[:-1]) / 2,
        _lat[-1] + (_lat[-1] - _lat[-2]) / 2,
    ]
    _figure, _axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    _fields = (
        (_maps.surface_temperature_difference, "RdBu_r", -4, 4, "Surface T (K)"),
        (_maps.sea_ice_concentration_difference, "RdBu", -0.6, 0.6, "Sea-ice concentration"),
    )
    for _axis, (_field, _cmap, _vmin, _vmax, _label) in zip(_axes[:2], _fields):
        _image = _axis.pcolormesh(
            _lon_edges, _lat_edges, _field[:, _order],
            cmap=_cmap, vmin=_vmin, vmax=_vmax, rasterized=True,
        )
        _figure.colorbar(_image, ax=_axis, fraction=0.035, pad=0.015, label=_label)
        _axis.contour(
            _lon, _lat, _maps.lsm[:, _order], levels=[0.5],
            colors="0.35", linewidths=0.6,
        )
    if _maps.edge_lat.size:
        _axes[1].plot(
            _wrap(_maps.edge_lon), _maps.edge_lat, "o",
            markersize=3.2, markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=0.7,
        )
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    _ocean_norm = Normalize(vmin=-1500, vmax=1500)
    for _row in range(_maps.lsg_lat.shape[0]):
        _x = _wrap(_maps.lsg_lon[_row])
        _sort = np.argsort(_x)
        _x = _x[_sort]
        _value = _maps.theta_150_300m_difference[_row, _sort] * 1000
        _value = np.where(_maps.wet_150_300m[_row, _sort], _value, np.nan)
        _x_edges = np.r_[_x - 2.5, _x[-1] + 2.5]
        _y = _maps.lsg_lat[_row].mean()
        _y_edges = [_y - 1.25, _y + 1.25]
        _land = np.ma.masked_equal(
            (~_maps.wet_150_300m[_row, _sort]).astype(float), 0,
        )
        _axes[2].pcolormesh(
            _x_edges, _y_edges, _land[None],
            cmap=ListedColormap(["0.82"]), vmin=0, vmax=1,
            rasterized=True,
        )
        _axes[2].pcolormesh(
            _x_edges, _y_edges, _value[None],
            cmap="RdBu_r", norm=_ocean_norm, rasterized=True,
        )
    _figure.colorbar(
        ScalarMappable(norm=_ocean_norm, cmap="RdBu_r"),
        ax=_axes[2], fraction=0.035, pad=0.015,
        label="150–300 m ocean T (mK)",
    )
    _mean_edge = np.ma.masked_invalid(
        np.where(np.abs(_lat[:, None]) > 20, _maps.mean_sea_ice_concentration, np.nan)[:, _order]
    )
    for _panel, _axis in enumerate(_axes):
        _axis.contour(
            _lon, _lat, _mean_edge, levels=[0.5],
            colors="black", linewidths=0.8, linestyles="dashed",
        )
        for _boundary in (-65, 20, 115):
            _axis.axvline(_boundary, color="0.55", linewidth=0.5, linestyle=":")
        _axis.axhline(0, color="0.7", linewidth=0.4)
        _axis.set(xlim=(-180, 180), ylim=(-80, 88), ylabel="Latitude (°)")
        _axis.text(
            0.005, 0.96, f"({'abc'[_panel]})", transform=_axis.transAxes,
            va="top", fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    _axes[-1].set_xlabel("Longitude (°)")
    _figure.suptitle(
        f"μ={selected_mu.value}, {cycle_preview.years[0]}–{cycle_preview.years[-1]}: "
        f"ice maximum − minimum ({_maps.composited_events} events)\n"
        f"{cycle_preview.detrend_method} detrend; "
        f"{cycle_preview.smooth_years}-year running mean; "
        f"median cycle {_maps.median_cycle_years:.1f} years",
    )
    _figure.tight_layout()
    _image_bytes = io.BytesIO()
    _figure.savefig(_image_bytes, format="png", dpi=300, bbox_inches="tight")
    _filename = (
        f"mu{selected_mu.value}_S1_global_maps_"
        f"{cycle_preview.years[0]}-{cycle_preview.years[-1]}.png"
    )
    _pathway_figure, _pathway_axes = plt.subplots(
        1, 5, figsize=(16, 4.6), sharex=True, sharey=True,
        constrained_layout=True,
    )
    _pathway_norm = Normalize(vmin=-1000, vmax=1000)
    _vector_lon = _wrap(_maps.lsg_vector_lon)
    _vector_mask = (
        (_maps.lsg_vector_lat >= -56) & (_maps.lsg_vector_lat <= 12)
        & (_vector_lon >= -72) & (_vector_lon <= 42)
        & (np.arange(_maps.lsg_vector_lat.shape[0])[:, None] % 2 == 0)
    )
    _phase_currents = current_view.value == "phase_anomaly"
    _current_scale = 0.45
    if _phase_currents:
        _speed = np.hypot(
            _maps.u_150_300m_snapshots[:, _vector_mask],
            _maps.v_150_300m_snapshots[:, _vector_mask],
        )
        _finite_speed = _speed[np.isfinite(_speed)]
        if _finite_speed.size:
            _current_scale = max(float(np.percentile(_finite_speed, 90)) * 22.5, 1e-4)
    for _phase_index, (_phase, _axis) in enumerate(
        zip(_maps.pathway_phases, _pathway_axes, strict=True)
    ):
        for _row in range(_maps.lsg_lat.shape[0]):
            _row_lat = _maps.lsg_lat[_row].mean()
            if _row_lat < -58 or _row_lat > 14:
                continue
            _x = _wrap(_maps.lsg_lon[_row])
            _sort = np.argsort(_x)
            _x = _x[_sort]
            _x_edges = np.r_[_x - 2.5, _x[-1] + 2.5]
            _y_edges = [_row_lat - 1.25, _row_lat + 1.25]
            _wet_row = _maps.wet_150_300m[_row, _sort]
            _land = np.ma.masked_equal((~_wet_row).astype(float), 0)
            _value = np.where(
                _wet_row,
                _maps.theta_150_300m_snapshots[_phase_index, _row, _sort] * 1000,
                np.nan,
            )
            _axis.pcolormesh(
                _x_edges, _y_edges, _land[None],
                cmap=ListedColormap(["0.82"]), vmin=0, vmax=1,
                rasterized=True,
            )
            _axis.pcolormesh(
                _x_edges, _y_edges, _value[None],
                cmap="RdBu_r", norm=_pathway_norm, rasterized=True,
            )
        _u = (
            _maps.u_150_300m_snapshots[_phase_index]
            if _phase_currents else _maps.u_mean_150_300m
        )
        _v = (
            _maps.v_150_300m_snapshots[_phase_index]
            if _phase_currents else _maps.v_mean_150_300m
        )
        _valid_vectors = _vector_mask & np.isfinite(_u) & np.isfinite(_v)
        _axis.quiver(
            _vector_lon[_valid_vectors],
            _maps.lsg_vector_lat[_valid_vectors],
            _u[_valid_vectors], _v[_valid_vectors],
            color="0.25", scale=_current_scale, width=0.0045,
            headwidth=3.5, headlength=4, alpha=0.75, zorder=4,
        )
        _axis.contour(
            _lon, _lat, _mean_edge, levels=[0.5],
            colors="black", linewidths=0.8, linestyles="dashed", zorder=5,
        )
        _axis.set(xlim=(-72, 42), ylim=(-56, 12))
        _phase_label = "0% (ice max)" if _phase == 0 else f"{_phase:+.0%}"
        _axis.text(
            0.03, 0.04, _phase_label, transform=_axis.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=6,
        )
    _pathway_axes[2].set_xlabel("Longitude (°)")
    _pathway_axes[0].set_ylabel("Latitude (°)")
    _pathway_figure.colorbar(
        ScalarMappable(norm=_pathway_norm, cmap="RdBu_r"),
        ax=_pathway_axes, orientation="horizontal", fraction=0.06,
        pad=0.12, shrink=0.65,
        label="150–300 m ocean T anomaly (mK)",
    )
    _pathway_figure.suptitle(
        f"μ={selected_mu.value}, {cycle_preview.years[0]}–{cycle_preview.years[-1]}: "
        f"S3 pathway snapshots ({_maps.composited_events} events)\n"
        f"{cycle_preview.detrend_method} detrend; "
        f"{cycle_preview.smooth_years}-year running mean; "
        f"arrows: {'phase-current anomalies' if _phase_currents else 'full-archive mean currents'}; "
        "dashed: mean ice edge",
    )
    _pathway_bytes = io.BytesIO()
    _pathway_figure.savefig(
        _pathway_bytes, format="png", dpi=300, bbox_inches="tight",
    )
    _pathway_filename = (
        f"mu{selected_mu.value}_S3_pathway_snapshots_"
        f"{cycle_preview.years[0]}-{cycle_preview.years[-1]}_"
        f"{'phase_currents' if _phase_currents else 'mean_currents'}.png"
    )
    mo.vstack([
        mo.md("### S1 global maps"),
        _figure,
        mo.download(
            data=_image_bytes.getvalue(), filename=_filename,
            mimetype="image/png", label="Download S1 PNG",
        ),
        mo.md("### S3 South Atlantic pathway snapshots"),
        _pathway_figure,
        mo.download(
            data=_pathway_bytes.getvalue(), filename=_pathway_filename,
            mimetype="image/png", label="Download S3 PNG",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vertical timing: ice edge and tropics

    The left panels show wet-volume-weighted zonal ocean temperature anomalies
    through 1000 m. The right panels compare South Atlantic box means at three
    ocean layers with the ocean-cell surface temperature, all in mK. Dots mark
    the warm peak of the first cycle harmonic. Phase is relative to each pair
    of neighboring ice maxima; 0% is an SH ice-area maximum.
    """)
    return


@app.cell
def _(
    LAYER_NAMES,
    REGIONS,
    archive_path,
    composite_ready,
    compute_vertical_timing,
    cycle_preview,
    io,
    mo,
    np,
    plt,
    run_composite,
    selected_mu,
):
    mo.stop(not run_composite.value, mo.md("Review the ice-cycle preview, then run the composite."))
    mo.stop(not composite_ready or cycle_preview is None, mo.md("Choose a window with complete cycles."))
    import json

    _timing = compute_vertical_timing(archive_path, cycle_preview)
    _figure, _axes = plt.subplots(
        2, 2, figsize=(12, 8), sharex="col",
        gridspec_kw={"width_ratios": [1.1, 1]},
        constrained_layout=True,
    )
    _output = {}
    for _region, (_name, _band, _box) in enumerate(REGIONS):
        _profile = _timing.profiles_mk[_region]
        _limit = np.nanmax(np.abs(_profile))
        _axis = _axes[_region, 0]
        _mesh = _axis.pcolormesh(
            _timing.phases * 100, _timing.depths, _profile,
            cmap="RdBu_r", vmin=-_limit, vmax=_limit,
            shading="nearest", rasterized=True,
        )
        _axis.plot(
            _timing.profile_warm_phase[_region] * 100,
            _timing.depths, "k.", markersize=5,
        )
        _axis.invert_yaxis()
        _axis.set_ylabel("Depth (m)")
        _axis.set_title(f"{_name}: zonal mean {abs(_band[1]):g}–{abs(_band[0]):g}°S")
        _figure.colorbar(_mesh, ax=_axis, label="Ocean T anomaly (mK)")

        _box_axis = _axes[_region, 1]
        _box_output = {}
        for _field, (_label, _color) in enumerate(zip(
            LAYER_NAMES, ("C0", "C1", "C2", "C3"), strict=True,
        )):
            _box_axis.plot(
                _timing.phases * 100,
                _timing.boxes_mk[_region, _field],
                color=_color, linewidth=1.3, label=_label,
            )
            _box_output[_label] = {
                "warm_peak_percent_of_cycle": round(
                    float(_timing.box_warm_phase[_region, _field] * 100), 1,
                ),
                "harmonic_amplitude_mK": round(
                    float(_timing.box_amplitude_mk[_region, _field]), 1,
                ),
            }
        _box_axis.axhline(0, color="0.7", linewidth=0.6)
        _box_axis.axvline(0, color="0.7", linewidth=0.6)
        _box_axis.set_ylabel("Temperature anomaly (mK)")
        _box_axis.set_title(
            f"{_name}: South Atlantic box {abs(_box[1]):g}–{abs(_box[0]):g}°S, "
            f"{abs(_box[2]):g}°W–{_box[3]:g}°E"
        )
        _box_axis.legend(fontsize="small", frameon=False)
        _output[_name.lower().replace(" ", "_")] = {
            "zonal_warm_peaks": {
                f"{_depth:g} m": {
                    "percent_of_cycle": round(float(_phase * 100), 1),
                    "harmonic_amplitude_mK": round(float(_amplitude), 1),
                }
                for _depth, _phase, _amplitude in zip(
                    _timing.depths,
                    _timing.profile_warm_phase[_region],
                    _timing.profile_amplitude_mk[_region], strict=True,
                )
            },
            "south_atlantic_box": _box_output,
        }
    for _axis in _axes[1]:
        _axis.set_xlabel("Phase after SH ice maximum (% of cycle)")
    _figure.suptitle(
        f"μ={selected_mu.value}, {cycle_preview.years[0]}–{cycle_preview.years[-1]}: "
        f"vertical timing ({_timing.event_count} events)\n"
        f"{cycle_preview.detrend_method} detrend; "
        f"{cycle_preview.smooth_years}-year running mean; "
        f"median cycle {_timing.median_cycle_years:.1f} years",
    )
    _png = io.BytesIO()
    _figure.savefig(_png, format="png", dpi=300, bbox_inches="tight")
    _stem = (
        f"mu{selected_mu.value}_vertical_timing_edge_vs_tropics_"
        f"{cycle_preview.years[0]}-{cycle_preview.years[-1]}"
    )
    _output["analysis"] = {
        "mu": selected_mu.value,
        "first_year": int(cycle_preview.years[0]),
        "last_year": int(cycle_preview.years[-1]),
        "events": _timing.event_count,
        "detrend": cycle_preview.detrend_method,
        "running_mean_years": cycle_preview.smooth_years,
        "median_cycle_years": _timing.median_cycle_years,
    }
    mo.vstack([
        _figure,
        mo.hstack([
            mo.download(
                data=_png.getvalue(), filename=f"{_stem}.png",
                mimetype="image/png", label="Download vertical-timing PNG",
            ),
            mo.download(
                data=json.dumps(_output, indent=2).encode(),
                filename=f"{_stem}.json", mimetype="application/json",
                label="Download timing JSON",
            ),
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
