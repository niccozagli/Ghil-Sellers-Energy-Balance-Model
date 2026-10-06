"""Marimo app for one South Atlantic PlaSim Koopman fit at a selected μ."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PlaSim: South Atlantic zonal Koopman analysis

    One KDMD fit at the selected lag using the selected μ raw-map
    archive and its South Atlantic basin masks (65°W–20°E, south of the equator).
    The state contains basin zonal-mean surface temperature and wet-volume
    zonal-mean upper-ocean temperature, with each ocean row detrended.
    The leading phase zero uses South Atlantic T21 sea-ice area. The harmonic
    contribution is composited on this same leading phase clock. The spectrum
    panel shows direct single-lag rates; multilag resonance estimates are
    deferred.
    """)
    return


@app.cell
def _():
    from gsebm.plasim_koopman_single import (
        available_mu_values,
        extract_eigenmode,
        fit_koopman,
        harmonic_contributions,
        leading_and_harmonic_eigenfunctions,
        leading_eigenfunction,
        load_fields,
    )

    return (
        available_mu_values,
        extract_eigenmode,
        fit_koopman,
        harmonic_contributions,
        leading_and_harmonic_eigenfunctions,
        leading_eigenfunction,
        load_fields,
    )


@app.cell
def _(available_mu_values, mo):
    _values = available_mu_values()
    if not _values:
        raise FileNotFoundError("No PlaSim raw-map archive with basin masks was found")
    selected_mu = mo.ui.dropdown(
        options=_values, value="1240" if "1240" in _values else _values[0],
        label="PlaSim μ",
    )
    selected_mu
    return (selected_mu,)


@app.cell
def _(selected_mu):
    analysis_mu = selected_mu.value
    stationary_start_year = 10_000#7000
    maximum_ocean_depth_m = 700.0
    snapshot_lag_years = 5
    factorization_rel_threshold = 1e-5
    koopman_rel_threshold = 5e-5
    maximum_training_snapshots = 10_000
    training_seed = 0
    return (
        analysis_mu,
        factorization_rel_threshold,
        koopman_rel_threshold,
        maximum_ocean_depth_m,
        maximum_training_snapshots,
        snapshot_lag_years,
        stationary_start_year,
        training_seed,
    )


@app.cell
def _(analysis_mu, load_fields, maximum_ocean_depth_m, stationary_start_year):
    fields = load_fields(stationary_start_year, maximum_ocean_depth_m, analysis_mu)
    return (fields,)


@app.cell
def _(
    factorization_rel_threshold,
    fields,
    fit_koopman,
    koopman_rel_threshold,
    maximum_training_snapshots,
    snapshot_lag_years,
    training_seed,
):
    koopman_spectrum, koopman_eigenvalues_per_year, retained_koopman_rank, _kernel_bandwidths = fit_koopman(
        fields,
        snapshot_lag_years,
        koopman_rel_threshold,
        maximum_training_snapshots,
        training_seed,
        factorization_rel_threshold,
    )
    return (
        koopman_eigenvalues_per_year,
        koopman_spectrum,
        retained_koopman_rank,
    )


@app.cell
def _(
    fields,
    koopman_eigenvalues_per_year,
    koopman_spectrum,
    leading_and_harmonic_eigenfunctions,
    leading_eigenfunction,
):
    try:
        (
            leading_complex_mode_index,
            complex_eigenfunction,
            second_harmonic_index,
            second_harmonic_eigenfunction,
            mode_scales,
        ) = leading_and_harmonic_eigenfunctions(
            koopman_spectrum, koopman_eigenvalues_per_year, fields
        )
    except ValueError as error:
        if "No stable mode lies within 10%" not in str(error):
            raise
        leading_complex_mode_index, complex_eigenfunction = leading_eigenfunction(
            koopman_spectrum, koopman_eigenvalues_per_year, fields
        )
        second_harmonic_index = None
        second_harmonic_eigenfunction = None
        mode_scales = None
    return (
        complex_eigenfunction,
        leading_complex_mode_index,
        mode_scales,
        second_harmonic_eigenfunction,
        second_harmonic_index,
    )


@app.cell
def _(
    complex_eigenfunction,
    fields,
    harmonic_contributions,
    koopman_spectrum,
    leading_complex_mode_index,
    maximum_training_snapshots,
    mode_scales,
    second_harmonic_eigenfunction,
    second_harmonic_index,
    snapshot_lag_years,
    training_seed,
):
    harmonic_data = None
    if second_harmonic_index is not None:
        harmonic_data = harmonic_contributions(
            koopman_spectrum,
            fields,
            leading_complex_mode_index,
            complex_eigenfunction,
            second_harmonic_index,
            second_harmonic_eigenfunction,
            mode_scales,
            lag=snapshot_lag_years,
            maximum_training_snapshots=maximum_training_snapshots,
            seed=training_seed,
        )
    return (harmonic_data,)


@app.cell(hide_code=True)
def _(
    analysis_mu,
    complex_eigenfunction,
    fields,
    koopman_eigenvalues_per_year,
    koopman_rel_threshold,
    leading_complex_mode_index,
    mo,
    retained_koopman_rank,
    second_harmonic_index,
    snapshot_lag_years,
):
    import numpy as np

    _eigenvalue = koopman_eigenvalues_per_year[leading_complex_mode_index]
    _harmonic_eigenvalue = (
        koopman_eigenvalues_per_year[second_harmonic_index]
        if second_harmonic_index is not None else None
    )
    _period = 2 * np.pi / _eigenvalue.imag
    mo.md(
        f"**μ={analysis_mu.replace('p', '.')} · lag={snapshot_lag_years} yr · "
        f"TSVD={koopman_rel_threshold:g} · rank={retained_koopman_rank}**  \n"
        f"{fields['years'].size} annual states "
        f"({fields['years'][0]}–{fields['years'][-1]}); "
        f"{fields['state'].shape[1]} South Atlantic zonal features. "
        f"Leading mode: λ={_eigenvalue.real:.5f}{_eigenvalue.imag:+.5f}i yr⁻¹, "
        f"period={_period:.1f} yr. "
        f"Eigenfunction RMS={np.sqrt(np.mean(np.abs(complex_eigenfunction) ** 2)):.3f}.  \n"
        + (
            f"Second harmonic: λ={_harmonic_eigenvalue.real:.5f}"
            f"{_harmonic_eigenvalue.imag:+.5f}i yr⁻¹, "
            f"frequency ratio={_harmonic_eigenvalue.imag / _eigenvalue.imag:.3f}."
            if _harmonic_eigenvalue is not None
            else "No stable mode was found within 10% of twice the leading frequency."
        )
    )
    return (np,)


@app.cell
def _(mo):
    refresh_figure = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        label="Reload figure code and redraw",
    )
    mo.vstack([
        mo.md(
            "Edit `make_figure`, `make_harmonic_contribution_figure`, "
            "`make_indexed_spectrum_figure`, or `make_selected_eigenmode_figure` in "
            "`src/gsebm/plasim_koopman_single.py`, "
            "then click the button. The fitted spectrum and eigenfunctions stay in memory."
        ),
        refresh_figure,
    ])
    return (refresh_figure,)


@app.cell
def _(
    complex_eigenfunction,
    fields,
    koopman_eigenvalues_per_year,
    koopman_rel_threshold,
    leading_complex_mode_index,
    refresh_figure,
    retained_koopman_rank,
    snapshot_lag_years,
):
    import importlib
    from gsebm import plasim_koopman_single as _analysis

    _refresh_count = refresh_figure.value
    importlib.reload(_analysis)
    figure_psi = _analysis.align_ice_peak(
        complex_eigenfunction, fields["ice_anomaly"]
    )
    figure = _analysis.make_figure(
        fields,
        koopman_eigenvalues_per_year,
        leading_complex_mode_index,
        figure_psi,
        lag=snapshot_lag_years,
        threshold=koopman_rel_threshold,
        rank=retained_koopman_rank,
    )

    figure_module = _analysis
    figure
    return figure_module, figure_psi


@app.cell
def _(
    figure_module,
    harmonic_data,
    koopman_rel_threshold,
    mo,
    retained_koopman_rank,
    snapshot_lag_years,
):
    if harmonic_data is None:
        _output = mo.md("The harmonic contribution figure needs a stable mode near $2\\omega$.")
    else:
        _output = figure_module.make_harmonic_contribution_figure(
            harmonic_data,
            lag=snapshot_lag_years,
            threshold=koopman_rel_threshold,
            rank=retained_koopman_rank,
        )
    _output
    return


@app.cell(hide_code=True)
def _(harmonic_data, mo, retained_koopman_rank):
    if harmonic_data is None:
        _output = mo.md("Harmonic diagnostics are unavailable for this fit.")
    else:
        _summary = harmonic_data["summary"]
        _rows = []
        for _name, _label, _unit in (
            ("I", "I", "10⁶ km²"),
            ("edge Ts", "edge $T_s$", "K"),
            ("theta 22-27S", "$\\theta$, 22–27°S", "mK"),
        ):
            _item = _summary[_name]
            _b1, _b2 = _item["B1"], _item["B2"]
            _rows.append(
                f"| {_label} ({_unit}) | {_b1.real:+.4g}{_b1.imag:+.4g}i | "
                f"{_b2.real:+.4g}{_b2.imag:+.4g}i | "
                f"{_item['amplitude_ratio']:.3f} | {_item['delta2']:+.3f} |"
            )
        _r2_first, _r2_both = _summary["ice_variance_fractions"]
        _ice_delta_difference = _summary["I"]["delta2"] - 2.1
        _reference = (
            f"Black ice composite $\\Delta_2={_summary['ice_observed_delta2']:+.3f}$ rad."
        )
        if harmonic_data["mu"] == "1240":
            _reference += (
                " Model-free reference for μ=1240 in this basin is about $+2.1$ rad. "
                f"The KDMD ice $\\Delta_2$ differs by {_ice_delta_difference:+.3f} rad."
            )
        _output = mo.md(
            f"**Harmonic diagnostics · rank {retained_koopman_rank} · "
            f"{harmonic_data['mode_source']}**  \n"
            "Fourier convention: $B_\\ell=\\langle C_\\ell(\\phi)"
            "e^{-i\\ell\\phi}\\rangle$ over 36 equally spaced phase bins.  \n"
            "| Observable | $B_1$ | $B_2$ | $|B_2|/|B_1|$ | "
            "$\\Delta_2$ (rad) |\n|---|---:|---:|---:|---:|\n"
            + "\n".join(_rows)
            + f"\n\nIce composite variance explained: "
            f"$I_1$ {_r2_first:.3f}; $I_1+I_2$ {_r2_both:.3f}. "
            f"Phase locking $R_2={_summary['phase_locking_R2']:.3f}$ "
            f"(fitted offset {_summary['phase_locking_offset']:+.3f} rad). "
            + _reference
        )
    _output
    return


@app.cell
def _(
    fields,
    figure_psi,
    koopman_eigenvalues_per_year,
    leading_complex_mode_index,
    np,
    second_harmonic_eigenfunction,
):
    import matplotlib.pyplot as plt

    _phase = np.mod(np.angle(figure_psi), 2 * np.pi)
    _edges = np.linspace(0, 2 * np.pi, 37)
    _centres = (_edges[:-1] + _edges[1:]) / 2
    _phase_rate = np.angle(
        figure_psi[1:] * np.conj(figure_psi[:-1])
    ) / np.diff(fields["years"])
    _counts, _ = np.histogram(_phase[:-1], bins=_edges)
    _rate_sum, _ = np.histogram(
        _phase[:-1], bins=_edges, weights=_phase_rate
    )
    _mean_rate = np.divide(
        _rate_sum,
        _counts,
        out=np.full(_counts.shape, np.nan, dtype=float),
        where=_counts > 0,
    )
    _omega = koopman_eigenvalues_per_year[leading_complex_mode_index].imag

    _fig = plt.figure(figsize=(8, 8), layout="constrained")
    _grid = _fig.add_gridspec(3, 1, height_ratios=(1, 1, 1.2))
    _hist_axis = _fig.add_subplot(_grid[0])
    _rate_axis = _fig.add_subplot(_grid[1], sharex=_hist_axis)
    _relation_axis = _fig.add_subplot(_grid[2])
    _hist_axis.hist(_phase, bins=_edges, color="#4472a0")
    _hist_axis.set(ylabel="Annual states", title=r"Distribution of arg $\psi_1$")
    _rate_axis.plot(_centres, _mean_rate, ".-", color="#4472a0", label="Bin mean")
    _rate_axis.axhline(
        _omega, color="#b44b38", linestyle="--",
        label=rf"$\omega = {_omega:.3f}$ rad yr$^{{-1}}$",
    )
    _rate_axis.set(
        xlim=(0, 2 * np.pi),
        xlabel=r"arg $\psi_1$",
        ylabel=r"Mean $\Delta$phase/$\Delta t$ (rad yr$^{-1}$)",
        title="Mean one-year phase advance by starting phase",
    )
    _rate_axis.set_xticks(
        np.linspace(0, 2 * np.pi, 5),
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    _rate_axis.legend(frameon=False)
    _hist_axis.tick_params(labelbottom=False)
    if second_harmonic_eigenfunction is not None:
        _harmonic_phase = np.mod(np.angle(second_harmonic_eigenfunction), 2 * np.pi)
        _relation_axis.scatter(
            _phase, _harmonic_phase, s=2, color="#4472a0",
            alpha=0.25, rasterized=True,
        )
        _reference_phase = np.linspace(0, np.pi, 200)
        for _offset in (0, np.pi):
            _relation_axis.plot(
                _reference_phase + _offset, 2 * _reference_phase,
                "k--", lw=1, label="2:1 reference" if _offset == 0 else None,
            )
        _relation_axis.set(
            xlim=(0, 2 * np.pi),
            ylim=(0, 2 * np.pi),
            xlabel=r"arg $\psi_1$",
            ylabel=r"arg $\psi_2$",
            title="Leading and second-harmonic phases",
        )
        _relation_axis.set_xticks(
            np.linspace(0, 2 * np.pi, 5),
            ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        )
        _relation_axis.set_yticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        _relation_axis.legend(frameon=False)
    else:
        _relation_axis.set_visible(False)
    _fig
    return


@app.cell
def _(
    figure_module,
    koopman_eigenvalues_per_year,
    leading_complex_mode_index,
    second_harmonic_index,
):
    indexed_spectrum = figure_module.make_indexed_spectrum_figure(
        koopman_eigenvalues_per_year,
        leading_complex_mode_index,
        second_harmonic_index,
    )
    indexed_spectrum
    return


@app.cell
def _(koopman_eigenvalues_per_year, mo):
    _options = {
        f"{_index}: {_rate.real:+.5f}{_rate.imag:+.5f}i yr⁻¹"
        + (" (real)" if abs(_rate.imag) < 1e-10 else ""): _index
        for _index, _rate in enumerate(koopman_eigenvalues_per_year)
    }
    chosen_mode = mo.ui.dropdown(
        options=_options, allow_select_none=True,
        searchable=True, label="Inspect KDMD eigenmode (zero-based index)",
    )
    chosen_mode
    return (chosen_mode,)


@app.cell
def _(
    chosen_mode,
    extract_eigenmode,
    fields,
    koopman_eigenvalues_per_year,
    koopman_spectrum,
    maximum_training_snapshots,
    snapshot_lag_years,
    training_seed,
):
    selected_mode = None
    selected_eigenvalue = None
    selected_eigenfunction = None
    selected_ice_mode = None
    selected_surface_mode = None
    selected_ocean_mode = None
    if chosen_mode.value is not None:
        selected_mode = extract_eigenmode(
            koopman_spectrum, koopman_eigenvalues_per_year, fields,
            chosen_mode.value, lag=snapshot_lag_years,
            maximum_training_snapshots=maximum_training_snapshots,
            seed=training_seed,
        )
        selected_eigenvalue = selected_mode["eigenvalue"]
        selected_eigenfunction = selected_mode["eigenfunction"]
        selected_ice_mode = selected_mode["ice_mode"]
        selected_surface_mode = selected_mode["surface_mode"]
        selected_ocean_mode = selected_mode["ocean_mode"]
    return (
        selected_eigenfunction,
        selected_eigenvalue,
        selected_ice_mode,
        selected_mode,
        selected_ocean_mode,
        selected_surface_mode,
    )


@app.cell(hide_code=True)
def _(mo, selected_mode):
    if selected_mode is None:
        _output = mo.md("Choose an index above to inspect its eigenfunction and modes.")
    else:
        _eigenvalue = selected_mode["eigenvalue"]
        _kind = "real" if selected_mode["is_real_mode"] else "complex"
        _ice_mode = selected_mode["ice_mode"]
        _output = mo.md(
            f"**Selected {_kind} mode {selected_mode['index']}** · "
            f"λ={_eigenvalue.real:+.5f}{_eigenvalue.imag:+.5f}i yr⁻¹ · "
            f"South Atlantic ice mode={_ice_mode.real:+.4g}{_ice_mode.imag:+.4g}i "
            "(10⁶ km²).  \n"
            "The eigenfunction has unit RMS; its mode coefficients use the inverse "
            "scaling. The arrays are available as `selected_eigenfunction`, "
            "`selected_surface_mode`, `selected_ocean_mode`, and `selected_ice_mode`. "
            "For a complex conjugate pair, use $2\\operatorname{Re}(v\\psi)$; "
            "for a real mode, use $\\operatorname{Re}(v\\psi)$."
        )
    _output
    return


@app.cell
def _(fields, figure_module, selected_mode):
    _output = None
    if selected_mode is not None:
        _output = figure_module.make_selected_eigenmode_figure(
            fields, selected_mode,
        )
    _output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
