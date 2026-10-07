"""Marimo app: irregularity of the South Atlantic oscillation around its Koopman cycle."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from gsebm import plasim_koopman_irregularity as irregularity
    from gsebm.plasim_koopman_single import available_mu_values, load_map_fields
    from gsebm.plasim_raw_maps import RAW_MAP_ROOTS

    return (
        RAW_MAP_ROOTS,
        available_mu_values,
        irregularity,
        load_map_fields,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PlaSim: irregularity of the South Atlantic oscillation

    The Koopman fit of `Koopman_analysis.py` (South Atlantic zonal $T_s$ and
    θ 0–700 m, lag 5 yr) gives a regular cycle: the leading eigenfunction
    $\psi_1$ and its harmonic $\psi_2$. Here that cycle is removed and the
    remainder is studied on the clock $\theta_1=\arg\psi_1$ (phase 0 = SA
    ice maximum). For an observable $G$, three residuals:

    - **ψ₁,ψ₂ modes**: $G-2\,\mathrm{Re}(a_1\psi_1)-2\,\mathrm{Re}(a_2\psi_2)$;
    - **phase composite**: $G-C_G(\theta_1)$, all harmonics of the mean cycle,
      so a sea-ice cell that always switches at the same phase is removed exactly;
    - **phase × |ψ₁| composite**: $G-C_G(\theta_1,\,|\psi_1|\text{ tercile})$,
      which also removes the dependence of the cycle on its strength.
      This is the residual used below unless stated.

    Each μ uses its final 4000 stationary years with identical fit settings.
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
def _(available_mu_values, irregularity, mo, selected_root):
    archive_root = selected_root.value
    mu_options = [
        _mu for _mu in available_mu_values(archive_root)
        if _mu in irregularity.ANALYSIS_WINDOWS
    ]
    mo.stop(
        not mu_options,
        mo.md(f"No archive with an analysis window in `{archive_root}`."),
    )
    selected_mu = mo.ui.dropdown(
        options=mu_options, value=mu_options[0], label="PlaSim μ",
    )
    selected_state = mo.ui.dropdown(
        options={
            "zonal Ts + θ 0–700 m": ("surface", "ocean"),
            "SA ice per row + θ 0–700 m": ("ice", "ocean"),
            "SA ice per row + Ts + θ 0–700 m": ("ice", "surface", "ocean"),
        },
        value="zonal Ts + θ 0–700 m", label="Koopman state",
    )
    _rows = "\n".join(
        f"| {_mu.replace('p', '.')} | {_start}–{_end} |"
        for _mu, (_start, _end) in irregularity.ANALYSIS_WINDOWS.items()
    )
    mo.vstack([
        mo.md(f"**Archive root:** `{archive_root}`"),
        mo.md("| μ | window |\n|---|---|\n" + _rows),
        mo.hstack([selected_mu, selected_state]),
    ])
    return archive_root, mu_options, selected_mu, selected_state


@app.cell
def _(archive_root, irregularity, selected_mu, selected_state):
    analysis_mu = selected_mu.value
    state_blocks = selected_state.value
    result = irregularity.irregularity_analysis(
        archive_root, analysis_mu, state_blocks=state_blocks,
    )
    return analysis_mu, result, state_blocks


@app.cell(hide_code=True)
def _(mo, result):
    _rate = result["rates"][result["leading_index"]]
    _summary = result["summary"]
    _fields = result["fields"]
    mo.md(
        f"**μ={result['mu'].replace('p', '.')} · {_fields['years'][0]}–{_fields['years'][-1]} · "
        f"rank {result['rank']}** · leading λ={_rate.real:.4f}{_rate.imag:+.4f}i yr⁻¹, "
        f"period {result['period']:.1f} yr.  \n"
        "Share of variance left in the residual "
        "(ψ₁,ψ₂ modes / phase composite / phase×|ψ₁| composite): "
        + "; ".join(
            f"{_block['label']} "
            + " / ".join(f"{_value:.0%}" for _value in _block["residuals"]["fractions"].values())
            for _block in result["blocks"].values()
        )
        + "; SA ice area "
        + " / ".join(f"{_value:.0%}" for _value in result["ice_residuals"]["fractions"].values())
        + "."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Irregularity of the clock

    The one-year phase advance and $|\psi_1|$ by phase, the phase diffusion
    $\mathrm{Var}[\theta(t+\tau)-\theta(t)]\approx 2D\tau$ (the oscillation
    loses its phase memory after about $1/D$ years), and the lengths of
    complete cycles between phase-zero passes.
    """)
    return


@app.cell
def _(analysis_mu, irregularity, result):
    irregularity.make_clock_figure(
        result["psi1"], result["fields"]["years"], result["omega"], mu=analysis_mu,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weak-cycle episodes

    Years where the 101-yr running mean of $|\psi_1|$ (unit RMS) falls
    below 0.75 are shaded. Check them before reading any whole-window
    irregularity number: a few episodes can dominate the phase diffusion,
    the cycle-length spread, and the return times.
    """)
    return


@app.cell
def _(irregularity, result):
    irregularity.make_amplitude_history_figure({result["mu"]: result})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Residual variance by phase

    Top: residual standard deviation by phase and latitude. Bottom: the
    weighted mean variance of the three residuals. The gap between grey and
    blue is the part of the regular cycle two harmonics cannot represent
    (sharp ice switches); between blue and magenta, the dependence of the
    cycle on its amplitude.
    """)
    return


@app.cell
def _(analysis_mu, irregularity, result):
    irregularity.make_state_residual_figure(
        result["blocks"], result["phase"], mu=analysis_mu,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Band-mean residuals: variance, skewness, memory, spectrum

    Variance is relative to each series' total variance (shading: one
    standard error). Lag-one memory is the slope of $r(t+1)$ on $r(t)$ for
    the years that start in each phase bin; a value growing towards 1 at a
    phase marks where disturbances linger.
    """)
    return


@app.cell
def _(analysis_mu, irregularity, result):
    irregularity.make_band_residual_figure(
        result["band_residuals"], result["phase"], result["fields"]["years"],
        mu=analysis_mu,
    )
    return


@app.cell(hide_code=True)
def _(mo, result):
    _rows = "\n".join(
        f"| {_name} | {_item['correlation']:+.2f} | {_item['length'].size} |"
        for _name, _item in result["next_cycle"].items()
    )
    mo.md(
        "**Does the residual near the ice minimum delay the cycle?** "
        "Correlation of the residual averaged within ±20° of phase ½ with the "
        "length of the cycle it falls in.\n\n"
        "| band | correlation | cycles |\n|---|---:|---:|\n" + _rows
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Sea-ice switches on the ψ₁ clock

    Annual SIC ≥ 0.5 per T21 ocean cell. For each cell: the mean phase at
    which ice appears and retreats, the jitter (circular standard deviation,
    in years) of those phases across cycles, and the number of onsets per
    cycle. A regular switch, however sharp, has zero jitter; a cell that
    switches only in some cycles has fewer than one onset per cycle.
    """)
    return


@app.cell
def _(archive_root, irregularity, load_map_fields, result):
    map_fields = load_map_fields(archive_root, result["mu"], result["fields"]["years"])
    ice_flips = irregularity.ice_flip_phases(
        irregularity.load_absolute_map(archive_root, result["mu"], result["fields"]["years"]),
        result["psi1"],
    )
    return ice_flips, map_fields


@app.cell
def _(analysis_mu, ice_flips, irregularity, map_fields, result):
    irregularity.make_flip_figure(
        ice_flips, map_fields, period=result["period"], mu=analysis_mu,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Residual maps by phase

    Standard deviation of each map's residual (phase × |ψ₁| composite
    removed, per cell) in equal phase windows of ψ₁, and over all years.
    """)
    return


@app.cell
def _(mo):
    map_phase_count = mo.ui.slider(
        start=2, stop=8, step=1, value=4, show_value=True, label="Phase windows",
    )
    map_basin_only = mo.ui.checkbox(value=True, label="South Atlantic basin only")
    mo.hstack([map_phase_count, map_basin_only])
    return map_basin_only, map_phase_count


@app.cell
def _(
    analysis_mu,
    irregularity,
    map_basin_only,
    map_fields,
    map_phase_count,
    result,
):
    irregularity.make_map_residual_figure(
        map_fields, result["psi1"], mu=analysis_mu,
        phase_count=map_phase_count.value, basin_only=map_basin_only.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Dynamics of the residual

    **Slow Koopman modes.** The fit's least damped modes with
    $|\mathrm{Im}\,\lambda|<\omega_1/4$ (the constant mode excluded), and how
    much of the residual EOFs their eigenfunctions explain.

    **Phase-dependent linear model.** The joint area-weighted EOFs of the zonal
    $T_s$ and θ residuals are fitted with
    $r(t+1)=A(\theta_1(t))\,r(t)+c(\theta_1(t))$, with $A$ and $c$ in low Fourier
    harmonics of $\theta_1$. Composing $A$ over one mean cycle gives Floquet
    multipliers: how much a disturbance off the regular cycle survives one
    period. A multiplier approaching 1 is a slowing of the return to the
    cycle; the spectral radius of $A(\theta)$ shows where along the cycle
    disturbances decay slowest.

    **Residual Koopman.** KDMD (lag 1 yr) on the phase-augmented state
    $[\text{residual EOFs}, \cos\theta_1, \sin\theta_1]$, keeping the clock
    so that phase-dependent dynamics are allowed. Each eigenfunction is
    classified by how much of it is a function of the clock alone
    ($e^{ik\theta_1}$): clock modes (rates near $ik\omega$, their decay is the
    phase diffusion) and transverse modes (the return of the residual to the
    cycle). Copies $\lambda+ik\omega$ of a transverse mode are excluded by
    requiring that its eigenfunction be carried linearly by the residual EOFs.
    """)
    return


@app.cell(hide_code=True)
def _(mo, result):
    _rates = result["rates"]
    _per_mode = result["slow_projection"]["per_mode"]
    _rows = "\n".join(
        f"| {_index} | {_rates[_index].real:+.4f} | {_rates[_index].imag:+.4f} | "
        f"{-1 / _rates[_index].real:.0f} | {_fraction:.1%} |"
        for _index, _fraction in zip(result["slow_indices"], _per_mode)
    )
    _floquet = result["floquet"]
    _eofs = result["eofs"]
    mo.md(
        "| mode | Re λ (yr⁻¹) | Im λ (yr⁻¹) | decay time (yr) | residual EOF variance explained |\n"
        "|---:|---:|---:|---:|---:|\n" + _rows
        + f"\n\nTogether: {result['slow_projection']['explained']:.1%}. "
        f"Residual EOFs: {_eofs['fractions'].size} retain "
        f"{_eofs['fractions'].sum():.0%} of the residual variance.  \n"
        f"Floquet multipliers per cycle (|μ|): "
        + ", ".join(f"{_value:.3f}" for _value in _floquet["per_cycle"][:5])
        + f"; slowest return {-1 / _floquet['rate'][0]:.0f} yr "
        f"(the decay rates in this table depend on the lag; compare μ at the same lag)."
    )
    return


@app.cell
def _(analysis_mu, irregularity, result):
    irregularity.make_floquet_figure(
        {analysis_mu.replace("p", "."): result["floquet"]},
        title=f"μ={analysis_mu.replace('p', '.')}: return to the cycle",
    )
    return


@app.cell
def _(irregularity, result):
    irregularity.make_residual_koopman_figure({result["mu"]: result})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Ice residual and the subtropical ocean

    The SA ice-area residual is correlated with the zonal θ 0–700 m residual
    of every ocean row at lags −15…+15 yr (negative: ice first), so the
    latitude is not chosen in advance. A small error of the ψ₁ clock alone
    makes all residuals covary (residual ≈ δθ·dC/dθ + α·C). The clock error
    δθ and amplitude error α are therefore estimated each year from the
    zonal $T_s$ rows and the ocean rows outside 15–40°S, and removed from the
    ice and every ocean row ("clock removed"). The 22–27°S band correlation
    is compared with 200 phase-randomized ice surrogates (95% range).
    """)
    return


@app.cell
def _(irregularity, mo, result):
    coupling = irregularity.coupling_analysis(result)
    _raw, _clean = coupling["raw"], coupling["clock removed"]
    mo.vstack([
        mo.md(
            f"Clock and amplitude errors explain "
            f"{coupling['ice variance explained by clock error']:.0%} of the ice-residual variance. "
            f"22–27°S band: raw {_raw['band_peak'][0]:+.2f} at {_raw['band_peak'][1]:+d} yr, "
            f"clock removed {_clean['band_peak'][0]:+.2f} at {_clean['band_peak'][1]:+d} yr "
            f"(95% surrogate {_clean['band_threshold']:.2f})."
        ),
        mo.hstack([
            irregularity.make_latitude_lag_figure({result["mu"]: coupling}, "raw"),
            irregularity.make_latitude_lag_figure({result["mu"]: coupling}),
        ]),
        irregularity.make_coupling_timeseries_figure(
            {result["mu"]: coupling}, {result["mu"]: result["fields"]["years"]},
        ),
    ])
    return (coupling,)


@app.cell
def _(archive_root, coupling, irregularity, map_fields, result):
    ice_regressions = irregularity.ice_regression_entry(
        archive_root, result, coupling, map_fields,
    )
    irregularity.make_regression_map_figure(ice_regressions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coupled ice–ocean residual Koopman mode

    The clock-removed residuals of the basin ice area per T21 row and of the
    zonal θ 0–700 m rows are reduced to joint EOFs (both blocks scaled
    equally). KDMD (lag 1 yr) is applied to [EOFs, cos θ₁, sin θ₁]. For each
    transverse mode the table gives its decay time and the R² with which its
    eigenfunction explains the ice-area residual and the 25–30°S gyre
    residual. The "coupled" mode maximizes the smaller of the two.
    """)
    return


@app.cell
def _(coupling, irregularity, mo, result):
    coupled_koopman = irregularity.coupled_residual_koopman(result, coupling)
    _rows = "\n".join(
        f"| {_mode['decay_time']:.1f} | "
        + (f"{_mode['period']:.0f}" if _mode["period"] < 1e5 else "real")
        + f" | {_mode['ice_r2']:.2f} | {_mode['gyre_r2']:.2f} |"
        for _mode in coupled_koopman["modes"]
    )
    mo.vstack([
        mo.md("| decay time (yr) | period (yr) | R² ice | R² gyre |\n|---:|---:|---:|---:|\n" + _rows),
        irregularity.make_coupled_mode_figure({result["mu"]: coupled_koopman}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Heat budget of the subtropical gyre

    South Atlantic basin, 0–700 m, scalar rows 18.75–28.75°S, between the
    vector-row faces at 31.25°S and 16.25°S. Terms in TW:
    - storage: the heat-content tendency;
    - surface: T21 net surface flux, down positive, over South Atlantic ocean
      cells in the band; over ice this is the flux into the ice surface;
    - advection: convergence of the Atlantic $\bar v\bar\theta$ proxy through the faces,
      relative to the box temperature;
    - residual: vertical exchange at 700 m, eddies, sub-annual covariance,
      and the half-row offset of the faces.

    The phase × |ψ₁| composite is removed from every term, which is then
    regressed on the clock-removed ice residual (left) and on the gyre
    heat-content residual itself (right).
    """)
    return


@app.cell
def _(archive_root, coupling, irregularity, mo, result):
    gyre_budget = irregularity.gyre_heat_budget(archive_root, result["mu"], result["fields"]["years"])
    _content = irregularity.cycle_residuals(
        gyre_budget["heat_content"][:, None], result["psi1"], None, None,
    )["amplitude"][:, 0]
    _regressions = {
        "clock removed": irregularity.budget_regression(
            gyre_budget, coupling["clock removed"]["ice"], result["psi1"],
        ),
        "gyre heat content": irregularity.budget_regression(gyre_budget, _content, result["psi1"]),
    }
    _label = result["mu"]
    mo.hstack([
        irregularity.make_budget_figure({_label: _regressions}),
        irregularity.make_budget_figure(
            {_label: _regressions}, kind="gyre heat content",
            title="per std of the gyre heat-content residual",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Comparison across μ

    Runs the same analysis for every μ with an analysis window (about 30 s
    each) and compares the scalar metrics: full window (diamond) and the
    four 1000-year blocks with ψ₁ held fixed (dots).
    """)
    return


@app.cell
def _(mo):
    run_comparison = mo.ui.run_button(label="Compare all μ")
    run_comparison
    return (run_comparison,)


@app.cell
def _(
    analysis_mu,
    archive_root,
    irregularity,
    mo,
    mu_options,
    result,
    run_comparison,
    state_blocks,
):
    mo.stop(not run_comparison.value, mo.md("Click to run the comparison."))
    comparison = {
        _mu: result if _mu == analysis_mu
        else irregularity.irregularity_analysis(archive_root, _mu, state_blocks=state_blocks)
        for _mu in mu_options
    }
    return (comparison,)


@app.cell
def _(archive_root, comparison, irregularity, mo):
    _couplings = {_mu: irregularity.coupling_analysis(_item) for _mu, _item in comparison.items()}
    _budgets = {}
    for _mu, _item in comparison.items():
        _budget = irregularity.gyre_heat_budget(archive_root, _mu, _item["fields"]["years"])
        _content = irregularity.cycle_residuals(
            _budget["heat_content"][:, None], _item["psi1"], None, None,
        )["amplitude"][:, 0]
        _budgets[_mu] = {
            "clock removed": irregularity.budget_regression(
                _budget, _couplings[_mu]["clock removed"]["ice"], _item["psi1"],
            ),
            "gyre heat content": irregularity.budget_regression(_budget, _content, _item["psi1"]),
        }
    _keys = list(next(iter(comparison.values()))["summary"])
    _rows = "\n".join(
        f"| {_key} | " + " | ".join(
            f"{_item['summary'][_key]:.4g}" for _item in comparison.values()
        ) + " |"
        for _key in _keys
    )
    mo.vstack([
        irregularity.make_comparison_figure({
            _mu: {"full": _item["summary"], "blocks": _item["block_summaries"]}
            for _mu, _item in comparison.items()
        }),
        irregularity.make_floquet_figure(
            {_mu.replace("p", "."): _item["floquet"] for _mu, _item in comparison.items()},
            title="Return to the cycle by μ",
        ),
        irregularity.make_residual_koopman_figure(comparison),
        irregularity.make_coupling_trend_figure(_couplings),
        irregularity.make_coupled_mode_figure({
            _mu: irregularity.coupled_residual_koopman(_item, _couplings[_mu])
            for _mu, _item in comparison.items()
        }),
        irregularity.make_latitude_lag_figure(_couplings),
        irregularity.make_budget_figure(_budgets),
        irregularity.make_budget_figure(
            _budgets, kind="gyre heat content",
            title="per std of the gyre heat-content residual",
        ),
        irregularity.make_amplitude_history_figure(comparison),
        mo.md(
            "| metric | " + " | ".join(_mu.replace("p", ".") for _mu in comparison)
            + " |\n|---|" + "---:|" * len(comparison) + "\n" + _rows
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Robustness

    Each analysis choice is changed one at a time:
    - refitted: main KDMD lag 3 and 7, each 2000-yr half of the window, and 18 composite bins;
    - main fit kept: 5 and 15 residual EOFs, residual-KDMD lag 2 and 5, and 1 and 3 Floquet harmonics.

    The clock-removed ice–ocean coupling is also checked under changes of how the
    clock error is estimated (surface rows only; excluded ocean band 20–35°S or
    10–45°S), the residual type, and the band (25–30°S). The whole sweep takes about
    10 minutes per μ; on the external drive, copy the archives locally first.
    """)
    return


@app.cell
def _(mo):
    run_robustness = mo.ui.run_button(label="Run robustness sweep for all μ")
    run_robustness
    return (run_robustness,)


@app.cell
def _(
    analysis_mu,
    archive_root,
    irregularity,
    mo,
    mu_options,
    result,
    run_robustness,
):
    mo.stop(not run_robustness.value, mo.md("Click to run the sweep."))
    with mo.status.spinner(title="Robustness sweep") as _spinner:
        robustness = {
            _mu: irregularity.robustness_sweep(
                archive_root, _mu, base=result if _mu == analysis_mu else None,
                progress=lambda _message: _spinner.update(_message),
            )
            for _mu in mu_options
        }
        coupling_robustness = {
            _mu: irregularity.coupling_robustness(
                archive_root, _mu, base=result if _mu == analysis_mu else None,
                progress=lambda _message: _spinner.update(_message),
            )
            for _mu in mu_options
        }
    return coupling_robustness, robustness


@app.cell
def _(coupling_robustness, irregularity, mo, robustness):
    _metrics = list(next(iter(robustness.values()))["baseline"])
    _variants = list(next(iter(robustness.values())))
    _tables = []
    for _metric in _metrics:
        _rows = "\n".join(
            f"| {_variant} | " + " | ".join(
                f"{robustness[_mu][_variant][_metric]:.4g}" for _mu in robustness
            ) + " |"
            for _variant in _variants
        )
        _tables.append(
            f"**{_metric}**\n\n| variant | "
            + " | ".join(_mu.replace("p", ".") for _mu in robustness)
            + " |\n|---|" + "---:|" * len(robustness) + "\n" + _rows
        )
    mo.vstack([
        irregularity.make_robustness_figure(robustness),
        mo.md("**Clock-removed ice–ocean coupling under one-at-a-time changes**"),
        irregularity.make_robustness_figure(coupling_robustness),
        mo.accordion({"Tables": mo.md("\n\n".join(_tables))}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Summary: methods, robust findings, interpretation

    Written 2026-10-07 from runs of this notebook's functions on seven μ.
    Numbers are for the Ts + θ state unless stated.

    ### A. Data
    1. Raw-map archives, schema v2, extracted 2026-10-05/06. 1240 and 1232.5
       are read from `data/Plasim`; the others were copied from the external
       drive to local disk one at a time.
    2. Windows, 4000 annual records each:
       - 1245 and 1235: 9690–13689. These runs start at year 4500. Each has
         an unreadable or non-finite block at 13690–13699, so the window ends
         before it.
       - 1242.5: 20630–24629; 1240: 13380–17379; 1237.5: 20270–24269;
         1233.75: 20530–24529; 1232.5: 20430–24429. These are the final 4000
         years. Except 1240 (from 5000), these runs start at year 15000.
       - That the final 4000 years have no drift was taken from an earlier
         drift test. It was not re-verified here.
    3. Integrity was checked here only for the year labels and for
       `surface_temperature`, `sea_ice_concentration` and `temperature_upper`
       (readable, finite) in these seven archives. Full scans covered only 1225
       and 1228.5.

    ### B. Koopman state and fit (`load_fields`, `fit_koopman`)
    4. South Atlantic basin masks: 65°W–20°E, south of the equator
       (`t21_south_atlantic`, `lsg_scalar_south_atlantic`).
    5. Zonal Ts: T21 `surface_temperature`, a plain mean over the basin cells
       of each T21 row. The mask holds ocean cells only (167 cells, none with
       lsm ≥ 0.5), including ice-covered ones.
       Time mean removed, no detrending. Row weight: Gaussian weight × number
       of cells, normalized.
    6. Zonal θ 0–700 m: LSG `temperature_upper` at the 10 levels between 0
       and 700 m. Each native row is a wet-volume-weighted mean over basin
       cells per level, then a layer-thickness-weighted mean over the levels
       that are wet in that row. Time mean and a linear trend are removed.
       Row weight: basin wet surface area, normalized.
    7. Ice area: basin `sea_ice_concentration` × T21 cell area, summed
       (10⁶ km²), time mean removed. In the "ice" state: the same per T21
       row, dropping rows that never vary. Row weight: basin area of the row.
    8. KDMD:
       - lag 5 yr; all 3995 snapshot pairs (the 10 000 cap is not reached);
       - Gaussian kernel with σ = 1; each block's row weights are divided by
         the squared median pairwise distance of that weighted block;
       - TSVD factorization threshold 1e-5 (eigh); Koopman rank threshold
         5e-5;
       - continuous-time rates from the lag-5 eigenvalues.
    9. ψ₁ is the stable complex mode (Re λ < 0, Im λ > 0) with the largest
       Re λ. ψ₂ is the stable mode with Im λ within 10% of 2 Im λ₁ and closest
       to 2λ₁. Both are evaluated on all window years and scaled to unit RMS.
       ψ₁ is rotated so that the 36-bin phase composite of the ice anomaly
       peaks at phase 0. ψ₂ is rotated to align with ψ₁².

    ### C. Residuals ("extra stuff", `cycle_residuals`)
    10. For each series G: G minus its mean in bins of (arg ψ₁: 36 bins) ×
        (|ψ₁| tercile over the window: 3). A joint bin with fewer than 3 years
        falls back to the phase-only mean. Other variants: the phase-only
        composite, and G − 2Re(a₁ψ₁) − 2Re(a₂ψ₂) with direct KDMD modes. The
        composite and ψ₁ come from the same window (in-sample).

    ### D. Clock diagnostics (ψ₁ itself, no residuals)
    11. Phase diffusion D: half the slope of
        Var[θ(t+τ) − θ(t)] against τ = 5–100 yr (unwrapped arg ψ₁).
    12. Cycle lengths: years between successive upward passes of arg ψ₁
        through 2πk (first pass only).
    13. Weak-cycle episodes: 101-yr running mean of |ψ₁| below 0.75. The mean
        uses full windows only, and the ends are padded with the nearest value.

    ### E. Residual dynamics
    14. Joint area-weighted EOFs of the zonal Ts and θ residuals, each block
        scaled to unit total weighted variance; 10 EOFs.
    15. Floquet model:
        - fit r(t+1) = A(θ)r(t) + c(θ), with A and c in Fourier harmonics
          (up to 2) of θ = arg ψ₁, by least squares;
        - compose A over N = round(P) annual steps along θ₀ + ωn;
        - return time = −N / ln|μ_max|.
    16. Residual KDMD:
        - state [EOFs, cos θ, sin θ], each block scaled by its median distance;
        - lag 1 yr, same thresholds as the main fit;
        - clock share = R² of each eigenfunction on e^{ikθ}, |k| ≤ 8;
        - "transverse" mode: clock share < 0.5 and R² on the EOFs ≥ 0.2.

    ### F. Ice–ocean relation (`coupling_analysis`)
    17. corr(θ residual of each ocean row at t, ice-area residual at t+L) for
        L = −15…+15. L < 0 means the ice comes first.
    18. Clock-error removal:
        - for each series, slope = periodic central difference of its 36-bin
          phase-only composite, and shape = the composite value;
        - for each year, a weighted least-squares fit of the stacked residual
          rows (all Ts rows, plus ocean rows outside 15–40°S) on
          [slope, shape] gives δ and α. Each block is scaled by its residual
          std and by √(row weight);
        - δ·slope + α·shape is removed from the ice residual and from every
          ocean row.
        - The residuals use the phase × amplitude composite, while slope and
          shape use the phase-only composite. This is a small inconsistency.
    19. Band correlation: the weighted mean of the 22–27°S rows; its peak over
        lags is compared with 200 Fourier-phase-randomized ice surrogates
        (95% of max |corr| over lags).
    20. Ice-leading maximum: the most negative correlation over all rows and
        L = −8…−2. It has **no** surrogate test, and it is a maximum over many
        rows and lags.
    21. Robustness variants:
        - clock estimated from Ts only;
        - excluded ocean band 20–35°S or 10–45°S;
        - phase-only composite;
        - band 25–30°S;
        - main lag 3 or 7;
        - each 2000-yr half;
        - 18 bins.

    ### G. Heat budget (`gyre_heat_budget`), indicative only
    22. Box: South Atlantic basin, 0–700 m, LSG scalar rows 18.75–28.75°S.
        - Heat content uses ρc_p = 1025 × 3990 J m⁻³ K⁻¹.
        - Storage is its centred annual difference.
    23. Surface flux: T21 rss + rls + hfss + hfls (all positive downward),
        over basin ocean cells on T21 rows 19.4°S, 24.9°S and 30.5°S.
        - These rows do not match the LSG box exactly.
        - Over ice this is the flux into the ice surface.
    24. Advection: the Atlantic `vbar*thetabar` proxy (65°W–20°E on vector
        longitudes) through the vector rows at 31.25°S and 16.25°S, 0–700 m,
        relative to the box-mean temperature.
        - It has no sub-annual covariance.
        - The faces are half a row off the box edges.
    25. Residual term = storage − surface − advection. Its std is about 21 TW,
        larger than storage (about 16 TW).
    26. Each term has its cycle composite removed and is regressed on the
        standardized clock-removed ice residual (lags −6…+12), or on the
        gyre heat-content residual.

    ### H. Other
    27. Regression maps: per-cell residuals of the map fields (ocean layers
        detrended), regressed on the standardized ice residual.
    28. Ice switches: annual SIC ≥ 0.5; the phase of a switch is the phase of
        ψ[t−1] + ψ[t]; circular standard deviation; cells with fewer than 3
        onsets are left out.
    29. Joint residual Koopman: the clock-removed residuals of ice per row and
        of θ rows, joint EOFs (10 or 15), residual KDMD (lag 1 or 3). The
        "coupled" mode maximizes min(R² ice, R² of the 25–30°S gyre).

    ### Robust findings
    - **Period** rises with decreasing μ: 43.7, 46.8, 50.4, 55.4, 61.2, 65.8
      and 70.0 yr (1245 → 1232.5). It is unchanged by the main lag, by window
      halves, and by including ice in the state.
    - **Share of ice-area variance outside the cycle**: about 0.2
      (1245–1237.5), then 0.42–0.47 (1235–1232.5).
    - **Ice–gyre covariability appears only at 1233.75 and 1232.5.**
      - At those two μ, the clock-removed ice residual on the 25–30°S rows
        leads a θ 0–700 m residual at 26–29°S by 2–4 yr.
      - The correlation is −0.27 to −0.55 in every variant except the Ts-only
        clock, and also with ice in the Koopman state.
      - Its sign: less ice, then a warmer gyre.
      - At 1245–1235 the band correlation is weak (|r| ≤ 0.2, near the 95%
        surrogate level of 0.08–0.14). The strongest ice-leading relation
        there sits elsewhere (around 11°S, 2-yr lag) and has not been
        interpreted.
    - **Where the ice is:** at 1232.5 the regressed ice anomaly sits on the
      T21 rows at 24.9°S and 30.5°S, and the gyre anomaly fills 15–30°S,
      40°W–0°, 0–300 m. At 1240 the ice anomaly is on the 36°S row and the
      ocean pattern is a weak dipole.

    ### Findings that are not robust or were wrong
    - Weak-cycle episodes occur at 1245 (18% of years) as much as at
      1233.75–1232.5 (20–24%). They are not a precursor.
    - Larger D and cycle-length spread appear only at 1233.75 and 1232.5,
      and only because of the episodes. On calm stretches D is 0.001–0.002 at
      every μ.
    - Return times show no monotonic trend at either lag: the Floquet model,
      the residual KDMD, and the joint ice–ocean residual Koopman mode
      (about 10 yr at lag 1 everywhere).
    - So there is no evidence of critical slowing in these measures.
    - The raw ice–ocean correlations were inflated by clock and amplitude
      errors, which explain 18–52% of the ice-residual variance (1235: −0.31
      raw, −0.14 corrected).
    - With a clock estimated from Ts only, the 1232.5 relation disappears,
      because Ts near the ice is set by the ice. The anomaly therefore has the
      form of a local shift of the cycle. Surface data alone cannot separate
      it from a clock error. The ice-including Koopman state removes much of
      this doubt but not all of it.
    - The joint residual Koopman mode explains at most a quarter of the ice
      residual and changes with the number of EOFs. No isolated coupled mode
      was found.
    - Two earlier statements from this work were wrong:
      - the ocean does not lead the ice (the ice leads);
      - intermittency is not specific to tipping.

    ### Physical interpretation (limited to what the analysis supports)
    - **What is established.** As μ decreases toward the transition, the
      oscillation slows down. Near tipping, the part of the sea-ice
      variability not explained by the cycle doubles. At the two μ closest to
      tipping, that irregular ice variability on the 25–30°S rows moves
      together with a persistent upper-ocean heat anomaly in the subtropical
      gyre interior, with the ice changing first.
    - **Plausible but not established.** At those μ the ice edge lies over the
      gyre (shown by the regression maps), and the ice and the upper ocean can
      exchange heat locally.
      - The budget is consistent with this: more ice goes with surface heat
        loss over 19–30°S at every μ (5–8 TW per std).
      - Only from 1235 down does storage respond (−3 to −5.5 TW, a
        heat-content drop of 0.5–0.9 × 10²¹ J), followed by advective
        convergence (+2 to +4 TW over 4–10 yr).
      - The budget does not close: the residual is larger than the storage.
        The surface term is partly a flux into ice, and the transport is a
        proxy.
      - We do not know why the same flux anomaly does not change gyre storage
        at higher μ.
    - **Not supported.** A slowing (critical slowing) of a cycle or of a
      coupled ice–ocean mode; episodes as precursors.

    ### Open issues
    - The ice-leading maximum has no surrogate test, and the row/lag scan is
      not corrected for multiple testing.
    - Clock error vs genuine local anomaly is still ambiguous. A test with a
      clock estimated far from the ice (deep ocean or Indo-Pacific) is needed.
    - The LSG ocean-side heat flux would give a cleaner budget than the T21
      surface flux.
    - Two run histories (start at 4500 vs 15000) are mixed, and windows are
      in-sample.
    - Integrity was only partly checked. Re-extraction is planned.
    """)
    return


if __name__ == "__main__":
    app.run()
