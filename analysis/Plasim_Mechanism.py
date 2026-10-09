"""Global equilibria, transitions, and the South Atlantic ice cycle in PlaSim–LSG."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Towards the snowball in PlaSim–LSG: global picture and the South Atlantic cycle

    **Goal.** Find what fingerprints the approach to the snowball transition
    as the solar constant μ (`GSOL0`) is lowered, using the PlaSim–LSG runs
    alone (no Koopman results). We start from the global picture (equilibria,
    energy budget, the transitions themselves) and then turn to the South
    Atlantic ice–ocean cycle, the largest variability near the ice edge.

    Sections:

    1. The model: grids and atmosphere–ice–ocean coupling.
    2. What can we trust: model limitations and two checks.
    3. Global equilibria and energy budget across all runs.
    4. Towards a reduced model: global and hemispheric tests, and the ocean's
       role in the Southern extratropics.
    5. Anatomy of the warm → cold and cold → snowball transitions.
    6. The South Atlantic ice cycle and its phase.

    Only statements that follow directly from the data are given as
    findings; interpretations are marked as hypotheses.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from gsebm import plasim_checks as checks
    from gsebm import plasim_global as global_picture
    from gsebm import plasim_mechanism as mechanism
    from gsebm import plasim_mechanism_plots as plots
    from gsebm import plasim_southern as southern
    from gsebm import plasim_transitions as transitions
    from gsebm.plasim_raw_maps import RAW_MAP_ROOTS

    return (
        RAW_MAP_ROOTS, checks, global_picture, mechanism, mo, plots, southern, transitions,
    )


@app.cell
def _(RAW_MAP_ROOTS, mo):
    selected_root = mo.ui.dropdown(options=RAW_MAP_ROOTS, value="repo", label="Archive root")
    selected_root
    return (selected_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The model

    **Atmosphere and sea ice.** PlaSim is a simplified atmosphere on a T21
    spectral grid: 32 Gaussian latitude rows about 5.5° apart and 64
    longitudes 5.6° apart. Under it sits a 50 m *mixed layer*, a slab of
    water whose temperature is the sea-surface temperature (SST), and a
    thermodynamic sea-ice model (ice grows and melts in place; it does not
    drift). This is the only sea-ice model, so the ice edge can only move in
    steps of one T21 row.

    **Ocean.** Below the mixed layer is LSG, a coarse 3-D ocean (72 × 76
    cells, rows 2.5° apart, 22 levels) that carries heat with currents and
    mixing. It has no ice of its own; it receives PlaSim's ice thickness only
    to set the water-column thickness.

    **How the ocean talks to the ice.** At each coupling step, LSG's top layer
    is reset to PlaSim's mixed-layer temperature (the freezing point under
    ice). The heat this moves, the *Newtonian coupling heat flux*, goes back
    to PlaSim as "flux from the deep ocean". In open water it warms or cools
    the mixed layer. Under ice it goes straight into melting or growing the
    ice from below. So warm water that LSG carries under the ice edge can
    stop ice from forming even when the air above is cold.

    **The subtropical gyre.** Winds drive a basin-wide anticlockwise loop in
    the South Atlantic: warm water flows south along South America, east near
    35–45°S, and back north along Africa. Its upper few hundred metres hold a
    large heat reservoir next to the ice edge.

    **Runs.** All runs have the AMOC collapsed. They come from three starting
    points: abrupt drops from the present-day state (1367), abrupt drops from
    the μ = 1265 state at year 4499, and continuations of the 1240 or 1230
    runs (see `plasim_global.RUNS`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What can we trust

    **Known model limitations.**

    * Sea ice lives on the T21 grid, so the edge advances in steps of about
      5.5° per row and sector. Tantet (2016) discarded the last few W m⁻²
      before the crisis in a swamp-ocean PlaSim for this reason.
    * Sea ice has no dynamics (no drift or export).
    * LSG is a coarse ocean with convective adjustment; such models produce
      millennial convection cycles whose size and period are model
      properties.
    * The ocean–ice coupling is crude: LSG's top layer is reset to the
      mixed-layer temperature and the excess heat melts ice from below.
    * The AMOC is collapsed in all runs (μ ≤ 1312).
    * Only annual means: no seasonal cycle.

    **Trust levels, for the warm → snowball question.**

    * *Trusted as a description of this model:* global integrals (Ts,
      planetary albedo, absorbed and emitted radiation, hemispheric ice
      area), the energy-budget split between runs (differences, so a constant
      energy error cancels), the warm branch and its reproducibility from
      three starting points, and the ocean releasing its heat at the ice edge.
    * *Suspect until tested:* the cold state and the staircase towards
      snowball, the jump in sensitivity in the last two warm steps,
      timescale estimates near the end of the branch, the millennial events
      (likely LSG convection cycles), and transition timing (one realization
      per μ, abrupt-change protocol).
    * *Most relevant:* hemispheric ice area (smoother than edge latitude),
      planetary albedo, global Ts, and the budget terms (albedo gain, λ,
      heat transport reaching the edge). Internal oscillations matter as
      possible triggers and as noise in any early-warning indicator.

    **Two checks, on annual values (no smoothing).**

    * *Check 1, grid signature.* (a) Window-mean zonal ocean ice
      concentration of each T21 row, and which rows are partly covered
      (0.05–0.95). (b) The change in hemispheric ice area across each jump,
      split into T21 row × sector (Atlantic 65°W–20°E, Indian 20–115°E,
      Pacific the rest); change = mean over 100 years after minus 100 years
      before (years given in the table). (c) Histograms of the annual
      ice-edge latitude in each window, with the T21 row latitudes.
    * *Check 2, energy closure.* Global integrals per year: TOA net
      `rst + rlut`; heat into the ocean `zonal_newtonian_coupling_heat_flux`
      × row wet area; ocean storage = year-to-year difference of the summed
      `zonal_ocean_heat_content` (LSG ρ = 1030 kg m⁻³, c_p = 4180 J kg⁻¹ K⁻¹)
      per 360-day year; sea-ice latent heat = year-to-year difference of
      `global_lsg_ice_volume` × 1030 kg m⁻³ × 3.344×10⁵ J kg⁻¹. All in W per
      m² of globe. Window means, and event-minus-quiet means for the 1312
      and 1250 events.
    """)
    return


@app.cell
def _(checks, global_picture, selected_root):
    # One archive at a time; sea-ice maps are reduced to rows × sectors.
    ice_rows = {label: checks.read_ice_rows(selected_root.value, label)
                for label in global_picture.RUNS}
    energy = {label: checks.read_energy_series(selected_root.value, label)
              for label in global_picture.RUNS}
    return energy, ice_rows


@app.cell
def _(checks, ice_rows, mo):
    mo.vstack([
        mo.md("**Check 1a.** Window-mean zonal ocean ice concentration per T21 row."),
        mo.ui.table(checks.coverage_table(ice_rows), selection=None),
    ])
    return


@app.cell
def _(checks, ice_rows, mo):
    _jumps = (
        ("1230 main step", "1230", (7750, 7849), (8150, 8249)),
        ("1230 second step", "1230", (9750, 9849), (10000, 10099)),
        ("1225 trigger", "1225", (7470, 7569), (7600, 7699)),
        ("1225 runaway", "1225", (7600, 7699), (7850, 7949)),
    )
    _table = []
    for _name, _label, _before, _after in _jumps:
        for _south in (True, False):
            _parts = checks.jump_contributions(ice_rows[_label], _before, _after, _south)
            _total = sum(part[2] for part in _parts)
            _table.append({
                "jump": _name, "hemisphere": "S" if _south else "N",
                "before": f"{_before[0]}–{_before[1]}", "after": f"{_after[0]}–{_after[1]}",
                "total change (10¹² m²)": round(_total, 2),
                "largest share": f"{_parts[0][2] / _total:.0%}",
                "five largest (sector, row, change)": "; ".join(
                    f"{sector[:3]} {lat:+.1f}: {change:+.2f}" for sector, lat, change in _parts[:5]
                ),
            })
    mo.vstack([
        mo.md("**Check 1b.** Change in hemispheric ice area across each jump, by row × sector."),
        mo.ui.table(_table, selection=None),
    ])
    return


@app.cell
def _(ice_rows, plots):
    plots.edge_histograms(ice_rows)
    return


@app.cell
def _(checks, energy, mo):
    _events = (("1312", (5250, 6249), (6500, 8999)), ("1312", (9250, 10249), (6500, 8999)),
               ("1250", (6100, 6399), (5000, 5999)), ("1250", (8500, 8799), (7000, 8399)))
    _differences = []
    for _label, _event, _quiet in _events:
        _d = checks.period_difference(energy[_label], _event, _quiet)
        _differences.append({"run": _label, "event": f"{_event[0]}–{_event[1]}",
                             "quiet": f"{_quiet[0]}–{_quiet[1]}",
                             **{f"Δ {key}": round(value, 3) for key, value in _d.items()}})
    mo.vstack([
        mo.md("**Check 2.** Window means of the global energy terms (W m⁻² of globe)."),
        mo.ui.table(checks.energy_table(energy), selection=None),
        mo.md("Event minus quiet periods (W m⁻² of globe)."),
        mo.ui.table(_differences, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the checks show

    * **The warm branch is not a row-by-row staircase in the zonal mean.**
      Along 1312 → 1232.5 two to three rows per hemisphere are partly
      covered, and each fills gradually: the 36°S row goes 0.01 (1265) →
      0.21 → 0.30 → 0.39 → 0.46 → 0.52 → 0.59 → 0.68 → 0.76 (1232.5).
    * **The warm → cold jump involves several rows in every sector.** The
      1230 main step adds 13.6 × 10¹² m² of Southern ice; no single row ×
      sector carries more than 30% (Pacific 19.4°S), with Pacific, Atlantic
      and Indian rows at 19–31°S all contributing. The Northern Hemisphere
      adds only 1.5 × 10¹² m². The second 1230 step (near 9900) is carried
      mostly by the North (2.9 vs 1.6 × 10¹² m²). So my earlier description
      "each step is one row filling in one sector" was wrong for this jump.
    * **The 1225 snowball trigger is the most row-like event:** the Atlantic
      19.4°S row carries 48% of the Southern change, with the 24.9°S rows in
      the Pacific and Indian sectors adding most of the rest. The runaway
      itself is spread over all rows and both hemispheres.
    * **Annual edges are not pinned to row latitudes.** Within each window
      the edge distribution is a single narrow peak (about ±1°) that moves
      with μ; it sits on a row latitude in some runs and between rows in
      others. 1312, 1288 and 1265 have skewed or two-humped Southern
      distributions from their events and oscillations.
    * **The model loses energy that is stored nowhere.** In every window
      the ocean storage and sea-ice latent heat are near zero, but the TOA
      net is −0.13 (1312) to −0.34 W m⁻² (1228.5) and −1.25 W m⁻² in the
      snowball (1225, still cooling: storage −0.07). Since the heat entering
      the ocean is small (−0.09 to +0.07 W m⁻²), most of this leak is on the
      PlaSim side (atmosphere, land, mixed layer). LSG itself is closer to
      closed: heat in minus storage is within ±0.07 W m⁻². The leak grows as
      the climate cools but changes by only ~0.2 W m⁻² along the warm
      branch, small next to the ~30 W m⁻² change in absorbed sunlight used
      in section 3.
    * **The millennial events are real energy exchanges within the model.**
      During the 1312 and 1250 events the TOA net falls by 0.23–0.27 W m⁻²,
      and the ocean loses heat at the same rate (storage changes by −0.24 to
      −0.28 W m⁻², heat into the ocean by −0.22 to −0.30 W m⁻²). Each 1312 event releases
      about 3 × 10²⁴ J from the ocean. The budget anomalies close; whether
      the events are realistic is a separate question (LSG convection).

    **Still open:** the cold state is a multi-row, multi-sector state, not a
    single-cell artefact, but grid effects at T21 resolution cannot be ruled
    out without a resolution change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Global equilibria and energy budget

    All 15 extracted runs, reduced to window means over their stationary
    periods (`plasim_global.RUNS`). Diamonds (1312, 1288, 1265) are abrupt
    drops from the equilibrated present-day run (1367) at year 1999. Black dots are
    abrupt drops from the μ = 1265 state at year 4499. Squares continue the 1240 run at year
    14999, and the triangle continues the 1230 run at year 14499.

    The ice edge is the latitude where zonal-mean ocean sea-ice concentration
    reaches 0.5. It is plotted as sin(latitude), which is the EBM's coordinate
    x. The global budget step splits the change in absorbed sunlight between
    two equilibria exactly, as ΔASR = ΔI (1 − ᾱ) − Ī Δα: a *direct* part
    (a dimmer Sun) and an *albedo* part (more ice and cloud). In equilibrium
    the outgoing longwave changes by the same amount, so
    ΔT = ΔOLR / λ, and `gain` = ΔASR / direct measures how much the albedo
    feedback amplifies the response.
    """)
    return


@app.cell
def _(global_picture, selected_root, southern, transitions):
    # One archive at a time: every valid year is read once; the equilibrium
    # means use only the analysis window of each run.
    run_series = {}
    states = {}
    modes = {}
    edges = {}
    boxes = {}
    for _label, _info in global_picture.RUNS.items():
        _full = global_picture.read_global_series(selected_root.value, _label, full=True)
        run_series[_label] = transitions.run_timeseries(selected_root.value, _label, _full)
        _window = global_picture.slice_series(_full, _info.window)
        states[_label] = global_picture.equilibrium_state(_window)
        modes[_label] = {
            hemisphere: {
                key: values[(_full.years >= _info.window[0]) & (_full.years <= _info.window[1])]
                for key, values in global_picture.hemispheric_modes(
                    _full.surface_temperature, _full.lat, _full.weight, hemisphere == "S",
                ).items()
            }
            for hemisphere in ("S", "N")
        }
        if _label != "1225":
            edges[_label], boxes[_label] = southern.edge_boxes(_full, _info.window)
    return boxes, edges, modes, run_series, states

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time series and analysis windows

    Every run at full length, **annual values, no smoothing**. In the
    overview the analysis window of each run is in full colour and the rest
    is faded; in the single-run view the window is shaded. The windows skip
    the adjustment after each abrupt change of μ.

    How each quantity is computed from the archive (one value per year):

    * **global Ts**: `surface_temperature` (PlaSim surface temperature over
      ocean, ice and land), averaged over the globe with the T21 Gaussian
      latitude weights.
    * **planetary albedo**: 1 − (global-mean net top-of-atmosphere
      shortwave `rst`) / (global-mean incoming shortwave `rst − rsut`).
    * **TOA imbalance**: global mean of `rst + rlut` (net downward
      radiation at the top of the atmosphere).
    * **S and N ice edge**: for each T21 row, the mean of
      `sea_ice_concentration` over ocean cells (`lsm < 0.5`). Going from the
      equator poleward, the edge is the latitude where this first reaches
      0.5, interpolated linearly between rows; 90° if no row reaches 0.5.
    * **deep θ**: LSG `zonal_potential_temperature` averaged over all
      levels below 1000 m, weighted by the wet volume of each row and level.

    The equilibrium values in the rest of this section are plain time means
    of the annual values over each window.
    """)
    return


@app.cell
def _(plots, run_series):
    fig, axes = plots.run_timeseries_overview(run_series)

    axes[0].set_ylim(bottom=250)
    axes[1].set_ylim(bottom=20)
    axes[2].set_ylim(bottom=25)
    fig
    return


@app.cell
def _(global_picture, mo):
    selected_run = mo.ui.dropdown(
        options={label.replace("p", "."): label for label in global_picture.RUNS},
        value="1232.5", label="Run",
    )
    selected_run
    return (selected_run,)


@app.cell
def _(plots, run_series, selected_run):
    plots.run_timeseries_detail(run_series[selected_run.value])
    return


@app.cell
def _(global_picture, mo, states):
    _order = sorted((label for label in states if label != "1235_new_IC"),
                    key=lambda label: -states[label].mu)
    steps = [global_picture.budget_step(states[a], states[b])
             for a, b in zip(_order[:-1], _order[1:])]
    mo.ui.table(
        [{key: round(value, 3) for key, value in step.items()} for step in steps],
        selection=None,
    )
    return (steps,)


@app.cell
def _(plots, states):
    plots.bifurcation_diagram(states)
    return


@app.cell
def _(plots, steps):
    plots.sensitivity_steps(steps)
    return


@app.cell
def _(plots, states):
    plots.zonal_change_profiles(
        states, (("1245", "1235"), ("1235", "1232p5"), ("1232p5", "1230")),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What section 3 shows (window means only)

    * **Warm branch from 1312 to 1232.5.** It is smooth: global Ts goes
      279.3 → 260.0 K, the Southern ice edge 68° → 33.6°S, the Northern edge
      52° → 36.5°N, and planetary albedo 0.340 → 0.392. Runs started from
      1367, from 1265 and from 1240 all fall on the same curve; the two 1235
      runs differ by 0.08 K.
    * **Below it, two separate states.** Cold (1230, 1228.5: 253 K, edges
      25°S and 30°N) and snowball (1225: 211 K). Whether the cold states are
      physical or held in place by the T21 rows is open (Tantet 2016
      discarded such states in a swamp-ocean PlaSim for that reason).
    * **Albedo amplification grows along the warm branch.** Warming per
      W m⁻² of μ: 0.21 K (1312→1288), 0.23, 0.26, 0.23, 0.26–0.28
      (1245→1235), then 0.46 and 0.40 in the last two steps. The albedo gain
      goes 2.1 → 2.5 → 3.7–4.1. The longwave response λ is not constant: it
      falls from 1.59 to about 1.37 W m⁻² K⁻¹ over the branch. The last two
      steps are 1.25 W m⁻² wide and coincide with the South Atlantic edge
      entering the 30.5°S T21 row (section 6).
    * **The Southern edge advances more slowly as it nears 35–40°S.** Edge
      shift per W m⁻² of μ: 0.68° (1312→1288), 0.39°, 0.31°, then
      0.14–0.27° from 1250 to 1235. Over the whole branch the edge crosses
      about six T21 rows.
    * **Ocean heat transport saturates at 1.5 PW.** Its Southern peak grows
      0.96 (1312) → 1.27 (1288) → 1.49 PW (1265) and then stays at
      1.50 PW at 17.5°S to the end of the warm branch (10°S on the cold
      branch). The Northern peak is 1.16 PW at 1312 and 0.85 PW from 1265
      down. The ocean heat crossing 35°S peaks at 1265–1288 (1.1 PW), then
      falls to 0.40 PW at 1232.5 and 0.08 PW on the cold branch. The ocean
      releases its heat at the ice edge, and the edge moves equatorward of
      35°S. The atmosphere carries most of the transport (4.3–5.1 PW
      across 35°S).
    * **Windows (see the annual time series above).** Most windows look
      stationary. The continuations from 1240 drift in deep θ for about
      5000 years after year 15000; their windows start after it levels off.
      The snowball run 1225 is still cooling in its window (deep θ
      271.4 → 271.1 K). In 1230 the deep ocean is flat in the window
      (about 271.8 K); before the window it has a second, smaller step near
      year 9900 after the main one near 7900. 1250 has episodic Southern ice
      events (around 6200 and 8600), and 1288 regular spikes in the Southern
      edge.
    * **1312 has a millennial oscillation.** Roughly every 4 kyr the South
      Pacific ice retreats strongly (edge 65° → 84°S) and the deep ocean
      swings by about 0.6 K. The window (4000–11999) averages over two such
      events.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Towards a reduced model: global and hemispheric tests, and the ocean's role

    The question: does the large-scale state behave like an energy balance
    model (EBM), and where does the ocean enter? Each run gives one
    equilibrium point; together they trace curves across μ. The informative
    part is the **direction of each run's year-to-year fluctuations** against
    those curves. If they coincide, the system's own variability moves it the
    same way the forced change towards snowball does.

    All values are annual (no smoothing), within each run's window; 1225
    (snowball) is left out. Grey: every year of every run; colour: 1312 and
    1250 (the runs with millennial ocean events); black: window means.

    ### 4a. Global albedo and hemispheric ice against global Ts

    Southern and Northern ice area (`sea_ice_concentration` × T21 cell area
    over ocean), and planetary albedo, against global Ts.
    """)
    return


@app.cell
def _(ice_rows, plots, run_series):
    plots.quasi_static_test(run_series, ice_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4b. Hemispheric mean and gradient

    Each hemisphere separately, from zonal-mean `surface_temperature` with the
    T21 Gauss–Legendre weights w_j (sum 1 per hemisphere) and x = sin(lat):

    * T_h = Σ w_j T_j (hemispheric mean);
    * ΔT_h = Gaussian-weighted mean over |lat| < 30° minus over |lat| > 30°
      (the two equal-area halves; on T21 no row boundary falls at 30°, so for
      a pure P₂ profile ΔT = −0.734 T₂ instead of −0.75);
    * T₂,h = 5 Σ w_j T_j P₂(|x_j|), with P₂(x) = (3x² − 1)/2, the coefficient
      of the two-mode EBM profile T₀ + T₂ P₂(x).
    """)
    return


@app.cell
def _(modes, plots):
    plots.hemispheric_mode_test(modes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4c. The ocean's role in the Southern extratropics

    Boxes that follow each run's window-mean Southern ice edge (zonal mean,
    as in section 3):

    * **cap**: from 10° equatorward of the edge to the pole. Its window-mean
      loss to space equals ocean release inside it + atmospheric heat carried
      across its one northern boundary + the model's energy leak, so
      *ocean share* = ocean release / loss to space is meaningful;
    * **band**: edge ± 10°. It has two boundaries, so only its ocean release
      is used (as a fraction of the cap's).

    Per year: ocean release = −Σ `zonal_newtonian_coupling_heat_flux` × wet
    row area over the LSG rows in the box (positive = heat leaving the ocean);
    TOA net = Σ (`rst` + `rlut`) × 2πR² w_j over the T21 rows in the box. Rows
    are assigned by centre latitude, so box boundaries move in whole rows;
    this can shift cap totals by about 10% (e.g. 1233.75 and 1232.5 gain the
    24.9°S row). The lead–lag figure correlates the Southern ice-area anomaly
    with each term at lags −60…+60 yr (anomaly = deviation from the window
    mean); no significance test.
    """)
    return


@app.cell
def _(boxes, edges, global_picture, mo, southern):
    mo.ui.table(
        southern.cap_table(boxes, edges, {label: global_picture.RUNS[label].window for label in boxes}),
        selection=None,
    )
    return


@app.cell
def _(boxes, ice_rows, plots):
    _southern_ice = {
        label: sum(ice_rows[label].sector_area.values())[:, ice_rows[label].lat < 0].sum(axis=1)
        for label in boxes
    }
    plots.southern_lag_figure(boxes, _southern_ice)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What section 4 shows

    **Findings.**

    * **Global albedo follows one curve against global Ts** for the
      equilibria, every run's year-to-year variability, and the 1312 and 1250
      ocean events. At the global level the EBM's albedo law holds, whatever
      moves the temperature.
    * **Southern Hemisphere: one curve.** ΔT_S and T₂,S against T_S lie on
      the equilibrium curve for all annual values, events included. Southern
      fluctuations move the climate in the direction of the forced change
      towards snowball.
    * **Northern Hemisphere: equilibria on a curve, fluctuations off it.**
      Within each run ΔT_N changes much more per K of T_N than along the
      equilibrium curve (at 1265 about 1.4 against 0.4 K/K, read by eye):
      Northern variability is concentrated at high latitudes and does not
      probe the direction of the transition. In the 1312 events the North
      does not move.
    * **The Southern profile is not pure P₂.** ΔT_S/T₂,S goes −0.60 (1312) →
      −0.69 (1232.5) → −0.77 (cold), against −0.734 for a pure P₂ profile on
      this grid; the North stays at −0.68 to −0.72.
    * **The ocean is a steady heat source at the ice edge.** In the cap it
      releases 1.2–1.4 PW from 1265 down, 0.24–0.31 of the cap's loss to
      space, on both the warm and the cold branch; 75–100% of that release
      is within ±10° of the edge.
    * **Ice and radiation vary together in the same year** (cap r = −0.35 to
      −0.90 at lag 0): the albedo effect.

    **Not established.**

    * That the ocean drives the year-to-year Southern ice variability near
      the end of the warm branch: in edge-following boxes the ice–release
      correlation is |r| ≤ 0.13 at all lags for 1245–1232.5 and the cold
      branch. At 1265 it is strong (about −0.57) but the sign of the lag flips
      between band and cap. Only the event runs (1312, 1250) show a strong
      ocean–ice link, at all lags.
    * An earlier fixed 30–90°S box gave an ocean share falling to 0.18 and
      0.03 and an ocean lead of a few years; both depended on the box and
      are withdrawn.

    **Reading for a reduced model (hypothesis).** Two hemispheres, the South
    possibly one-dimensional along the curve above, the North with an extra
    fluctuating variable, plus a steady ocean heat source at the Southern edge
    and a slow ocean reservoir for the millennial events.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Anatomy of the transitions

    Two runs change state after the abrupt drop from μ = 1265 at year 4499:

    * μ = 1230 goes from warm to cold around year 7900;
    * μ = 1225 goes from cold to snowball around year 7800.

    We follow ice edges per ocean sector and hemisphere, land snow, the
    tropical upper ocean (0–700 m, 20°S–20°N), the deep ocean (> 1000 m), and
    the zonal ocean heat transport.

    The figures show annual values. The *onset* of a series is computed as
    follows: smooth the annual values with an 11-year running mean, fit a
    linear trend to the reference period, and take the first later year from
    which the smoothed series stays more than 4 reference standard
    deviations away from that trend for 30 consecutive years. Smooth series, such as ocean
    temperatures, cross this threshold earlier than noisy ones, such as ice
    edges, so only gaps of more than a few decades are meaningful. The
    northern "Indian" sector is mostly Asia and is left out.
    """)
    return


@app.cell
def _(selected_root, transitions):
    # One archive at a time; only zonal reductions are kept.
    transition_series = {
        "1230": transitions.read_transition_series(selected_root.value, "1230"),
        "1225": transitions.read_transition_series(selected_root.value, "1225"),
        "1232p5": transitions.read_transition_series(
            selected_root.value, "1232p5", (20430, 24300),
        ),
    }
    transition_results = {
        label: transitions.transition_indices(series)
        for label, series in transition_series.items()
    }
    return transition_results, transition_series


@app.cell
def _(plots, transition_results):
    plots.transition_overview(transition_results["1230"], (4600, 11000),
                              "μ = 1230 (abrupt drop from 1265)")
    return


@app.cell
def _(plots, transition_results):
    plots.transition_overview(transition_results["1225"], (4600, 7830),
                              "μ = 1225 (abrupt drop from 1265)")
    return


@app.cell
def _(mo, plots, transition_results, transitions):
    _cases = (
        ("1230", (7500, 8450), (7300, 7800), "1230 warm → cold"),
        ("1225", (7300, 7830), (7000, 7500), "1225 cold → snowball"),
    )
    _figures = []
    for _label, _years, _reference, _title in _cases:
        _result = transition_results[_label]
        _names = [name for name in _result.indices
                  if "N land" not in name and "deep" not in name and "35°N" not in name]
        _onsets = {
            name: transitions.onset_year(_result.years, _result.indices[name], _reference)
            for name in _names
        }
        _figures.append(plots.onset_comparison(_result, _years, _reference, _onsets, _title))
    mo.vstack(_figures)
    return


@app.cell
def _(plots, transition_series, transitions):
    plots.transport_and_edges([
        transitions.window_profile(transition_series["1232p5"], (20430, 24300), "warm eq. 1232.5"),
        transitions.window_profile(transition_series["1230"], (7300, 7800), "1230 before jump"),
        transitions.window_profile(transition_series["1230"], (12000, 16200), "cold eq. 1230"),
        transitions.window_profile(transition_series["1225"], (6900, 7500), "1225 before trigger"),
        transitions.window_profile(transition_series["1225"], (7600, 7760), "1225 slide"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What section 5 shows

    **Findings.**

    * **The Southern Hemisphere moves first; the Northern follows.** In 1230
      the southern ice edges (Pacific, Indian), the tropical upper ocean, and
      global Ts all change within about 100 years of year 7900. The northern
      edges follow 300–400 years later. In 1225 the first event is a sharp
      step: the Atlantic 19.4°S row freezes over in about 15 years
      (7580–7595). The tropical ocean and global Ts follow within about
      40 years, the other sectors and the North within about 100 years, and
      the runaway to snowball comes about 200 years later (~7790–7810).
    * **The deep ocean is still cooling when the runs change state.** After
      the drop from 1265 the deep ocean (> 1000 m) cools for thousands of
      years. At the warm → cold step it is at 272.30 K (1230) and 272.46 K
      (1225, Pacific step); at the snowball trigger it is at 271.71 K (1225).
      The 1232.5 equilibrium sits at 272.77 K. The warm period of the 1230
      run was therefore a slow transient, not an equilibrium.
    * **Ocean heat release follows the ice edge.** The ocean's poleward
      transport peaks at 17.5°S on the warm branch and 10°S on the cold
      branch. The strongest release (30–40 W m⁻²) sits at the ice-edge rows:
      31°S (warm), 26°S (cold), 19°S (1225 just before the runaway). The edge
      lies 16° poleward of the transport peak on the warm branch, 15.6° on
      the cold branch, 14° before the 1225 trigger, and 11° during the final
      slide.

    **Hypotheses, not established.**

    * The deep-ocean drift times the transitions (a slow–fast picture). This
      is inferred from two runs.
    * A large ice cap is stable only while the ocean converges heat at its
      edge (Rose & Marshall 2009). The snowball runaway would then start when
      the edge nears the ocean-transport peak.
    * Variance and autocorrelation before the transitions: our estimates do
      not separate the oscillation reliably, so they are not used.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The South Atlantic ice cycle

    Runs μ = 1245, 1240 (before the change near 1235), 1235, and 1233.75,
    1232.5 (after), each over its 4000-year window in `ANALYSIS_WINDOWS`.

    **The cycle phase.** We add up the South Atlantic sea-ice area (red
    outline in the maps), smooth it with an 11-year running mean, and find its
    maxima. The phase runs linearly from 0 at one maximum to 1 at the next, so
    cycles of different lengths are stretched to a common time axis. A *phase
    composite* is the average of any field over all years in the same phase
    bin: it shows the typical cycle and removes the year-to-year noise.
    """)
    return


@app.cell
def _(mechanism, mo, selected_root):
    _missing = [
        mu for mu in mechanism.MECHANISM_MUS
        if not mechanism.archive_path(selected_root.value, mu).exists()
    ]
    mo.stop(bool(_missing), mo.md(f"Missing archives for μ = {', '.join(_missing)}."))
    # Archives are read one at a time; only the sea-ice maps are loaded here.
    summaries = {
        mu: mechanism.summarize_cycle(
            mechanism.read_ice_series(mechanism.archive_path(selected_root.value, mu), mu)
        )
        for mu in mechanism.MECHANISM_MUS
    }
    mo.ui.table(mechanism.cycle_table(summaries), selection=None)
    return (summaries,)


@app.cell
def _(plots, summaries):
    plots.ice_edge_maps(summaries, ("1245", "1235", "1232p5"))
    return


@app.cell
def _(plots, summaries):
    plots.ice_area_excerpts(summaries)
    return


@app.cell
def _(plots, summaries):
    plots.cycle_shapes(summaries)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What section 6 shows

    * **A very cold climate.** In annual mean, the South Atlantic is fully
      ice-covered from about 41°S to the pole. The edge of the ice sits at
      30–36°S, right on top of the subtropical gyre. The whole oscillation
      happens in the one or two T21 rows at the edge.
    * **The edge moves north as μ drops.** At μ = 1245 the 36°S row is
      partly covered across the basin. At 1232.5 it is almost full, and the
      variable edge has moved to 30.5°S (and 24.9°S near 0–10°E).
    * **Period and regularity.** The median cycle grows from 43 to 69 years.
      At 1245 and 1240 the cycle is sharply periodic. From 1235 down it is
      irregular: maxima vary in height and some cycles have double humps.
      There the maxima, and hence the phase, are less well defined.
    * **Shape of the cycle.** At high μ the cycle looks like a relaxation
      oscillation: a quick drop just after the maximum, a slow decline to a
      sharp minimum, then a fast regrowth over about 15 years. At low μ the
      minimum is shallower and broader, and the regrowth takes about 30
      years. The years-after-maximum panel suggests most of the extra length
      sits after the minimum (not yet tested).
    """)
    return


if __name__ == "__main__":
    app.run()
