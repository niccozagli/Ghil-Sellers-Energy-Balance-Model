"""PlaSim-LSG step responses: linear and quadratic response of one step ensemble, start by start."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from gsebm.paths import get_data_dir
    from gsebm.plasim_indicators import latitude_path, load_indicators, series_path

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
    return get_data_dir, latitude_path, load_indicators, mo, np, plt, series_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PlaSim-LSG step responses

    Step ensembles: from start states s taken in a parent's stationary run at
    GSOL0 = μ, one member at μ + δ and one at μ − δ (`PLASIM_INFO/DATA.md` §3).
    Per start and year t since the step (t = label − start label), with
    x(μ ± δ) = x₀ ± R₁δ + ½R₂δ² (`PLASIM_INFO/MODEL.md` §2):

    - **linear** R₁,s(t) = (x₋ − x₊) / (2δ), per W m⁻², signed as the
      response to cooling;
    - **quadratic** R₂,s(t) = (x₊ + x₋ − 2⟨x₀⟩) / δ², per (W m⁻²)², where
      ⟨x₀⟩ is the parent's time mean over its agreed stationary window
      (`DATA.md` §4).

    Each figure shows the mean over the starts that reach year t (line) and
    every start (light lines). Fewer starts reach the later years; the
    number is written where it changes. Annual values, no smoothing.

    Data: the experiment files collected on JASMIN (`Plasim_Indicators.md`
    §7) in `data/Plasim/experiments/`, one per ensemble, and the parents'
    series and latitude files (§6) in `data/Plasim/indicators/`.
    """)
    return


@app.cell
def _(get_data_dir, load_indicators, np, plt, series_path):
    PANELS = {
        "tas global": "K",
        "ice area SH": "10$^6$ km$^2$",
        "T 0-700 m": "K",
        "T 700-2025 m": "K",
        "T 2025 m-bottom": "K",
    }
    # The panels of each figure: surface and ocean.
    SURFACE = ["tas global", "ice area SH"]
    OCEAN = ["T 0-700 m", "T 700-2025 m", "T 2025 m-bottom"]
    # Panel -> variable of the series files (Plasim_Indicators.md §6).
    VARIABLES = {"tas global": "tas_global", "ice area SH": "ice_area_sh", "T 0-700 m": "t_0_700",
                 "T 700-2025 m": "t_700_2025", "T 2025 m-bottom": "t_2025_bottom"}
    # Agreed stationary windows of the parents, in labels (DATA.md §4, 2026-10-28).
    WINDOWS = {
        "1245": (7500, 14999),
        "1242p5": (21000, 27499),
        "1240": (11000, 17379),
        "1237p5": (18000, 24999),
        "1235": (9500, 17499),
        "1233p75": (20000, 29999),
        "1232p5": (21000, 29999),
    }
    DATA = get_data_dir(create=False) / "Plasim"
    EXPERIMENTS = DATA / "experiments"
    PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
    # The experiment file of each ensemble (collect_plasim_experiment.py; 1237.5 collected with --name).
    ENSEMBLE_NAMES = {"1245": "step_members_1245", "1240": "step_members_1240", "1237p5": "step_members_1237p5",
                      "1235": "step_members_1235", "1233p75": "step_members_1233p75"}

    def number(text):
        """1233p75 -> 1233.75 (the 'p' in run names is a decimal point)."""
        return float(text.replace("p", "."))

    # The series of every member of each ensemble whose experiment file is on the laptop.
    ensemble_series = {mu: load_indicators(EXPERIMENTS / f"{name}_series.nc") for mu, name in ENSEMBLE_NAMES.items()
                       if (EXPERIMENTS / f"{name}_series.nc").is_file()}

    def find_members(mu):
        """Step members of the ensemble at mu: {(delta, start label): {"p": plus member, "m": minus member}},
        members given by their index in the experiment file; control members are left out."""
        series = ensemble_series[mu]
        base_mu = float(series.attrs["base_mu"])
        members = {}
        for k in range(series.sizes["member"]):
            role = str(series["role"].values[k])
            if role == "control":
                continue
            delta = round(abs(float(series["gsol0"].values[k]) - base_mu), 4)
            members.setdefault((delta, int(series["start_label"].values[k])), {})[role[0]] = k
        return members

    def member_values(mu, k):
        """Annual panels of member k of the ensemble at mu over the years it has run (index 0 = year 1)."""
        row = ensemble_series[mu].isel(member=k)
        years_run = int(row["years_run"])
        return {name: row[variable].values[:years_run].astype(float) for name, variable in VARIABLES.items()}

    def stationary_mean(mu):
        """<x0>: time mean of the parent over its agreed stationary window, from its series file."""
        first, last = WINDOWS[mu]
        run = f"{PREFIX}{mu}"
        series = load_indicators(series_path(DATA / "indicators" / run, run, "spinup"))
        years = series["year"].values
        inside = (years >= first) & (years <= last)
        assert inside.sum() == last - first + 1, "parent series do not cover the window"
        return {name: float(series[variable].values[inside].mean()) for name, variable in VARIABLES.items()}

    def pad(rows):
        """Stack rows of different lengths into (start, year since step, ...), padding with NaN."""
        array = np.full((len(rows), max(len(row) for row in rows), *rows[0].shape[1:]), np.nan, dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            array[i, :len(row)] = row
        return array

    def responses(mu, starts, delta, x0):
        """R1_s(t) and R2_s(t) for every start and panel, as arrays (start, year since step)."""
        linear = {name: [] for name in PANELS}
        quadratic = {name: [] for name in PANELS}
        for start, pair in sorted(starts.items()):
            plus, minus = member_values(mu, pair["p"]), member_values(mu, pair["m"])
            # Both members are aligned by years since the step; use the years they share.
            shared = min(len(plus["tas global"]), len(minus["tas global"]))
            for name in PANELS:
                x_plus = plus[name][:shared]
                x_minus = minus[name][:shared]
                linear[name].append((x_minus - x_plus) / (2 * delta))
                quadratic[name].append((x_plus + x_minus - 2 * x0[name]) / delta**2)
        return ({name: pad(rows) for name, rows in linear.items()},
                {name: pad(rows) for name, rows in quadratic.items()})

    def plot(response, title, per, panels, time_scale="linear"):
        """One panel per indicator in panels: mean over starts (line) and each start (light); time axis linear or log."""
        count = np.sum(~np.isnan(response[panels[0]]), axis=0)
        t = np.arange(1, count.size + 1)
        changes = [i for i in range(1, count.size) if count[i] != count[i - 1]]
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 0.8 + 2.6 * len(panels)), sharex=True)
        for ax, name in zip(axes, panels):
            unit = PANELS[name]
            for row in response[name]:
                ax.plot(t, row, color="#2a6fb0", alpha=0.15, linewidth=0.6)
            ax.plot(t, np.nanmean(response[name], axis=0), color="#2a6fb0", linewidth=2)
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            for i in changes:
                ax.axvline(t[i], color="0.6", linestyle=":", linewidth=0.8)
            ax.set_ylabel(f"{name}\n[{unit} per {per}]")
            ax.grid(alpha=0.3, linewidth=0.5)
        for i in [0] + changes:
            axes[0].annotate(f"{count[i]} starts", (t[i], 1.0), xycoords=("data", "axes fraction"),
                             xytext=(3, -14), textcoords="offset points", fontsize=10, color="0.3")
        axes[-1].set_xscale(time_scale)  # t starts at 1, so a log axis is defined
        axes[-1].set_xlabel("years since the step")
        axes[0].set_title(title)
        fig.tight_layout()
        return fig

    ensembles = sorted(ensemble_series, key=number)
    return (
        DATA,
        ENSEMBLE_NAMES,
        EXPERIMENTS,
        OCEAN,
        PANELS,
        PREFIX,
        SURFACE,
        WINDOWS,
        ensembles,
        find_members,
        number,
        pad,
        plot,
        responses,
        stationary_mean,
    )


@app.cell
def _(ensembles, mo, number):
    chosen_mu = mo.ui.dropdown(options={f"GSOL0 {number(mu):g}": mu for mu in ensembles},
                               value=f"GSOL0 {number('1233p75'):g}", label="ensemble")
    chosen_mu
    return (chosen_mu,)


@app.cell
def _(chosen_mu, find_members, mo):
    members = find_members(chosen_mu.value)
    _deltas = sorted({delta for delta, _ in members})
    chosen_delta = mo.ui.dropdown(options={f"{d:g}": d for d in _deltas}, value=f"{_deltas[0]:g}", label="δ")
    chosen_delta
    return chosen_delta, members


@app.cell
def _(
    WINDOWS,
    chosen_delta,
    chosen_mu,
    members,
    mo,
    np,
    number,
    responses,
    stationary_mean,
):
    mu, delta = chosen_mu.value, chosen_delta.value
    # A start is used when both its plus and its minus member with this delta exist.
    starts = {start: pair for (d, start), pair in members.items() if d == delta and "p" in pair and "m" in pair}
    x0 = stationary_mean(mu)
    linear, quadratic = responses(mu, starts, delta, x0)

    _count = np.sum(~np.isnan(linear["tas global"]), axis=0)
    _reach = ", ".join(f"{n} up to year {np.flatnonzero(_count == n).max() + 1}" for n in sorted(set(_count), reverse=True))
    _x0 = ", ".join(f"{name} {value:.3f}" for name, value in x0.items())
    mo.md(f"""
    **GSOL0 {number(mu):g} ± {delta:g}:** {len(starts)} starts, labels {min(starts)}–{max(starts)};
    starts reaching year t: {_reach}.

    ⟨x₀⟩ over labels {WINDOWS[mu][0]}–{WINDOWS[mu][1]}: {_x0}.
    """)
    return delta, linear, mu, quadratic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear response
    """)
    return


@app.cell
def _(mo):
    time_scale = mo.ui.radio(options=["linear", "log"], value="linear", label="time axis", inline=True)
    time_scale
    return (time_scale,)


@app.cell
def _(SURFACE, delta, linear, mu, number, plot, time_scale):
    plot(linear, f"Linear response (x$_-$ - x$_+$)/(2$\\delta$), GSOL0 {number(mu):g} $\\pm$ {delta:g}: surface",
         "W m$^{-2}$", SURFACE, time_scale.value)
    return


@app.cell
def _(OCEAN, delta, linear, mu, number, plot, time_scale):
    plot(linear, f"Linear response (x$_-$ - x$_+$)/(2$\\delta$), GSOL0 {number(mu):g} $\\pm$ {delta:g}: ocean",
         "W m$^{-2}$", OCEAN, time_scale.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quadratic response

    With a single ⟨x₀⟩, each start keeps the offset of its own state from
    ⟨x₀⟩ (natural variability), so the light lines are shifted up or down,
    most visibly in the deep ocean. The mean at t ≈ 0 shows how far the
    starts' mean state is from ⟨x₀⟩; R₂ itself is zero at t = 0.
    """)
    return


@app.cell
def _(SURFACE, delta, mu, number, plot, quadratic, time_scale):
    plot(quadratic,
         f"Quadratic response (x$_+$ + x$_-$ - 2$\\langle$x$_0\\rangle$)/$\\delta^2$, GSOL0 {number(mu):g} $\\pm$ {delta:g}: surface",
         "(W m$^{-2}$)$^2$", SURFACE, time_scale.value)
    return


@app.cell
def _(OCEAN, delta, mu, number, plot, quadratic, time_scale):
    plot(quadratic,
         f"Quadratic response (x$_+$ + x$_-$ - 2$\\langle$x$_0\\rangle$)/$\\delta^2$, GSOL0 {number(mu):g} $\\pm$ {delta:g}: ocean",
         "(W m$^{-2}$)$^2$", OCEAN, time_scale.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison between μ

    Linear response R₁ of every ensemble whose experiment file is on the
    laptop, one colour per μ: mean over starts (line) and each start (faint). Where
    an ensemble has two step sizes (1237.5), the smaller δ is dashed. The
    number of starts drops with time differently in each ensemble (see the
    legend). Two time ranges: the first 300 years with every ensemble, and
    2000 years with the ensembles run longer than 300 years (1233.75, 1235,
    1240). The time axis follows the switch above.
    """)
    return


@app.cell
def _(ENSEMBLE_NAMES, ensembles, find_members, mo, number, responses, stationary_mean):
    # Linear response of every ensemble and step size: {(mu, delta): {panel: (start, year)}}
    compared = {}
    for _mu in ensembles:
        _members = find_members(_mu)
        _x0 = stationary_mean(_mu)
        for _delta in sorted({d for d, _ in _members}):
            _starts = {s: p for (d, s), p in _members.items() if d == _delta and "p" in p and "m" in p}
            compared[(_mu, _delta)] = responses(_mu, _starts, _delta, _x0)[0]

    _skipped = [f"{number(mu):g}" for mu in ENSEMBLE_NAMES if mu not in ensembles]
    mo.md("No experiment file on the laptop, left out: " + (", ".join(_skipped) if _skipped else "none") + ".")
    return (compared,)


@app.cell
def _(PANELS, compared, np, number, plt, time_scale):
    # One fixed colour per ensemble (palette order of the dataviz skill), so adding an ensemble repaints nothing.
    COLOURS = {"1233p75": "#2a78d6", "1235": "#eb6834", "1237p5": "#1baf7a", "1240": "#eda100", "1245": "#e87ba4"}

    def comparison_figure(panels, title, years, longer_than=0):
        """Linear response of each ensemble in panels over the first ``years`` years: faint starts first, then the
        means on top. Only ensembles whose longest starts run more than ``longer_than`` years are drawn."""
        deltas_of = {}
        for mu, delta in compared:
            deltas_of.setdefault(mu, []).append(delta)
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.6 + 2.6 * len(panels)), sharex=True,
                                 layout="constrained")
        for (mu, delta), full in compared.items():
            if full[panels[0]].shape[1] <= longer_than:
                continue
            response = {name: full[name][:, :years] for name in panels}
            style = "--" if len(deltas_of[mu]) > 1 and delta == min(deltas_of[mu]) else "-"
            count = np.sum(~np.isnan(response[panels[0]]), axis=0)
            t = np.arange(1, count.size + 1)
            label = f"{number(mu):g} $\\pm$ {delta:g}: {count[0]} starts, {count[-1]} to year {count.size}"
            for ax, name in zip(axes, panels):
                for row in response[name]:
                    ax.plot(t, row, color=COLOURS[mu], alpha=0.1, linewidth=0.4, linestyle=style)
                ax.plot(t, np.nanmean(response[name], axis=0), color=COLOURS[mu], linewidth=2, linestyle=style,
                        label=label, zorder=3)
        for ax, name in zip(axes, panels):
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.set_ylabel(f"{name}\n[{PANELS[name]} per W m$^{{-2}}$]")
            ax.grid(alpha=0.3, linewidth=0.5)
        axes[-1].set_xscale(time_scale.value)
        axes[-1].set_xlabel("years since the step")
        # Title at the top, legend below the panels (constrained layout makes room for both).
        fig.suptitle(title)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=11, frameon=False)
        return fig

    return (comparison_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First 300 years, all ensembles
    """)
    return


@app.cell
def _(SURFACE, comparison_figure):
    comparison_figure(SURFACE, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), first 300 years: surface", 300)
    return


@app.cell
def _(OCEAN, comparison_figure):
    comparison_figure(OCEAN, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), first 300 years: ocean", 300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2000 years, the ensembles run longer than 300 years
    """)
    return


@app.cell
def _(SURFACE, comparison_figure):
    comparison_figure(SURFACE, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), 2000 years: surface", 2000,
                      longer_than=300)
    return


@app.cell
def _(OCEAN, comparison_figure):
    comparison_figure(OCEAN, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), 2000 years: ocean", 2000,
                      longer_than=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Response by latitude

    The same response without the global and hemispheric averaging, per
    latitude: surface air temperature, sea-ice fraction of the ocean (both on
    the T21 rows), and ocean temperature in two layers (volume mean per LSG
    row). One step size per ensemble, the one with the longest runs (1237.5 ± 1).

    **Parent difference** at a latitude: (⟨x⟩(μ_low) − ⟨x⟩(μ_high)) / (μ_high − μ_low),
    the time means over the agreed windows of the two neighbouring parents,
    per W m⁻². It is what the response would reach if the parents are
    stationary and the sensitivity is constant between them. Neighbours: 1240
    from 1242.5 and 1237.5, 1237.5 from 1240 and 1235, 1235 from 1237.5 and
    1232.5, 1233.75 from 1235 and 1232.5; none for 1245 (the 1250 parent has
    only a short window).

    Time axes are logarithmic in this section, annual values, no smoothing.
    Averages over stretches of years are named where they are used. Tables use
    the long starts only, so the same starts at every time; ± is the standard
    error over starts, treating them as independent, which they are not, so
    it is too small.
    """)
    return


@app.cell
def _(
    DATA,
    ENSEMBLE_NAMES,
    EXPERIMENTS,
    PREFIX,
    WINDOWS,
    ensembles,
    find_members,
    latitude_path,
    load_indicators,
    np,
    pad,
):
    # One step size per ensemble: the one with the longest runs.
    LATITUDE_DELTA = {"1245": 2.0, "1240": 2.0, "1237p5": 1.0, "1235": 2.0, "1233p75": 1.5}
    FIELD_UNITS = {"tas": "K", "sea ice": "fraction", "T 0-700 m": "K", "T 2025 m-bottom": "K"}
    # Field -> variable of the latitude files (Plasim_Indicators.md §6-7).
    FIELD_VARIABLES = {"tas": "tas", "sea ice": "sic", "T 0-700 m": "t_0_700", "T 2025 m-bottom": "t_2025_bottom"}

    ensemble_fields = {mu: load_indicators(EXPERIMENTS / f"{ENSEMBLE_NAMES[mu]}_latitude.nc") for mu in ensembles}

    def parent_by_latitude(mu):
        """Time mean of the parent over its agreed stationary window, per latitude, from its latitude file."""
        first, last = WINDOWS[mu]
        run = f"{PREFIX}{mu}"
        fields = load_indicators(latitude_path(DATA / "indicators" / run, run, "spinup"))
        years = fields["year"].values
        inside = (years >= first) & (years <= last)
        assert inside.sum() == last - first + 1, "parent latitude file does not cover the window"
        return {name: np.nanmean(fields[variable].values[inside], axis=0) for name, variable in FIELD_VARIABLES.items()}

    def one_sided_responses(mu, delta, x0):
        """Cooling (x_- - <x0>)/delta and warming (<x0> - x_+)/delta of every start, each {field: (start, year, latitude)}.
        Their mean is the centred response."""
        fields = ensemble_fields[mu]
        cooling, warming = {name: [] for name in FIELD_UNITS}, {name: [] for name in FIELD_UNITS}
        starts = {s: p for (d, s), p in find_members(mu).items() if d == delta and "p" in p and "m" in p}
        for _start, pair in sorted(starts.items()):
            plus, minus = fields.isel(member=pair["p"]), fields.isel(member=pair["m"])
            # Aligned by years since the step; use the years both members have run.
            shared = min(int(plus["years_run"]), int(minus["years_run"]))
            for name, variable in FIELD_VARIABLES.items():
                cooling[name].append(((minus[variable].values[:shared] - x0[name]) / delta).astype(np.float32))
                warming[name].append(((x0[name] - plus[variable].values[:shared]) / delta).astype(np.float32))
        return ({name: pad(rows) for name, rows in cooling.items()},
                {name: pad(rows) for name, rows in warming.items()})

    parents = {mu: parent_by_latitude(mu) for mu in WINDOWS}
    one_sided = {mu: one_sided_responses(mu, delta, parents[mu])
                 for mu, delta in LATITUDE_DELTA.items() if mu in ensembles}
    _any = next(iter(ensemble_fields.values()))
    latitudes = {name: (_any["t21_lat"].values if variable in ("tas", "sic") else _any["lsg_lat"].values)
                 for name, variable in FIELD_VARIABLES.items()}
    return FIELD_UNITS, LATITUDE_DELTA, latitudes, one_sided, parents


@app.cell
def _(FIELD_UNITS, LATITUDE_DELTA, latitudes, np, number, one_sided, parents, plt):
    # The ensembles with starts run to 2000 years.
    LONG = ["1240", "1235", "1233p75"]
    # The parents around each ensemble: (warmer, colder). None around 1245: the 1250 window is short.
    NEIGHBOURS = {"1240": ("1242p5", "1237p5"), "1237p5": ("1240", "1235"), "1235": ("1237p5", "1232p5"),
                  "1233p75": ("1235", "1232p5")}
    # The rows where the response is largest in the south (ice edge, 36S; upper ocean, 31S; deep, 51S) and
    # their mirror in the north.
    SOUTH = [("tas", -35.9), ("sea ice", -35.9), ("T 0-700 m", -31.25), ("T 2025 m-bottom", -51.25)]
    NORTH = [("tas", 35.9), ("sea ice", 35.9), ("T 0-700 m", 31.25), ("T 2025 m-bottom", 31.25)]
    STRETCHES = [(11, 20), (41, 60), (101, 150), (451, 550), (951, 1050), (1401, 1500), (1901, 2000)]
    MAP_LIMITS = {"tas": 1.2, "sea ice": 0.07, "T 0-700 m": 0.25, "T 2025 m-bottom": 0.14}
    MU_COLOURS = {"1233p75": "#2a78d6", "1235": "#eb6834", "1237p5": "#1baf7a", "1240": "#eda100", "1245": "#e87ba4"}

    def centred(mu, name):
        """Centred response (x_- - x_+)/(2 delta), (start, year, latitude): the mean of cooling and warming."""
        cooling, warming = one_sided[mu]
        return (cooling[name] + warming[name]) / 2

    def long_starts(mu):
        """Which starts of mu run to year 2000."""
        return ~np.isnan(one_sided[mu][0]["tas"][:, 1999, 0])

    def row_of(name, latitude):
        """Index of the grid row of the field nearest to the latitude."""
        return int(np.argmin(np.abs(latitudes[name] - latitude)))

    def parent_difference(mu, name):
        """(<x>(colder neighbour) - <x>(warmer neighbour)) / (difference in GSOL0), per latitude."""
        warmer, colder = NEIGHBOURS[mu]
        return (parents[colder][name] - parents[warmer][name]) / (number(warmer) - number(colder))

    def stretch_table(series, divide_by=1.0):
        """Table cells of one (start, year) array: mean and SE over starts of each stretch, and R(1950)/R(500)."""
        cells, means = [], {}
        for first, last in STRETCHES:
            per_start = series[:, first - 1:last].mean(axis=1) / divide_by
            means[first] = per_start.mean()
            cells.append(f"{means[first]:.3g} ± {per_start.std(ddof=1) / np.sqrt(len(per_start)):.1g}")
        return cells + [f"{means[1901] / means[451]:.2f}"]

    def table_head(first_columns):
        names = [f"{first}–{last}" for first, last in STRETCHES] + ["R(1950)/R(500)"]
        head = "| " + " | ".join(first_columns + names) + " |"
        return head + "\n|" + "---|" * (len(first_columns) + len(names))

    def fraction_table(places):
        """Markdown table: the centred response as a fraction of the parent difference, long starts."""
        lines = [table_head(["Field", "Latitude", "μ", "Parent difference"])]
        for name, latitude in places:
            row = row_of(name, latitude)
            for mu in LONG:
                difference = parent_difference(mu, name)[row]
                series = centred(mu, name)[long_starts(mu)][:, :, row]
                cells = [name, f"{latitude:g}", f"{number(mu):g}", f"{difference:+.3g}"]
                lines.append("| " + " | ".join(cells + stretch_table(series, difference)) + " |")
        return "\n".join(lines)

    def one_sided_table(places):
        """Markdown table: cooling and warming responses, per W m^-2, long starts."""
        lines = [table_head(["Field", "Latitude", "μ", "Side"])]
        for name, latitude in places:
            row = row_of(name, latitude)
            for mu in LONG:
                for side, responses_of_side in zip(("cooling", "warming"), one_sided[mu]):
                    series = responses_of_side[name][long_starts(mu)][:, :, row]
                    cells = [name, f"{latitude:g}", f"{number(mu):g}", side]
                    lines.append("| " + " | ".join(cells + stretch_table(series)) + " |")
        return "\n".join(lines)

    def latitude_map():
        """Latitude-time map of the centred response, mean over the starts reaching year t, one column per ensemble."""
        ensembles_here = [mu for mu in LATITUDE_DELTA if mu in one_sided]
        fig, axes = plt.subplots(len(FIELD_UNITS), len(ensembles_here), figsize=(17, 12), sharex=True, sharey="row",
                                 layout="constrained", squeeze=False)
        for row, (name, unit) in zip(axes, FIELD_UNITS.items()):
            for ax, mu in zip(row, ensembles_here):
                mean = np.nanmean(centred(mu, name), axis=0)
                mesh = ax.pcolormesh(np.arange(1, mean.shape[0] + 1), latitudes[name], mean.T, shading="nearest",
                                     cmap="RdBu_r", vmin=-MAP_LIMITS[name], vmax=MAP_LIMITS[name])
                ax.set_xscale("log")
                ax.set_xlim(1, 2000)
                ax.set_ylim(-80, 85)
                ax.grid(alpha=0.3, linewidth=0.5)
            row[0].set_ylabel(f"{name}\nlatitude")
            fig.colorbar(mesh, ax=row, label=f"{unit} per W m$^{{-2}}$", shrink=0.9, aspect=25)
        for ax, mu in zip(axes[0], ensembles_here):
            ax.set_title(f"{number(mu):g} $\\pm$ {LATITUDE_DELTA[mu]:g}")
        for ax in axes[-1]:
            ax.set_xlabel("years since the step")
        fig.suptitle("Centred response by latitude, mean over the starts reaching year t")
        return fig

    def latitude_profiles():
        """Centred response against latitude over stretches of years, long starts, with the parent difference."""
        stretches = [(1, 10), (41, 60), (101, 150), (451, 550), (951, 1050), (1901, 2000)]
        colours = plt.cm.viridis(np.linspace(0.92, 0.05, len(stretches)))
        fig, axes = plt.subplots(len(FIELD_UNITS), len(LONG), figsize=(16, 13), sharex=True, sharey="row",
                                 layout="constrained", squeeze=False)
        for row, (name, unit) in zip(axes, FIELD_UNITS.items()):
            for ax, mu in zip(row, LONG):
                series = centred(mu, name)[long_starts(mu)]
                for (first, last), colour in zip(stretches, colours):
                    ax.plot(latitudes[name], np.nanmean(series[:, first - 1:last], axis=(0, 1)), color=colour,
                            linewidth=1.5, label=f"years {first}-{last}")
                ax.plot(latitudes[name], parent_difference(mu, name), color="k", linestyle="--", linewidth=1.5,
                        label="parent difference")
                ax.axhline(0.0, color="0.5", linewidth=0.7)
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.set_xlim(-80, 85)
            row[0].set_ylabel(f"{name}\n[{unit} per W m$^{{-2}}$]")
        for ax, mu in zip(axes[0], LONG):
            ax.set_title(f"{number(mu):g} $\\pm$ {LATITUDE_DELTA[mu]:g}, {long_starts(mu).sum()} long starts")
        for ax in axes[-1]:
            ax.set_xlabel("latitude")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle("Centred response against latitude, averages over stretches of years")
        return fig

    def time_shapes(places, title):
        """At each place: the centred response (left) and its fraction of the parent difference (right), annual,
        mean over the starts reaching year t."""
        fig, axes = plt.subplots(len(places), 2, figsize=(14, 2.6 * len(places) + 1), sharex=True,
                                 layout="constrained", squeeze=False)
        for row_axes, (name, latitude) in zip(axes, places):
            row = row_of(name, latitude)
            for mu in LATITUDE_DELTA:
                if mu not in one_sided:
                    continue
                mean = np.nanmean(centred(mu, name)[:, :, row], axis=0)
                t = np.arange(1, mean.size + 1)
                row_axes[0].plot(t, mean, color=MU_COLOURS[mu], linewidth=0.9, label=f"{number(mu):g}")
                if mu in NEIGHBOURS:
                    row_axes[1].plot(t, mean / parent_difference(mu, name)[row], color=MU_COLOURS[mu], linewidth=0.9)
            row_axes[0].set_ylabel(f"{name}, {latitude:g}\n[{FIELD_UNITS[name]} per W m$^{{-2}}$]")
            row_axes[1].set_ylim(-0.2, 1.3)
            row_axes[1].axhline(1.0, color="0.5", linewidth=0.7)
            for ax in row_axes:
                ax.set_xscale("log")
                ax.grid(alpha=0.3, linewidth=0.5)
        axes[0, 0].set_title("centred response")
        axes[0, 1].set_title("fraction of the parent difference")
        axes[0, 0].legend(ncol=5, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("years since the step")
        fig.suptitle(title)
        return fig

    def one_sided_figure(places, title):
        """At each place and for each long ensemble: cooling, warming and centred response, annual, long starts."""
        fig, axes = plt.subplots(len(places), len(LONG), figsize=(16, 2.6 * len(places) + 1), sharex=True, sharey="row",
                                 layout="constrained", squeeze=False)
        for row_axes, (name, latitude) in zip(axes, places):
            row = row_of(name, latitude)
            for ax, mu in zip(row_axes, LONG):
                cooling, warming = one_sided[mu]
                long = long_starts(mu)
                t = np.arange(1, 2001)
                ax.plot(t, cooling[name][long][:, :2000, row].mean(axis=0), color="#2a78d6", linewidth=0.8, label="cooling")
                ax.plot(t, warming[name][long][:, :2000, row].mean(axis=0), color="#eb6834", linewidth=0.8, label="warming")
                ax.plot(t, centred(mu, name)[long][:, :2000, row].mean(axis=0), color="k", linewidth=0.8, label="centred")
                ax.axhline(0.0, color="0.5", linewidth=0.7)
                ax.set_xscale("log")
                ax.grid(alpha=0.3, linewidth=0.5)
            row_axes[0].set_ylabel(f"{name}, {latitude:g}\n[{FIELD_UNITS[name]} per W m$^{{-2}}$]")
        for ax, mu in zip(axes[0], LONG):
            ax.set_title(f"{number(mu):g} $\\pm$ {LATITUDE_DELTA[mu]:g}")
        for ax in axes[-1]:
            ax.set_xlabel("years since the step")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(title)
        return fig

    def state_against_gsol0():
        """Time-mean state against GSOL0 at the southern (left) and northern (right) rows: parents over their windows
        (black), and the plus and minus members of the long starts over years 1901-2000 (mean and SE over starts)."""
        fig, axes = plt.subplots(len(SOUTH), 2, figsize=(14, 2.8 * len(SOUTH) + 1), sharex=True,
                                 layout="constrained", squeeze=False)
        for row_axes, places in zip(axes, zip(SOUTH, NORTH)):
            for ax, (name, latitude) in zip(row_axes, places):
                row = row_of(name, latitude)
                mus = sorted(parents, key=number)
                ax.plot([number(mu) for mu in mus], [parents[mu][name][row] for mu in mus], color="k", marker="o",
                        markersize=5, linewidth=1, label="parents")
                for mu in LONG:
                    delta = LATITUDE_DELTA[mu]
                    cooling, warming = one_sided[mu]
                    long = long_starts(mu)
                    # x_- = <x0> + delta * cooling and x_+ = <x0> - delta * warming.
                    minus = parents[mu][name][row] + delta * cooling[name][long][:, 1900:2000, row].mean(axis=1)
                    plus = parents[mu][name][row] - delta * warming[name][long][:, 1900:2000, row].mean(axis=1)
                    for gsol0, member, marker in ((number(mu) - delta, minus, "v"), (number(mu) + delta, plus, "^")):
                        ax.errorbar(gsol0, member.mean(), yerr=member.std(ddof=1) / np.sqrt(len(member)), linestyle="none",
                                    marker=marker, markersize=8, color=MU_COLOURS[mu])
                ax.set_ylabel(f"{name}, {latitude:g}\n[{FIELD_UNITS[name]}]")
                ax.grid(alpha=0.3, linewidth=0.5)
        for ax in axes[-1]:
            ax.set_xlabel("GSOL0 [W m$^{-2}$]")
        handles = [plt.Line2D([], [], color="k", marker="o", label="parents")]
        for mu in LONG:
            label = f"{number(mu):g} $\\pm$ {LATITUDE_DELTA[mu]:g}, years 1901-2000"
            handles.append(plt.Line2D([], [], color=MU_COLOURS[mu], marker="^", linestyle="none", label=label))
        fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=11, frameon=False)
        fig.suptitle("State against GSOL0 ($\\blacktriangle$ plus member, $\\blacktriangledown$ minus member)")
        return fig

    return (
        NORTH,
        SOUTH,
        fraction_table,
        latitude_map,
        latitude_profiles,
        one_sided_figure,
        one_sided_table,
        state_against_gsol0,
        time_shapes,
    )


@app.cell
def _(latitude_map):
    latitude_map()
    return


@app.cell
def _(latitude_profiles):
    latitude_profiles()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time shape at single latitudes

    R(1950)/R(500) is the ratio of the means over years 1901–2000 and 451–550.
    It uses no parent. The stretch of year 500 was picked after seeing the
    curves.
    """)
    return


@app.cell
def _(SOUTH, time_shapes):
    time_shapes(SOUTH, "Southern rows: ice edge, upper ocean and deep ocean")
    return


@app.cell
def _(SOUTH, fraction_table, mo):
    mo.md("**South: fraction of the parent difference, over stretches of years**\n\n" + fraction_table(SOUTH))
    return


@app.cell
def _(NORTH, time_shapes):
    time_shapes(NORTH, "Northern rows: ice edge, upper ocean and deep ocean")
    return


@app.cell
def _(NORTH, fraction_table, mo):
    mo.md("**North: fraction of the parent difference, over stretches of years**\n\n" + fraction_table(NORTH))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cooling and warming separately

    The centred response is the linear response up to order δ². Cooling
    (x₋ − ⟨x₀⟩)/δ and warming (⟨x₀⟩ − x₊)/δ are R₁ ± ½δR₂ and their mean is
    the centred response, so this asks whether the two sides agree: whether
    the response is still linear at this δ. It matters near the edge, where δ
    is not small against the distance to it (1233.75 − 1.5 = 1232.25).

    Both use ⟨x₀⟩, so unlike the centred response they keep the offset of the
    starts' mean state from ⟨x₀⟩ (with opposite signs), which shows at t ≈ 0,
    mostly in the deep ocean.
    """)
    return


@app.cell
def _(SOUTH, one_sided_figure):
    one_sided_figure(SOUTH, "Cooling and warming, southern rows, long starts")
    return


@app.cell
def _(SOUTH, mo, one_sided_table):
    mo.md("**South: cooling and warming, per W m⁻², over stretches of years**\n\n" + one_sided_table(SOUTH))
    return


@app.cell
def _(NORTH, one_sided_figure):
    one_sided_figure(NORTH, "Cooling and warming, northern rows, long starts")
    return


@app.cell
def _(NORTH, mo, one_sided_table):
    mo.md("**North: cooling and warming, per W m⁻², over stretches of years**\n\n" + one_sided_table(NORTH))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State against GSOL0

    The local sensitivity is the slope of this picture. The members add six
    values of GSOL0 between the parents. They have not fully adjusted by year
    2000, and each has adjusted from its own parent: a minus member is still
    too warm and a plus member still too cold, most in the deep ocean.
    """)
    return


@app.cell
def _(state_against_gsol0):
    state_against_gsol0()
    return


if __name__ == "__main__":
    app.run()
