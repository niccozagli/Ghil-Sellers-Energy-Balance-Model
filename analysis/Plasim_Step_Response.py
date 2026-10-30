"""PlaSim-LSG step responses: linear and quadratic response of one step ensemble, start by start."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from gsebm.paths import get_data_dir
    from gsebm.plasim_indicators import area_mean, area_total, load_indicators, product_stem, volume_mean

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
    return (
        area_mean,
        area_total,
        get_data_dir,
        load_indicators,
        mo,
        np,
        plt,
        product_stem,
        re,
        volume_mean,
    )


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
    Indicators as in `Plasim_Indicators.md` §4 and `stationarity_tests.py`.
    """)
    return


@app.cell
def _(
    area_mean,
    area_total,
    get_data_dir,
    load_indicators,
    np,
    plt,
    product_stem,
    re,
    volume_mean,
):
    BANDS = {"0-700 m": slice(0, 10), "700-2025 m": slice(10, 17), "2025 m-bottom": slice(17, 22)}
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
    STEPS = DATA / "indicators_steps"
    MEMBER = re.compile(
        r"CONTROL_360ppm_T21L10_10000Y_MU_(?P<mu>[0-9p]+)_STEP_(?P<sign>[pm])(?P<delta>[0-9p]+)_from(?P<start>\d+)")

    def number(text):
        """1233p75 -> 1233.75 (the 'p' in run names is a decimal point)."""
        return float(text.replace("p", "."))

    def find_members(mu):
        """Step members of the ensemble at mu: {(delta, start label): {"p": plus run, "m": minus run}}."""
        members = {}
        for folder in sorted(STEPS.iterdir()):
            match = MEMBER.fullmatch(folder.name)
            if match is None or match["mu"] != mu:
                continue
            members.setdefault((number(match["delta"]), int(match["start"])), {})[match["sign"]] = folder.name
        return members

    def indicators(folder):
        """Label years and the annual indicators of the run whose products are in folder."""
        stem = folder / product_stem(folder.name, "spinup")
        zonal = load_indicators(stem.with_name(stem.name + "_zonal.nc"))
        sections = load_indicators(stem.with_name(stem.name + "_sections.nc"))
        t = sections["t_section"]
        series = {
            "tas global": area_mean(zonal["tas_zonal"], zonal),
            "ice area SH": area_total(zonal["sic_ocean_zonal"], zonal, "sh") / 1e12,
            **{f"T {band}": volume_mean(t, sections, levels) for band, levels in BANDS.items()},
        }
        return zonal["year"].values, {name: np.asarray(v.values, dtype=float) for name, v in series.items()}

    def stationary_mean(mu):
        """<x0>: time mean of the parent over its agreed stationary window."""
        first, last = WINDOWS[mu]
        years, series = indicators(DATA / "indicators" / f"CONTROL_360ppm_T21L10_10000Y_MU_{mu}")
        inside = (years >= first) & (years <= last)
        assert inside.sum() == last - first + 1, "parent product does not cover the window"
        return {name: float(values[inside].mean()) for name, values in series.items()}

    def pad(rows):
        """Stack rows of different lengths into (start, year since step, ...), padding with NaN."""
        array = np.full((len(rows), max(len(row) for row in rows), *rows[0].shape[1:]), np.nan, dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            array[i, :len(row)] = row
        return array

    def responses(starts, delta, x0):
        """R1_s(t) and R2_s(t) for every start and panel, as arrays (start, year since step)."""
        linear = {name: [] for name in PANELS}
        quadratic = {name: [] for name in PANELS}
        for start, pair in sorted(starts.items()):
            years_plus, plus = indicators(STEPS / pair["p"])
            years_minus, minus = indicators(STEPS / pair["m"])
            # Both members of a start run over the same labels; use the years they share.
            last = min(years_plus[-1], years_minus[-1])
            for name in PANELS:
                x_plus = plus[name][years_plus <= last]
                x_minus = minus[name][years_minus <= last]
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

    ensembles = sorted({m["mu"] for f in STEPS.iterdir() if (m := MEMBER.fullmatch(f.name))} & set(WINDOWS),
                       key=number)
    return (
        BANDS,
        DATA,
        OCEAN,
        PANELS,
        STEPS,
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
    linear, quadratic = responses(starts, delta, x0)

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

    Linear response R₁ of every ensemble whose files are all on the laptop,
    one colour per μ: mean over starts (line) and each start (faint). Where
    an ensemble has two step sizes (1237.5), the smaller δ is dashed. The
    number of starts drops with time differently in each ensemble (see the
    legend). Two time ranges: the first 300 years with every ensemble, and
    2000 years with the ensembles run longer than 300 years (1233.75, 1235,
    1240). The time axis follows the switch above.
    """)
    return


@app.cell
def _(
    STEPS,
    ensembles,
    find_members,
    mo,
    number,
    product_stem,
    responses,
    stationary_mean,
):
    def is_complete(mu):
        """Whether every step member of mu has both its zonal and its sections file on the laptop."""
        for pair in find_members(mu).values():
            for run in pair.values():
                stem = product_stem(run, "spinup")
                if not all((STEPS / run / f"{stem}_{kind}.nc").is_file() for kind in ("zonal", "sections")):
                    return False
        return True

    # Checked once, so the plotted and the left-out lists agree while a download is still running.
    _complete = [mu for mu in ensembles if is_complete(mu)]

    # Linear response of every complete ensemble and step size: {(mu, delta): {panel: (start, year)}}
    compared = {}
    for _mu in _complete:
        _members = find_members(_mu)
        _x0 = stationary_mean(_mu)
        for _delta in sorted({d for d, _ in _members}):
            _starts = {s: p for (d, s), p in _members.items() if d == _delta and "p" in p and "m" in p}
            compared[(_mu, _delta)] = responses(_starts, _delta, _x0)[0]

    _skipped = [f"{number(mu):g}" for mu in ensembles if mu not in _complete]
    mo.md("Not complete on the laptop yet, left out: " + (", ".join(_skipped) if _skipped else "none") + ".")
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
def _(BANDS, DATA, STEPS, WINDOWS, compared, find_members, load_indicators, np, pad, product_stem):
    # One step size per ensemble: the one with the longest runs.
    LATITUDE_DELTA = {"1245": 2.0, "1240": 2.0, "1237p5": 1.0, "1235": 2.0, "1233p75": 1.5}
    FIELD_UNITS = {"tas": "K", "sea ice": "fraction", "T 0-700 m": "K", "T 2025 m-bottom": "K"}

    def band_by_latitude(sections, levels):
        """Volume mean of the ocean temperature over the levels, per LSG row (NaN where a row has no water)."""
        t = sections["t_section"].isel(lsg_level=levels)
        volume = sections["wet_volume"].isel(lsg_level=levels)
        total = volume.sum("lsg_level")
        return (t * volume).sum("lsg_level") / total.where(total > 0)

    def latitude_fields(folder):
        """Label years, annual fields (year, latitude) and their latitudes for the run whose products are in folder."""
        stem = folder / product_stem(folder.name, "spinup")
        zonal = load_indicators(stem.with_name(stem.name + "_zonal.nc"))
        sections = load_indicators(stem.with_name(stem.name + "_sections.nc"))
        fields = {
            "tas": zonal["tas_zonal"],
            "sea ice": zonal["sic_ocean_zonal"],
            "T 0-700 m": band_by_latitude(sections, BANDS["0-700 m"]),
            "T 2025 m-bottom": band_by_latitude(sections, BANDS["2025 m-bottom"]),
        }
        grid = {name: (zonal["t21_lat"].values if name in ("tas", "sea ice") else sections["lsg_lat"].values)
                for name in fields}
        return zonal["year"].values, {name: field.values for name, field in fields.items()}, grid

    def parent_by_latitude(mu):
        """Time mean of the parent over its agreed stationary window, per latitude."""
        first, last = WINDOWS[mu]
        years, fields, _ = latitude_fields(DATA / "indicators" / f"CONTROL_360ppm_T21L10_10000Y_MU_{mu}")
        inside = (years >= first) & (years <= last)
        assert inside.sum() == last - first + 1, "parent product does not cover the window"
        return {name: np.nanmean(field[inside], axis=0) for name, field in fields.items()}

    def one_sided_responses(mu, delta, x0):
        """Cooling (x_- - <x0>)/delta and warming (<x0> - x_+)/delta of every start, each {field: (start, year, latitude)}.
        Their mean is the centred response."""
        cooling, warming = {name: [] for name in FIELD_UNITS}, {name: [] for name in FIELD_UNITS}
        starts = {s: p for (d, s), p in find_members(mu).items() if d == delta and "p" in p and "m" in p}
        for _start, pair in sorted(starts.items()):
            _, plus, _ = latitude_fields(STEPS / pair["p"])
            _, minus, _ = latitude_fields(STEPS / pair["m"])
            for name in FIELD_UNITS:
                last = min(len(plus[name]), len(minus[name]))
                cooling[name].append(((minus[name][:last] - x0[name]) / delta).astype(np.float32))
                warming[name].append(((x0[name] - plus[name][:last]) / delta).astype(np.float32))
        return ({name: pad(rows) for name, rows in cooling.items()},
                {name: pad(rows) for name, rows in warming.items()})

    # Only the ensembles complete on the laptop (see the comparison above).
    _available = {mu for mu, _ in compared}
    parents = {mu: parent_by_latitude(mu) for mu in WINDOWS}
    one_sided = {mu: one_sided_responses(mu, delta, parents[mu])
                 for mu, delta in LATITUDE_DELTA.items() if mu in _available}
    _, _, latitudes = latitude_fields(DATA / "indicators" / "CONTROL_360ppm_T21L10_10000Y_MU_1240")
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
