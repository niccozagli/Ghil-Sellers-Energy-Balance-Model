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
        "1240": (11000, 17379),
        "1237p5": (18000, 24999),
        "1235": (9500, 17499),
        "1233p75": (20000, 29999),
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
        """Stack rows of different lengths into (start, year since step), padding with NaN."""
        array = np.full((len(rows), max(len(row) for row in rows)), np.nan)
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
        OCEAN,
        PANELS,
        STEPS,
        SURFACE,
        WINDOWS,
        ensembles,
        find_members,
        number,
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
    1000 years with the ensembles run longer than 300 years (1233.75, 1235,
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
    ### 1000 years, the ensembles run longer than 300 years
    """)
    return


@app.cell
def _(SURFACE, comparison_figure):
    comparison_figure(SURFACE, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), 1000 years: surface", 1000,
                      longer_than=300)
    return


@app.cell
def _(OCEAN, comparison_figure):
    comparison_figure(OCEAN, "Linear response (x$_-$ - x$_+$)/(2$\\delta$), 1000 years: ocean", 1000,
                      longer_than=300)
    return


if __name__ == "__main__":
    app.run()
