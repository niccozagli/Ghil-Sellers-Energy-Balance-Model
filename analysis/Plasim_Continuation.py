"""PlaSim-LSG continuation of the warm branch below GSOL0 1232.25: each start's three arms, and the departure check."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from gsebm.paths import get_data_dir
    from gsebm.plasim_indicators import load_indicators, series_path

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})
    return get_data_dir, load_indicators, mo, np, plt, series_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Continuation of the warm branch below GSOL0 1232.25

    Eighteen states, the − members of the 1233.75 ± 1.5 ensemble (GSOL0
    1232.25) at 2000 years since their step (`PLASIM_INFO/DATA.md` §3.2), each
    continued in three arms:

    - **1232.25**: the same run, continued (control);
    - **1232** and **1231.75**: GSOL0 lowered by 0.25 and 0.5 at year 2000.

    Time is in years since the 1233.75 step, so the arms start at 2000 and
    the control has its first 2000 years as well. Annual values, no smoothing,
    except the block means of the departure check, which are named there.
    The indicators are read from each run's annual series file, computed on
    JASMIN (`Plasim_Indicators.md` §6), in `data/Plasim/indicators_steps/`
    and, for the parents, `data/Plasim/indicators/`.

    **Two kinds of result, kept apart.** The first three sections are a
    *continuation exercise*: they follow the states of the three arms, which
    start from runs that have not settled (the grey runs are still adjusting
    to their own step), so differences between arms are not responses. The
    last section is a *response*: only the grey runs' step from the
    stationary 1233.75 parent qualifies.
    """)
    return


@app.cell
def _(get_data_dir, load_indicators, np, series_path):
    DATA = get_data_dir(create=False) / "Plasim"
    STEPS = DATA / "indicators_steps"
    PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
    STARTS = list(range(20009, 26810, 400))  # step labels of the 18 source members
    ARMS = {"1232.25": "#555555", "1232": "#eb6834", "1231.75": "#2a78d6"}
    # Shown name -> variable of the series file (Plasim_Indicators.md §6).
    SERIES = {
        "tas global": "tas_global", "tas SH": "tas_sh", "tas NH": "tas_nh",
        "SH ice area": "ice_area_sh", "NH ice area": "ice_area_nh", "ACC": "acc",
        "T 0-700 m": "t_0_700", "T 700-2025 m": "t_700_2025", "T 2025 m-bottom": "t_2025_bottom",
    }
    UNITS = {
        "tas global": "K", "tas SH": "K", "tas NH": "K",
        "SH ice area": "10$^6$ km$^2$", "NH ice area": "10$^6$ km$^2$", "ACC": "Sv",
        "T 0-700 m": "K", "T 700-2025 m": "K", "T 2025 m-bottom": "K",
    }

    def run_name(arm, start):
        """The run holding this arm of this start: the source member itself for the control."""
        if arm == "1232.25":
            return f"{PREFIX}1233p75_STEP_m1p5_from{start}"
        return f"{PREFIX}1232p25_STEP_{'m0p25' if arm == '1232' else 'm0p5'}_from{start + 2000}"

    def read_series(run, root=STEPS):
        """Label years and annual indicators of one run, from its series file in root/run."""
        series = load_indicators(series_path(root / run, run, "spinup"))
        return series["year"].values, {name: series[variable].values for name, variable in SERIES.items()}

    def indicators(run, root=STEPS):
        """Annual indicators of one run."""
        return read_series(run, root)[1]

    def parent_mean(mu, first, last):
        """Time mean of a parent run over its agreed stationary window, in labels (DATA.md §4)."""
        years, values = read_series(f"{PREFIX}{mu}", DATA / "indicators")
        inside = (years >= first) & (years <= last)
        assert inside.sum() == last - first + 1, "parent series do not cover the window"
        return {name: float(v[inside].mean()) for name, v in values.items()}

    def years_since_step(arm, length):
        """Control: 1, 2, ...; the lowered arms start at 2001."""
        return np.arange(1, length + 1) if arm == "1232.25" else np.arange(2001, 2001 + length)

    # data[arm][start] = (years since the step, {indicator: annual values})
    data = {}
    for _arm in ARMS:
        data[_arm] = {}
        for _start in STARTS:
            _values = indicators(run_name(_arm, _start))
            data[_arm][_start] = (years_since_step(_arm, len(_values["ACC"])), _values)

    # Members have different lengths (a stage can stop some before others), so the last year differs.
    LAST_YEAR = max(int(years[-1]) for arm in data.values() for years, _ in arm.values())

    def on_years(arm, name, first, last):
        """The indicator of every start of an arm on the years first..last since the step, as (start, year);
        NaN where a start has no value (not reached yet, or before the arm's step)."""
        table = np.full((len(STARTS), last - first + 1), np.nan)
        for k, start in enumerate(STARTS):
            years, values = data[arm][start]
            keep = (years >= first) & (years <= last)
            table[k, years[keep] - first] = values[name][keep]
        return table

    # For the response of the grey runs: the + members of the same 1233.75 starts (GSOL0 1235.25, 2000 years) and
    # the parents around (agreed windows, DATA.md §4).
    plus = {s: indicators(f"{PREFIX}1233p75_STEP_p1p5_from{s}") for s in STARTS}
    parents = {"1233.75": parent_mean("1233p75", 20000, 29999), "1232.5": parent_mean("1232p5", 21000, 29999)}
    return ARMS, LAST_YEAR, STARTS, UNITS, data, on_years, parents, plus


@app.cell
def _(UNITS, mo):
    indicator = mo.ui.dropdown(options=list(UNITS), value="T 0-700 m", label="indicator")
    first_year = mo.ui.slider(start=0, stop=2000, step=100, value=1500, label="plot from year")
    mo.hstack([indicator, first_year], justify="start")
    return first_year, indicator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuation: all starts

    *Continuation exercise, not a response.* Each start (thin) and the mean
    over the starts of each arm that reach each year (thick), annual values.
    The members may have different lengths; the mean is then over fewer
    starts at the end.
    """)
    return


@app.cell
def _(ARMS, LAST_YEAR, UNITS, data, first_year, indicator, np, on_years, plt):
    def all_starts(name, since):
        fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
        for arm, colour in ARMS.items():
            for years, values in data[arm].values():
                keep = years > since
                ax.plot(years[keep], values[name][keep], color=colour, linewidth=0.3, alpha=0.6)
            # Mean over the starts that reach each year, from the lowered arms' step on.
            table = on_years(arm, name, 2001, LAST_YEAR)
            reached = np.isfinite(table).sum(axis=0)
            mean = np.where(reached > 0, np.nansum(table, axis=0) / np.maximum(reached, 1), np.nan)
            ax.plot(np.arange(2001, LAST_YEAR + 1), mean, color=colour, linewidth=2,
                    label=f"GSOL0 {arm}, mean ({reached.max()} to {reached[reached > 0].min()} starts)")
        ax.axvline(2000, color="k", linewidth=0.8, linestyle=":")
        ax.set_xlabel("years since the 1233.75 step")
        ax.set_ylabel(f"{name} [{UNITS[name]}]")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=10, ncol=3)
        return fig

    all_starts(indicator.value, first_year.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuation: each start

    *Continuation exercise, not a response.* The three arms of each start,
    annual values, on a common vertical axis.
    """)
    return


@app.cell
def _(ARMS, STARTS, UNITS, data, first_year, indicator, plt):
    def each_start(name, since):
        fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), sharex=True, sharey=True, layout="constrained")
        for ax, start in zip(axes.flat, STARTS):
            for arm, colour in ARMS.items():
                years, values = data[arm][start]
                keep = years > since
                ax.plot(years[keep], values[name][keep], color=colour, linewidth=0.5)
            ax.axvline(2000, color="k", linewidth=0.6, linestyle=":")
            ax.set_title(f"start {start}", fontsize=10)
            ax.grid(alpha=0.3, linewidth=0.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{name} [{UNITS[name]}]")
        fig.suptitle(f"{name}: control 1232.25 (grey), 1232 (orange), 1231.75 (blue)")
        return fig

    each_start(indicator.value, first_year.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuation: departure check

    *Continuation exercise, not a response.* The years after 2000, up to the
    longest run, are cut into consecutive blocks; each start's block means are
    compared with the range (minimum to maximum) of the block means of the
    controls that cover the same block. A block counts only where a run covers
    all of it. A start counts as **outside** in a block when all the chosen
    indicators are outside that range. Agreed beforehand (`DATA.md` §3.2):
    ACC and SH ice area, 100-year blocks, three outside in a row. **At the
    end** says whether the start is outside in its own last block, which
    separates an excursion that came back from one that has not.
    """)
    return


@app.cell
def _(UNITS, mo):
    chosen = mo.ui.multiselect(options=list(UNITS), value=["ACC", "SH ice area"], label="indicators")
    block = mo.ui.slider(start=50, stop=200, step=50, value=100, label="block length (years)")
    in_a_row = mo.ui.slider(start=1, stop=5, step=1, value=3, label="blocks outside in a row")
    mo.vstack([chosen, mo.hstack([block, in_a_row], justify="start")])
    return block, chosen, in_a_row


@app.cell
def _(ARMS, LAST_YEAR, STARTS, block, chosen, data, in_a_row, np):
    BLOCKS = [(first, first + block.value - 1)
              for first in range(2001, LAST_YEAR - block.value + 2, block.value)]

    def block_means(arm, start, name):
        """Mean of each block of years since the step; NaN for a block the run does not cover in full."""
        years, values = data[arm][start]
        means = []
        for a, b in BLOCKS:
            inside = (years >= a) & (years <= b)
            means.append(values[name][inside].mean() if inside.sum() == b - a + 1 else np.nan)
        return np.array(means)

    # The controls' range of block means, per indicator: (minimum, maximum) over the controls covering the block.
    control_range = {}
    for _name in chosen.value:
        _means = np.array([block_means("1232.25", s, _name) for s in STARTS])
        _covered = np.isfinite(_means).any(axis=0)
        control_range[_name] = (np.where(_covered, np.nanmin(np.where(np.isfinite(_means), _means, np.inf), axis=0),
                                         np.nan),
                                np.where(_covered, np.nanmax(np.where(np.isfinite(_means), _means, -np.inf), axis=0),
                                         np.nan))

    def outside(arm, start):
        """Per block: whether every chosen indicator is outside the controls' range."""
        result = np.ones(len(BLOCKS), dtype=bool)
        for name, (low, high) in control_range.items():
            means = block_means(arm, start, name)
            result &= (means < low) | (means > high)
        return result

    def first_run(flags, length):
        """First year of the first run of ``length`` blocks outside in a row, or None."""
        count = 0
        for k, flag in enumerate(flags):
            count = count + 1 if flag else 0
            if count == length:
                return BLOCKS[k - length + 1][0]
        return None

    def blocks_covered(arm, start):
        """How many blocks the run covers in full."""
        years, _ = data[arm][start]
        return sum(years[-1] >= b for _, b in BLOCKS)

    rows = []
    for _arm in [a for a in ARMS if a != "1232.25"]:
        for _start in STARTS:
            _flags = outside(_arm, _start)
            _covered = blocks_covered(_arm, _start)
            rows.append({"GSOL0": _arm, "start": _start, "years run": int(data[_arm][_start][0][-1]) - 2000,
                         "blocks outside": int(_flags.sum()),
                         f"{in_a_row.value} in a row from": first_run(_flags, in_a_row.value),
                         "outside at the end": bool(_flags[_covered - 1]) if _covered else None})
    return BLOCKS, block_means, control_range, rows


@app.cell
def _(ARMS, in_a_row, mo, rows):
    _key = f"{in_a_row.value} in a row from"
    _summary = ", ".join(
        f"GSOL0 {arm}: {sum(r[_key] is not None for r in rows if r['GSOL0'] == arm)} of 18 flagged, "
        f"{sum(r['outside at the end'] for r in rows if r['GSOL0'] == arm)} outside at the end"
        for arm in ARMS if arm != "1232.25")
    mo.vstack([mo.md(_summary + "."), mo.ui.table(rows, selection=None, page_size=36)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The check itself: the controls' range of block means (grey band) and each
    start's block means (lines); flagged starts are drawn thick.
    """)
    return


@app.cell
def _(ARMS, BLOCKS, STARTS, UNITS, block_means, control_range, in_a_row, np, plt, rows):
    def check_figure():
        names = list(control_range)
        arms = [a for a in ARMS if a != "1232.25"]
        middle = np.array([(a + b) / 2 for a, b in BLOCKS])
        flagged = {(r["GSOL0"], r["start"]) for r in rows if r[f"{in_a_row.value} in a row from"] is not None}
        fig, axes = plt.subplots(len(names), len(arms), figsize=(14, 3.2 * len(names) + 1), sharex=True,
                                 sharey="row", layout="constrained", squeeze=False)
        for row_axes, name in zip(axes, names):
            low, high = control_range[name]
            for ax, arm in zip(row_axes, arms):
                ax.fill_between(middle, low, high, color="0.8", step="mid", label="controls' range")
                for start in STARTS:
                    thick = (arm, start) in flagged
                    ax.plot(middle, block_means(arm, start, name), color=ARMS[arm], marker="o", markersize=3,
                            linewidth=2 if thick else 0.6, alpha=1 if thick else 0.5,
                            label=f"start {start}" if thick else None)
                ax.set_title(f"GSOL0 {arm}")
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.legend(fontsize=8)
            row_axes[0].set_ylabel(f"{name} [{UNITS[name]}]\nblock mean")
        for ax in axes[-1]:
            ax.set_xlabel("years since the 1233.75 step (block middle)")
        return fig

    check_figure() if control_range else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Response: the grey runs, from the stationary 1233.75 parent

    A response needs a stationary starting state. The grey runs are the
    1233.75 − 1.5 members, stepped from the stationary 1233.75 parent, so
    theirs is a response; the lowered arms start from the grey runs, which
    have not settled, so none is computed for them. Signed as the response to
    cooling, per W m⁻² (`MODEL.md` §2):

    - cooling side (x₋ − ⟨x₀⟩)/1.5 to the end of the grey runs, with ⟨x₀⟩
      the 1233.75 parent's mean over its window;
    - centred (x₋ − x₊)/3 to 2000 years, where the + members (GSOL0 1235.25)
      stop.

    Dashed: the parents' slope (⟨x⟩(1232.5) − ⟨x⟩(1233.75))/1.25, where the
    response would end if the sensitivity were the same down to 1232.25; a
    reference only. Thin lines: each start of the cooling side; thick: means
    over the starts that reach each year (the table gives how many). Annual
    values.
    """)
    return


@app.cell
def _(mo):
    response_time = mo.ui.radio(options=["linear", "log"], value="log", label="time axis", inline=True)
    response_time
    return (response_time,)


@app.cell
def _(ARMS, STARTS, UNITS, data, indicator, np, on_years, parents, plt, plus, response_time):
    def grey_cooling(name):
        """Cooling side of the grey runs, (start, year since the step) from year 1 to the longest grey run."""
        last = max(int(data["1232.25"][s][0][-1]) for s in STARTS)
        return (on_years("1232.25", name, 1, last) - parents["1233.75"][name]) / 1.5

    def response_figure(name, scale):
        """Grey runs: cooling side to their end and centred response to 2000 years, with the parents' slope."""
        slope = (parents["1232.5"][name] - parents["1233.75"][name]) / 1.25
        cooling = grey_cooling(name)
        centred = np.array([(data["1232.25"][s][1][name][:2000] - plus[s][name][:2000]) / 3.0 for s in STARTS])
        t = np.arange(1, cooling.shape[1] + 1)
        fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
        for row in cooling:
            ax.plot(t, row, color=ARMS["1232.25"], linewidth=0.3, alpha=0.4)
        ax.plot(t, np.nanmean(cooling, axis=0), color=ARMS["1232.25"], linewidth=2,
                label="cooling side (x$_-$ - <x$_0$>)/1.5")
        ax.plot(t[:2000], centred.mean(axis=0), color="#1baf7a", linewidth=2, label="centred (x$_-$ - x$_+$)/3")
        ax.axhline(slope, color="k", linestyle="--", linewidth=1.2, label="parents' slope 1233.75 to 1232.5")
        ax.axhline(0.0, color="0.5", linewidth=0.7)
        ax.set_xscale(scale)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=9)
        ax.set_title("grey runs: step 1233.75 to 1232.25 (centred: 1233.75 $\\pm$ 1.5)")
        ax.set_xlabel("years since the 1233.75 step")
        ax.set_ylabel(f"{name}\n[{UNITS[name]} per W m$^{{-2}}$]")
        return fig

    response_figure(indicator.value, response_time.value)
    return (grey_cooling,)


@app.cell
def _(STARTS, UNITS, data, grey_cooling, mo, np, parents, plus):
    STRETCHES = [(951, 1050), (1901, 2000), (2901, 3000), (3901, 4000), (4901, 5000)]

    def stretch_mean(table, first, last):
        """Mean over the starts that cover the whole stretch of years, and how many they are."""
        if table.shape[1] < last:
            return np.nan, 0
        full = np.isfinite(table[:, first - 1:last]).all(axis=1)
        return (table[full, first - 1:last].mean() if full.any() else np.nan), int(full.sum())

    def response_table():
        """Means over starts of the grey runs' responses over stretches of years, against the parents' slope."""
        heads = [f"cooling {a}–{b}" for a, b in STRETCHES] + ["centred 951–1050", "centred 1901–2000"]
        lines = ["| Indicator | parents' slope | " + " | ".join(heads) + " |", "|---" * (len(heads) + 2) + "|"]
        counts = []
        for name in UNITS:
            slope = (parents["1232.5"][name] - parents["1233.75"][name]) / 1.25
            cooling = grey_cooling(name)
            centred = np.array([(data["1232.25"][s][1][name][:2000] - plus[s][name][:2000]) / 3.0 for s in STARTS])
            cells = [f"{slope:+.3g}"]
            counts = []
            for a, b in STRETCHES:
                value, n = stretch_mean(cooling, a, b)
                cells.append(f"{value:+.3g}" if n else "–")
                counts.append(n)
            cells += [f"{centred[:, a - 1:b].mean():+.3g}" for a, b in ((951, 1050), (1901, 2000))]
            lines.append(f"| {name} [{UNITS[name]}] | " + " | ".join(cells) + " |")
        reach = ", ".join(f"{a}–{b}: {n}" for (a, b), n in zip(STRETCHES, counts))
        return "\n".join(lines) + f"\n\nGrey runs covering each cooling stretch: {reach}."

    mo.md("**Responses per W m⁻², means over the starts**\n\n" + response_table())
    return


if __name__ == "__main__":
    app.run()
