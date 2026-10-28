"""PlaSim-LSG exploratory analysis: phase diagram of all runs, and the indicators of one run."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from gsebm.paths import get_data_dir
    from gsebm.plasim_indicators import (
        area_mean, area_total, load_indicators, product_stem, read_manifest, volume_mean,
    )

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
        read_manifest,
        volume_mean,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PlaSim-LSG exploratory analysis

    1. **One run:** annual indicators of one run, to find stationary windows
       by eye.
    2. **Phase diagram:** time-mean global `ts` of the runs against GSOL0.

    Definitions are in `Plasim_Indicators.md` §4; the stationary windows and
    the reasons for them in `PLASIM_INFO/DATA.md` §4 and
    `PLASIM_INFO/STATIONARITY.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One run

    Annual values of one run, one figure per group of indicators, to find by
    eye a window in which the run looks stationary. No smoothing or
    detrending. Definitions are in `Plasim_Indicators.md` §4. The time axis
    is the **year since the start of the run** (0 = first year of output);
    the model years and file-name labels are given in the header.

    The slider sets a candidate window: every figure shows only the years
    inside it, with the y axes scaled to those years. Nothing is computed
    from the window.
    """)
    return


@app.cell
def _(get_data_dir, read_manifest):
    INDICATOR_DIR = get_data_dir(create=False) / "Plasim" / "indicators"

    def short_name(run):
        """Run name without the common prefix: '1232p5_cold', or 'base'."""
        return run.split("_MU_")[1] if "_MU_" in run else "base"

    def run_manifest(run_dir):
        """Manifest of a run folder, whichever phase it holds."""
        for phase in ("spinup", "stationary"):
            manifest = read_manifest(run_dir, run_dir.name, phase)
            if manifest is not None:
                return manifest
        raise FileNotFoundError(f"no indicator manifest in {run_dir}")

    manifests = {path.name: run_manifest(path) for path in INDICATOR_DIR.iterdir() if path.is_dir()}
    manifests = dict(sorted(manifests.items(), key=lambda item: (item[1]["gsol0"], item[0])))
    return INDICATOR_DIR, manifests, short_name


@app.cell
def _(manifests, mo, short_name):
    chosen = mo.ui.dropdown(
        options={f"{short_name(run)}  (GSOL0 {manifest['gsol0']:g})": run
                 for run, manifest in manifests.items()},
        value="1240  (GSOL0 1240)",
        label="run",
    )
    chosen
    return (chosen,)


@app.cell
def _(INDICATOR_DIR, chosen, load_indicators, manifests, product_stem):
    run = chosen.value
    manifest = manifests[run]
    _stem = product_stem(run, manifest["phase"])
    zonal = load_indicators(INDICATOR_DIR / run / f"{_stem}_zonal.nc")
    sections = load_indicators(INDICATOR_DIR / run / f"{_stem}_sections.nc")
    first_year, last_year = int(zonal.model_year[0]), int(zonal.model_year[-1])

    # Time axis of the plots: years since the run's first year of output
    def _from_start(dataset):
        run_year = dataset["model_year"].values - first_year
        return dataset.assign_coords(run_year=("year", run_year)).swap_dims(year="run_year")

    zonal, sections = _from_start(zonal), _from_start(sections)
    return first_year, last_year, manifest, run, sections, zonal


@app.cell
def _(first_year, last_year, mo):
    window = mo.ui.range_slider(
        start=0, stop=last_year - first_year, step=10,
        value=[0, last_year - first_year],
        label="candidate window (years since start)", show_value=True, full_width=True,
    )
    window
    return (window,)


@app.cell
def _(first_year, last_year, manifest, mo, run, short_name, window, zonal):
    _offset = int(zonal.year[0]) - first_year
    mo.md(
        f"**{short_name(run)}**, GSOL0 {manifest['gsol0']:g} W m$^{{-2}}$, phase `{manifest['phase']}`: "
        f"model years {first_year}–{last_year} ({last_year - first_year + 1} years; labels "
        f"{first_year + _offset}–{last_year + _offset}). "
        f"Window: years {window.value[0]}–{window.value[1]} since the start "
        f"(model years {first_year + window.value[0]}–{first_year + window.value[1]})."
    )
    return


@app.cell
def _(plt, window):
    # Same colour for a hemisphere in every figure; other lines take C0, C1, ...
    COLOURS = {"NH": "C3", "SH": "C0"}

    def group_figure(title, panels):
        """One column of panels sharing the time axis (years since the start).

        ``panels`` is a list of (y label, [(line label, series), ...]); each
        series is an annual DataArray on ``run_year``.
        """
        fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.7 * len(panels)), sharex=True,
                                 layout="constrained", squeeze=False)
        for ax, (ylabel, lines) in zip(axes[:, 0], panels):
            for number, (label, values) in enumerate(lines):
                colour = COLOURS.get(label, f"C{number}")
                values = values.sel(run_year=slice(*window.value))
                ax.plot(values.run_year, values, color=colour, linewidth=0.7, label=label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if len(lines) > 1:
                legend = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
                for handle in legend.get_lines():
                    handle.set_linewidth(2.5)
        axes[-1, 0].set_xlabel("years since start of run")
        fig.suptitle(title)
        return fig

    return (group_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Surface climate

    Area means with Gaussian weights; `tas` and `ts` over land and ocean,
    `sst` (the 50 m slab) over ocean only.
    """)
    return


@app.cell
def _(area_mean, group_figure, zonal):
    def _surface(region):
        return [("tas", area_mean(zonal["tas_zonal"], zonal, region)),
                ("ts", area_mean(zonal["ts_zonal"], zonal, region))]

    group_figure("Surface climate", [
        ("global (K)", _surface("global")),
        ("NH (K)", _surface("nh")),
        ("SH (K)", _surface("sh")),
        ("global sst (K)", [("sst", area_mean(zonal["sst_ocean_zonal"], zonal, ocean_only=True))]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Energy balance

    Global means, positive down (into the climate system or the ocean):
    net TOA flux N = `rst` + `rlut`; surface net flux = `rss` + `rls` +
    `hfss` + `hfls`; heat flux into LSG's top layer (`flukhea`, mean over
    LSG's wet surface). Planetary albedo = −mean(`rsut`) / mean(`rst` − `rsut`).
    The dashed lines mark zero.
    """)
    return


@app.cell
def _(area_mean, group_figure, zonal):
    _incoming = area_mean(zonal["rst_zonal"] - zonal["rsut_zonal"], zonal)
    _wet_area = zonal["lsg_row_wet_area"]
    _flukhea = (zonal["flukhea_zonal"] * _wet_area).sum("lsg_lat") / _wet_area.sum()
    _fig = group_figure("Energy balance", [
        ("net TOA (W m$^{-2}$)", [("N", area_mean(zonal["rst_zonal"] + zonal["rlut_zonal"], zonal))]),
        ("surface net (W m$^{-2}$)", [("surface", area_mean(zonal["surface_net_flux_zonal"], zonal))]),
        ("into LSG (W m$^{-2}$)", [("flukhea", _flukhea)]),
        ("planetary albedo", [("albedo", -area_mean(zonal["rsut_zonal"], zonal) / _incoming)]),
    ])
    for _ax in _fig.axes[:3]:
        _ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Sea ice

    Sea-ice area Σ A_j f_j `sic`_j and ice volume Σ A_j f_j `sit`_j per
    hemisphere, over T21 ocean points (`sic` is the fraction of the year a
    point is ice-covered).
    """)
    return


@app.cell
def _(area_total, group_figure, zonal):
    group_figure("Sea ice", [
        ("area (10$^{6}$ km$^2$)", [
            ("NH", area_total(zonal["sic_ocean_zonal"], zonal, "nh") / 1e12),
            ("SH", area_total(zonal["sic_ocean_zonal"], zonal, "sh") / 1e12),
        ]),
        ("volume (10$^{3}$ km$^3$)", [
            ("NH", area_total(zonal["sit_ocean_zonal"], zonal, "nh") / 1e12),
            ("SH", area_total(zonal["sit_ocean_zonal"], zonal, "sh") / 1e12),
        ]),
    ])
    return


@app.cell
def _():
    # Depth bands as whole LSG layers (Plasim_Indicators.md §4), by level position
    BANDS = {
        "0–700 m": slice(0, 10),
        "700–2025 m": slice(10, 17),
        "2025 m–bottom": slice(17, 22),
    }
    return (BANDS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Ocean temperature by depth

    Volume means of LSG `t` over whole layers, global; for the deep band also
    per hemisphere.
    """)
    return


@app.cell
def _(BANDS, group_figure, sections, volume_mean):
    _t = sections["t_section"]
    group_figure("Ocean temperature", [
        (f"{_band} (K)", [("global", volume_mean(_t, sections, _levels))])
        for _band, _levels in BANDS.items()
    ] + [
        ("2025 m–bottom (K)", [
            ("NH", volume_mean(_t, sections, BANDS["2025 m–bottom"], "nh")),
            ("SH", volume_mean(_t, sections, BANDS["2025 m–bottom"], "sh")),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Salt and sea level

    Volume-mean salinity per band, global; Atlantic salinity 0–700 m
    (Börner et al. Atlantic mask); global mean sea level (`zeta` over LSG's
    wet surface). Total salt content is not plotted: it is a conservation
    check, recorded per run in the manifest (`salt_content_relative_range`).
    """)
    return


@app.cell
def _(BANDS, group_figure, sections, volume_mean, zonal):
    _s = sections["s_section"]
    _wet_area = zonal["lsg_row_wet_area"]
    group_figure("Salt and sea level", [
        *[(f"{_band} (‰)", [("global", volume_mean(_s, sections, _levels))]) for _band, _levels in BANDS.items()],
        ("Atlantic 0–700 m (‰)", [("Atlantic", volume_mean(sections["s_atlantic_section"], sections,
                                                          BANDS["0–700 m"], atlantic=True))]),
        ("sea level (m)", [("zeta", (zonal["zeta_zonal"] * _wet_area).sum("lsg_lat") / _wet_area.sum())]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 Ocean circulation

    AMOC strength (Börner et al. 2026): maximum of `psi_atlantic` over all
    depths and the rows in 46–66 °N, and over depth at the row nearest
    26 °N. When the AMOC is off the maximum can sit at depth, in a weak deep
    cell. ACC: Drake Passage transport from the barotropic streamfunction.
    """)
    return


@app.cell
def _(group_figure, np, sections, zonal):
    _psi = sections["psi_atlantic"]
    _lat = sections["lsg_lat"]
    _row_26 = int(np.argmin(np.abs(_lat.values - 26.0)))
    group_figure("Ocean circulation", [
        ("AMOC (Sv)", [
            ("46–66 °N", _psi.where((_lat >= 46) & (_lat <= 66)).max(("lsg_level", "lsg_lat"))),
            (f"{float(_lat[_row_26]):.2f} °N", _psi.isel(lsg_lat=_row_26).max("lsg_level")),
        ]),
        ("ACC (Sv)", [("ACC", zonal["acc"])]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Phase diagram

    Time mean of global `ts` (Gaussian-weighted area mean of the annual zonal
    mean, `area_mean`) against GSOL0. Plain averages, no uncertainties.
    Global `tas` looks the same and is in the table.

    - **Filled markers:** runs with an agreed stationary window. In 1250 and
      1312 the window spans whole cycles of a Southern Ocean oscillation; in
      1336 it holds for global tas/ts only.
    - **Faded open markers:** runs without a window (still drifting, too
      short, or snowball), averaged over their last `LAST_YEARS` years. The
      last 500 years of 1227 still contain the end of its collapse.
    - **Lines** join the runs of the warm branch, the cold branch and the
      snowball; the runs between branches (1231p5, 1230p5, 1227) have
      markers only. The branches are a working reading of the runs
      (`STATIONARITY.md`), not established. Left out: the check 1235_new_IC
      (converging to the 1235 state) and 1231p5_bis, 1230p5_bis (repeats of
      1231p5 and 1230p5 from another start).

    Windows are in years since the run's first year of output, end inclusive.
    """)
    return


@app.cell
def _():
    # Agreed stationary windows (years since start, end inclusive), DATA.md §4
    WINDOWS = {
        "base": (6000, 9999),
        "1336": (3000, 4999),
        "1312": (1250, 8749),
        "1288": (2000, 9999),
        "1265": (1000, 9999),
        "1250": (1640, 3999),
        "1245": (3000, 10499),
        "1242p5": (6000, 12499),
        "1240": (6000, 12379),
        "1237p5": (3000, 9999),
        "1235": (5000, 12999),
        "1233p75": (5000, 14999),
        "1232p5": (6000, 14999),
        "1230": (8500, 12499),
    }
    # Runs without a window: mean over their last LAST_YEARS years
    LAST_YEARS = 500
    # Runs joined by a line in the phase diagram (working reading, STATIONARITY.md)
    BRANCHES = {
        "warm": ["base", "1336", "1312", "1288", "1265", "1250", "1245", "1242p5", "1240",
                 "1237p5", "1235", "1233p75", "1232p5"],
        "cold": ["1235_cold", "1232p5_cold", "1230", "1228p5"],
        "snowball": ["1226", "1225", "1216", "1192", "1168"],
    }
    BRANCH_COLOURS = {"warm": "C3", "cold": "C0", "snowball": "C7"}
    # Checks left out of the phase diagram
    LEFT_OUT = {"1235_new_IC", "1231p5_bis", "1230p5_bis"}
    return BRANCHES, BRANCH_COLOURS, LAST_YEARS, LEFT_OUT, WINDOWS


@app.cell
def _(
    INDICATOR_DIR,
    LAST_YEARS,
    LEFT_OUT,
    WINDOWS,
    area_mean,
    load_indicators,
    manifests,
    product_stem,
    short_name,
):
    def time_means(run, manifest):
        """Window, and time-mean global tas and ts, of one run."""
        zonal = load_indicators(INDICATOR_DIR / run / f"{product_stem(run, manifest['phase'])}_zonal.nc")
        length = zonal.sizes["year"]
        stationary = short_name(run) in WINDOWS
        first, last = WINDOWS[short_name(run)] if stationary else (length - LAST_YEARS, length - 1)
        window = zonal.isel(year=slice(first, last + 1))
        return {
            "run": short_name(run),
            "GSOL0": float(manifest["gsol0"]),
            "stationary": stationary,
            "window": f"{first}–{last}",
            "tas": float(area_mean(window["tas_zonal"], window).mean()),
            "ts": float(area_mean(window["ts_zonal"], window).mean()),
        }

    means = {short_name(run): time_means(run, manifest) for run, manifest in manifests.items()
             if short_name(run) not in LEFT_OUT}
    return (means,)


@app.cell
def _(BRANCHES, BRANCH_COLOURS, means, plt):
    def draw(ax, labels=False):
        """Branch lines and one marker per run: filled if stationary, faded open if not."""
        for branch, runs in BRANCHES.items():
            points = sorted((means[run] for run in runs), key=lambda row: row["GSOL0"])
            ax.plot([row["GSOL0"] for row in points], [row["ts"] for row in points],
                    color=BRANCH_COLOURS[branch], linewidth=1.2, label=f"{branch} branch")
        colour_of = {run: BRANCH_COLOURS[branch] for branch, runs in BRANCHES.items() for run in runs}
        for row in means.values():
            colour = colour_of.get(row["run"], "0.4")
            style = (dict(color=colour) if row["stationary"]
                     else dict(color=colour, markerfacecolor="none", alpha=0.5))
            ax.plot(row["GSOL0"], row["ts"], marker="o", markersize=7, linestyle="none", **style)
            if labels:
                ax.annotate(row["run"], (row["GSOL0"], row["ts"]), xytext=(4, 2),
                            textcoords="offset points", fontsize=7, color=colour)

    _fig, _ax = plt.subplots(figsize=(11, 7), layout="constrained")
    draw(_ax)
    _ax.plot([], [], marker="o", color="0.3", linestyle="none", label="stationary window")
    _ax.plot([], [], marker="o", color="0.3", markerfacecolor="none", alpha=0.5, linestyle="none",
             label="no window (last years)")
    _ax.set_xlabel("GSOL0 (W m$^{-2}$)")
    _ax.set_ylabel("global ts (K)")
    _ax.grid(alpha=0.3)
    _ax.legend(loc="upper left", frameon=False)

    _inset = _ax.inset_axes([0.56, 0.07, 0.41, 0.4])
    draw(_inset, labels=True)
    _inset.set_xlim(1225.5, 1251.5)
    _inset.set_ylim(251.5, 266.5)
    _inset.tick_params(labelsize=8)
    _inset.grid(alpha=0.3)
    _ax.indicate_inset_zoom(_inset, edgecolor="0.5")
    _fig
    return


if __name__ == "__main__":
    app.run()
