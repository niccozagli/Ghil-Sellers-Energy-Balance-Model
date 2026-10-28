"""PlaSim-LSG indicators of one run, grouped by climate component, to locate stationary windows by eye."""

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
    # PlaSim-LSG indicators of one run

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
    ## 1. Surface climate

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
    ## 2. Energy balance

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
    ## 3. Sea ice

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
    ## 4. Ocean temperature by depth

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
    ## 5. Salt and sea level

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
    ## 6. Ocean circulation

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


if __name__ == "__main__":
    app.run()
