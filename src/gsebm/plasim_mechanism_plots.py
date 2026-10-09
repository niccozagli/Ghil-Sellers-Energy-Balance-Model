"""Figures for the South Atlantic mechanism notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gsebm.plasim_mechanism import CycleSummary, mu_value, phase_composite, wrapped_longitude


def _colors(count: int) -> np.ndarray:
    return plt.cm.viridis_r(np.linspace(0.1, 0.9, count))


def ice_edge_maps(summaries: dict[str, CycleSummary], mus: tuple[str, ...]) -> plt.Figure:
    """Window-mean sea-ice concentration around the South Atlantic."""
    figure, axes = plt.subplots(1, len(mus), figsize=(4.2 * len(mus), 3.6), sharey=True,
                                constrained_layout=True)
    for axis, mu in zip(np.atleast_1d(axes), mus):
        series = summaries[mu].series
        lon = wrapped_longitude(series.t21_lon)
        order = np.argsort(lon)
        lat = series.t21_lat
        mesh = axis.pcolormesh(lon[order], lat, series.mean_concentration[:, order],
                               cmap="Blues", vmin=0, vmax=1, shading="nearest")
        axis.contour(lon[order], lat, series.south_atlantic_mask[:, order].astype(float),
                     levels=[0.5], colors="crimson", linewidths=1)
        axis.set(xlim=(-90, 40), ylim=(-62, -10), title=f"μ = {mu_value(mu):g}",
                 xlabel="longitude (°E)")
        axis.set_yticks(lat[(lat > -62) & (lat < -10)])
        axis.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("T21 row latitude (°)")
    figure.colorbar(mesh, ax=axes, label="mean sea-ice concentration")
    return figure


def ice_area_excerpts(summaries: dict[str, CycleSummary], years: int = 400) -> plt.Figure:
    """South Atlantic ice area with the detected maxima, first `years` of each window."""
    figure, axes = plt.subplots(len(summaries), 1, figsize=(10, 1.6 * len(summaries)),
                                sharey=True, constrained_layout=True)
    for axis, (mu, summary), color in zip(axes, summaries.items(), _colors(len(summaries))):
        cycle = summary.cycle
        span = slice(0, years)
        axis.plot(cycle.years[span], cycle.detrended_area[span], color="0.75", lw=0.7)
        axis.plot(cycle.years[span], cycle.smoothed_area[span], color=color, lw=1.6)
        peaks = cycle.peak_indices[cycle.peak_indices < years]
        axis.plot(cycle.years[peaks], cycle.smoothed_area[peaks], "kv", ms=5)
        axis.set_ylabel(f"{mu_value(mu):g}")
        axis.grid(alpha=0.3)
    axes[0].set_title("South Atlantic ice-area anomaly (10¹² m²): annual (grey), "
                      "11-yr mean (colour), maxima (▼)")
    axes[-1].set_xlabel("model year")
    return figure


def cycle_shapes(summaries: dict[str, CycleSummary], bins: int = 20) -> plt.Figure:
    """Cycle-length distributions and the composite ice-area cycle."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    colors = _colors(len(summaries))
    for (mu, summary), color in zip(summaries.items(), colors):
        label = f"{mu_value(mu):g}"
        lengths = summary.cycle.cycle_lengths
        axes[0].hist(lengths, bins=np.arange(25, 111, 4), histtype="step", lw=1.6,
                     color=color, label=label)
        composite = phase_composite(summary.cycle.detrended_area, summary.phase, bins)
        centres = (np.arange(bins) + 0.5) / bins
        axes[1].plot(centres, composite, color=color, lw=1.8, label=label)
        axes[2].plot(centres * np.median(lengths), composite, color=color, lw=1.8, label=label)
    axes[0].set(xlabel="years between ice maxima", ylabel="cycles", title="Cycle lengths")
    axes[1].set(xlabel="phase (0 = ice maximum)", ylabel="10¹² m²",
                title="Composite SA ice-area anomaly")
    axes[2].set(xlabel="years after maximum (phase × median length)",
                title="Same composite in years")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].legend(title="μ")
    return figure


_GROUP_STYLE = {
    "1367 at 1999": dict(marker="D", color="C2", mfc="none", label="abrupt drop from 1367"),
    "1265 at 4499": dict(marker="o", color="k", label="abrupt drop from 1265"),
    "1240 at 14999": dict(marker="s", color="C0", mfc="none", label="continued from 1240"),
    "1230 at 14499": dict(marker="^", color="C3", mfc="none", label="continued from 1230"),
}


def bifurcation_diagram(states: dict) -> plt.Figure:
    """Window means against μ, marked by where each run started."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    seen = set()
    for state in states.values():
        style = dict(_GROUP_STYLE[state.initial_condition])
        label = style.pop("label")
        seen_before = label in seen
        seen.add(label)
        label = None if seen_before else label
        axes[0].plot(state.mu, state.global_temperature, ls="none", ms=7, label=label, **style)
        axes[1].plot(state.mu, np.sin(np.radians(state.south_edge)), ls="none", ms=7, **style)
        axes[1].plot(state.mu, np.sin(np.radians(state.north_edge)), ls="none", ms=5,
                     alpha=0.5, **style)
        axes[2].plot(state.mu, state.planetary_albedo, ls="none", ms=7, **style)
    axes[0].set(ylabel="global mean Ts (K)", title="Global surface temperature")
    axes[1].set(ylabel="sin(ice-edge latitude)",
                title="Ice edge, as in the EBM (large: S, small: N)")
    axes[2].set(ylabel="planetary albedo", title="Planetary albedo")
    for axis in axes:
        axis.set_xlabel("μ = GSOL0 (W m⁻²)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    return figure


def sensitivity_steps(steps: list[dict]) -> plt.Figure:
    """Temperature sensitivity, albedo gain, and longwave response per μ step."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    centre = np.array([0.5 * (s["from μ"] + s["to μ"]) for s in steps])
    for axis, key, title in zip(
        axes,
        ("ΔT per W m⁻² of μ (K)", "gain", "λ = ΔOLR/ΔT (W m⁻² K⁻¹)"),
        ("Warming per W m⁻² of μ", "Albedo amplification (ΔASR / direct)",
         "Longwave response λ"),
    ):
        values = np.array([s[key] for s in steps])
        axis.plot(centre, values, "ko-")
        axis.set(xlabel="μ (step midpoint)", title=title, yscale="log" if key != "λ = ΔOLR/ΔT (W m⁻² K⁻¹)" else "linear")
        axis.grid(alpha=0.3, which="both")
    return figure


def zonal_change_profiles(states: dict, pairs: tuple[tuple[str, str], ...]) -> plt.Figure:
    """Zonal climate of selected runs and the change per W m⁻² of μ between pairs."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    show = [label for label in ("1245", "1235", "1232p5", "1230", "1225") if label in states]
    for label, color in zip(show, _colors(len(show))):
        state = states[label]
        x = np.sin(np.radians(state.lat))
        axes[0, 0].plot(x, state.zonal_temperature, color=color, label=f"{state.mu:g}")
        axes[0, 1].plot(x, state.zonal_ocean_ice, color=color)
        faces = np.sin(np.radians(0.5 * (state.lat[1:] + state.lat[:-1])))
        axes[0, 2].plot(faces, state.total_transport[:-1], color=color)
        lsg_faces = np.sin(np.radians(state.lsg_lat + 1.25))
        axes[0, 2].plot(lsg_faces, state.ocean_transport, color=color, ls="--")
    axes[0, 0].set(ylabel="Ts (K)", title="Zonal surface temperature")
    axes[0, 0].legend(title="μ", fontsize=8)
    axes[0, 1].set(ylabel="ocean ice concentration", title="Zonal sea ice")
    axes[0, 2].set(ylabel="PW", title="Northward transport: total (solid), ocean (dashed)")
    for (warm, cold), style in zip(pairs, ("-", "--", ":")):
        a, b = states[warm], states[cold]
        dmu = b.mu - a.mu
        x = np.sin(np.radians(a.lat))
        name = f"{a.mu:g}→{b.mu:g}"
        axes[1, 0].plot(x, (b.zonal_temperature - a.zonal_temperature) / dmu, "k", ls=style,
                        label=name)
        axes[1, 1].plot(x, (b.zonal_absorbed - a.zonal_absorbed) / dmu, "C1", ls=style)
        axes[1, 1].plot(x, (b.zonal_outgoing - a.zonal_outgoing) / dmu, "C0", ls=style)
        axes[1, 2].plot(x, ((b.zonal_absorbed - b.zonal_outgoing)
                            - (a.zonal_absorbed - a.zonal_outgoing)) / dmu, "C2", ls=style)
    axes[1, 0].set(ylabel="K per W m⁻²", title="ΔTs per W m⁻² of μ")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(ylabel="W m⁻² per W m⁻²", title="Δ absorbed sunlight (orange), ΔOLR (blue)")
    axes[1, 2].set(ylabel="W m⁻² per W m⁻²", title="Δ net heating = −Δ transport divergence")
    for axis in axes.flat:
        axis.set_xlabel("sin(latitude)")
        axis.grid(alpha=0.3)
    return figure


def transition_overview(result, years: tuple[int, int], title: str) -> plt.Figure:
    """Annual transition indices of one run over a year range (no smoothing)."""
    select = (result.years >= years[0]) & (result.years <= years[1])
    y = result.years[select]
    panels = (
        (("global Ts",), "K"),
        (("S edge atlantic", "S edge indian", "S edge pacific"), "° (S)"),
        (("N edge atlantic", "N edge pacific"), "° (N)"),
        (("tropical θ 0–700 m", "deep θ >1000 m"), "K"),
        (("ocean transport 20°S", "ocean transport 35°S", "ocean transport 35°N"), "PW"),
    )
    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 12), sharex=True,
                                constrained_layout=True)
    for axis, (names, unit) in zip(axes, panels):
        twin = None
        for k, name in enumerate(names):
            values = result.indices[name][select]
            target = axis
            if name == "deep θ >1000 m":
                twin = axis.twinx()
                target = twin
            target.plot(y, values, color=f"C{k}", lw=0.6, label=name)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.3)
        handles = axis.get_legend_handles_labels()
        if twin is not None:
            extra = twin.get_legend_handles_labels()
            handles = (handles[0] + extra[0], handles[1] + extra[1])
        axis.legend(*handles, fontsize=8, loc="best")
    axes[0].set_title(f"{title} (annual values)")
    axes[-1].set_xlabel("model year")
    return figure


def onset_comparison(result, years: tuple[int, int], reference: tuple[int, int],
                     onsets: dict[str, int | None], title: str) -> plt.Figure:
    """Annual departures from the pre-transition linear trend, scaled to ±1.

    The trend is fitted to the annual values in `reference`. Onset years in
    the legend come from `plasim_transitions.onset_year` (see its docstring).
    """
    select = (result.years >= years[0]) & (result.years <= years[1])
    ref = (result.years >= reference[0]) & (result.years <= reference[1])
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, onset in sorted(onsets.items(), key=lambda item: item[1] or 1e9):
        values = result.indices[name]
        trend = np.polyfit(result.years[ref], values[ref], 1)
        departure = (values - np.polyval(trend, result.years))[select]
        scale = np.nanmax(np.abs(departure)) or 1.0
        axis.plot(result.years[select], departure / scale, lw=0.6, label=f"{name} ({onset})")
    axis.axvspan(*reference, color="0.9", zorder=0)
    axis.set(xlabel="model year", ylabel="departure from pre-transition trend (scaled)",
             title=f"{title}: onset years in brackets; grey = reference period")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.3)
    return figure


def transport_and_edges(profiles: list) -> plt.Figure:
    """Ocean heat transport and uptake profiles with the ice edges marked."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for profile, color in zip(profiles, _colors(len(profiles))):
        axes[0].plot(profile.faces, profile.transport, color=color, label=profile.name)
        for edge in (-profile.south_edge, profile.north_edge):
            axes[0].axvline(edge, color=color, ls=":", lw=1)
        axes[1].plot(profile.lsg_lat, profile.uptake, color=color)
    axes[0].set(xlabel="latitude", ylabel="PW",
                title="Northward ocean heat transport (dotted: ice edges)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="latitude", ylabel="W m⁻² into the ocean", xlim=(-60, 60),
                title="Coupling heat flux (negative: ocean releases heat)")
    for axis in axes:
        axis.grid(alpha=0.3)
    return figure


def _run_colors(labels) -> dict:
    from gsebm.plasim_global import run_mu

    mus = np.array([run_mu(label) for label in labels])
    scaled = (mus - mus.min()) / max(np.ptp(mus), 1e-9)
    return {label: plt.cm.viridis(value) for label, value in zip(labels, scaled)}


def run_timeseries_overview(timeseries: dict) -> tuple[plt.Figure, plt.Axes]:
    """All runs at full length, annual values; the analysis window is drawn darker."""
    names = ("global Ts (K)", "S ice edge (°)", "N ice edge (°)", "deep θ >1000 m (K)")
    colors = _run_colors(list(timeseries))
    figure, axes = plt.subplots(len(names), 1, figsize=(12, 13), sharex=True,
                                constrained_layout=True)
    for label, series in timeseries.items():
        inside = (series.years >= series.window[0]) & (series.years <= series.window[1])
        for axis, name in zip(axes, names):
            values = series.indices[name]
            axis.plot(series.years, values, color=colors[label], lw=0.4, alpha=0.6)
            axis.plot(series.years[inside], values[inside], color=colors[label], lw=0.6,
                      label=label.replace("p", ".") if name == names[0] else None)
    for axis, name in zip(axes, names):
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    axes[0].set_title("All runs, annual values; full colour = analysis window, faded = rest")
    axes[0].legend(ncol=8, fontsize=7, title="μ")
    axes[-1].set_xlabel("model year")
    return (figure , axes)


def run_timeseries_detail(series) -> plt.Figure:
    """Every index of one run, annual values, with the analysis window shaded."""
    from gsebm.plasim_global import RUNS

    names = list(series.indices)
    figure, axes = plt.subplots(len(names), 1, figsize=(12, 2.0 * len(names)), sharex=True,
                                constrained_layout=True)
    for axis, name in zip(axes, names):
        axis.plot(series.years, series.indices[name], color="k", lw=0.4)
        axis.axvspan(*series.window, color="C0", alpha=0.12, zorder=0)
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    info = RUNS[series.label]
    axes[0].set_title(
        f"μ = {series.label.replace('p', '.')}, started from {info.initial_condition}; "
        f"annual values; shaded = analysis window {info.window[0]}–{info.window[1]}"
    )
    axes[-1].set_xlabel("model year")
    return figure


def edge_histograms(rows: dict) -> plt.Figure:
    """Annual ice-edge latitudes in each run's window, with T21 row latitudes in red."""
    from gsebm.plasim_checks import annual_edges
    from gsebm.plasim_global import RUNS

    figure, axes = plt.subplots(len(rows), 2, figsize=(12, 1.8 * len(rows)),
                                constrained_layout=True)
    for (label, data), pair in zip(rows.items(), np.atleast_2d(axes)):
        start, end = RUNS[label].window
        inside = (data.years >= start) & (data.years <= end)
        row_lat = np.sort(np.abs(data.lat[data.lat < 0]))
        for axis, (hemisphere, edges) in zip(pair, annual_edges(data).items()):
            values = edges[inside]
            values = values[values < 89]
            if values.size == 0:
                axis.set_title(f"{label.replace('p', '.')} {hemisphere}: no edge", fontsize=8)
                continue
            axis.hist(values, bins=np.arange(0, 90, 0.25), color="k")
            for latitude in row_lat:
                axis.axvline(latitude, color="r", lw=0.6)
            axis.set_xlim(max(0, values.min() - 3), min(90, values.max() + 3))
            axis.set_title(f"{label.replace('p', '.')} {hemisphere} edge, annual values in window",
                           fontsize=8)
    return figure


def _window_mask(years: np.ndarray, label: str) -> np.ndarray:
    from gsebm.plasim_global import RUNS

    start, end = RUNS[label].window
    return (years >= start) & (years <= end)


def _equilibrium_overlay(axis, x_by_run: dict, y_by_run: dict, highlight=("1312", "1250")):
    """Annual values (grey; highlighted runs in colour) and window means (black)."""
    for label in x_by_run:
        if label not in highlight:
            axis.plot(x_by_run[label], y_by_run[label], ".", color="0.8", ms=1, zorder=0)
    for label, color in zip(highlight, ("C3", "C0")):
        if label in x_by_run:
            axis.plot(x_by_run[label], y_by_run[label], ".", color=color, ms=1.5, alpha=0.5,
                      label=f"{label} annual", zorder=1)
    means = np.array([[np.nanmean(x_by_run[k]), np.nanmean(y_by_run[k])] for k in x_by_run])
    order = np.argsort(means[:, 0])
    axis.plot(means[order, 0], means[order, 1], "ko-", ms=4, lw=1,
              label="window means, all runs", zorder=2)
    for label, (x, y) in zip(x_by_run, means):
        axis.annotate(label.replace("p", ".").replace("_new_IC", "*"), (x, y), fontsize=6,
                      xytext=(3, 3), textcoords="offset points")
    axis.grid(alpha=0.3)


def quasi_static_test(run_series: dict, ice_rows: dict) -> plt.Figure:
    """Southern ice, Northern ice and planetary albedo against global Ts (annual, windows)."""
    labels = [label for label in run_series if label != "1225"]
    ts, south, north, albedo = {}, {}, {}, {}
    for label in labels:
        series, rows = run_series[label], ice_rows[label]
        inside = _window_mask(series.years, label)
        ts[label] = series.indices["global Ts (K)"][inside]
        albedo[label] = series.indices["planetary albedo"][inside]
        total = sum(rows.sector_area.values())
        south[label] = total[:, rows.lat < 0].sum(axis=1)[inside]
        north[label] = total[:, rows.lat > 0].sum(axis=1)[inside]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, values, name in zip(
        axes, (south, north, albedo),
        ("Southern ice area (10¹² m²)", "Northern ice area (10¹² m²)", "planetary albedo"),
    ):
        _equilibrium_overlay(axis, ts, values)
        axis.set(xlabel="global Ts (K)", ylabel=name)
    axes[0].legend(fontsize=8, markerscale=6)
    return figure


def hemispheric_mode_test(modes: dict) -> plt.Figure:
    """ΔT_h and T₂,h against the hemispheric mean T_h, annual values in each window."""
    labels = [label for label in modes if label != "1225"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, hemisphere in enumerate(("S", "N")):
        name = "Southern" if hemisphere == "S" else "Northern"
        mean = {label: modes[label][hemisphere]["mean"] for label in labels}
        for column, (key, ylabel) in enumerate((
            ("delta", "ΔT = tropics − extratropics (K)"), ("p2", "T₂, P₂ coefficient (K)"),
        )):
            values = {label: modes[label][hemisphere][key] for label in labels}
            _equilibrium_overlay(axes[row, column], mean, values)
            axes[row, column].set(xlabel=f"{name} hemisphere mean Ts (K)", ylabel=ylabel,
                                  title=f"{name} hemisphere")
    axes[0, 0].legend(fontsize=8, markerscale=6)
    return figure


def southern_lag_figure(boxes: dict, ice: dict, max_lag: int = 60) -> plt.Figure:
    """Lagged correlation of Southern ice with ocean release and TOA net, band and cap."""
    from gsebm.plasim_southern import lagged_correlation

    labels = [label for label in boxes if label != "1225"]
    colors = _run_colors(labels)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True, sharex=True)
    for column, kind in enumerate(("band", "cap")):
        for label in labels:
            box = boxes[label][kind]
            inside = _window_mask(box.years, label)
            for row, values in enumerate((box.ocean_release, box.toa_net)):
                lags, r = lagged_correlation(ice[label][inside], values[inside], max_lag)
                axes[row, column].plot(lags, r, color=colors[label], lw=1,
                                       label=label.replace("p", "."))
        axes[0, column].set_title(f"{kind}: corr(Southern ice(t), ocean release(t + k))")
        axes[1, column].set_title(f"{kind}: corr(Southern ice(t), TOA net(t + k))")
        axes[1, column].set_xlabel("lag k (years); k > 0: the second variable follows the ice")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.axhline(0, color="k", lw=0.5)
        axis.axvline(0, color="k", lw=0.5)
    axes[0, 0].legend(ncol=5, fontsize=7)
    return figure
