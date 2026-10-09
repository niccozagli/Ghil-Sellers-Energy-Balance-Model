"""Global equilibrium picture of the PlaSim μ runs: branches and energy budget.

Quantities are window means over each run's stationary period. They are
chosen to match the Ghil–Sellers EBM: global surface temperature, ice-edge
latitude per hemisphere, planetary albedo, and the zonal top-of-atmosphere
budget with its implied poleward heat transport.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

from gsebm.plasim_exploratory import EARTH_RADIUS_M
from gsebm.plasim_koopman_irregularity import ANALYSIS_WINDOWS
from gsebm.plasim_mechanism import EXPERIMENT_PREFIX


@dataclass(frozen=True)
class RunInfo:
    """Where a run started and the stationary window used for its means."""

    initial_condition: str
    window: tuple[int, int]


# Initial conditions confirmed by the user (2026-10-07). Every run starts
# from the state of another run at the given year and changes μ abruptly:
# "1367 at 1999" is the equilibrated present-day run, "1265 at 4499" the
# μ = 1265 run (itself started from 1367), "1240 at 14999" and
# "1230 at 14499" continue the 1240 and 1230 runs. Windows skip each run's
# adjustment after the change.
RUNS = {
    "1312": RunInfo("1367 at 1999", (4000, 11999)),
    "1288": RunInfo("1367 at 1999", (4000, 11999)),
    "1265": RunInfo("1367 at 1999", (4000, 11999)),
    "1250": RunInfo("1265 at 4499", (6500, 9429)),
    "1245": RunInfo("1265 at 4499", ANALYSIS_WINDOWS["1245"]),
    "1242p5": RunInfo("1240 at 14999", ANALYSIS_WINDOWS["1242p5"]),
    "1240": RunInfo("1265 at 4499", ANALYSIS_WINDOWS["1240"]),
    "1237p5": RunInfo("1240 at 14999", ANALYSIS_WINDOWS["1237p5"]),
    "1235": RunInfo("1265 at 4499", ANALYSIS_WINDOWS["1235"]),
    "1235_new_IC": RunInfo("1240 at 14999", (16000, 19998)),
    "1233p75": RunInfo("1240 at 14999", ANALYSIS_WINDOWS["1233p75"]),
    "1232p5": RunInfo("1240 at 14999", ANALYSIS_WINDOWS["1232p5"]),
    "1230": RunInfo("1265 at 4499", (12369, 16368)),
    "1228p5": RunInfo("1230 at 14499", (15400, 16898)),
    "1225": RunInfo("1265 at 4499", (8500, 9498)),
}
READ_BLOCK = 10


def run_mu(label: str) -> float:
    """Return the solar constant of a run label such as '1235_new_IC'."""
    return float(label.split("_")[0].replace("p", "."))


def run_archive(root: Path, label: str) -> Path:
    """Return the raw-map archive of a run label under an explicit root."""
    name = f"{EXPERIMENT_PREFIX}{label}"
    return Path(root) / name / f"{name}_spinup_raw_maps.nc"


@dataclass(frozen=True)
class GlobalSeries:
    """Annual zonal-mean fields of one run over its window (T21 rows unless noted)."""

    label: str
    years: np.ndarray
    lat: np.ndarray
    weight: np.ndarray
    surface_temperature: np.ndarray
    ocean_ice: np.ndarray
    absorbed_shortwave: np.ndarray
    insolation: np.ndarray
    outgoing_longwave: np.ndarray
    lsg_lat: np.ndarray
    lsg_row_area: np.ndarray
    ocean_heat_uptake: np.ndarray
    unreadable_years: tuple[int, ...]


def _read_zonal(dataset, start: int, stop: int, reduce_lon: bool) -> tuple[np.ndarray, list[int]]:
    """Read rows start:stop in small blocks; unreadable blocks become NaN."""
    blocks, bad = [], []
    for first in range(start, stop, READ_BLOCK):
        last = min(first + READ_BLOCK, stop)
        try:
            values = np.asarray(dataset[first:last], dtype=float)
            if reduce_lon:
                values = values.mean(axis=-1)
        except OSError:
            shape = (last - first,) + dataset.shape[1:-1 if reduce_lon else None]
            values = np.full(shape, np.nan)
            bad.append(first)
        blocks.append(values)
    return np.concatenate(blocks), bad


def valid_record_range(years: np.ndarray) -> tuple[int, int]:
    """Return the first and last model year of the leading valid records.

    Records with year 0 (unwritten blocks at the end of an archive) and any
    records after the first break in the year sequence are excluded.
    """
    years = np.asarray(years, dtype=int)
    valid = np.flatnonzero(years > 0)
    if valid.size == 0:
        raise ValueError("No valid annual records.")
    first = int(valid[0])
    breaks = np.flatnonzero(np.diff(years[first:]) != 1)
    last = first + (int(breaks[0]) if breaks.size else years.size - 1 - first)
    return int(years[first]), int(years[last])


def read_global_series(root: Path, label: str, full: bool = False) -> GlobalSeries:
    """Read the zonal-mean energy-budget fields of one run.

    By default only the stationary window in `RUNS` is read; `full=True`
    reads every valid annual record, including the initial adjustment.
    """
    archive = run_archive(root, label)
    with h5py.File(archive, "r") as source:
        years = np.asarray(source["year"][:], dtype=int)
        start, end = valid_record_range(years) if full else RUNS[label].window
        index = np.flatnonzero((years >= start) & (years <= end))
        if index.size == 0 or not np.all(np.diff(years[index]) == 1):
            raise ValueError(f"Window {start}–{end} is missing or not consecutive in {archive}")
        a, b = int(index[0]), int(index[-1]) + 1
        lat = np.asarray(source["t21_lat"][:], dtype=float)
        weight = np.asarray(source["t21_gaussian_weight"][:], dtype=float)
        ocean = np.asarray(source["lsm"][:], dtype=float) < 0.5
        fields, bad = {}, []
        for name in ("surface_temperature", "rst", "rsut", "rlut"):
            fields[name], missing = _read_zonal(source[name], a, b, reduce_lon=True)
            bad += missing
        sic, missing = _read_zonal(source["sea_ice_concentration"], a, b, reduce_lon=False)
        bad += missing
        uptake, missing = _read_zonal(source["zonal_newtonian_coupling_heat_flux"], a, b, False)
        bad += missing
        lsg_lat = np.asarray(source["lsg_lat"][:], dtype=float)
        row_area = np.asarray(source["wet_surface_area"][:], dtype=float)
    ocean_cells = ocean.sum(axis=1)
    ocean_ice = np.where(
        ocean_cells > 0, (sic * ocean).sum(axis=2) / np.maximum(ocean_cells, 1), np.nan,
    )
    return GlobalSeries(
        label=label,
        years=years[a:b],
        lat=lat,
        weight=weight / weight.sum(),
        surface_temperature=fields["surface_temperature"],
        ocean_ice=ocean_ice,
        absorbed_shortwave=fields["rst"],
        insolation=fields["rst"] - fields["rsut"],
        outgoing_longwave=-fields["rlut"],
        lsg_lat=lsg_lat,
        lsg_row_area=row_area,
        ocean_heat_uptake=uptake,
        unreadable_years=tuple(sorted({int(years[k]) for k in bad})),
    )


def slice_series(series: GlobalSeries, years: tuple[int, int]) -> GlobalSeries:
    """Return the part of a series between two model years (inclusive)."""
    keep = (series.years >= years[0]) & (series.years <= years[1])
    if not keep.any():
        raise ValueError(f"No records of {series.label} in {years[0]}–{years[1]}.")
    return replace(
        series,
        years=series.years[keep],
        surface_temperature=series.surface_temperature[keep],
        ocean_ice=series.ocean_ice[keep],
        absorbed_shortwave=series.absorbed_shortwave[keep],
        insolation=series.insolation[keep],
        outgoing_longwave=series.outgoing_longwave[keep],
        ocean_heat_uptake=series.ocean_heat_uptake[keep],
        unreadable_years=tuple(y for y in series.unreadable_years if years[0] <= y <= years[1]),
    )


def legendre_p2(x: np.ndarray) -> np.ndarray:
    """Second Legendre polynomial P₂(x) = (3x² − 1) / 2."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (3 * x**2 - 1)


def hemispheric_modes(zonal: np.ndarray, lat: np.ndarray, weight: np.ndarray,
                      south: bool) -> dict[str, np.ndarray]:
    """Hemispheric mean, tropics-minus-extratropics difference, and P₂ coefficient.

    `zonal` is (time, row) on the T21 Gaussian rows and `weight` the
    Gauss–Legendre weights (sum 2, one per hemisphere). With x = sin(lat)
    and the rows of one hemisphere:

    * mean: T_h = Σ w_j T_j / Σ w_j
    * delta: Gaussian-weighted mean over |lat| < 30° minus over |lat| > 30°
      (the two equal-area halves of the hemisphere, x = ½)
    * p2: T₂,h = 5 Σ w_j T_j P₂(|x_j|) / Σ w_j, the coefficient of P₂ in the
      hemisphere (exact for T = T₀ + T₂ P₂ up to the quadrature).
    """
    zonal = np.atleast_2d(np.asarray(zonal, dtype=float))
    lat = np.asarray(lat, dtype=float)
    weight = np.asarray(weight, dtype=float)
    rows = lat < 0 if south else lat > 0
    w = weight[rows] / weight[rows].sum()
    values = zonal[:, rows]
    x = np.abs(np.sin(np.radians(lat[rows])))
    tropics = np.abs(lat[rows]) < 30
    mean = values @ w
    delta = (values[:, tropics] @ w[tropics]) / w[tropics].sum() - (
        values[:, ~tropics] @ w[~tropics]) / w[~tropics].sum()
    return {"mean": mean, "delta": delta, "p2": 5 * values @ (w * legendre_p2(x))}


def ice_edge_latitude(ocean_ice: np.ndarray, lat: np.ndarray, south: bool,
                      threshold: float = 0.5) -> np.ndarray:
    """Return the absolute latitude where zonal ocean ice first reaches `threshold`.

    Rows are scanned from the equator poleward and the crossing is linearly
    interpolated between rows. Ice at the first row gives the latitude of that
    row, never 0, so a snowball reads as the equatorward-most row.
    """
    ocean_ice = np.atleast_2d(np.asarray(ocean_ice, dtype=float))
    lat = np.asarray(lat, dtype=float)
    rows = np.flatnonzero(lat < 0) if south else np.flatnonzero(lat > 0)
    rows = rows[np.argsort(np.abs(lat[rows]))]
    absolute = np.abs(lat[rows])
    edges = np.full(ocean_ice.shape[0], 90.0)
    for t, profile in enumerate(np.nan_to_num(ocean_ice[:, rows], nan=0.0)):
        covered = np.flatnonzero(profile >= threshold)
        if covered.size == 0:
            continue
        k = covered[0]
        if k == 0:
            edges[t] = absolute[0]
            continue
        fraction = (threshold - profile[k - 1]) / (profile[k] - profile[k - 1])
        edges[t] = absolute[k - 1] + fraction * (absolute[k] - absolute[k - 1])
    return edges


def ocean_northward_transport(uptake: np.ndarray, row_area: np.ndarray) -> np.ndarray:
    """Return the northward ocean transport (PW) at the north face of each row.

    `uptake` is the surface heat flux into the ocean (W m⁻²) on rows ordered
    south to north, `row_area` their wet areas (m²). The area-weighted mean
    is removed so that the transport vanishes at the northern boundary.
    """
    flux = np.asarray(uptake, dtype=float) * np.asarray(row_area, dtype=float)
    flux -= flux.sum() * np.asarray(row_area, dtype=float) / np.sum(row_area)
    return np.cumsum(flux) / 1e15


def implied_northward_transport(net_down: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return the northward energy transport (PW) at the faces between rows.

    `net_down` is a zonal-mean net downward flux (W m⁻²) on rows ordered from
    north to south (the T21 order) or south to north; the global-mean
    imbalance is removed first so that the transport vanishes at both poles.
    Rows are integrated from the southernmost row northward.
    """
    net_down = np.asarray(net_down, dtype=float)
    weight = np.asarray(weight, dtype=float) / np.sum(weight)
    area = 4 * np.pi * EARTH_RADIUS_M**2 * weight
    balanced = net_down - np.sum(net_down * weight)
    return np.cumsum(balanced * area) / 1e15


@dataclass(frozen=True)
class EquilibriumState:
    """Window means of one run."""

    label: str
    mu: float
    initial_condition: str
    global_temperature: float
    planetary_albedo: float
    insolation: float
    absorbed_shortwave: float
    outgoing_longwave: float
    south_edge: float
    north_edge: float
    lat: np.ndarray
    zonal_temperature: np.ndarray
    zonal_absorbed: np.ndarray
    zonal_outgoing: np.ndarray
    zonal_ocean_ice: np.ndarray
    total_transport: np.ndarray
    lsg_lat: np.ndarray
    ocean_transport: np.ndarray
    unreadable_years: tuple[int, ...]


def time_mean(field: np.ndarray) -> np.ndarray:
    """Mean over the time axis of the finite values; NaN where there are none.

    Rows without ocean (the southernmost T21 rows over Antarctica) have no
    ocean-ice value in any year and stay NaN, without a warning.
    """
    field = np.asarray(field, dtype=float)
    finite = np.isfinite(field)
    counts = finite.sum(axis=0)
    sums = np.where(finite, field, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def equilibrium_state(series: GlobalSeries) -> EquilibriumState:
    """Reduce a run's window to its equilibrium quantities (plain time means)."""
    w = series.weight
    temperature, absorbed, insolation, outgoing, ocean_ice = (
        time_mean(field) for field in (
            series.surface_temperature, series.absorbed_shortwave, series.insolation,
            series.outgoing_longwave, series.ocean_ice,
        )
    )
    south_to_north = np.argsort(series.lat)
    total = implied_northward_transport(
        (absorbed - outgoing)[south_to_north], w[south_to_north],
    )
    uptake = np.nan_to_num(time_mean(series.ocean_heat_uptake))
    # Ocean transport from the surface uptake on LSG rows (south to north). In
    # steady state the northward transport at a face equals the heat taken up
    # south of it: release under the ice must be supplied from the north.
    order = np.argsort(series.lsg_lat)
    ocean = ocean_northward_transport(uptake[order], series.lsg_row_area[order])
    return EquilibriumState(
        label=series.label,
        mu=run_mu(series.label),
        initial_condition=RUNS[series.label].initial_condition,
        global_temperature=float(temperature @ w),
        planetary_albedo=float(1 - (absorbed @ w) / (insolation @ w)),
        insolation=float(insolation @ w),
        absorbed_shortwave=float(absorbed @ w),
        outgoing_longwave=float(outgoing @ w),
        south_edge=float(np.nanmean(ice_edge_latitude(series.ocean_ice, series.lat, True))),
        north_edge=float(np.nanmean(ice_edge_latitude(series.ocean_ice, series.lat, False))),
        lat=series.lat[south_to_north],
        zonal_temperature=temperature[south_to_north],
        zonal_absorbed=absorbed[south_to_north],
        zonal_outgoing=outgoing[south_to_north],
        zonal_ocean_ice=ocean_ice[south_to_north],
        total_transport=total,
        lsg_lat=series.lsg_lat[order],
        ocean_transport=ocean,
        unreadable_years=series.unreadable_years,
    )


def budget_step(warm: EquilibriumState, cold: EquilibriumState) -> dict[str, float]:
    """Split the global absorbed-sunlight change between two equilibria.

    ΔASR = ΔI (1 − ᾱ) − Ī Δα exactly, with bars denoting the two-state mean.
    The first term is the direct effect of the dimmer Sun, the second the
    albedo (ice and cloud) response. In equilibrium ΔOLR ≈ ΔASR, and
    `gain` = ΔASR / direct says how much the albedo response amplifies it.
    """
    d_insolation = cold.insolation - warm.insolation
    d_albedo = cold.planetary_albedo - warm.planetary_albedo
    mean_albedo = 0.5 * (cold.planetary_albedo + warm.planetary_albedo)
    mean_insolation = 0.5 * (cold.insolation + warm.insolation)
    direct = d_insolation * (1 - mean_albedo)
    albedo = -mean_insolation * d_albedo
    d_temperature = cold.global_temperature - warm.global_temperature
    d_outgoing = cold.outgoing_longwave - warm.outgoing_longwave
    return {
        "from μ": warm.mu,
        "to μ": cold.mu,
        "ΔT (K)": d_temperature,
        "ΔT per W m⁻² of μ (K)": d_temperature / (cold.mu - warm.mu),
        "direct ΔASR (W m⁻²)": direct,
        "albedo ΔASR (W m⁻²)": albedo,
        "ΔOLR (W m⁻²)": d_outgoing,
        "gain": (direct + albedo) / direct,
        "λ = ΔOLR/ΔT (W m⁻² K⁻¹)": d_outgoing / d_temperature,
    }
