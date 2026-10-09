"""Southern extratropics (30–90°S): ocean heat release, radiation, and sea ice.

Annual values only. Each grid uses its own weights: T21 Gaussian weights for
radiation, LSG wet row areas for the ocean coupling flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsebm.plasim_exploratory import EARTH_RADIUS_M
from gsebm.plasim_global import GlobalSeries

BOX_EDGE_LAT = -30.0


@dataclass(frozen=True)
class BoxSeries:
    """Annual Southern-extratropics terms of one run (W, and m² of box area)."""

    label: str
    years: np.ndarray
    ocean_release: np.ndarray
    toa_net: np.ndarray
    atmosphere_area: float
    ocean_area: float


def box_series(series: GlobalSeries,
               bounds: tuple[float, float] = (-90.0, BOX_EDGE_LAT)) -> BoxSeries:
    """Ocean heat release and TOA net radiation in a latitude band, per year.

    Rows are selected by their centre latitude, `bounds[0] <= lat < bounds[1]`,
    separately on the T21 and LSG grids. The default is 30–90°S.

    * `ocean_release`: −Σ `zonal_newtonian_coupling_heat_flux` × wet row
      area over the LSG rows (positive = heat leaving the ocean).
    * `toa_net`: Σ (`rst` + `rlut`) × 2πR² w_j over the T21 rows
      (negative = the box loses energy to space).
    """
    t21 = (series.lat >= bounds[0]) & (series.lat < bounds[1])
    lsg = (series.lsg_lat >= bounds[0]) & (series.lsg_lat < bounds[1])
    row_area = 2 * np.pi * EARTH_RADIUS_M**2 * series.weight / series.weight.sum() * 2
    net = series.absorbed_shortwave - series.outgoing_longwave
    return BoxSeries(
        label=series.label,
        years=series.years,
        ocean_release=-(np.nan_to_num(series.ocean_heat_uptake[:, lsg]) @ series.lsg_row_area[lsg]),
        toa_net=net[:, t21] @ row_area[t21],
        atmosphere_area=float(row_area[t21].sum()),
        ocean_area=float(series.lsg_row_area[lsg].sum()),
    )


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of x(t) with y(t + k) for k = −max_lag … max_lag.

    Positive k: y follows x. Anomalies are taken from each series' own mean.
    """
    x = np.asarray(x, dtype=float) - np.nanmean(x)
    y = np.asarray(y, dtype=float) - np.nanmean(y)
    lags = np.arange(-max_lag, max_lag + 1)
    values = []
    for k in lags:
        a, b = (x[:x.size - k], y[k:]) if k >= 0 else (x[-k:], y[:y.size + k])
        ok = np.isfinite(a) & np.isfinite(b)
        values.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return lags, np.asarray(values)


EDGE_BOX_WIDTH = 10.0


def edge_boxes(series: GlobalSeries, window: tuple[int, int],
               width: float = EDGE_BOX_WIDTH) -> tuple[float, dict[str, BoxSeries]]:
    """Boxes that follow a run's mean Southern ice edge.

    The edge is the window mean of the annual zonal-mean Southern edge. The
    *band* covers edge ± `width`; the *cap* covers everything poleward of
    (edge − `width`), i.e. `width` degrees equatorward of the edge to the pole.
    """
    from gsebm.plasim_global import ice_edge_latitude

    inside = (series.years >= window[0]) & (series.years <= window[1])
    edge = float(np.mean(ice_edge_latitude(series.ocean_ice[inside], series.lat, south=True)))
    return edge, {
        "band": box_series(series, (-edge - width, -edge + width)),
        "cap": box_series(series, (-90.0, -edge + width)),
    }


def cap_table(boxes: dict, edges: dict, windows: dict) -> list[dict[str, object]]:
    """Window means of ocean release and TOA net in the cap and band (plain means)."""
    table = []
    for label, kinds in boxes.items():
        start, end = windows[label]
        entry: dict[str, object] = {"run": label.replace("p", "."),
                                    "S edge (°S)": round(edges[label], 1)}
        for kind, box in kinds.items():
            inside = (box.years >= start) & (box.years <= end)
            release = float(np.nanmean(box.ocean_release[inside]))
            toa = float(np.nanmean(box.toa_net[inside]))
            entry[f"{kind} release (PW)"] = round(release / 1e15, 3)
            entry[f"{kind} TOA net (PW)"] = round(toa / 1e15, 3)
            if kind == "cap":
                entry["cap ocean share"] = round(release / -toa, 2)
        entry["band / cap release"] = round(entry["band release (PW)"] / entry["cap release (PW)"], 2)
        table.append(entry)
    return table
