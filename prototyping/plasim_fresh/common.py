"""Shared readers for the cached annual series (plain quantities only)."""

from pathlib import Path

import numpy as np

from gsebm.plasim_global import ice_edge_latitude, run_mu

CACHE = Path("/Users/niccolo/.claude/jobs/3fe717ba/tmp/annual")
EARTH_RADIUS = 6.371e6
PARENT = {
    "1312": ("1367", 1999), "1288": ("1367", 1999), "1265": ("1367", 1999),
    "1250": ("1265", 4499), "1245": ("1265", 4499), "1240": ("1265", 4499),
    "1235": ("1265", 4499), "1230": ("1265", 4499), "1225": ("1265", 4499),
    "1242p5": ("1240", 14999), "1237p5": ("1240", 14999), "1235_new_IC": ("1240", 14999),
    "1233p75": ("1240", 14999), "1232p5": ("1240", 14999), "1228p5": ("1230", 14499),
}
# Year of the step in the run's own year labels. The 1240 and 1228.5 records are labelled 500 yr ahead
# of their parents (checked: their first-year deep-ocean θ matches the parent at 4499 / 14499).
STEP_YEAR = {lab: y0 for lab, (_, y0) in PARENT.items()}
STEP_YEAR.update({"1240": 4999, "1228p5": 14999})


def load(label):
    d = dict(np.load(CACHE / f"{label}.npz"))
    d["label"] = label
    d["mu"] = run_mu(label)
    return d


def row_ocean_area(d, sector=0):
    """Ocean area (m²) of each T21 row in a sector (0 all, 1 Atl, 2 Ind, 3 Pac)."""
    row_area = 4 * np.pi * EARTH_RADIUS**2 * d["gw"] / d["gw"].sum()
    return row_area * d["ocean_counts"][:, sector] / 64.0


def south_edge(d, sector=0, threshold=0.5):
    return ice_edge_latitude(d["sic"][:, :, sector], d["lat"], south=True, threshold=threshold)


def south_ice_area(d, sector=0, lat_max=0.0):
    """Ice-covered ocean area (m²) south of -lat_max (annual-mean cover × ocean area)."""
    rows = d["lat"] < -lat_max
    return np.nansum(d["sic"][:, rows, sector] * row_ocean_area(d, sector)[rows], axis=1)


def equivalent_edge(d, sector=0):
    """Latitude (°S) at which the ocean area poleward equals the Southern ice area.

    Threshold-free: rows are filled from the pole; the edge is interpolated in
    area within the last partly filled row (row spans taken as midpoints
    between Gaussian latitudes).
    """
    lat = d["lat"]
    south = np.flatnonzero(lat < 0)
    south = south[np.argsort(lat[south])]  # pole first
    area = row_ocean_area(d, sector)[south]
    absl = np.abs(lat[south])
    # row boundaries (absolute latitude), pole side and equator side
    mids = 0.5 * (absl[:-1] + absl[1:])
    pole_side = np.concatenate([[90.0], mids])
    eq_side = np.concatenate([mids, [0.0]])
    cum = np.concatenate([[0.0], np.cumsum(area)])
    ice = south_ice_area(d, sector)
    k = np.clip(np.searchsorted(cum, ice, side="right") - 1, 0, len(area) - 1)
    frac = np.where(area[k] > 0, (ice - cum[k]) / np.where(area[k] > 0, area[k], 1), 0)
    return pole_side[k] + np.clip(frac, 0, 1) * (eq_side[k] - pole_side[k])


def lsg_layer_mean(d, layer, lat_lo=-90.0, lat_hi=0.0):
    """Wet-volume-weighted θ of one layer (0: 0–700, 1: 700–2025, 2: >2025 m) over LSG rows."""
    rows = (d["lsg_lat"] >= lat_lo) & (d["lsg_lat"] < lat_hi)
    w = d["layer_volume"][rows, layer]
    x = d["theta_layers"][:, rows, layer]
    return np.nansum(x * w, axis=1) / w[np.isfinite(x[0])].sum()


def global_ts(d):
    w = d["gw"] / d["gw"].sum()
    return d["ts"] @ w


def band_cover(d, sectors=(2, 3), lats=(-30.46, -36.0)):
    """Ocean-area-weighted annual ice cover over the given rows and sectors (1 Atl, 2 Ind, 3 Pac)."""
    rows = [int(np.argmin(np.abs(d["lat"] - x))) for x in lats]
    num, den = 0.0, 0.0
    for s in sectors:
        area = row_ocean_area(d, s)
        for j in rows:
            num = num + np.nan_to_num(d["sic"][:, j, s]) * area[j]
            den += area[j]
    out = num / den
    return np.where(np.isnan(d["sic"][:, rows[0], sectors[0]]), np.nan, out)


def block_means(years, x, start, stop, length=100):
    """Non-overlapping block means of x over [start, stop) in model years; NaN-aware."""
    edges = np.arange(start, stop - length + 1, length)
    vals = np.array([np.nanmean(x[(years >= a) & (years < a + length)]) for a in edges])
    return edges, vals


def mean_between(years, x, a, b):
    """Mean of x over model years a..b inclusive (NaN-aware)."""
    k = (years >= a) & (years <= b)
    return float(np.nanmean(x[k]))


CELLS = Path("/Users/niccolo/.claude/jobs/3fe717ba/tmp/cells")


def sector_sw(label):
    """Absorbed TOA shortwave south of 20°S per sector (Atl, Ind, Pac), PW, annual, full record."""
    z = np.load(CELLS / f"{label}.npz")
    d = load(label)
    row_area = 4 * np.pi * EARTH_RADIUS**2 * d["gw"] / d["gw"].sum()
    lon = z["lon"]
    masks = [((lon >= 295) | (lon < 20)), ((lon >= 20) & (lon < 115)), ((lon >= 115) & (lon < 295))]
    frac = np.array([m.sum() / 64 for m in masks])
    rows = z["lat"] < -20
    sw = np.stack([(z["rst_sector"][:, rows, s] * row_area[rows] * frac[s]).sum(axis=1) / 1e15 for s in range(3)], axis=1)
    return z["years"], sw


def sector_ice_area(d, sectors):
    """Southern ice area (10¹² m²) in the given sectors (1 Atl, 2 Ind, 3 Pac), annual."""
    south = d["lat"] < 0
    tot = 0.0
    for s in sectors:
        tot = tot + np.nansum(d["sic"][:, south, s] * row_ocean_area(d, s)[south], axis=1)
    return tot / 1e12
