"""Plain look: window-mean annual ice cover (fraction of the year covered) per T21 cell,
Southern rows 14–58°S, and the distribution of annual cell values in the edge rows."""

import matplotlib.pyplot as plt
import numpy as np

from common import load
from gsebm.plasim_global import RUNS

OUT = "../../figures/plasim_fresh"
labels = ["1265", "1245", "1240", "1235", "1233p75", "1232p5", "1230", "1228p5"]
fig, axes = plt.subplots(len(labels), 2, figsize=(14, 2.0 * len(labels)),
                         gridspec_kw={"width_ratios": [3, 1]})
for i, lab in enumerate(labels):
    d = load(lab)
    w0, w1 = RUNS[lab].window
    keep = (d["years"] >= w0) & (d["years"] <= w1)
    lat_s = d["lat"][d["lat"] < 0]
    sic = d["sic_map_south"][keep].astype(float)  # (t, 16, 64)
    ocean = d["ocean"][d["lat"] < 0]
    mean = np.where(ocean, np.nanmean(sic, axis=0), np.nan)
    rows = (lat_s > -60) & (lat_s < -13)
    ax = axes[i, 0]
    im = ax.pcolormesh(d["lon"], lat_s[rows], mean[rows], vmin=0, vmax=1, cmap="Blues_r", shading="nearest")
    ax.set_ylabel(f"μ={d['mu']:g}\nlat", fontsize=8)
    ax.set_yticks(lat_s[rows]); ax.set_yticklabels([f"{abs(x):.1f}" for x in lat_s[rows]], fontsize=6)
    for lon in (20, 115, 295):
        ax.axvline(lon, color="r", lw=0.5)
    # distribution of annual cell values in the two rows bracketing the mean zonal edge
    zonal = np.nanmean(np.where(ocean, mean, np.nan), axis=1)
    edge_rows = [k for k in range(len(lat_s)) if rows[k] and 0.05 < zonal[k] < 0.95]
    vals = sic[:, edge_rows, :][:, :, ocean[edge_rows].any(axis=0)] if edge_rows else np.array([])
    vals = np.concatenate([sic[:, k, ocean[k]].ravel() for k in edge_rows]) if edge_rows else np.array([])
    axes[i, 1].hist(vals, bins=np.linspace(0, 1, 21), color="#2a78d6")
    axes[i, 1].set_title("annual cell values, rows " + ", ".join(f"{abs(lat_s[k]):.1f}" for k in edge_rows), fontsize=7)
    axes[i, 1].tick_params(labelsize=6)
axes[-1, 0].set_xlabel("longitude (°E); red: sector limits 20°E, 115°E, 65°W")
fig.colorbar(im, ax=axes[:, 0], shrink=0.4, label="window-mean fraction of year ice-covered")
fig.suptitle("Window-mean annual ice cover per T21 ocean cell (plain time means)")
fig.savefig(f"{OUT}/edge_maps.png", dpi=105)
print("ok")
