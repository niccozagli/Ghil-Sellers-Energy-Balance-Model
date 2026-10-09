"""Plain look at slow (100-yr-mean) natural fluctuations: time series and where they live."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import global_ts, load, lsg_layer_mean, sector_ice_area, block_means

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
SM = TMP + "slowmaps/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
OUT = "../../figures/plasim_fresh"
BLUE, ORANGE, AQUA, GREY, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
WIN = {"1230": (11600, 16369)}

# ---- Fig A: time series of 100-yr means (z-scores) ----
runs = ["1245", "1240", "1235", "1232p5", "1230"]
fig, axes = plt.subplots(len(runs), 1, figsize=(13, 13))
for ax, lab in zip(axes, runs):
    d = load(lab); y = d["years"]; a, b = WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))
    edge = SH["EDGE"][lab]
    rows = (d["lsg_lat"] >= -edge) & (d["lsg_lat"] <= -edge + 10); wv = d["layer_volume"][rows, 0]
    W = np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()
    series = {"Indian/Pacific ice area": (-sector_ice_area(d, (2, 3)), AQUA, "-"),
              "Atlantic ice area": (-sector_ice_area(d, (1,)), BLUE, "-"),
              "water next to the edge, 0–700 m": (W, ORANGE, "-"),
              "Southern Ocean 700–2025 m": (lsg_layer_mean(d, 1, -90, 0), "k", "-"),
              "global surface temperature": (global_ts(d), YELLOW, "--")}
    for name, (x, c, ls) in series.items():
        st, bm = block_means(y, x, a, b + 1, 100)
        z = (bm - np.nanmean(bm)) / np.nanstd(bm)
        ax.plot(st + 50, z, ls, color=c, lw=1.3, label=name + (" (sign flipped: up = less ice)" if "ice" in name else ""))
    ax.set_title(f"μ = {d['mu']:g}: 100-year means, each standardised", fontsize=9); ax.grid(alpha=0.3, lw=0.4)
    ax.set_ylabel("z-score")
axes[0].legend(fontsize=7, ncol=3, frameon=False)
axes[-1].set_xlabel("model year")
fig.tight_layout(); fig.savefig(f"{OUT}/slow_timeseries.png", dpi=100); plt.close(fig)

# ---- Fig B: where the slow fluctuations live, and where the forced change lives ----
def load_sm(lab):
    return dict(np.load(SM + lab + ".npz"))


A, B = load_sm("1240"), load_sm("1232p5")
fields = [("sic", "ice cover (fraction of year)", "t21"), ("ts", "surface temperature (K)", "t21"),
          ("th0_100", "ocean 0–100 m (K)", "lsg"), ("th300_600", "ocean 300–600 m (K)", "lsg"),
          ("t1025_2000", "ocean 1025–2000 m (K)", "lsg")]
fig, axes = plt.subplots(len(fields), 3, figsize=(17, 17))
for i, (key, name, grid) in enumerate(fields):
    sdA = np.nanstd(A[key], axis=0); sdB = np.nanstd(B[key], axis=0)
    forced = (np.nanmean(B[key], axis=0) - np.nanmean(A[key], axis=0)) / (1232.5 - 1240.0)  # per W m⁻² (sign: per unit μ increase)
    forced = -forced  # change per W m⁻² of μ *decrease*
    vmax = np.nanpercentile(np.concatenate([sdA[np.isfinite(sdA)], sdB[np.isfinite(sdB)]]), 99)
    fmax = np.nanpercentile(np.abs(forced[np.isfinite(forced)]), 99)
    for j, (fld, title, cmap, lim) in enumerate(((sdA, "slow fluctuations, μ = 1240 (sd of 100-yr means)", "magma", (0, vmax)),
                                                (sdB, "slow fluctuations, μ = 1232.5 (sd of 100-yr means)", "magma", (0, vmax)),
                                                (forced, "forced change per W m⁻² of μ decrease (1240 → 1232.5)", "RdBu_r", (-fmax, fmax)))):
        ax = axes[i, j]
        if grid == "t21":
            lat, lon = A["t21_lat"], A["t21_lon"]; s = lat < -5
            im = ax.pcolormesh(lon, lat[s], fld[s], cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest")
            if key == "sic":
                ax.contour(lon, lat[s], np.nanmean(B["sic"], axis=0)[s], levels=[0.5], colors="c", linewidths=0.8)
        else:
            lat, lon = A["lsg_lat2d"], np.mod(A["lsg_lon2d"], 360); s = lat < -5
            im = ax.scatter(lon[s], lat[s], c=fld[s], s=9, marker="s", cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_ylim(-80, -5); ax.set_xlim(0, 360); fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name}: {title}", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("longitude (°E)")
fig.suptitle("Where the slow natural fluctuations live, and where the forced change lives (Southern Hemisphere)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/slow_maps.png", dpi=95); plt.close(fig)
print("ok")
