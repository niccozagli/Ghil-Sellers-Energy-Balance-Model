"""Figures for the slow-fluctuation analysis."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import global_ts, load, lsg_layer_mean, sector_ice_area
from slow_analysis import (FIELDS, RUNS, WARMC, maps, index_blocks, regress_map, forced_map, window, W_series, runmean)

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
OUT = "../../figures/plasim_fresh"
R = json.load(open(TMP + "slow_analysis.json")); SK = json.load(open(TMP + "skew.json"))
BLUE, ORANGE, AQUA, GREY, YELLOW, MAG, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#eda100", "#e87ba4", "#4a3aa7"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"sic": "ice cover", "ts": "surface temperature", "rst": "absorbed sunlight", "th0_100": "ocean 0–100 m",
         "th300_600": "ocean 300–600 m", "t1025_2000": "ocean 1025–2000 m"}
lab_txt = lambda l: f"{float(l.replace('p', '.').split('_')[0]):g}" + (" (2nd)" if "new" in l else "")

# ---- Fig 1: maps for 1235 ----
lab = "1235"; z = maps(lab, 100); I = index_blocks(lab, z["block_start"], 100, "Ts")
show = ["sic", "ts", "th0_100", "th300_600", "t1025_2000"]
fig, axes = plt.subplots(len(show), 2, figsize=(14, 16))
for i, key in enumerate(show):
    nat = regress_map(z[key], I, False); frc = forced_map(lab, key)
    lim = np.nanpercentile(np.abs(np.concatenate([nat[np.isfinite(nat)], frc[np.isfinite(frc)]])), 98)
    for j, (fld, tt) in enumerate(((nat, "slow natural fluctuation (century means regressed on global Ts)"),
                                   (frc, "forced change between neighbouring settled states"))):
        ax = axes[i, j]
        if key in ("sic", "ts", "rst"):
            lat, lon = z["t21_lat"], z["t21_lon"]; s = lat < -5
            im = ax.pcolormesh(lon, lat[s], fld[s], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        else:
            lat, lon = z["lsg_lat2d"], np.mod(z["lsg_lon2d"], 360); s = lat < -5
            im = ax.scatter(lon[s], lat[s], c=fld[s], s=10, marker="s", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_ylim(-80, -5); ax.set_xlim(0, 360); fig.colorbar(im, ax=ax, shrink=0.8, label="per K of global Ts")
        r = R["S1"][lab]["L100_Ts_d0"][key]["r"]
        ax.set_title(f"{NAMES[key]}: {tt}" + (f"\npattern correlation with the forced change r = {r:+.2f}" if j == 0 else ""), fontsize=8)
axes[-1, 0].set_xlabel("longitude (°E)"); axes[-1, 1].set_xlabel("longitude (°E)")
fig.suptitle("μ = 1235: the slow natural fluctuation and the forced change, both per K of global surface temperature", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/slow_patterns.png", dpi=95); plt.close(fig)

# ---- Fig 2: summary ----
fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
x = np.arange(len(RUNS)); cols = {"sic": BLUE, "ts": ORANGE, "rst": YELLOW, "th0_100": AQUA, "th300_600": MAG, "t1025_2000": VIO}
for key, c in cols.items():
    v = [R["S1"][l]["L100_Ts_d0"][key]["r"] for l in RUNS]
    lo = [min(R["S1"][l][vk][key]["r"] for vk in R["S1"][l]) for l in RUNS]; hi = [max(R["S1"][l][vk][key]["r"] for vk in R["S1"][l]) for l in RUNS]
    axes[0].plot(x, v, "o-", color=c, label=NAMES[key], lw=1.2, ms=4)
    axes[1].plot(x, [R["S1"][l]["L100_Ts_d0"][key]["amp"] for l in RUNS], "o-", color=c, lw=1.2, ms=4)
axes[0].axhline(0.7, color=GREY, ls=":"); axes[0].set_ylabel("pattern correlation, natural vs forced"); axes[0].legend(fontsize=7, frameon=False)
axes[0].set_title("Does the slow fluctuation look like the forced change?")
axes[1].axhline(1, color=GREY, ls=":"); axes[1].set_ylabel("amplitude, natural / forced (per K of global Ts)")
axes[1].set_title("…and with the same strength?")
idx_cols = {"less Indian/Pacific ice": AQUA, "edge water 0–700": ORANGE, "EP 0–100": BLUE, "EP 300–600": VIO, "EA 300–600": MAG}
for j, (name, c) in enumerate(idx_cols.items()):
    v = [R["S2"][l]["rm50_d0"][name][0] for l in RUNS]
    lo = [min(R["S2"][l][vk][name][0] for vk in R["S2"][l]) for l in RUNS]; hi = [max(R["S2"][l][vk][name][0] for vk in R["S2"][l]) for l in RUNS]
    off = (j - 2) * 0.1
    axes[2].vlines(x + off, lo, hi, color=c, alpha=0.3, lw=4); axes[2].plot(x + off, v, "o", color=c, ms=5, label=name)
axes[2].axhline(0, color=GREY, lw=0.6); axes[2].set_ylim(-160, 160)
axes[2].set_ylabel("lag of maximum correlation with global Ts (yr)\n(positive: the index moves first)")
axes[2].set_title("Who moves first? (EP/EA: eastern South Pacific/Atlantic)"); axes[2].legend(fontsize=7, frameon=False)
for key, c, nm in (("Ts", ORANGE, "global surface temperature"), ("lessIPice", AQUA, "less Indian/Pacific ice")):
    v = [SK[l][f"{key}_L100_d0_hNone"] for l in RUNS]
    ks = [k for k in SK["1245"] if k.startswith(key) and "_L1_" not in k]
    lo = [min(SK[l][k] for k in ks) for l in RUNS]; hi = [max(SK[l][k] for k in ks) for l in RUNS]
    axes[3].vlines(x + (0.08 if key == "Ts" else -0.08), lo, hi, color=c, alpha=0.3, lw=5)
    axes[3].plot(x + (0.08 if key == "Ts" else -0.08), v, "o-", color=c, ms=5, label=nm)
axes[3].axhline(0, color=GREY, lw=0.8); axes[3].set_ylabel("skewness of century means\n(positive: warm/ice-retreat excursions dominate)")
axes[3].set_title("Which way do the big slow excursions go? [exploratory]"); axes[3].legend(fontsize=7, frameon=False); axes[3].set_ylim(-2, 3)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels([lab_txt(l) for l in RUNS], rotation=50, fontsize=7.5); ax.grid(alpha=0.3, lw=0.4)
    ax.axvspan(8.5, 10.5, color="0.92", zorder=0)
fig.suptitle("Slow (century-scale) natural fluctuations across μ (shaded: cold states; bars: range over robustness variants)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/slow_summary.png", dpi=105); plt.close(fig)

# ---- Fig 3: event timing ----
events = [("1245", 11200, "1245 warm spell"), ("1237p5", 17900, "1237.5 warm spell"), ("1240", 9100, "1240 spike (unlike the others)"),
          ("1230", 15800, "1230 cold state: warm spell"), ("1232p5", 23700, "1232.5 cold spell (z ≈ −2.4)")]
fig, axes = plt.subplots(1, len(events), figsize=(22, 4.4), sharey=True)
for ax, (lab, s0, title) in zip(axes, events):
    d = load(lab); y = d["years"]; a0, a1 = window(lab); k = (y >= a0) & (y <= a1); yy = y[k]
    reg = dict(np.load(TMP + f"regional/{lab}.npz")); kr = (reg["years"] >= a0) & (reg["years"] <= a1)
    ser = {"global Ts": (global_ts(d)[k], "k", "-"), "less Indian/Pacific ice": (-sector_ice_area(d, (2, 3))[k], AQUA, "-"),
           "less Atlantic ice": (-sector_ice_area(d, (1,))[k], BLUE, "-"), "edge water 0–700 m": (W_series(lab)[k], ORANGE, "-"),
           "EP 0–100 m": (reg["EP_0_100"][kr], VIO, "--"), "EP 300–600 m": (reg["EP_300_600"][kr], MAG, "--"),
           "SH ocean 700–2025 m": (lsg_layer_mean(d, 1, -90, 0)[k], GREY, ":")}
    t = yy - (s0 + 50); sel = (t >= -300) & (t <= 300)
    for nm, (xv, c, ls) in ser.items():
        r = runmean(xv, 30); zz = (r - np.nanmean(r)) / np.nanstd(r)
        ax.plot(t[sel], zz[sel], ls, color=c, lw=1.3, label=nm)
    ax.axvspan(-50, 50, color="0.9", zorder=0); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlabel("years from the centre of the event block")
axes[0].set_ylabel("30-yr running mean, standardised"); axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.suptitle("The largest slow events: which variables move first? (single events, description only)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/slow_events.png", dpi=105); plt.close(fig)
print("ok")
