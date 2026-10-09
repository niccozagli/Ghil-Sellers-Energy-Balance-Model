"""Figures for the swing analysis."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import load
from swings import RUNS, STAT, SH, rows, peaks, TMP

sw = json.load(open(TMP + "swings.json")); res = sw["res"]
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
labels = RUNS + [k for k in res if k.startswith("1230:") and k != "1230:7600"]
txt = [(f"{float(l.replace('p', '.').split('_')[0]):g}" + (" (2nd)" if "new" in l else "")) if not l.startswith("1230")
       else f"1230 creep\n{l[5:]}" for l in labels]
x = np.arange(len(labels))

# ---- Fig 1: measures across μ and along the creep ----
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
ax = axes[0]
ax.plot(x, [res[l]["base"]["mean"] for l in labels], "o-", color=GREY, label="average cover")
ax.plot(x, [res[l]["base"]["h50"] for l in labels], "o-", color=BLUE, alpha=0.6, label="typical swing peak (median)")
ax.plot(x, [res[l]["base"]["h90"] for l in labels], "o-", color=BLUE, label="large swing peak (90th percentile)")
ax.axhline(1, color="k", lw=0.8); ax.plot(len(labels) - 0.2, sw["hj"], "*", color=ORANGE, ms=14, label="the swing that started the jump")
ax.set_ylabel("Atlantic 30.5°S row: fraction of the year ice-covered"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, frameon=False)
ax.set_title("Filling of the row and height of the swings")
ax = axes[1]
ax.errorbar(x, [res[l]["base"]["R"] for l in labels], yerr=[2 * np.nan_to_num(res[l]["base"]["R_se"]) for l in labels],
            fmt="o", color=AQUA, capsize=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("rise of Indian/Pacific 30.5°S cover in the 20 yr after an\nAtlantic peak vs the 20 yr before (fraction of year)")
ax.set_title("Indian/Pacific response to each Atlantic swing (±2 s.e.)"); ax.set_ylim(-0.02, 0.03)
ax = axes[2]
ax.errorbar(x, [res[l]["base"]["Rn"] for l in labels], yerr=[2 * np.nan_to_num(res[l]["base"]["Rn_se"]) for l in labels],
            fmt="o", color=AQUA, capsize=3)
ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.1, 0.12)
ax.set_ylabel("Indian/Pacific response per unit of Atlantic swing")
ax.set_title("…per unit of swing (±2 s.e.)")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(txt, rotation=60, fontsize=7.5); ax.grid(alpha=0.3, lw=0.4)
    ax.axvspan(len(RUNS) - 0.5, len(labels) - 0.5, color="0.93", zorder=0)
fig.suptitle("Atlantic swings into the 30.5°S row, along the warm branch (all stationary years) and along the μ = 1230 creep "
             "before the jump (shaded, 500-yr segments)", fontsize=10)
fig.tight_layout(); fig.savefig("../../figures/plasim_fresh/swings_measures.png", dpi=110); plt.close(fig)

# ---- Fig 2: composites around Atlantic peaks + the jump swing ----
lags = np.arange(-60, 121)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharex=True)
cmap = plt.get_cmap("viridis")
sel = ["1245", "1240", "1235", "1232p5"]
for i, lab in enumerate(sel + ["creep"]):
    if lab == "creep":
        d = load("1230"); yy = d["years"]; k = (yy >= 5100) & (yy <= 7823); P = sw["Pc"]; col = ORANGE; name = "1230 creep (5100–7823)"
    else:
        d = load(lab); yy = d["years"]; k = (yy >= STAT[lab]["0.95"]) & (yy <= STAT[lab]["end"]); P = int(round(SH["A"][lab]["full"]["per_acf"]))
        col = cmap(i / 4); name = f"μ = {d['mu']:g}"
    xa, ya = rows(d); xs, ys = xa[k], ya[k]; m, my = np.nanmean(xs), np.nanmean(ys)
    pk = [p for p in peaks(xs, int(round(P / 2)), m) if p + lags[0] >= 0 and p + lags[-1] < len(xs)]
    cx = np.nanmean([xs[p + lags] - m for p in pk], axis=0); cyy = np.nanmean([ys[p + lags] - my for p in pk], axis=0)
    axes[0].plot(lags, cx, color=col, lw=1.5, label=f"{name} ({len(pk)} swings)")
    axes[1].plot(lags, cyy, color=col, lw=1.5, label=name)
# the jump swing (single event), anomalies relative to the 500 yr before it
d = load("1230"); yy = d["years"]; xa, ya = rows(d); tj = int(sw["tj"])
kk = (yy >= tj - 500) & (yy < tj); m, my = np.nanmean(xa[kk]), np.nanmean(ya[kk])
idx = np.flatnonzero(yy == tj)[0]
axes[0].plot(lags, xa[idx + lags] - m, color="k", lw=1.0, ls="--", label=f"the jump swing ({tj}), single event")
axes[1].plot(lags, ya[idx + lags] - my, color="k", lw=1.0, ls="--", label="the jump swing, single event")
axes[0].set_ylabel("Atlantic 30.5°S row cover, anomaly"); axes[1].set_ylabel("Indian/Pacific 30.5°S row cover, anomaly")
for ax in axes:
    ax.axvline(0, color=GREY, lw=0.6); ax.axhline(0, color=GREY, lw=0.6); ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlabel("years from an Atlantic peak")
axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("Atlantic row: average swing (annual values)")
axes[1].set_title("Indian/Pacific row around the same swings")
fig.suptitle("Average Atlantic swing and the Indian/Pacific response, compared with the swing that started the μ = 1230 jump", fontsize=10)
fig.tight_layout(); fig.savefig("../../figures/plasim_fresh/swings_composite.png", dpi=110); plt.close(fig)
print("ok")
