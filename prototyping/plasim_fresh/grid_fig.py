"""Plain look: where the Southern ice edge sits relative to the T21 grid rows, for every settled run."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import equivalent_edge, load, row_ocean_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

warm = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
cold = ["1230", "1228p5"]
ROWS = [47.1, 41.5, 36.0, 30.5, 24.9, 19.4]
# single cells at the equatorward tip of the ice: eastern South Pacific and eastern South Atlantic
CELLS = [(24.9, 270.0), (24.9, 275.6), (19.4, 270.0), (24.9, 0.0), (24.9, 354.4)]

res = {}
for lab in warm + cold:
    d = load(lab)
    a, b = ST[lab]["0.95"], ST[lab]["end"]
    k = (d["years"] >= a) & (d["years"] <= b) & np.isfinite(d["sic"][:, 0, 0])
    out = {"mu": float(d["mu"])}
    for s, name in ((0, "all"), (1, "atl"), (2, "ind"), (3, "pac")):
        e = equivalent_edge(d, s)[k]
        out[name] = (float(np.mean(e)), float(np.percentile(e, 5)), float(np.percentile(e, 95)))
    area = row_ocean_area(d, 0)
    for r in ROWS:
        j = int(np.argmin(np.abs(d["lat"] + r)))
        out[f"row{r}"] = float(np.nanmean(d["sic"][k, j, 0]))
    cmap = np.nanmean(d["sic_map_south"][k], axis=0)
    slat = d["lat"][16:]
    for r, lon in CELLS:
        j = int(np.argmin(np.abs(slat + r)))
        i = int(np.argmin(np.abs(d["lon"] - lon)))
        out[f"cell{r}_{lon}"] = float(cmap[j, i])
    res[lab] = out
json.dump(res, open(TMP + "grid_fig.json", "w"), indent=1)

# row spans (absolute latitude): midpoints between Gaussian latitudes
d = load("1240")
absl = np.sort(np.abs(d["lat"][d["lat"] < 0]))
mids = 0.5 * (absl[:-1] + absl[1:])

fig, axes = plt.subplots(1, 3, figsize=(19, 5.0))
ax = axes[0]
for i, r in enumerate(absl):
    if 17 < r < 66:
        lo = mids[i - 1] if i > 0 else 0.0
        hi = mids[i] if i < len(mids) else 90.0
        ax.axhspan(lo, hi, color="0.93" if i % 2 else "0.86", zorder=0, lw=0)
        ax.text(1321, r, f"T21 row {r:.1f}°S", va="center", fontsize=7, color=GREY)
for name, c, lbl, dx in (("all", "k", "all sectors", 0.0), ("atl", BLUE, "Atlantic", -0.5),
                         ("ind", AQUA, "Indian", 0.0), ("pac", ORANGE, "Pacific", 0.5)):
    for group, filled in ((warm, True), (cold, False)):
        mu = np.array([res[l]["mu"] for l in group]) + dx
        m = np.array([res[l][name][0] for l in group])
        lo = m - np.array([res[l][name][1] for l in group])
        hi = np.array([res[l][name][2] for l in group]) - m
        ax.errorbar(mu, m, yerr=[lo, hi], fmt="o" if filled else "s", ms=4.5, color=c, lw=1, capsize=0,
                    mfc=c if filled else "white", label=lbl if filled else None, zorder=3)
ax.set_ylim(67, 17)
ax.set_ylabel("ice edge, equivalent-area latitude (°S)\n(dot: settled mean; bar: 5–95% of annual values)")
ax.set_title("The approach to the end of the warm branch lies within one grid row")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.2, 0.75))

ax = axes[1]
cols = {47.1: "0.7", 41.5: GREY, 36.0: ORANGE, 30.5: BLUE, 24.9: AQUA, 19.4: "k"}
for r in ROWS:
    for group, filled in ((warm, True), (cold, False)):
        mu = [res[l]["mu"] for l in group]
        v = [res[l][f"row{r}"] for l in group]
        ax.plot(mu, v, "o-" if filled else "s--", color=cols[r], ms=4.5, lw=1.2, mfc=cols[r] if filled else "white",
                label=f"{r}°S row" if filled else None)
ax.set_ylabel("annual ice cover of the row\n(fraction of the year, ocean area-weighted, all sectors)")
ax.set_title("The rows fill one at a time")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

ax = axes[2]
cc = {(24.9, 270.0): ORANGE, (24.9, 275.6): "#b04a1f", (19.4, 270.0): "k", (24.9, 0.0): BLUE, (24.9, 354.4): "#164f91"}
for cell in CELLS:
    r, lon = cell
    where = "eastern S. Pacific" if 200 < lon < 300 else "eastern S. Atlantic"
    lonlab = f"{lon:.1f}°E" if lon < 180 else f"{360 - lon:.1f}°W"
    for group, filled in ((warm, True), (cold, False)):
        mu = [res[l]["mu"] for l in group]
        v = [res[l][f"cell{r}_{lon}"] for l in group]
        ax.plot(mu, v, "o-" if filled else "s--", color=cc[cell], ms=4.5, lw=1.2, mfc=cc[cell] if filled else "white",
                label=f"{r}°S, {lonlab} ({where})" if filled else None)
ax.set_ylabel("annual ice cover of one grid cell\n(fraction of the year)")
ax.set_title("The tip of the advance is a handful of single cells")
ax.legend(fontsize=7.5, frameon=False, loc="center left")

for ax in axes:
    ax.set_xlim(1322, 1224)
    ax.axvspan(1258, 1233, color=AQUA, alpha=0.10, lw=0, zorder=0)
    ax.axvline(1231.25, color="k", ls=":", lw=1)
    ax.set_xlabel("μ (W m⁻²); filled: warm states, open: cold states")
    ax.grid(alpha=0.3, lw=0.4)
axes[0].text(1245.5, 17.6, "same distance from the end\nas Tantet's slowing-down\nwindow (1290–1265)", ha="center",
             va="top", fontsize=7, color=GREY)
axes[1].text(1230.6, 0.62, "warm → cold\ntransition", ha="left", fontsize=7, color=GREY)
axes[2].set_xlim(1252, 1226)
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/grid_rows.png", dpi=110)
plt.close(fig)
for lab in warm + cold:
    r = res[lab]
    print(lab, "edge all %.1f (%.1f–%.1f)" % r["all"], " atl %.1f ind %.1f pac %.1f" % (r["atl"][0], r["ind"][0], r["pac"][0]),
          " rows:", " ".join(f"{x}:{r[f'row{x}']:.2f}" for x in ROWS))
