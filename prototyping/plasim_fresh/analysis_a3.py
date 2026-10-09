"""Analysis A, figures: zoomed (H, E) and (H, C) in 100-yr block means; 1230 jump onset by sector."""

import matplotlib.pyplot as plt
import numpy as np

from common import PARENT, band_cover, block_means, equivalent_edge, global_ts, load, lsg_layer_mean
from gsebm.plasim_global import RUNS

OUT = "../../figures/plasim_fresh"
labels = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1225"]
cmap = plt.get_cmap("plasma"); norm = plt.Normalize(1224, 1247)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for lab in labels:
    d = load(lab); _, y0 = PARENT[lab]
    H, E, C = lsg_layer_mean(d, 1, -90, 0), equivalent_edge(d), band_cover(d)
    stop = {"1230": 9000, "1225": 6000}.get(lab, int(d["years"][-1]) + 1)
    start = y0 + 1 if lab in ("1230", "1225") or y0 == 14999 else RUNS[lab].window[0]
    if lab == "1240":
        start = 9000
    yb, Hb = block_means(d["years"], H, start, stop)
    _, Eb = block_means(d["years"], E, start, stop)
    _, Cb = block_means(d["years"], C, start, stop)
    c = cmap(norm(d["mu"]))
    ls = "--" if lab == "1235_new_IC" else "-"
    for ax, K in zip(axes, (Eb, Cb)):
        ax.plot(Hb, K, ls, color=c, lw=1.0, marker="o", ms=2.5, label=f"{d['mu']:g}" + (" new IC" if "new" in lab else ""))
        w0, w1 = RUNS[lab].window
        kw = (yb >= w0) & (yb < w1)
        if lab not in ("1225",):
            ax.plot(np.nanmean(Hb[kw]), np.nanmean(K[kw]), "s", color=c, ms=9, mec="k", mew=0.8)
for ax in axes:
    ax.set_xlim(273.8, 272.0); ax.grid(alpha=0.3, lw=0.4); ax.set_xlabel("SH ocean θ 700–2025 m (K), 100-yr means")
axes[0].set_ylim(37.8, 25); axes[0].set_ylabel("equivalent-area S edge (°S)")
axes[1].set_ylim(0.0, 0.85); axes[1].set_ylabel("Indo-Pacific band cover")
axes[0].legend(fontsize=7, ncol=2, title="μ; squares = window means", title_fontsize=7)
fig.suptitle("Ice vs slow ocean index, 100-yr means after each step (transients from 1240 start at the step; "
             "1230/1225 to their jumps)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/A_ice_vs_ocean_zoom.png", dpi=110); plt.close(fig)

# 1230 jump onset: annual values 7400–8200, sector edges, band covers, H, upper ocean, global Ts
d = load("1230"); y = d["years"]; k = (y >= 7300) & (y <= 8200)
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
cols = {1: ("Atlantic", "#2a78d6"), 2: ("Indian", "#eb6834"), 3: ("Pacific", "#1baf7a")}
for s, (n, c) in cols.items():
    axes[0].plot(y[k], equivalent_edge(d, s)[k], lw=0.7, color=c, label=n)
    axes[1].plot(y[k], band_cover(d, sectors=(s,))[k], lw=0.7, color=c, label=n)
axes[0].invert_yaxis(); axes[0].set_ylabel("sector equiv. edge (°S)"); axes[0].legend(fontsize=7, ncol=3)
axes[1].set_ylabel("band cover 30.5+36°S"); axes[1].legend(fontsize=7, ncol=3)
r24 = int(np.argmin(np.abs(d["lat"] + 24.92)))
for s, (n, c) in cols.items():
    axes[2].plot(y[k], d["sic"][k, r24, s], lw=0.7, color=c, label=n)
axes[2].set_ylabel("cover 24.9°S row"); axes[2].legend(fontsize=7, ncol=3)
axes[3].plot(y[k], lsg_layer_mean(d, 1, -90, 0)[k], lw=0.8, color="k", label="SH 700–2025 m")
ax3 = axes[3].twinx(); ax3.plot(y[k], lsg_layer_mean(d, 0, -90, 0)[k], lw=0.6, color="#e34948")
axes[3].set_ylabel("SH θ 700–2025 m (K, black)"); ax3.set_ylabel("SH θ 0–700 m (K, red)")
axes[4].plot(y[k], global_ts(d)[k], lw=0.6, color="k"); axes[4].set_ylabel("global Ts (K)")
for ax in axes:
    ax.grid(alpha=0.3, lw=0.4); ax.axvline(7824, color="0.4", lw=0.8, ls=":")
axes[-1].set_xlabel("model year (dotted: jump onset 7824, pre-registered rule)")
fig.suptitle("1230: annual values around the warm → cold jump")
fig.tight_layout(); fig.savefig(f"{OUT}/A_1230_jump.png", dpi=110); plt.close(fig)
print("ok")
