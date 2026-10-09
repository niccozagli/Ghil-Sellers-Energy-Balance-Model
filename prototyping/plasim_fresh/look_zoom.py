"""Plain look: 1230/1225 dwell by sector; 1265 and 1240 edges by sector (annual values)."""

import matplotlib.pyplot as plt
import numpy as np

from common import equivalent_edge, global_ts, load, lsg_layer_mean

OUT = "../../figures/plasim_fresh"
SECT = {1: ("Atlantic", "#2a78d6"), 2: ("Indian", "#eb6834"), 3: ("Pacific", "#1baf7a")}


def panel_set(axes, d, years):
    keep = (d["years"] >= years[0]) & (d["years"] <= years[1])
    t = d["years"][keep]
    for s, (name, c) in SECT.items():
        axes[0].plot(t, equivalent_edge(d, s)[keep], lw=0.5, color=c, label=name)
    axes[0].plot(t, equivalent_edge(d, 0)[keep], lw=0.6, color="k", label="zonal")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("equiv.-area edge (°S)", fontsize=8)
    axes[1].plot(t, global_ts(d)[keep], lw=0.5, color="k")
    axes[1].set_ylabel("global Ts (K)", fontsize=8)
    axes[2].plot(t, lsg_layer_mean(d, 0, -90, 0)[keep], lw=0.5, color="#2a78d6", label="SH 0–700 m")
    ax2 = axes[2]
    ax2.set_ylabel("SH θ 0–700 m (K)", fontsize=8)
    axes[3].plot(t, lsg_layer_mean(d, 1, -90, 0)[keep], lw=0.6, color="#eb6834")
    axes[3].set_ylabel("SH θ 700–2025 m (K)", fontsize=8)
    axes[0].set_title(f"μ = {d['mu']:g}, years {years[0]}–{years[1]}", fontsize=9)
    axes[0].legend(fontsize=7, ncol=4)
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.4)


cases = [("1230", (4500, 8500)), ("1225", (4500, 7900)), ("1235", (4500, 8500))]
fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex="col")
for j, (lab, yrs) in enumerate(cases):
    panel_set(axes[:, j], load(lab), yrs)
fig.suptitle("Transients from 1265 (step at 4499): sector edges and Southern ocean (annual values)")
fig.tight_layout()
fig.savefig(f"{OUT}/zoom_dwell.png", dpi=105)
plt.close(fig)

cases = [("1265", (6000, 7500)), ("1240", (14000, 15500)), ("1232p5", (21000, 22500))]
fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex="col")
for j, (lab, yrs) in enumerate(cases):
    panel_set(axes[:, j], load(lab), yrs)
fig.suptitle("Stationary runs, 1500-yr excerpts: sector edges and Southern ocean (annual values)")
fig.tight_layout()
fig.savefig(f"{OUT}/zoom_stationary.png", dpi=105)
plt.close(fig)
print("ok")
