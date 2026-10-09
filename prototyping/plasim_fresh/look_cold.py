"""Plain look at the cold state: 1230 (settled 11600–16369), 1228.5 (15000–16899), and 1225's cold stage
(5700–7590, before the snowball). Annual values; raw periodograms."""

import matplotlib.pyplot as plt
import numpy as np

from common import band_cover, equivalent_edge, global_ts, load, lsg_layer_mean

OUT = "../../figures/plasim_fresh"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
cases = [("1230", (11600, 16369), "1230 cold (settled)"), ("1228p5", (15000, 16899), "1228.5 cold (settled)"),
         ("1225", (5700, 7600), "1225 cold stage, before the snowball (not settled)")]
fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex="col")
for j, (lab, (a, b), title) in enumerate(cases):
    d = load(lab); y = d["years"]; k = (y >= a) & (y <= b)
    for s, (n, c) in {1: ("Atlantic", BLUE), 2: ("Indian", ORANGE), 3: ("Pacific", AQUA)}.items():
        axes[0, j].plot(y[k], equivalent_edge(d, s)[k], lw=0.5, color=c, label=n)
    axes[0, j].invert_yaxis(); axes[0, j].set_title(title, fontsize=9)
    for s, (n, c) in {1: ("Atlantic", BLUE), 2: ("Indian", ORANGE), 3: ("Pacific", AQUA)}.items():
        axes[1, j].plot(y[k], band_cover(d, sectors=(s,), lats=(-19.38, -24.92))[k], lw=0.5, color=c)
    axes[2, j].plot(y[k], lsg_layer_mean(d, 1, -90, 0)[k], lw=0.8, color="k")
    axes[3, j].plot(y[k], global_ts(d)[k], lw=0.4, color="k")
axes[0, 0].legend(fontsize=7, ncol=3, frameon=False)
for i, lab in enumerate(["sector ice edge (°S)", "ice cover of the 19.4 + 24.9°S rows", "SH ocean θ 700–2025 m (K)", "global Ts (K)"]):
    axes[i, 0].set_ylabel(lab, fontsize=8)
for ax in axes.ravel():
    ax.grid(alpha=0.3, lw=0.4)
fig.suptitle("Plain look: the cold state (annual values)")
fig.tight_layout(); fig.savefig(f"{OUT}/look_cold.png", dpi=95); plt.close(fig)

# raw periodograms of the band cover, by sector, cold vs a warm run for comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, (lab, (a, b), title, lats) in zip(axes, [("1232p5", (19300, 24429), "1232.5 warm (30.5 + 36°S rows)", (-30.46, -36.0)),
                                                ("1230", (11600, 16369), "1230 cold (19.4 + 24.9°S rows)", (-19.38, -24.92)),
                                                ("1225", (5700, 7600), "1225 cold stage (19.4 + 24.9°S rows)", (-19.38, -24.92))]):
    d = load(lab); y = d["years"]; k = (y >= a) & (y <= b)
    for s, (n, c) in {1: ("Atlantic", BLUE), 2: ("Indian", ORANGE), 3: ("Pacific", AQUA)}.items():
        x = band_cover(d, sectors=(s,), lats=lats)[k]; x = np.nan_to_num(x - np.nanmean(x))
        f = np.fft.rfftfreq(len(x))[1:]; p = np.abs(np.fft.rfft(x))[1:] ** 2 / len(x)
        ax.loglog(f, p, lw=0.4, color=c, label=n)
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, lw=0.4, which="both"); ax.set_xlabel("frequency (1/yr)")
    for per in (50, 100, 1000):
        ax.axvline(1 / per, color=GREY, lw=0.6, ls=":")
axes[0].set_ylabel("raw periodogram"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/look_cold_spectra.png", dpi=95); plt.close(fig)
for lab, (a, b), title in cases:
    d = load(lab); y = d["years"]; k = (y >= a) & (y <= b)
    for s, n in ((1, "Atl"), (2, "Ind"), (3, "Pac")):
        x = band_cover(d, sectors=(s,), lats=(-19.38, -24.92))[k]
        print(f"{title[:28]:28s} {n}: band mean {np.nanmean(x):.3f}  sd {np.nanstd(x):.3f}")
