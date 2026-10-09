"""Simple figures for the review (annual values unless stated; 100-yr means where stated)."""

import matplotlib.pyplot as plt
import numpy as np

from common import PARENT, block_means, equivalent_edge, global_ts, load, lsg_layer_mean
from gsebm.plasim_global import RUNS

OUT = "../../figures/plasim_fresh"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: the equilibrium states
warm = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
cold = ["1230", "1228p5"]
snow = ["1225"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for labs, col, name in ((warm, ORANGE, "warm states"), (cold, BLUE, "cold states"), (snow, GREY, "snowball")):
    mu, T, E = [], [], []
    for lab in labs:
        d = load(lab); w0, w1 = RUNS[lab].window; k = (d["years"] >= w0) & (d["years"] <= w1)
        mu.append(d["mu"]); T.append(np.nanmean(global_ts(d)[k])); E.append(np.nanmean(equivalent_edge(d)[k]))
    axes[0].plot(mu, T, "-o", color=col, ms=6, lw=1.5, label=name)
    axes[1].plot(mu, E, "-o", color=col, ms=6, lw=1.5, label=name)
axes[0].set_ylabel("global mean surface temperature (K)")
axes[1].set_ylabel("Southern sea-ice edge (° latitude S)\n(latitude enclosing the same area as the ice)")
axes[1].invert_yaxis()
for ax in axes:
    ax.set_xlabel("solar constant μ (W m⁻²)   [present day: 1367]")
    ax.invert_xaxis(); ax.grid(alpha=0.3, lw=0.5); ax.legend(frameon=False)
fig.suptitle("The model's long-term states (each dot = mean of one run's settled period)")
fig.tight_layout(); fig.savefig(f"{OUT}/review_fig1_states.png", dpi=120); plt.close(fig)

# Fig 2: natural variability at fixed μ (600-yr excerpts)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for ax, (lab, y0) in zip(axes, [("1265", 6000), ("1240", 14000), ("1232p5", 21000)]):
    d = load(lab); k = (d["years"] >= y0) & (d["years"] < y0 + 600)
    ax.plot(d["years"][k], equivalent_edge(d, 3)[k], color=AQUA, lw=0.9, label="Pacific sector")
    ax.plot(d["years"][k], equivalent_edge(d, 1)[k], color=BLUE, lw=0.9, label="Atlantic sector")
    ax.plot(d["years"][k], equivalent_edge(d, 2)[k], color=ORANGE, lw=0.9, label="Indian sector")
    ax.invert_yaxis(); ax.set_title(f"μ = {d['mu']:g}"); ax.set_xlabel("model year"); ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("ice edge by sector (° S)"); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Natural variability at fixed μ: 600 years of annual values of the Southern ice edge, by ocean sector")
fig.tight_layout(); fig.savefig(f"{OUT}/review_fig2_variability.png", dpi=120); plt.close(fig)

# Fig 3: response to a step in μ (from the 1240 state at year 14999), log time axis
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
p = load("1240"); kp = (p["years"] > 14899) & (p["years"] <= 14999)
for lab, col in (("1237p5", ORANGE), ("1232p5", BLUE)):
    d = load(lab); t = d["years"] - 14999
    axes[0].semilogx(t, equivalent_edge(d), color=col, lw=0.6, alpha=0.8, label=f"μ 1240 → {d['mu']:g}")
    axes[1].semilogx(t, lsg_layer_mean(d, 1, -90, 0), color=col, lw=1.0, label=f"μ 1240 → {d['mu']:g}")
    w0, w1 = RUNS[lab].window; kw = (d["years"] >= w0) & (d["years"] <= w1)
    axes[0].axhline(np.nanmean(equivalent_edge(d)[kw]), color=col, ls="--", lw=0.8)
    axes[1].axhline(np.nanmean(lsg_layer_mean(d, 1, -90, 0)[kw]), color=col, ls="--", lw=0.8)
axes[0].axhline(np.nanmean(equivalent_edge(p)[kp]), color=GREY, ls=":", lw=1, label="before the step")
axes[1].axhline(np.nanmean(lsg_layer_mean(p, 1, -90, 0)[kp]), color=GREY, ls=":", lw=1, label="before the step")
axes[0].invert_yaxis(); axes[0].set_ylabel("Southern ice edge (° S)")
axes[1].set_ylabel("Southern Ocean temperature, 700–2025 m (K)")
for ax in axes:
    ax.set_xlabel("years after the step (log scale)"); ax.grid(alpha=0.3, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=8); ax.axvspan(1, 50, color="0.9", zorder=0)
axes[0].text(3, axes[0].get_ylim()[1] + 0.1, "fast stage", fontsize=9, color=GREY, va="top")
fig.suptitle("After a step down in μ: a fast stage (decades, shaded) and a slow stage that follows the ocean "
             "(dashed: final means)")
fig.tight_layout(); fig.savefig(f"{OUT}/review_fig3_step.png", dpi=120); plt.close(fig)

# Fig 4: the 1230 run
d = load("1230"); y = d["years"]; k = (y >= 4450) & (y <= 8600)
E, H = equivalent_edge(d), lsg_layer_mean(d, 1, -90, 0)
yb, Eb = block_means(y, E, 4500, 8600); _, Hb = block_means(y, H, 4500, 8600)
e32 = load("1232p5"); w0, w1 = RUNS["1232p5"].window; k32 = (e32["years"] >= w0) & (e32["years"] <= w1)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(y[k], E[k], color="0.7", lw=0.5, label="annual")
axes[0].plot(yb + 50, Eb, "o-", color=BLUE, ms=3.5, lw=1.2, label="100-yr means")
axes[0].axhline(np.nanmean(equivalent_edge(e32)[k32]), color=ORANGE, ls="--", lw=1,
                label="settled edge of the coldest warm run (μ 1232.5)")
axes[0].invert_yaxis(); axes[0].set_ylabel("Southern ice edge (° S)"); axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(y[k], H[k], color=BLUE, lw=1.0)
axes[1].axhline(np.nanmean(lsg_layer_mean(e32, 1, -90, 0)[k32]), color=ORANGE, ls="--", lw=1)
axes[1].set_ylabel("Southern Ocean temperature,\n700–2025 m (K)")
for ax in axes:
    ax.grid(alpha=0.3, lw=0.5)
    for x in (7200, 7824):
        ax.axvline(x, color=GREY, ls=":", lw=1)
axes[1].set_xlabel("model year (μ lowered from 1265 to 1230 at year 4499)")
axes[0].text(7210, 25.5, "ocean stops\ncooling", fontsize=8, color=GREY)
axes[0].text(7835, 25.5, "jump\nstarts", fontsize=8, color=GREY)
fig.suptitle("The μ = 1230 run: the edge creeps past the last warm state together with the ocean, "
             "pauses, then jumps")
fig.tight_layout(); fig.savefig(f"{OUT}/review_fig4_1230.png", dpi=120); plt.close(fig)
print("ok")
