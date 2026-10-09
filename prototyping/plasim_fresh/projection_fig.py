"""Figure: projection of slow natural fluctuations onto the full forced response, across μ."""

import json

import matplotlib.pyplot as plt
import numpy as np

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
P = json.load(open(TMP + "projection.json"))["RES"]; S = json.load(open(TMP + "slow_analysis.json"))["S1"]
BLUE, ORANGE, AQUA, GREY, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#4a3aa7"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
runs = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]
txt = [f"{float(l.replace('p', '.').split('_')[0]):g}" + (" (2nd)" if "new" in l else "") for l in runs]
x = np.arange(len(runs)); prim = P["L100_common_forced_d0"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
ax = axes[0]
ax.errorbar(x, [prim[l]["f_full"] for l in runs], yerr=[2 * np.nan_to_num(prim[l]["f_full_se"]) for l in runs], fmt="o", color="k", capsize=3,
            label="full forced pattern (all fields)")
ax.plot(x, [prim[l]["fk"]["t1025_2000"] for l in runs], "s", color=VIO, label="deep ocean alone (1025–2000 m)")
ax.plot(x, [prim[l]["fk"]["ts"] for l in runs], "^", color=ORANGE, label="surface temperature alone")
lo = [min(P[v][l]["f_full"] for v in P) for l in runs]; hi = [max(P[v][l]["f_full"] for v in P) for l in runs]
ax.vlines(x, lo, hi, color="k", alpha=0.2, lw=6)
ax.set_ylabel("share of slow (century) natural variance\nlying along the forced-response pattern"); ax.legend(fontsize=7, frameon=False)
ax.set_title("How much of the slow variability looks like the forced response?")
ax = axes[1]
ax.errorbar(x, [prim[l]["g"] for l in runs], yerr=[2 * np.nan_to_num(prim[l]["g_se"]) for l in runs], fmt="o", color=VIO, capsize=3)
lo = [min(P[v][l]["g"] for v in P) for l in runs]; hi = [max(P[v][l]["g"] for v in P) for l in runs]
ax.vlines(x, lo, hi, color=VIO, alpha=0.2, lw=6); ax.axhline(0, color=GREY, lw=0.6); ax.axhline(1, color=GREY, lw=0.6, ls=":")
ax.set_ylabel("deep gain g: forced-pattern amplitude in the deep ocean\nper unit of forced-pattern amplitude at the surface")
ax.set_title("Does the deep ocean join the slow surface swings?\n(1 = as much as in the forced response; 0 = not at all)")
ax = axes[2]
for key, c, nm in (("t1025_2000", VIO, "deep ocean 1025–2000 m"), ("th300_600", AQUA, "ocean 300–600 m"), ("ts", ORANGE, "surface temperature")):
    v = [S[l]["L100_Ts_d0"][key]["r"] for l in runs]
    lo = [min(S[l][vk][key]["r"] for vk in S[l]) for l in runs]; hi = [max(S[l][vk][key]["r"] for vk in S[l]) for l in runs]
    ax.vlines(x, lo, hi, color=c, alpha=0.2, lw=6); ax.plot(x, v, "o-", color=c, label=nm)
ax.set_ylabel("pattern correlation: slow fluctuation vs forced change"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Shape only (from the previous analysis)")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(txt, rotation=45, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    ax.axvspan(7.5, 9.5, color="0.92", zorder=0); ax.set_xlabel("μ (W m⁻²); cold states shaded")
fig.suptitle("Slow natural fluctuations against the full forced response (error bars ±2 s.e. from 1000-yr segments; pale bars: range over 16 variants)",
             fontsize=10)
fig.tight_layout(); fig.savefig("../../figures/plasim_fresh/projection.png", dpi=110); plt.close(fig)
print("ok")
