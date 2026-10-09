"""Summary figure: cold state against warm state, and the 1225 creep towards the snowball."""

import json

import matplotlib.pyplot as plt
import numpy as np

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
R = json.load(open(TMP + "cold_compare.json"))
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
warm = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
names = ["1245", "1242.5", "1240", "1237.5", "1235", "1235 (2nd)", "1233.75", "1232.5", "1230 cold", "1228.5 cold"]
labs = warm + ["1230", "1228p5"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
ax = axes[0]
x = np.arange(len(labs))
for q, c, n in (("ip", AQUA, "Indian/Pacific ice"), ("atl", BLUE, "Atlantic ice")):
    v = [R["R1"][l][q]["acf"][1] for l in labs]  # lag 2
    ax.plot(x, [R["R1"][l][q]["acf"][2] for l in labs], "o-", color=c, label=f"{n}, lag 5 yr")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=50, fontsize=7.5); ax.axvspan(7.5, 9.5, color="0.92", zorder=0)
ax.set_ylabel("autocorrelation of Southern ice area at 5 yr"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Memory of the ice: warm runs and cold states (shaded)")
ax = axes[1]
lat = [36.0, 24.9, 19.4]; v = [-R["R3"]["rst"]["warm36"], -R["R3"]["rst"]["cold25"], -R["R3"]["rst"]["cold19"]]
ax.bar([0, 1, 2], v, color=[ORANGE, BLUE, BLUE], width=0.6)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["36°S row\n(warm states)", "24.9°S row\n(cold states)", "19.4°S row\n(cold states)"])
ax.set_ylabel("sunlight lost per unit of annual ice cover (W m⁻²)"); ax.set_title("Each unit of ice costs more sunlight nearer the tropics")
for i, val in enumerate(v):
    ax.text(i, val + 2, f"{val:.0f}", ha="center")
ax = axes[2]
steps = ["1245→\n1242.5", "1242.5→\n1240", "1240→\n1237.5", "1237.5→\n1235", "1235→\n1233.75", "1233.75→\n1232.5", "1230→\n1228.5\n(cold)"]
vals = [0.86, 0.99, 0.60, 0.85, 2.30, 1.86, R["cold_sens"]["ice"]]
ax.bar(range(len(vals)), vals, color=[ORANGE] * 6 + [BLUE], width=0.6)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(steps, fontsize=7)
ax.set_ylabel("Southern ice added per W m⁻² of μ (10¹² m²)"); ax.set_title("How easily the ice advances between settled states")
ax = axes[3]
segs = [r["start"] for r in R["R5"]]; t = [s + 150 for s in segs]
ax.plot(t, [r["ip_sd_det"] for r in R["R5"]], "o-", color=AQUA, label="Indian/Pacific ice: size of fluctuations (sd, 10¹² m²)")
ax.plot(t, [r["atl_sd_det"] for r in R["R5"]], "o-", color=BLUE, label="Atlantic ice: size of fluctuations")
ax.plot(t, [r["ip_acf_det"][1] for r in R["R5"]], "s--", color=AQUA, label="Indian/Pacific ice: memory at 5 yr")
ax.plot(t, [r["atl_acf_det"][1] for r in R["R5"]], "s--", color=BLUE, label="Atlantic ice: memory at 5 yr")
ax.axvline(7590, color="k", ls=":", lw=1); ax.text(7560, 1.0, "snowball\nrunaway", ha="right", fontsize=7)
ax.set_xlabel("model year (μ = 1225 run, cold stage; 300-yr segments, trend removed)"); ax.legend(fontsize=6.5, frameon=False, loc="upper right")
ax.set_title("Approaching the snowball (1225 run)"); ax.set_ylim(-0.3, 1.15)
for a in axes:
    a.grid(alpha=0.3, lw=0.4)
fig.tight_layout(); fig.savefig("../../figures/plasim_fresh/cold_vs_warm.png", dpi=110); plt.close(fig)
print("ok")
