"""Figures for the sharpened analysis (all stationary data; robustness shown)."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import band_cover, load
from gsebm.plasim_global import run_mu

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
OUT = "../../figures/plasim_fresh"
R = json.load(open(TMP + "sharpen.json")); STAT = json.load(open(TMP + "stationary.json"))
A, B, C = R["A"], R["B"], R["C"]
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
lab_txt = lambda l: f"{run_mu(l):g}" + (" (2nd)" if l == "1235_new_IC" else "")

# ---- Fig S1: Atlantic oscillation --------------------------------------------------------------------
runs = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
x = np.arange(len(runs))
fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
ax = axes[0]
ax.errorbar(x - 0.1, [A[l]["full"]["per_pg"] for l in runs], yerr=[2 * np.nan_to_num(A[l]["per_pg_seg"][1]) for l in runs],
            fmt="o", color=BLUE, capsize=3, label="spectral peak")
ax.errorbar(x + 0.1, [A[l]["full"]["per_acf"] for l in runs], yerr=[2 * np.nan_to_num(A[l]["per_acf_seg"][1]) if l != "1250" else 0 for l in runs],
            fmt="s", color=ORANGE, capsize=3, label="autocorrelation maximum")
vmin = [min(v["per_pg"] for v in A[l]["variants"].values()) for l in runs]; vmax = [max(v["per_pg"] for v in A[l]["variants"].values()) for l in runs]
ax.vlines(x - 0.1, vmin, vmax, color=BLUE, alpha=0.25, lw=6, label="range over robustness variants")
ax.set_ylabel("period (yr)"); ax.set_title("Period of the Atlantic ice–gyre oscillation"); ax.legend(fontsize=7, frameon=False)
ax = axes[1]
ax.errorbar(x, [A[l]["full"]["reg"] for l in runs], yerr=[2 * np.nan_to_num(A[l]["reg_seg"][1]) for l in runs], fmt="o", color=BLUE, capsize=3)
vmin = [min(v["reg"] for v in A[l]["variants"].values()) for l in runs]; vmax = [max(v["reg"] for v in A[l]["variants"].values()) for l in runs]
ax.vlines(x, vmin, vmax, color=BLUE, alpha=0.25, lw=6)
ax.set_ylabel("autocorrelation at one period\n(1 = perfectly regular)"); ax.set_title("Regularity of the oscillation")
ax = axes[2]
means, p99 = [], []
for l in runs:
    d = load(l); y = d["years"]; k = (y >= STAT[l]["0.95"]) & (y <= STAT[l]["end"])
    a30 = d["sic"][k, int(np.argmin(np.abs(d["lat"] + 30.46))), 1]; means.append(np.nanmean(a30)); p99.append(np.nanpercentile(a30, 99))
ax.plot(x, means, "o-", color=GREY, label="average cover")
ax.errorbar(x, p99, yerr=[2 * np.nan_to_num(A[l]["reach99_seg"][1]) for l in runs], fmt="o-", color=BLUE, capsize=3, label="99th percentile (largest swings)")
ax.set_ylabel("Atlantic 30.5°S row: fraction of the year ice-covered"); ax.legend(fontsize=7, frameon=False)
ax.set_title("How far the swings reach into the next row")
ax = axes[3]
ax.errorbar(x - 0.1, [A[l]["leak"] for l in runs], yerr=[2 * np.nan_to_num(A[l]["leak_seg"][1]) for l in runs], fmt="o", color=BLUE, capsize=3, label="annual values")
ax.errorbar(x + 0.1, [A[l]["leak_rm10"] for l in runs], yerr=[2 * np.nan_to_num(A[l]["leak_rm10_seg"][1]) for l in runs], fmt="s", color=ORANGE, capsize=3, label="10-yr running means")
ax.set_ylabel("max correlation, Atlantic ice → Indian/Pacific ice\n(lags 0 to one period)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("How strongly the Indian/Pacific ice follows")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels([lab_txt(l) for l in runs], rotation=45, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlabel("μ (W m⁻²)"); ax.axvspan(-0.5, 0.5, color="0.92", zorder=0)
fig.suptitle("The Atlantic ice–gyre oscillation along the warm branch (all stationary years; error bars ±2 s.e. from 1000-yr segments; "
             "1250, shaded, contains millennial events)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/sharpen_atlantic.png", dpi=110); plt.close(fig)

# ---- Fig S2: slow gain ------------------------------------------------------------------------------
runs = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]
x = np.arange(len(runs))
fig, ax = plt.subplots(figsize=(12, 4.6))
nat = np.array([-B[l]["base"]["slope"] for l in runs]); se = np.array([B[l]["base"]["se"] for l in runs])
f = np.array([-B[l]["forced"] for l in runs])
vmin = [min(-v["slope"] for v in B[l]["variants"].values() if np.isfinite(v["slope"])) for l in runs]
vmax = [max(-v["slope"] for v in B[l]["variants"].values() if np.isfinite(v["slope"])) for l in runs]
ax.vlines(x, vmin, vmax, color=BLUE, alpha=0.25, lw=8, label="natural, range over 36 robustness variants")
ax.errorbar(x, nat, yerr=2 * se, fmt="o", color=BLUE, capsize=3, label="natural: 100-yr means within the run (±2 s.e.)")
ax.plot(x, f, "s", color="k", ms=7, label="between neighbouring settled states")
for xi, l in zip(x, runs):
    ax.annotate(f"r={B[l]['base']['r']:+.2f}", (xi, -1.2), ha="center", fontsize=7, color=GREY)
ax.set_xticks(x); ax.set_xticklabels([lab_txt(l) for l in runs], rotation=45, fontsize=8)
ax.set_ylabel("extra Indian/Pacific ice per K of cooling\nof the water next to the edge (10¹² m² per K)")
ax.set_ylim(-2.5, 24); ax.grid(alpha=0.3, lw=0.4); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_xlabel("μ (W m⁻²); cold states at the right")
ax.set_title("Slow ice–ocean coupling: natural fluctuations against the differences between settled states (all stationary years)")
fig.tight_layout(); fig.savefig(f"{OUT}/sharpen_gain.png", dpi=110); plt.close(fig)

# ---- Fig S3: slow albedo variance and decadal memory -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, base, ylab, logy in ((axes[0], "sw_tot_L50", "sd of 50-yr means of sunlight absorbed south of 20°S (PW)", True),
                             (axes[1], "acf10", "10-yr autocorrelation of the Indian/Pacific ice area", False)):
    val = [C[l][f"{base}_d0"] for l in runs]; se = [2 * np.nan_to_num(C[l][f"{base}_d0_seg"][1]) for l in runs]
    keys = [f"{base}_d0", f"{base}_d1", f"{base}_d0_q0.9_h0", f"{base}_d0_q0.99_h0", f"{base}_d0_q0.95_h1"]
    vmin = [min(C[l][k] for k in keys) for l in runs]; vmax = [max(C[l][k] for k in keys) for l in runs]
    ax.vlines(x, vmin, vmax, color=BLUE, alpha=0.25, lw=8, label="range over robustness variants")
    ax.errorbar(x, val, yerr=se, fmt="o", color=BLUE, capsize=3, label="all stationary years (±2 s.e.)")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([lab_txt(l) for l in runs], rotation=45, fontsize=8)
    ax.set_ylabel(ylab); ax.grid(alpha=0.3, lw=0.4, which="both"); ax.set_xlabel("μ (W m⁻²); cold states at the right")
    ax.axvspan(-0.5, 3.5, color="0.94", zorder=0)
axes[0].legend(fontsize=7.5, frameon=False)
axes[0].set_title("Slow albedo fluctuations (shaded: runs with events or the ~110-yr oscillation)")
axes[1].set_title("Decadal memory of the Indian/Pacific ice")
fig.tight_layout(); fig.savefig(f"{OUT}/sharpen_variance_memory.png", dpi=110); plt.close(fig)
print("ok")
