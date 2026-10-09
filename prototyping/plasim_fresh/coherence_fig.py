"""Figure: coherence of slow fluctuations across μ, and each index's correlation with SH surface temperature."""

import json

import matplotlib.pyplot as plt
import numpy as np

from coherence import IDX, RUNS, blocks

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
R = json.load(open(TMP + "coherence.json"))
NAMES = ["less Atlantic ice", "less Indian ice", "less Pacific ice", "water next to edge 0–700 m", "E Pacific 0–100 m",
         "E Pacific 300–600 m", "E Atlantic 0–100 m", "E Atlantic 300–600 m", "SH ocean 700–2025 m"]
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
txt = [f"{float(l.replace('p', '.').split('_')[0]):g}" + (" (2nd)" if "new" in l else "") for l in RUNS]
x = np.arange(len(RUNS))
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
ax = axes[0]
for m, c, nm in (("C1 mean corr", "k", "mean correlation among the 10 Southern indices"), ("C1 λ1/N", "#52514e", "share of the leading pattern (λ1/N)")):
    v = [R["L50_d0"][l][m][0] for l in RUNS]; se = [2 * np.nan_to_num(R["L50_d0"][l][m][1]) for l in RUNS]
    lo = [min(R[k][l][m][0] for k in R) for l in RUNS]; hi = [max(R[k][l][m][0] for k in R) for l in RUNS]
    ax.vlines(x, lo, hi, color=c, alpha=0.15, lw=7)
    ax.errorbar(x, v, yerr=se, fmt="o-" if m == "C1 mean corr" else "s--", color=c, capsize=3, label=nm)
ax.set_ylabel("coherence of 50-yr means"); ax.legend(fontsize=7, frameon=False); ax.set_title("Do the parts of the Southern system move together?")
ax = axes[1]
for m, c in (("C2 Ts 20–50°S", "#eb6834"), ("C2 ocean 0–100 m", "#1baf7a"), ("C2 ice cover", "#2a78d6")):
    v = [R["L50_d0"][l][m][0] for l in RUNS]; se = [2 * np.nan_to_num(R["L50_d0"][l][m][1]) for l in RUNS]
    ax.errorbar(x, v, yerr=se, fmt="o-", color=c, capsize=3, label=m.replace("C2 ", ""))
ax.set_ylabel("mean correlation between grid cells (50-yr means)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Do neighbouring places fluctuate together? (spatial correlation)")
ax = axes[2]
for j, (nm, c) in enumerate(zip(NAMES, COLS)):
    r = []
    for l in RUNS:
        B = np.column_stack([blocks(IDX[l][:, i], 50) for i in range(IDX[l].shape[1])])
        ok = np.all(np.isfinite(B), axis=1); r.append(np.corrcoef(B[ok, j], B[ok, -1])[0, 1])
    ax.plot(x, r, "o-", color=c, lw=1.2, ms=4, label=nm)
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("correlation with Southern Hemisphere surface temperature (50-yr means)")
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right"); ax.set_title("Which parts swing with the rest?")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(txt, rotation=45, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    ax.axvspan(7.5, 9.5, color="0.92", zorder=0); ax.set_xlabel("μ (W m⁻²); cold states shaded")
fig.suptitle("Slow (50-yr-mean) fluctuations: do they become more correlated towards the warm → cold transition? "
             "(error bars ±2 s.e. from 1000-yr segments; pale bars: range over variants)", fontsize=10)
fig.tight_layout(); fig.savefig("../../figures/plasim_fresh/coherence.png", dpi=110); plt.close(fig)
print("ok")
