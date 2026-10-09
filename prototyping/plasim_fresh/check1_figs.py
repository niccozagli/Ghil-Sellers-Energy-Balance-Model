"""Figures for check 1."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import PARENT
from gsebm.plasim_global import RUNS

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
O10 = TMP + "ocean10/"
OUT = "../../figures/plasim_fresh"
TF = 271.25
S = np.load(TMP + "check1_series.npy", allow_pickle=True).item()
R = json.load(open(TMP + "check1.json")); SENS = json.load(open(TMP + "check1_timing_sens.json"))
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ALL = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC",
       "1233p75", "1232p5", "1230", "1228p5", "1225"]

# ---- Fig 1: sections for every settled state ----------------------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True, sharey=True)
for ax, lab in zip(axes.ravel(), ALL):
    z = np.load(O10 + lab + ".npz"); w0, w1 = RUNS[lab].window
    k = (z["block_start"] >= w0) & (z["block_start"] + 9 <= w1)
    lat, vlat, db = z["lsg_lat"], z["lsg_vector_lat"], z["depth_bounds"]
    zc, zi = db.mean(axis=1), db[:, 1]
    th = np.nanmean(z["potential_temperature"][k], axis=0) - TF
    psi = np.nancumsum(np.nanmean(z["meridional_volume_transport"][k], axis=0), axis=1) / 1e6
    s, sv = lat < -5, vlat < -5
    im = ax.pcolormesh(lat[s], zc, th[s].T, vmin=0, vmax=5, cmap="viridis", shading="nearest")
    cs = ax.contour(vlat[sv], zi, psi[sv].T, levels=[-50, -40, -30, -20, -10, 10, 20, 30], colors="w", linewidths=0.7)
    ax.clabel(cs, fontsize=6, fmt="%d")
    E = S[lab]["E_set"]
    if np.isfinite(E):
        ax.axvline(-E, color="#e34948", lw=1.6, ls="--")
        lk = R["A1"][lab]["Lk"]
        ax.plot(-lk, 60, "v", color="#e34948", ms=8, mec="w", clip_on=False)
    ax.set_ylim(4000, 0); ax.set_xlim(-76, -5)
    name = f"μ = {S[lab]['mu']:g}" + (" (2nd run)" if lab == "1235_new_IC" else "")
    ax.set_title(name + ("  (snowball)" if lab == "1225" else (f"  edge {E:.1f}°S" if np.isfinite(E) else "")), fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("latitude")
for ax in axes[:, 0]:
    ax.set_ylabel("depth (m)")
cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
cb.set_label("ocean temperature above freezing (K, colour)")
fig.suptitle("Southern Ocean in every settled state (zonal means). Colour: temperature above freezing. White lines: overturning "
             "streamlines in Sv (positive = clockwise: north at the top, sinking on the north side; negative = anticlockwise).\n"
             "Red dashed line: ice edge. Red triangle: latitude of strongest convection (0–1000 m).", fontsize=10)
fig.savefig(f"{OUT}/check1_sections_all.png", dpi=100, bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: summary across μ ----------------------------------------------------------------------
labs = ALL[:-1]
mu = np.array([S[l]["mu"] for l in labs]); cold = np.array([l in ("1230", "1228p5") for l in labs])
g = lambda key: np.array([R["A1"][l][key] for l in labs])
E = np.array([S[l]["E_set"] for l in labs])
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
ax = axes[0]
for m, c, n in ((~cold, None, "warm"), (cold, None, "cold")):
    mk = "o" if n == "warm" else "s"
    ax.plot(mu[m], E[m], mk, color="k", ms=6, label=f"ice edge ({n})")
    ax.plot(mu[m], g("B")[m], mk, color=BLUE, mfc="none", ms=8, mew=1.5, label=f"overturning boundary ({n})")
    ax.plot(mu[m], g("Lk")[m], mk, color=ORANGE, ms=4, label=f"strongest convection ({n})")
ax.invert_yaxis(); ax.set_ylabel("latitude (°S)"); ax.legend(fontsize=6.5, frameon=False, ncol=1)
ax.set_title("Where things sit")
ax = axes[1]
ax.plot(mu[~cold], g("Tedge")[~cold], "o-", color=ORANGE, label="at the edge, 100–700 m")
ax.plot(mu[cold], g("Tedge")[cold], "s", color=ORANGE)
ax.plot(mu[~cold], (g("H") - TF)[~cold], "o-", color=BLUE, label="whole SH, 700–2025 m")
ax.plot(mu[cold], (g("H") - TF)[cold], "s", color=BLUE)
ax.set_ylabel("ocean temperature above freezing (K)"); ax.legend(fontsize=7, frameon=False); ax.set_ylim(0, None)
ax.set_title("How much heat above freezing is left")
ax = axes[2]
ax.plot(mu[~cold], g("pole")[~cold], "o-", color=BLUE, label="poleward cell (clockwise)")
ax.plot(mu[~cold], -g("equ")[~cold], "o-", color=ORANGE, label="equatorward cell (anticlockwise)")
ax.plot(mu[cold], g("pole")[cold], "s", color=BLUE); ax.plot(mu[cold], -g("equ")[cold], "s", color=ORANGE)
ax.set_ylabel("overturning strength (Sv)"); ax.legend(fontsize=7, frameon=False); ax.set_title("Strength of the two cells")
ax = axes[3]
ax.plot(mu[~cold], -g("release")[~cold], "o-", color="k", label="heat released within ±5° of the edge (PW)")
ax.plot(mu[cold], -g("release")[cold], "s", color="k")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(mu[~cold], g("dS")[~cold], "o--", color=AQUA); ax2.plot(mu[cold], g("dS")[cold], "s", color=AQUA)
ax2.set_ylabel("salinity contrast at the edge, 700–1025 m minus 50–100 m (psu)", color=AQUA, fontsize=8)
ax.set_ylabel("heat release at the edge (PW, black)"); ax.set_title("Heat released at the edge; salinity layering")
for ax in axes:
    ax.set_xlabel("solar constant μ (W m⁻²)"); ax.invert_xaxis(); ax.grid(alpha=0.3, lw=0.4)
fig.suptitle("Check 1A across μ (circles: warm states; squares: cold states)")
fig.tight_layout(); fig.savefig(f"{OUT}/check1_summary_mu.png", dpi=110); plt.close(fig)

# ---- Fig 3: timing dot plot -----------------------------------------------------------------------
runs = ["1288", "1265", "1245", "1235", "1233p75", "1232p5"]
cols = {"W": ORANGE, "Hu": AQUA, "H": BLUE, "Hd": YELLOW}
names = {"W": "warm upper water next to the edge (0–700 m)", "Hu": "SH ocean 0–700 m", "H": "SH ocean 700–2025 m",
         "Hd": "global ocean below 2025 m"}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, src, title in ((axes[0], "pre", "halfway time, pre-set rule (10-yr blocks, must stay above 0.5 for 50 yr)"),
                       (axes[1], "sens", "added check: 50-yr blocks, first crossing")):
    for i, lab in enumerate(runs):
        get = (lambda q: R["B1"][lab][q]["t50s"]) if src == "pre" else (lambda q: SENS[lab][q])
        off = {"W": 0.24, "Hu": 0.12, "H": -0.12, "Hd": -0.24}
        for q, c in cols.items():
            ax.plot(get(q), i + off[q], "o", color=c, ms=7, label=names[q] if i == 0 else None)
        ax.plot(get("E"), i, "D", color="k", ms=8, label="ice edge (slow part)" if i == 0 else None)
        ax.axhline(i - 0.4, color="0.85", lw=0.5)
    ax.set_xscale("log"); ax.set_xlim(60, 2500); ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.set_xlabel("years after the step (log scale)"); ax.set_title(title, fontsize=9)
axes[0].set_yticks(range(len(runs)))
axes[0].set_yticklabels([f"1367 → 1288" if l == "1288" else (f"1367 → 1265" if l == "1265" else
                         f"{PARENT[l][0]} → {S[l]['mu']:g}") for l in runs])
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("Check 1B: when does each quantity complete half of its slow adjustment after a step in μ? "
             "(runs where the ice signal is above the noise)")
fig.tight_layout(); fig.savefig(f"{OUT}/check1_timing.png", dpi=110); plt.close(fig)

# ---- Fig 4: progress curves for two runs -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, lab in zip(axes, ("1245", "1232p5")):
    s = S[lab]; st = s["starts"]; y0 = PARENT[lab][1]; t = st + 5 - y0
    for q, c in list(cols.items()) + [("E", "k")]:
        o = R["B1"][lab][q]
        p = (s["ser"][q] - o["fast"]) / (o["final"] - o["fast"])
        ax.semilogx(t, p, color=c, lw=1.6 if q == "E" else 1.2, alpha=0.7 if q == "E" else 1,
                    label=("ice edge (slow part, 10-yr means)" if q == "E" else names[q]))
    ax.axhline(0.5, color=GREY, ls=":", lw=1); ax.axhline(1, color=GREY, lw=0.6); ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylim(-0.5, 1.5); ax.set_xlim(10, 9000); ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.set_title(f"μ {PARENT[lab][0]} → {S[lab]['mu']:g}"); ax.set_xlabel("years after the step (log scale)")
axes[0].set_ylabel("progress of the slow adjustment\n(0 = level 20–50 yr after the step, 1 = new settled level)")
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("Check 1B: the slow adjustment of the ice and of each ocean layer (10-yr means)")
fig.tight_layout(); fig.savefig(f"{OUT}/check1_progress.png", dpi=110); plt.close(fig)

# ---- Fig 5: slopes ---------------------------------------------------------------------------------
labs = [l for l in ALL[:-1]]
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(labs))
f = np.array([R["C1"][l]["forced"] for l in labs]); sl = np.array([R["C1"][l]["slow"] for l in labs])
nat = np.array([R["C1"][l]["natural"] for l in labs]); se = np.array([R["C1"][l]["se"] for l in labs])
ax.plot(x, f, "s", color="k", ms=7, label="between neighbouring settled states")
ax.plot(x, sl, "^", color=ORANGE, ms=8, label="slow stage after the step")
ax.errorbar(x, nat, yerr=2 * se, fmt="o", color=BLUE, ms=6, capsize=3, label="natural: 100-yr means within the run (±2 s.e.)")
for xi, l in zip(x, labs):
    ax.annotate(R["C1"][l]["verdict"].replace("agrees, weak r", "agrees,\nweak r").replace("no slow link", "no slow\nlink")
                .replace("not resolvable", "not\nresolvable"), (xi, -5.5), fontsize=6.5, ha="center", va="top", color=GREY)
ax.set_ylim(-9, 30); ax.axhline(0, color=GREY, lw=0.6)
ax.set_ylabel("ice edge change per K of ocean (700–2025 m) change\n(° of latitude per K)")
ax.set_xticks(x); ax.set_xticklabels([f"{S[l]['mu']:g}" + ("\n(2nd)" if l == "1235_new_IC" else "") for l in labs])
ax.set_xlabel("solar constant μ (W m⁻²); runs evenly spaced, cold states at the right"); ax.grid(alpha=0.3, lw=0.4)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_title("Check 1C: does the ice follow the ocean the same way in natural fluctuations as between states?")
fig.tight_layout(); fig.savefig(f"{OUT}/check1_slopes.png", dpi=110); plt.close(fig)
print("ok")
