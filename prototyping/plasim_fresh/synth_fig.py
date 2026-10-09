"""Synthesis figure: candidate fingerprints of the approach to the warm → cold transition, across μ.
All quantities were computed in earlier agreed analyses; oscillation periods are raw-periodogram peaks."""

import json

import matplotlib.pyplot as plt
import numpy as np

from analysis_c import sector_sw  # noqa: E402  (prints its own tables when imported)
from common import band_cover, equivalent_edge, load
from gsebm.plasim_global import RUNS

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
OUT = "../../figures/plasim_fresh"
R1 = json.load(open(TMP + "check1.json"))
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
WARM = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
COLD = ["1230", "1228p5"]


def win(lab):
    d = load(lab); w0, w1 = RUNS[lab].window; k = (d["years"] >= w0) & (d["years"] <= w1)
    return d, k


def peak_period(x, lo=25, hi=200):
    x = np.nan_to_num(x - np.nanmean(x)); f = np.fft.rfftfreq(len(x)); p = np.abs(np.fft.rfft(x)) ** 2
    k = (f > 1 / hi) & (f < 1 / lo)
    return 1 / f[k][np.argmax(p[k])]


rows = {}
for lab in WARM + COLD:
    d, k = win(lab)
    w = d["gw"] / d["gw"].sum()
    rows[lab] = dict(mu=d["mu"], Ts=float(np.nanmean(d["ts"][k] @ w)))
    south = d["lat"] < 0
    from common import row_ocean_area
    rows[lab]["iceS"] = float(np.nansum(np.nanmean(d["sic"][k, :, 0], axis=0)[south] * row_ocean_area(d)[south]) / 1e12)
    if lab in ("1288", "1265"):
        rows[lab]["period"] = peak_period(equivalent_edge(d, 3)[k]); rows[lab]["osc"] = "Pacific"
    elif lab not in COLD and lab != "1312":
        rows[lab]["period"] = peak_period(band_cover(d, sectors=(1,))[k]); rows[lab]["osc"] = "Atlantic"
    years, sw = sector_sw(lab)
    w0, w1 = RUNS[lab].window; kk = (years >= w0) & (years <= w1); a = sw[kk] - np.nanmean(sw[kk], axis=0)
    m = (len(a) // 50) * 50; b = np.nanmean(a[:m].reshape(-1, 50, 3), axis=1)
    rows[lab]["sd50_tot"] = float(np.nanstd(b.sum(axis=1), ddof=1)); rows[lab]["sd50_pac"] = float(np.nanstd(b[:, 2], ddof=1))
    if lab in WARM[4:]:
        c = band_cover(d)[k]; c = c - np.nanmean(c); n = len(c)

        def acf10(x):
            x = x - np.nanmean(x); return float(np.sum(x[:-10] * x[10:]) / (len(x) - 10) / np.var(x))
        rows[lab]["acf10"] = acf10(c); rows[lab]["acf10_h"] = (acf10(c[: n // 2]), acf10(c[n // 2:]))
for lab in rows:
    if lab in R1["A1"]:
        rows[lab]["Tedge"] = R1["A1"][lab]["Tedge"]; rows[lab]["H"] = R1["A1"][lab]["H"] - 271.25
        rows[lab]["equ"] = -R1["A1"][lab]["equ"]
print({k: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()} for k, r in rows.items()})

mu = lambda labs: np.array([rows[l]["mu"] for l in labs])
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
regimes = [(1316, 1262, "events and the ~110-yr\nPacific oscillation"), (1262, 1231.3, "overturning locked to the edge;\nAtlantic cycle"),
           (1231.3, 1226, "cold")]
for ax in axes.ravel():
    for i, (a, b, t) in enumerate(regimes):
        ax.axvspan(b, a, color=["0.96", "0.90", "0.82"][i], zorder=0)
    ax.set_xlim(1316, 1226); ax.grid(alpha=0.3, lw=0.4); ax.set_xlabel("solar constant μ (W m⁻²)")
for (a, b, t) in regimes:
    axes[0, 0].text(0.5 * (a + b), 1.02, t, transform=axes[0, 0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=GREY)

# 1: what we want to anticipate
ax = axes[0, 0]
mids, sens = [], []
seq = WARM
for a, b in zip(seq[:-1], seq[1:]):
    mids.append(0.5 * (rows[a]["mu"] + rows[b]["mu"])); sens.append((rows[b]["iceS"] - rows[a]["iceS"]) / (rows[a]["mu"] - rows[b]["mu"]))
ax.step(mids, sens, where="mid", color="k"); ax.plot(mids, sens, "o", color="k", ms=4)
ax.set_ylabel("Southern ice area added per W m⁻² of μ\n(10¹² m², between neighbouring warm states)")
ax.set_title("What to anticipate: the equilibrium sensitivity", fontsize=9, pad=28)
# 2: heat reserve
ax = axes[0, 1]
for labs, mk in ((WARM, "o-"), (COLD, "s")):
    ax.plot(mu(labs), [rows[l]["Tedge"] for l in labs], mk, color=ORANGE, label="water at the edge, 100–700 m" if mk == "o-" else None)
    ax.plot(mu(labs), [rows[l]["H"] for l in labs], mk, color=BLUE, label="Southern Ocean, 700–2025 m" if mk == "o-" else None)
ax.set_ylabel("ocean temperature above freezing (K)"); ax.set_ylim(0, None); ax.legend(fontsize=7, frameon=False)
ax.set_title("Mean state: the ocean's heat reserve above freezing", fontsize=9, pad=28)
# 3: overturning
ax = axes[0, 2]
ax.plot(mu(WARM), [rows[l]["equ"] for l in WARM], "o-", color=ORANGE); ax.plot(mu(COLD), [rows[l]["equ"] for l in COLD], "s", color=ORANGE)
ax.set_ylabel("circulation bringing warm water to the edge\n(equatorward overturning cell, Sv)")
ax.set_title("Mean state: the ocean circulation feeding the edge", fontsize=9, pad=28)
# 4: oscillation period
ax = axes[1, 0]
for osc, col in (("Pacific", AQUA), ("Atlantic", BLUE)):
    labs = [l for l in rows if rows[l].get("osc") == osc]
    ax.plot(mu(labs), [rows[l]["period"] for l in labs], "o-" if osc == "Atlantic" else "D", color=col, label=f"{osc} oscillation")
ax.set_ylabel("period of the dominant oscillation (yr)\n(raw periodogram peak)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Natural variability: the clock of the dominant oscillation", fontsize=9)
# 5: low-frequency variance
ax = axes[1, 1]
for labs, mk in ((WARM, "o-"), (COLD, "s")):
    ax.semilogy(mu(labs), [rows[l]["sd50_tot"] for l in labs], mk, color="k", label="all sectors" if mk == "o-" else None)
    ax.semilogy(mu(labs), [rows[l]["sd50_pac"] for l in labs], mk, color=AQUA, label="Pacific sector" if mk == "o-" else None)
ax.set_ylabel("sd of 50-yr means of sunlight absorbed\nsouth of 20°S (PW, log scale)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Natural variability: slow (≥ 50 yr) albedo fluctuations [hint]", fontsize=9)
# 6: decadal persistence
ax = axes[1, 2]
labs = WARM[4:]
ax.plot(mu(labs), [rows[l]["acf10"] for l in labs], "o-", color=BLUE, label="whole window")
for l in labs:
    h = rows[l]["acf10_h"]; ax.plot([rows[l]["mu"]] * 2, h, "-", color=BLUE, alpha=0.4, lw=3)
ax.set_ylabel("10-yr autocorrelation of the Indo-Pacific\nice band (30.5 + 36°S); bars = two halves")
ax.set_title("Natural variability: decadal memory of the Indo-Pacific ice [hint]", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.suptitle("Along the warm branch towards the collapse: what changes in the mean state and in the natural variability "
             "(squares: cold states)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/synthesis_fingerprints.png", dpi=110); plt.close(fig)
