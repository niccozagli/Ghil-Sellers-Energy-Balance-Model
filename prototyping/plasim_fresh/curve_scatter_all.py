"""Curve-and-scatter check over the whole warm branch (criteria: plan_curve.md, extension section)."""

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import equivalent_edge, load, lsg_layer_mean, sector_ice_area, south_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
SEQ = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
EVENTS = {"1312", "1250", "1245"}
RUNS = SEQ + ["1235_new_IC"]

D = {}
for lab in RUNS:
    d = load(lab)
    y = d["years"]
    m = (y >= ST[lab]["0.95"]) & (y <= ST[lab]["end"])
    bad = ~np.isfinite(d["sic"][:, 0, 0])
    ser = {"ice": south_ice_area(d) / 1e12, "H": lsg_layer_mean(d, 1, -90, 0), "edge": equivalent_edge(d),
           "atl": sector_ice_area(d, (1,)), "ind": sector_ice_area(d, (2,)), "pac": sector_ice_area(d, (3,))}
    for v in ser.values():
        v[bad] = np.nan
    D[lab] = {"d": d, "mu": float(d["mu"]), "m": m, "bad": bad, "ser": {k: v[m] for k, v in ser.items()}}
    D[lab]["mean"] = {k: float(np.nanmean(v)) for k, v in D[lab]["ser"].items()}


def w_edge(lab, rows_lab):
    """0–700 m θ from rows_lab's mean edge to 10° equatorward, evaluated in run lab (stationary years)."""
    d = D[lab]["d"]
    e = D[rows_lab]["mean"]["edge"]
    rows = (d["lsg_lat"] >= -e) & (d["lsg_lat"] <= -e + 10)
    wv = d["layer_volume"][rows, 0]
    w = np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()
    w[D[lab]["bad"]] = np.nan
    return w[D[lab]["m"]]


def neighbours(lab):
    base = "1235" if lab == "1235_new_IC" else lab
    i = SEQ.index(base)
    return SEQ[max(i - 1, 0)], SEQ[min(i + 1, len(SEQ) - 1)]


def series(lab, key):
    return w_edge(lab, lab) if key == "W" else D[lab]["ser"][key]


def sens(lab, key):
    a, b = neighbours(lab)
    if key == "W":
        xa, xb = np.nanmean(w_edge(a, lab)), np.nanmean(w_edge(b, lab))
    else:
        xa, xb = D[a]["mean"][key], D[b]["mean"][key]
    return (xb - xa) / (D[a]["mu"] - D[b]["mu"])


def prep(lab, key, detrend):
    x = series(lab, key).copy()
    if detrend:
        y = D[lab]["d"]["years"][D[lab]["m"]]
        ok = np.isfinite(x)
        x[ok] -= np.polyval(np.polyfit(y[ok], x[ok], 1), y[ok])
    return x


def blocks(x, L):
    n = len(x) // L
    return np.array([np.nanmean(x[i * L:(i + 1) * L]) if np.isfinite(x[i * L:(i + 1) * L]).any() else np.nan for i in range(n)])


def measures(ai, ao, s_ice):
    ok = np.isfinite(ai) & np.isfinite(ao)
    ai, ao = ai[ok] - ai[ok].mean(), ao[ok] - ao[ok].mean()
    along, across = (ai + ao) / 2, (ai - ao) / 2
    m2 = np.std(along, ddof=1) * s_ice
    return {"M2": m2, "r": np.corrcoef(ai, ao)[0, 1], "ratio": np.std(along, ddof=1) / np.std(across, ddof=1),
            "R": m2 ** 2 / s_ice, "sd_ice": np.std(ai, ddof=1) * s_ice}


def run_measures(lab, ocean, L, detrend):
    s_i, s_o = sens(lab, "ice"), sens(lab, ocean)
    ai = blocks(prep(lab, "ice", detrend), L) / s_i
    ao = blocks(prep(lab, ocean, detrend), L) / s_o
    per = 1000 // L
    segs = [measures(ai[k * per:(k + 1) * per], ao[k * per:(k + 1) * per], s_i) for k in range(len(ai) // per)]
    segs = [s for s in segs if np.isfinite(list(s.values())).all()]
    out = {"S_ice": s_i, "S_ocean": s_o, "whole": measures(ai, ao, s_i)}
    for q in segs[0]:
        v = np.array([s[q] for s in segs])
        out[q] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))))
    return out


RES, VER = {}, {}
for ocean, L, det in itertools.product(("H", "W"), (100, 50), (False, True)):
    key = f"{ocean}|L{L}|{'detr' if det else 'trend'}"
    R = {lab: run_measures(lab, ocean, L, det) for lab in RUNS}
    RES[key] = R
    rho = spearmanr([R[l]["M2"][0] ** 2 for l in SEQ], [R[l]["S_ice"] for l in SEQ])[0]
    rho_sd = spearmanr([R[l]["sd_ice"][0] ** 2 for l in SEQ], [R[l]["S_ice"] for l in SEQ])[0]
    on = sum((R[l]["r"][0] >= 0.5) and (R[l]["ratio"][0] >= 2) for l in SEQ)
    VER[key] = {"rho_M2sq_S": float(rho), "rho_sdice_sq_S": float(rho_sd), "follows": bool(rho >= 0.8), "on_curve_runs": int(on),
                "on_curve": bool(on >= 8)}

# per-sector sd of 100-yr means (descriptive); plus a spread that ignores rare events (1.4826 × median absolute deviation)
SECT = {lab: {s: float(np.nanstd(blocks(D[lab]["ser"][s], 100), ddof=1)) for s in ("atl", "ind", "pac", "ice")} for lab in RUNS}
for lab in RUNS:
    b = blocks(D[lab]["ser"]["ice"], 100)
    b = b[np.isfinite(b)]
    SECT[lab]["ice_robust"] = float(1.4826 * np.median(np.abs(b - np.median(b))))
LABELS = {"1312", "1288", "1265", "1250", "1245", "1237p5", "1232p5"}
json.dump({"res": RES, "ver": VER, "sect": SECT, "edge": {l: D[l]["mean"]["edge"] for l in RUNS}},
          open(TMP + "curve_scatter_all.json", "w"), indent=1, default=float)

P = "H|L100|trend"
print("run        edge   S_ice   M2(H)        r(H)   M2(W)        r(W)  ratio(W)  sd100 ice | sd100 Atl Ind Pac")
for lab in RUNS:
    h, w = RES[P][lab], RES["W|L100|trend"][lab]
    print(f"{lab:11s} {D[lab]['mean']['edge']:.1f} {h['S_ice']:6.2f}  {h['M2'][0]:.2f}±{h['M2'][1]:.2f}  {h['r'][0]:5.2f}  "
          f"{w['M2'][0]:.2f}±{w['M2'][1]:.2f}  {w['r'][0]:5.2f}  {w['ratio'][0]:.2f}   {SECT[lab]['ice']:.2f} | "
          f"{SECT[lab]['atl']:.2f} {SECT[lab]['ind']:.2f} {SECT[lab]['pac']:.2f}{'  (events)' if lab in EVENTS else ''}")
print()
for k, v in VER.items():
    print(k, v)

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
cmap = plt.cm.plasma
col = {l: cmap(0.85 * (1312 - D[l]["mu"]) / (1312 - 1232.5)) for l in RUNS}
edge = {l: D[l]["mean"]["edge"] for l in RUNS}
lat = D["1240"]["d"]["lat"]
absl = np.sort(np.abs(lat[lat < 0]))
mids = 0.5 * (absl[:-1] + absl[1:])


def rows_bg(ax):
    for i, r in enumerate(absl):
        if 30 < r < 66:
            lo = mids[i - 1] if i > 0 else 0.0
            hi = mids[i] if i < len(mids) else 90.0
            ax.axvspan(lo, hi, color="0.95" if i % 2 else "0.88", zorder=0, lw=0)
    ax.set_xlim(65, 31)
    ax.set_xlabel("mean ice edge, equivalent-area latitude (°S); grey bands: T21 rows")


fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
ax = axes[0]
ax.plot([D[l]["mean"]["H"] for l in SEQ], [D[l]["mean"]["ice"] for l in SEQ], "-", color="0.6", lw=1, zorder=1)
for l in RUNS:
    ax.scatter(blocks(D[l]["ser"]["H"], 100), blocks(D[l]["ser"]["ice"], 100), s=9, color=col[l], alpha=0.6, lw=0, zorder=2)
    ax.scatter([D[l]["mean"]["H"]], [D[l]["mean"]["ice"]], s=50, marker="s", color=col[l], edgecolor="k", lw=0.7, zorder=3,
               label=l.replace("p", ".").replace("_new_IC", " (2nd)"))
ax.invert_xaxis()
ax.set_xlabel("Southern ocean θ, 700–2025 m (K) — colder to the right")
ax.set_ylabel("Southern ice area (10¹² m²)")
ax.set_title("Whole warm branch: settled means (squares)\nand 100-yr means of the stationary years (dots)")
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
ax.grid(alpha=0.3, lw=0.4)

ax = axes[1]
rows_bg(ax)
for s, c, n in (("atl", BLUE, "Atlantic"), ("ind", AQUA, "Indian"), ("pac", ORANGE, "Pacific")):
    ax.plot([edge[l] for l in SEQ], [SECT[l][s] for l in SEQ], "o-", color=c, label=n, ms=4.5)
ax.plot([edge[l] for l in SEQ], [SECT[l]["ice"] for l in SEQ], "o-", color="k", label="all sectors", ms=4.5)
ax.plot([edge[l] for l in SEQ], [SECT[l]["ice_robust"] for l in SEQ], "o--", color="k", mfc="white", ms=4.5,
        label="all sectors, ignoring rare events (median-based)")
ax.set_yscale("log")
for l in [x for x in SEQ if x in LABELS]:
    ax.annotate(l.replace("p", "."), (edge[l], SECT[l]["ice"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5, color=GREY)
ax.set_ylabel("sd of 100-yr means of ice area (10¹² m²)")
ax.set_title("Slow ice fluctuations by sector (log scale)")
ax.legend(fontsize=7, frameon=False, loc="lower left")

ax = axes[2]
rows_bg(ax)
for key, c, n in ((P, "k", "spread along the curve, deep-ocean version"), ("W|L100|trend", BLUE, "spread along the curve, upper-ocean version")):
    ax.errorbar([edge[l] for l in SEQ], [RES[key][l]["M2"][0] for l in SEQ], yerr=[RES[key][l]["M2"][1] for l in SEQ],
                fmt="o-", color=c, capsize=0, ms=4.5, label=n)
ax.plot([edge[l] for l in SEQ if l in EVENTS], [RES[P][l]["M2"][0] for l in SEQ if l in EVENTS], "o", mfc="none", mec=ORANGE,
        ms=11, mew=1.3, label="run with rare large events")
ax.set_ylabel("sd along the curve, 1000-yr segments (10¹² m² of ice)")
ax.set_yscale("log")
ax.set_title("Natural spread along the curve (log scale)")
ax.legend(fontsize=7, frameon=False, loc="lower left")

ax = axes[3]
rows_bg(ax)
ax.plot([edge[l] for l in SEQ], [RES[P][l]["S_ice"] for l in SEQ], "s-", color=ORANGE, ms=5)
for l in [x for x in SEQ if x in LABELS | {"1235", "1233p75"}]:
    ax.annotate(l.replace("p", "."), (edge[l], RES[P][l]["S_ice"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5, color=GREY)
ax.set_ylabel("sensitivity: Southern ice added per W m⁻² (10¹² m²)")
ax.set_title("Forced sensitivity (centred differences of settled means)")
for a in axes:
    a.grid(alpha=0.3, lw=0.4)
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/curve_scatter_all.png", dpi=110)
plt.close(fig)
