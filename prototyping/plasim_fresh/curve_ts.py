"""Curve-and-scatter check in the (T_S, ΔT_S) plane of analysis/Plasim_Mechanism.py (criteria: plan_curve.md, second extension)."""

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import equivalent_edge, load
from gsebm.plasim_global import hemispheric_modes

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
SEQ = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
LATE7 = SEQ[4:]
REF = ["1245", "1242p5", "1240", "1237p5"]
END2 = ["1233p75", "1232p5"]
RUNS = SEQ + ["1235_new_IC"]

D = {}
for lab in RUNS:
    d = load(lab)
    y = d["years"]
    m = (y >= ST[lab]["0.95"]) & (y <= ST[lab]["end"])
    bad = ~np.isfinite(d["ts"][:, 0])
    ser = {}
    for h, south in (("S", True), ("N", False)):
        hm = hemispheric_modes(d["ts"], d["lat"], d["gw"], south)
        rows = d["lat"] < 0 if south else d["lat"] > 0
        w = d["gw"][rows]
        trop = np.abs(d["lat"][rows]) < 30
        ser[f"T{h}"], ser[f"D{h}"] = hm["mean"], hm["delta"]
        ser[f"trop{h}"] = d["ts"][:, rows][:, trop] @ w[trop] / w[trop].sum()
        ser[f"ext{h}"] = d["ts"][:, rows][:, ~trop] @ w[~trop] / w[~trop].sum()
    ser["edge"] = equivalent_edge(d)
    for v in ser.values():
        v[bad] = np.nan
    D[lab] = {"mu": float(d["mu"]), "y": y[m], "ser": {k: v[m] for k, v in ser.items()}}
    D[lab]["mean"] = {k: float(np.nanmean(v)) for k, v in D[lab]["ser"].items()}


def neighbours(lab):
    base = "1235" if lab == "1235_new_IC" else lab
    i = SEQ.index(base)
    return SEQ[max(i - 1, 0)], SEQ[min(i + 1, len(SEQ) - 1)]


def sens(lab, key):
    a, b = neighbours(lab)
    return (D[b]["mean"][key] - D[a]["mean"][key]) / (D[a]["mu"] - D[b]["mu"])


def prep(lab, key, detrend):
    x = D[lab]["ser"][key].copy()
    if detrend:
        ok = np.isfinite(x)
        x[ok] -= np.polyval(np.polyfit(D[lab]["y"][ok], x[ok], 1), D[lab]["y"][ok])
    return x


def blocks(x, L):
    if L == 1:
        return x
    n = len(x) // L
    return np.array([np.nanmean(x[i * L:(i + 1) * L]) if np.isfinite(x[i * L:(i + 1) * L]).any() else np.nan for i in range(n)])


def measures(a1, a2, s1):
    ok = np.isfinite(a1) & np.isfinite(a2)
    a1, a2 = a1[ok] - a1[ok].mean(), a2[ok] - a2[ok].mean()
    along, across = (a1 + a2) / 2, (a1 - a2) / 2
    m2 = np.std(along, ddof=1) * abs(s1)
    return {"M2": m2, "r": np.corrcoef(a1, a2)[0, 1], "ratio": np.std(along, ddof=1) / np.std(across, ddof=1), "R": m2 ** 2 / abs(s1),
            "M1": np.std(along, ddof=1)}


def run_measures(lab, h, L, detrend):
    sT, sD = sens(lab, f"T{h}"), sens(lab, f"D{h}")
    aT = blocks(prep(lab, f"T{h}", detrend), L) / sT
    aD = blocks(prep(lab, f"D{h}", detrend), L) / sD
    per = 1000 // L
    segs = [measures(aT[k * per:(k + 1) * per], aD[k * per:(k + 1) * per], sT) for k in range(len(aT) // per)]
    segs = [s for s in segs if np.isfinite(list(s.values())).all()]
    out = {"S_T": sT, "S_D": sD, "whole": measures(aT, aD, sT)}
    for q in segs[0]:
        v = np.array([s[q] for s in segs])
        out[q] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))))
    # tropics/extratropics ratio k: forced, and natural (regression of tropics on extratropics)
    a, b = neighbours(lab)
    out["k_forced"] = (D[b]["mean"][f"trop{h}"] - D[a]["mean"][f"trop{h}"]) / (D[b]["mean"][f"ext{h}"] - D[a]["mean"][f"ext{h}"])
    xt, xe = blocks(prep(lab, f"trop{h}", detrend), L), blocks(prep(lab, f"ext{h}", detrend), L)
    ok = np.isfinite(xt) & np.isfinite(xe)
    xe0, xt0 = xe[ok] - xe[ok].mean(), xt[ok] - xt[ok].mean()
    out["k_natural"] = float(xe0 @ xt0 / (xe0 @ xe0))
    return out


RES, VER = {}, {}
for h in ("S", "N"):
    for L, det in itertools.product((1, 10, 50, 100), (False, True)):
        key = f"{h}|L{L}|{'detr' if det else 'trend'}"
        R = {lab: run_measures(lab, h, L, det) for lab in RUNS}
        RES[key] = R
        on = sum((R[l]["r"][0] >= 0.5) and (R[l]["ratio"][0] >= 2) for l in SEQ)
        rho = spearmanr([R[l]["M2"][0] ** 2 for l in SEQ], [abs(R[l]["S_T"]) for l in SEQ])[0]
        ref = np.mean([R[l]["M2"][0] for l in REF])
        ref_se = np.sqrt(sum(R[l]["M2"][1] ** 2 for l in REF)) / len(REF)
        late = all(R[l]["M2"][0] - ref > 2 * np.hypot(R[l]["M2"][1], ref_se) for l in END2)
        rho7 = spearmanr(np.arange(7), [R[l]["M2"][0] for l in LATE7])[0]
        VER[key] = {"on_curve_runs": int(on), "on_curve": bool(on >= 8), "rho_M2sq_S": float(rho), "follows": bool(rho >= 0.8),
                    "late_2se": bool(late), "rho_late7": float(rho7), "grows_end": bool(late and rho7 >= 0.8)}
json.dump({"res": RES, "ver": VER, "edge": {l: D[l]["mean"]["edgeS"] if "edgeS" in D[l]["mean"] else D[l]["mean"]["edge"] for l in RUNS}},
          open(TMP + "curve_ts.json", "w"), indent=1, default=float)

for P in ("S|L1|trend", "S|L100|trend", "N|L1|trend"):
    print(P)
    print("run        edge   S_T    S_D     k_forced k_nat   r      along/across  M2 (K)        M1 (W m-2)")
    for lab in RUNS:
        o = RES[P][lab]
        print(f"  {lab:11s} {D[lab]['mean']['edge']:.1f} {o['S_T']:.3f} {o['S_D']:7.3f}  {o['k_forced']:6.2f}  {o['k_natural']:6.2f}  "
              f"{o['r'][0]:5.2f}  {o['ratio'][0]:5.2f}   {o['M2'][0]:.3f}±{o['M2'][1]:.3f}  {o['M1'][0]:.2f}")
print()
for k, v in VER.items():
    print(k, v)
print("\ndetrend check, S annual M2 trend vs detr:", [(l, round(RES["S|L1|trend"][l]["M2"][0], 4), round(RES["S|L1|detr"][l]["M2"][0], 4)) for l in ("1312", "1245", "1232p5")])
for P in ("S|L1|trend", "S|L100|trend"):
    ref = np.mean([RES[P][l]["R"][0] for l in ["1245", "1242p5", "1240", "1237p5", "1235"]])
    print(P, "R end / mean(1245-1235):", [round(RES[P][l]["R"][0] / ref, 2) for l in END2])

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
cmap = plt.cm.plasma
col = {l: cmap(0.85 * (1312 - D[l]["mu"]) / (1312 - 1232.5)) for l in RUNS}
edge = {l: D[l]["mean"]["edge"] for l in RUNS}
d0 = load("1240")
absl = np.sort(np.abs(d0["lat"][d0["lat"] < 0]))
mids = 0.5 * (absl[:-1] + absl[1:])


def rows_bg(ax):
    for i, r in enumerate(absl):
        if 30 < r < 66:
            lo = mids[i - 1] if i > 0 else 0.0
            hi = mids[i] if i < len(mids) else 90.0
            ax.axvspan(lo, hi, color="0.95" if i % 2 else "0.88", zorder=0, lw=0)
    ax.set_xlim(65, 31)
    ax.set_xlabel("mean ice edge, equivalent-area latitude (°S); grey bands: T21 rows")


def slope(lab, L):
    t, dd = blocks(D[lab]["ser"]["TS"], L), blocks(D[lab]["ser"]["DS"], L)
    ok = np.isfinite(t) & np.isfinite(dd)
    t0, d1 = t[ok] - t[ok].mean(), dd[ok] - dd[ok].mean()
    return float(t0 @ d1 / (t0 @ t0))


fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
ax = axes[0]
ax.plot([D[l]["mean"]["TS"] for l in SEQ], [D[l]["mean"]["DS"] for l in SEQ], "-", color="0.5", lw=1, zorder=1)
for l in RUNS:
    ax.scatter(D[l]["ser"]["TS"], D[l]["ser"]["DS"], s=1.5, color=col[l], alpha=0.25, lw=0, zorder=2, rasterized=True)
    ax.scatter([D[l]["mean"]["TS"]], [D[l]["mean"]["DS"]], s=50, marker="s", color=col[l], edgecolor="k", lw=0.7, zorder=3,
               label=l.replace("p", ".").replace("_new_IC", " (2nd)"))
ax.set_xlabel("Southern Hemisphere mean Ts, T_S (K)")
ax.set_ylabel("ΔT_S = tropics − extratropics (K)")
ax.set_title("Southern (T_S, ΔT_S): settled means (squares)\nand annual values of the stationary years (dots)")
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
ax.grid(alpha=0.3, lw=0.4)

ax = axes[1]
rows_bg(ax)
ax.plot([edge[l] for l in SEQ], [sens(l, "DS") / sens(l, "TS") for l in SEQ], "s-", color=ORANGE, label="forced (between settled states)")
ax.plot([edge[l] for l in SEQ], [slope(l, 1) for l in SEQ], "o-", color="k", label="natural, annual values")
ax.plot([edge[l] for l in SEQ], [slope(l, 100) for l in SEQ], "o--", color=BLUE, label="natural, 100-yr means")
ax.axhline(-2, color=GREY, ls=":", lw=1)
ax.text(64, -1.96, "−2: change in the extratropics only", fontsize=7, color=GREY, va="bottom")
ax.set_ylabel("slope dΔT_S / dT_S")
ax.set_ylim(-2.1, -0.5)
ax.set_title("Natural fluctuations are steeper (more extratropical)\nthan the forced change")
ax.legend(fontsize=7, frameon=False, loc="lower right")

ax = axes[2]
rows_bg(ax)
for key, c, n, f in (("S|L1|trend", "k", "annual values", "o-"), ("S|L100|trend", BLUE, "100-yr means", "o--")):
    ax.errorbar([edge[l] for l in SEQ], [RES[key][l]["M2"][0] for l in SEQ], yerr=[RES[key][l]["M2"][1] for l in SEQ],
                fmt=f, color=c, capsize=0, ms=4.5, label=n)
ax.set_yscale("log")
ax.set_ylabel("sd along the curve, 1000-yr segments (K of T_S)")
ax.set_title("Natural spread along the curve (log scale)")
ax.legend(fontsize=7, frameon=False, loc="lower left")

ax = axes[3]
rows_bg(ax)
ax.plot([edge[l] for l in SEQ], [abs(sens(l, "TS")) for l in SEQ], "s-", color=ORANGE)
for l in ("1312", "1265", "1250", "1240", "1235", "1233p75", "1232p5"):
    ax.annotate(l.replace("p", "."), (edge[l], abs(sens(l, "TS"))), textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=6.5, color=GREY)
ax.set_ylabel("sensitivity: Southern cooling per W m⁻² (K)")
ax.set_title("Forced sensitivity of T_S")
for a in axes:
    a.grid(alpha=0.3, lw=0.4)
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/curve_ts.png", dpi=110)
plt.close(fig)
