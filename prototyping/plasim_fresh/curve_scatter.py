"""Natural fluctuations along the curve of settled states (criteria: Claude/plasim_mechanism/plan_curve.md)."""

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import band_cover, load, lsg_layer_mean, south_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
SEQ = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]  # the seven judged runs
REF = ["1245", "1242p5", "1240", "1237p5"]
REF_R = ["1245", "1242p5", "1240", "1237p5", "1235"]
END2 = ["1233p75", "1232p5"]
PLOT = ["1312", "1288", "1265", "1250"] + SEQ[:5] + ["1235_new_IC"] + SEQ[5:] + ["1230", "1228p5"]


def w_fix(d):
    rows = (d["lsg_lat"] >= -38.0) & (d["lsg_lat"] <= -23.0)
    wv = d["layer_volume"][rows, 0]
    return np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()


D = {}
for lab in PLOT:
    d = load(lab)
    y = d["years"]
    bad = ~np.isfinite(d["sic"][:, 0, 0])
    ser = {"ice": south_ice_area(d) / 1e12, "ipb": band_cover(d), "H": lsg_layer_mean(d, 1, -90, 0), "W": w_fix(d)}
    for v in ser.values():
        v[bad] = np.nan
    m = (y >= ST[lab]["0.95"]) & (y <= ST[lab]["end"])
    D[lab] = {"mu": float(d["mu"]), "y": y[m], "ser": {k: v[m] for k, v in ser.items()}}
    D[lab]["mean"] = {k: float(np.nanmean(v)) for k, v in D[lab]["ser"].items()}


def sens(lab, key):
    """Forced change of `key` per W m⁻² towards lower μ (centred; one-sided at 1245 and 1232.5)."""
    base = "1235" if lab == "1235_new_IC" else lab
    i = SEQ.index(base)
    a, b = SEQ[max(i - 1, 0)], SEQ[min(i + 1, len(SEQ) - 1)]
    return (D[b]["mean"][key] - D[a]["mean"][key]) / (D[a]["mu"] - D[b]["mu"])


def prep(lab, key, detrend):
    x = D[lab]["ser"][key].copy()
    if detrend:
        y = D[lab]["y"]
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
    m1 = np.std(along, ddof=1)
    m2 = m1 * s_ice
    return {"M1": m1, "M2": m2, "r": np.corrcoef(ai, ao)[0, 1], "ratio": m1 / np.std(across, ddof=1), "R": m2 ** 2 / s_ice}


def run_measures(lab, ice, ocean, L, detrend):
    s_i, s_o = sens(lab, ice), sens(lab, ocean)
    ai = blocks(prep(lab, ice, detrend), L) / s_i
    ao = blocks(prep(lab, ocean, detrend), L) / s_o
    whole = measures(ai, ao, s_i)
    per = 1000 // L
    segs = [measures(ai[k * per:(k + 1) * per], ao[k * per:(k + 1) * per], s_i) for k in range(len(ai) // per)]
    segs = [s for s in segs if np.isfinite(list(s.values())).all()]
    out = {"whole": whole, "S_ice": s_i, "S_ocean": s_o, "n_seg": len(segs)}
    for q in whole:
        v = np.array([s[q] for s in segs])
        out[q] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) >= 2 else np.nan)
    return out


VARIANTS = list(itertools.product(("ice", "ipb"), ("H", "W"), (100, 50), (False, True)))
RES = {}
verdicts = {}
for ice, ocean, L, det in VARIANTS:
    key = f"{ice}|{ocean}|L{L}|{'detr' if det else 'trend'}"
    R = {lab: run_measures(lab, ice, ocean, L, det) for lab in SEQ + ["1235_new_IC"]}
    RES[key] = R
    on_curve = sum((R[l]["r"][0] >= 0.5) and (R[l]["ratio"][0] >= 2) for l in SEQ)
    ref = np.mean([R[l]["M2"][0] for l in REF])
    ref_se = np.sqrt(sum(R[l]["M2"][1] ** 2 for l in REF)) / len(REF)
    late_ok = all(R[l]["M2"][0] - ref > 2 * np.hypot(R[l]["M2"][1], ref_se) for l in END2)
    rho = spearmanr(np.arange(len(SEQ)), [R[l]["M2"][0] for l in SEQ])[0]
    rref = np.mean([R[l]["R"][0] for l in REF_R])
    rr = [R[l]["R"][0] / rref for l in END2]
    v_r = "consistent" if all(0.7 <= x <= 1.3 for x in rr) else ("less than predicted" if all(x < 0.7 for x in rr) else "mixed")
    verdicts[key] = {"on_curve_runs": int(on_curve), "on_curve": bool(on_curve >= 5), "M2_late_2se": bool(late_ok),
                     "M2_spearman": float(rho), "M2_grows": bool(late_ok and rho >= 0.8), "R_rel_end": [float(x) for x in rr],
                     "R_verdict": v_r}

json.dump({"res": RES, "verdicts": verdicts}, open(TMP + "curve_scatter.json", "w"), indent=1, default=float)

# ---------------- print ----------------
P = "ice|H|L100|trend"
print("Primary variant", P)
print("run      S_ice   S_H     M1 (W m-2)      M2 (1e12 m2)    r(ice,ocean)  along/across   R")
for lab in SEQ + ["1235_new_IC"]:
    o = RES[P][lab]
    print(f"{lab:11s} {o['S_ice']:.2f} {o['S_ocean']:.3f}  {o['M1'][0]:.2f}±{o['M1'][1]:.2f}  {o['M2'][0]:.2f}±{o['M2'][1]:.2f}  "
          f"{o['r'][0]:.2f}±{o['r'][1]:.2f}  {o['ratio'][0]:.2f}  {o['R'][0]:.3f}  | whole-run M2 {o['whole']['M2']:.2f} r {o['whole']['r']:.2f}")
print("\nVerdicts per variant")
for k, v in verdicts.items():
    print(f"{k:22s} on-curve runs {v['on_curve_runs']}/7  M2 late>2SE {v['M2_late_2se']}  rho {v['M2_spearman']:.2f}  "
          f"grows {v['M2_grows']}  R end/ref {v['R_rel_end'][0]:.2f},{v['R_rel_end'][1]:.2f} -> {v['R_verdict']}")
print("\nsummary: on_curve all variants:", all(v["on_curve"] for v in verdicts.values()),
      "| M2 grows all variants:", all(v["M2_grows"] for v in verdicts.values()),
      "| R verdicts:", {x: sum(v["R_verdict"] == x for v in verdicts.values()) for x in ("consistent", "mixed", "less than predicted")})

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
cmap = plt.cm.plasma
mus = [D[l]["mu"] for l in PLOT]
ZOOM = ["1250"] + SEQ[:5] + ["1235_new_IC"] + SEQ[5:]
col = {l: cmap(0.85 * (1250 - D[l]["mu"]) / (1250 - 1232.5)) for l in ZOOM}
fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
for ax, ocean, name in ((axes[0], "H", "Southern ocean θ, 700–2025 m (K)"), (axes[1], "W", "ocean θ 0–700 m, 23–38°S (K)")):
    warm = [l for l in ZOOM if l != "1235_new_IC"]
    ax.plot([D[l]["mean"][ocean] for l in warm], [D[l]["mean"]["ice"] for l in warm], "-", color="0.6", lw=1, zorder=1)
    for l in ZOOM:
        xb, yb = blocks(D[l]["ser"][ocean], 100), blocks(D[l]["ser"]["ice"], 100)
        ax.scatter(xb, yb, s=10, color=col[l], alpha=0.7, lw=0, zorder=2)
        ax.scatter([D[l]["mean"][ocean]], [D[l]["mean"]["ice"]], s=55, marker="s", color=col[l], edgecolor="k", lw=0.7, zorder=3,
                   label=l.replace("p", ".").replace("_new_IC", " (2nd)"))
    ax.invert_xaxis()
    ax.set_xlabel(name + " — colder to the right")
    ax.set_ylabel("Southern ice area (10¹² m²)")
    ax.grid(alpha=0.3, lw=0.4)
axes[0].set_title("Against the 700–2025 m ocean: within a run,\nthe ice moves with little deep-ocean change")
axes[1].set_title("Against the upper ocean next to the edge:\nthe fluctuations lie along the curve")
axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left", title="squares: settled means\ndots: 100-yr means, stationary years", title_fontsize=7)

ax = axes[2]
x = [D[l]["mu"] for l in SEQ]
for k in verdicts:
    if k in (P, "ice|W|L100|trend") or k.startswith("ipb"):
        continue
    ax.plot(x, [RES[k][l]["M2"][0] for l in SEQ], color="0.75", lw=0.8)
ax.errorbar(x, [RES[P][l]["M2"][0] for l in SEQ], yerr=[RES[P][l]["M2"][1] for l in SEQ], fmt="o-", color="k", capsize=0,
            label="primary: total ice & 700–2025 m, 100-yr means")
ax.plot([D["1235_new_IC"]["mu"]], [RES[P]["1235_new_IC"]["M2"][0]], "o", mfc="white", color="k", label="1235, second run")
W2 = "ice|W|L100|trend"
ax.errorbar(x, [RES[W2][l]["M2"][0] for l in SEQ], yerr=[RES[W2][l]["M2"][1] for l in SEQ], fmt="o-", color="#2a78d6", capsize=0,
            label="total ice & upper ocean 0–700 m, 100-yr means")
ax.plot([], [], color="0.75", label="the 6 other total-ice variants")
ax.set_xlim(1246.5, 1231)
ax.set_xlabel("μ (W m⁻²)")
ax.set_ylabel("spread along the curve, sd (10¹² m² of ice)")
ax.set_title("Spread along the curve (100-yr means, 1000-yr segments):\nU-shaped, no steady growth")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.3, lw=0.4)

ax = axes[3]
s_ref = np.mean([RES[P][l]["S_ice"] for l in REF_R])
v_ref = np.mean([RES[P][l]["M2"][0] ** 2 for l in REF_R])
ax.axhspan(0.7, 1.3, color="0.92", lw=0)
ax.plot(x, [RES[P][l]["S_ice"] / s_ref for l in SEQ], "s-", color="#eb6834", label="sensitivity S (ice added per W m⁻²)")
ax.plot(x, [RES[P][l]["M2"][0] ** 2 / v_ref for l in SEQ], "o-", color="k", label="natural variance along the curve (deep-ocean version)")
v_ref_w = np.mean([RES["ice|W|L100|trend"][l]["M2"][0] ** 2 for l in REF_R])
ax.plot(x, [RES["ice|W|L100|trend"][l]["M2"][0] ** 2 / v_ref_w for l in SEQ], "o-", color="#2a78d6", label="natural variance along the curve (upper-ocean version)")
ax.plot(x, [RES[P][l]["R"][0] / np.mean([RES[P][m]["R"][0] for m in REF_R]) for l in SEQ], "d--", color="0.45",
        label="variance / S, deep-ocean version (fixed in the 1D picture)")
ax.set_xlim(1246.5, 1231)
ax.set_xlabel("μ (W m⁻²)")
ax.set_ylabel("relative to the mean over 1245–1235")
ax.set_title("Sensitivity flat, then up; variance falls, then rises:\nthey do not move together")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.grid(alpha=0.3, lw=0.4)
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/curve_scatter.png", dpi=110)
plt.close(fig)
