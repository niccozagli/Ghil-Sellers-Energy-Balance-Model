"""Step response against natural autocorrelation, linearity, slow part at every μ (criteria: plan_step_response.md)."""

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np

from common import PARENT, STEP_YEAR, equivalent_edge, load, lsg_layer_mean, south_ice_area
from gsebm.plasim_global import hemispheric_modes

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
STEPS = ["1250", "1245", "1240", "1235", "1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5"]
FAR = ["1312", "1288", "1265"]
WARM = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
VARS = {"ice": "Southern ice area", "TS": "Southern mean Ts", "H": "Southern ocean 700–2025 m"}
LAGS = (30, 100, 300, 1000)

D = {}
for lab in sorted(set(STEPS + FAR + WARM + ["1240", "1265"])):
    d = load(lab)
    bad = ~np.isfinite(d["sic"][:, 0, 0])
    ser = {"ice": south_ice_area(d) / 1e12, "TS": hemispheric_modes(d["ts"], d["lat"], d["gw"], True)["mean"],
           "H": lsg_layer_mean(d, 1, -90, 0), "edge": equivalent_edge(d)}
    for v in ser.values():
        v[bad] = np.nan
    y = d["years"]
    st = (y >= ST[lab]["0.95"]) & (y <= ST[lab]["end"])
    D[lab] = {"y": y, "ser": ser, "st": st, "mu": float(d["mu"]),
              "mean": {k: float(np.nanmean(v[st])) for k, v in ser.items()}}


def acf(x, lags):
    x = x - np.nanmean(x)
    var = np.nanmean(x * x)
    out = []
    for k in lags:
        a, b = x[:-k], x[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.mean(a[ok] * b[ok]) / var) if ok.sum() > 10 else np.nan)
    return np.array(out)


def centred(t_years, x, t, h):
    k = (t_years >= t - h) & (t_years <= t + h)
    return float(np.nanmean(x[k])) if k.any() else np.nan


def block_sd(x, L):
    n = len(x) // L
    b = np.array([np.nanmean(x[i * L:(i + 1) * L]) for i in range(n)])
    return float(np.nanstd(b[np.isfinite(b)], ddof=1))


def step_case(lab, q, x0_kind, L):
    par, y0p = PARENT[lab]
    P, S = D[par], D[lab]
    if x0_kind == "parent_mean":
        x0 = P["mean"][q]
    else:
        k = (P["y"] > y0p - 50) & (P["y"] <= y0p)
        x0 = float(np.nanmean(P["ser"][q][k]))
    xinf = S["mean"][q]
    t = S["y"] - STEP_YEAR[lab]
    x = S["ser"][q]
    phi = {lag: (xinf - centred(t, x, lag, L // 2)) / (xinf - x0) for lag in LAGS + (50,)}
    rho = dict(zip(LAGS, acf(x[S["st"]], LAGS)))
    eps = block_sd(x[S["st"]], L) / abs(xinf - x0)
    # half-time and 90%-time from 50-yr centred means of G
    tt = np.arange(25, int(t.max()) - 25, 5)
    G = np.array([1 - (xinf - centred(t, x, s, 25)) / (xinf - x0) for s in tt])
    t_half = float(tt[np.argmax(G >= 0.5)]) if (G >= 0.5).any() else np.nan
    t_90 = float(tt[np.argmax(G >= 0.9)]) if (G >= 0.9).any() else np.nan
    return {"x0": x0, "xinf": xinf, "dx": xinf - x0, "phi": phi, "rho": rho, "eps": eps, "t_half": t_half, "t_90": t_90,
            "overshoot": bool(np.nanmax(G) > 1.2), "G_t": tt.tolist(), "G": G.tolist()}


RES = {}
for x0k, L in itertools.product(("parent_mean", "pre50"), (10, 30)):
    key = f"{x0k}|L{L}"
    RES[key] = {lab: {q: step_case(lab, q, x0k, L) for q in VARS} for lab in STEPS}

# ---------------- Q1 ----------------
Q1 = {}
for key, R in RES.items():
    for q in ("ice", "TS"):
        judged = [l for l in STEPS if R[l][q]["eps"] < 0.25]
        slower = [l for l in judged if all(R[l][q]["phi"][g] - R[l][q]["rho"][g] > 2 * R[l][q]["eps"] for g in (100, 300))]
        holds = [l for l in judged if all(abs(R[l][q]["phi"][g] - R[l][q]["rho"][g]) <= 2 * R[l][q]["eps"] for g in LAGS)]
        n = len(judged)
        v = "slower" if n and len(slower) >= 2 / 3 * n else ("holds" if n and len(holds) >= 2 / 3 * n else "mixed")
        Q1[f"{key}|{q}"] = {"judged": judged, "slower": slower, "holds": holds, "verdict": v}

# ---------------- Q2 ----------------
P = RES["parent_mean|L10"]
Q2 = {}
for q in VARS:
    out = {}
    for name, (a, b) in {"same_target_1235": ("1235", "1235_new_IC"), "mirror": ("1242p5", "1237p5")}.items():
        ra, rb = P[a][q], P[b][q]
        ok = ra["eps"] < 0.25 and rb["eps"] < 0.25
        rat = [max(ra[k], rb[k]) / min(ra[k], rb[k]) if np.isfinite([ra[k], rb[k]]).all() else np.nan for k in ("t_half", "t_90")]
        out[name] = {"judged": bool(ok), "t_half": (ra["t_half"], rb["t_half"]), "t_90": (ra["t_90"], rb["t_90"]),
                     "ratios": rat, "linear": bool(ok and all(np.isfinite(rat)) and max(rat) <= 1.5)}
    Q2[q] = out

# ---------------- Q3 ----------------
far = {}
for lab in FAR:
    S = D[lab]; t = S["y"] - STEP_YEAR[lab]
    out = {}
    for q in ("ice", "TS", "H"):
        x = S["ser"][q]; xinf = S["mean"][q]
        first = float(np.nanmean(x[(t >= 1) & (t <= (5 if q == "H" else 10))]))
        out[q] = {"share_after_50_upper": (xinf - centred(t, x, 50, 5)) / (xinf - first)}
        if q == "H":
            tt = np.arange(25, 3000, 5)
            G = np.array([1 - (xinf - centred(t, x, s, 25)) / (xinf - first) for s in tt])
            out[q]["t_half"] = float(tt[np.argmax(G >= 0.5)]) if (G >= 0.5).any() else np.nan
    far[lab] = out


def sens(lab, q):
    i = WARM.index(lab)
    a, b = WARM[max(i - 1, 0)], WARM[min(i + 1, len(WARM) - 1)]
    return (D[b]["mean"][q] - D[a]["mean"][q]) / (D[a]["mu"] - D[b]["mu"])


excite = {}
for lab in WARM:
    S = D[lab]
    bi = np.array([np.nanmean(S["ser"]["ice"][S["st"]][i * 100:(i + 1) * 100]) for i in range(S["st"].sum() // 100)])
    bh = np.array([np.nanmean(S["ser"]["H"][S["st"]][i * 100:(i + 1) * 100]) for i in range(S["st"].sum() // 100)])
    ok = np.isfinite(bi) & np.isfinite(bh)
    ai, ah = bi[ok] / sens(lab, "ice"), bh[ok] / sens(lab, "H")
    excite[lab] = {"edge": S["mean"]["edge"], "r": float(np.corrcoef(ai, ah)[0, 1]),
                   "sd_ratio": float(np.std(ah, ddof=1) / np.std(ai, ddof=1))}

json.dump({"res": RES, "q1": Q1, "q2": Q2, "far": far, "excite": excite}, open(TMP + "step_response.json", "w"), indent=1, default=float)

# ---------------- print ----------------
print("Primary (X0 = parent mean, 10-yr means). Φ(t) / ρ(t) at 30, 100, 300, 1000 yr; ε; Φ(50); t½; t0.9")
for q in VARS:
    print(f"\n{VARS[q]}")
    for lab in STEPS:
        r = P[lab][q]
        dm = D[lab]["mu"] - D[PARENT[lab][0]]["mu"]
        pr = "  ".join(f"{r['phi'][g]:5.2f}/{r['rho'][g]:5.2f}" for g in LAGS)
        print(f"  {lab:11s} Δμ {dm:6.2f}  {pr}  ε {r['eps']:.2f}  Φ50 {r['phi'][50]:.2f}  t½ {r['t_half']:5.0f}  t90 {r['t_90']:5.0f}"
              f"{'  overshoot' if r['overshoot'] else ''}")
print("\nQ1 verdicts")
for k, v in Q1.items():
    print(f"  {k:24s} judged {len(v['judged'])}  slower {len(v['slower'])}  holds {len(v['holds'])}  -> {v['verdict']}")
print("\nQ2")
for q, v in Q2.items():
    print(f"  {q}: {v}")
print("\nQ3 far runs (parent unknown):", json.dumps(far, default=float))
print("\nQ3 natural excitation of the deep part (100-yr means):")
for lab, v in excite.items():
    print(f"  {lab:8s} edge {v['edge']:.1f}  r(ice,H) {v['r']:.2f}  sd(a_H)/sd(a_ice) {v['sd_ratio']:.2f}")

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
big = ["1250", "1245", "1240", "1235"]
small = ["1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5"]
col = {l: plt.cm.Oranges(0.45 + 0.15 * i) for i, l in enumerate(big)}
col.update({l: plt.cm.Blues(0.4 + 0.13 * i) for i, l in enumerate(small)})
tgrid = np.unique(np.round(np.logspace(np.log10(5), np.log10(3000), 60)).astype(int))
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
for ax, q in zip(axes[0], VARS):
    for lab in STEPS:
        S = D[lab]; t = S["y"] - STEP_YEAR[lab]; r = P[lab][q]
        phi = [(r["xinf"] - centred(t, S["ser"][q], g, 5)) / (r["xinf"] - r["x0"]) for g in tgrid]
        dm = D[lab]["mu"] - D[PARENT[lab][0]]["mu"]
        ax.plot(tgrid, phi, "-", color=col[lab], lw=1.4, label=f"{lab.replace('p', '.').replace('_new_IC', ' (2nd)')} ({dm:+g})")
        ax.plot(tgrid, acf(S["ser"][q][S["st"]], tgrid), ":", color=col[lab], lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, 1.4)
    ax.set_xlabel("years after the step (solid) / lag (dotted)")
    ax.set_ylabel("remaining fraction Φ (solid); natural autocorrelation ρ (dotted)")
    ax.set_title(f"{VARS[q]}: step response vs natural decay")
    ax.grid(alpha=0.3, lw=0.4)
axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left", title="target (step in W m⁻²)", title_fontsize=7)

ax = axes[1, 0]
for q, m, c in (("ice", "o", "k"), ("TS", "s", ORANGE), ("H", "D", BLUE)):
    xs = [D[l]["mu"] - D[PARENT[l][0]]["mu"] for l in STEPS]
    ax.scatter(xs, [P[l][q]["t_half"] for l in STEPS], marker=m, color=c, s=30, label=f"{VARS[q]}, t½", zorder=3)
    ax.scatter(xs, [P[l][q]["t_90"] for l in STEPS], marker=m, facecolor="white", edgecolor=c, s=30, label=f"{VARS[q]}, t₀.₉", zorder=3)
ax.set_yscale("log")
ax.set_xlabel("size of the step Δμ (W m⁻²)")
ax.set_ylabel("years after the step (50-yr means)")
ax.set_title("Time to complete half (filled) and 90% (open) of the response:\nsmall pushes relax more slowly")
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
ax.grid(alpha=0.3, lw=0.4)

ax = axes[1, 1]
for q, m, c in (("ice", "o", "k"), ("TS", "s", ORANGE)):
    ax.scatter([D[l]["mu"] for l in STEPS], [P[l][q]["phi"][50] for l in STEPS], marker=m, color=c, s=32, label=f"{VARS[q]} (known start)")
ax.scatter([D[l]["mu"] for l in FAR], [far[l]["TS"]["share_after_50_upper"] for l in FAR], marker="s", facecolor="white",
           edgecolor=ORANGE, s=32, label="Southern mean Ts, 1367 → (start unknown; upper bound)")
ax.axhline(0.3, color=GREY, ls=":", lw=1)
ax.invert_xaxis()
ax.set_xlabel("target μ (W m⁻²)")
ax.set_ylabel("share of the change still to come 50 yr after the step")
ax.set_title("The slow part is present at every μ")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.grid(alpha=0.3, lw=0.4)

ax = axes[1, 2]
ax.plot([excite[l]["edge"] for l in WARM], [excite[l]["r"] for l in WARM], "o-", color="k")
for l in WARM:
    if l in ("1312", "1288", "1265", "1250", "1240", "1235", "1232p5"):
        ax.annotate(l.replace("p", "."), (excite[l]["edge"], excite[l]["r"]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=GREY)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(65, 31)
ax.set_xlabel("mean ice edge, equivalent-area latitude (°S)")
ax.set_ylabel("r(ice, 700–2025 m ocean), 100-yr means of the stationary years")
ax.set_title("How much natural variability moves the deep part with the ice")
ax.grid(alpha=0.3, lw=0.4)
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/step_response.png", dpi=110)
plt.close(fig)
