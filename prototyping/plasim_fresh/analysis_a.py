"""Analysis A: is the active ice slaved to the slow ocean state? (annual values; block means as stated)

H  = SH ocean θ 700–2025 m (wet-volume weighted, all SH LSG rows)
E  = equivalent-area Southern edge (°S, threshold-free)
C  = Indo-Pacific band cover (rows 30.5 + 36.0°S, Indian + Pacific ocean cells, area-weighted)
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import (PARENT, band_cover, block_means, equivalent_edge, load, lsg_layer_mean,
                    mean_between)
from gsebm.plasim_global import RUNS, run_mu

OUT = "../../figures/plasim_fresh"
TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
LABELS = ["1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75",
          "1232p5", "1230", "1225", "1228p5"]

S = {}
for lab in LABELS + ["1288", "1312"]:
    d = load(lab)
    S[lab] = dict(years=d["years"], H=lsg_layer_mean(d, 1, -90, 0), Hu=lsg_layer_mean(d, 0, -90, 0),
                  E=equivalent_edge(d), C=band_cover(d), CA=band_cover(d, sectors=(1,)), mu=d["mu"])

# ---- (a) fast stage and (b) slow stage, per step run --------------------------------------
rows = []
for lab in ["1250", "1245", "1240", "1235", "1230", "1225", "1242p5", "1237p5", "1235_new_IC",
            "1233p75", "1232p5", "1228p5"]:
    parent, y0 = PARENT[lab]
    p, s = S[parent], S[lab]
    dmu = s["mu"] - run_mu(parent)
    par = {k: mean_between(p["years"], p[k], y0 - 99, y0) for k in ("H", "E", "C")}
    fast = {k: mean_between(s["years"], s[k], y0 + 20, y0 + 50) for k in ("H", "E", "C")}
    w0, w1 = RUNS[lab].window
    win = {k: mean_between(s["years"], s[k], w0, w1) for k in ("H", "E", "C")}
    r = dict(run=lab, parent=parent, dmu=dmu,
             fast_dE=fast["E"] - par["E"], fast_dC=fast["C"] - par["C"], fast_dH=fast["H"] - par["H"],
             fast_sens_E=-(fast["E"] - par["E"]) / dmu, fast_sens_C=(fast["C"] - par["C"]) / -dmu,
             slow_dE=win["E"] - fast["E"], slow_dC=win["C"] - fast["C"], slow_dH=win["H"] - fast["H"])
    r["slow_ratio_E"] = -r["slow_dE"] / r["slow_dH"] if abs(r["slow_dH"]) > 0.03 else np.nan
    r["slow_ratio_C"] = -r["slow_dC"] / r["slow_dH"] if abs(r["slow_dH"]) > 0.03 else np.nan
    r["total_sens_E"] = -(win["E"] - par["E"]) / dmu
    r["slow_share_E"] = r["slow_dE"] / (win["E"] - par["E"])
    rows.append(r)
print("Fast stage = years 20–50 after the step minus parent's last 100 yr; slow = window minus fast.")
print("Sign: E sensitivity in ° equatorward per W m⁻² of μ decrease; C per W m⁻² decrease;")
print("slow ratios per K of H cooling (° equatorward / K, cover / K).")
hdr = f"{'run':12s}{'Δμ':>7s}{'fast ΔH':>9s}{'fastE°/W':>9s}{'fastC/W':>9s}{'slowΔH':>8s}{'slow E/K':>9s}{'slow C/K':>9s}{'totE°/W':>9s}{'slowshare':>10s}"
print(hdr)
for r in rows:
    print(f"{r['run']:12s}{r['dmu']:7.2f}{r['fast_dH']:9.3f}{r['fast_sens_E']:9.3f}{r['fast_sens_C']:9.4f}"
          f"{r['slow_dH']:8.3f}{r['slow_ratio_E']:9.2f}{r['slow_ratio_C']:9.3f}{r['total_sens_E']:9.3f}{r['slow_share_E']:10.2f}")

# ---- (c) the 1235 pair: ice at equal H after year 300 ------------------------------------
pair = {}
for lab in ("1235", "1235_new_IC"):
    s = S[lab]; _, y0 = PARENT[lab]
    good = np.isfinite(s["H"])
    a0, a1 = y0 + 300, int(s["years"][good][-1]) + 1
    e, H = block_means(s["years"], s["H"], a0, a1)
    _, E = block_means(s["years"], s["E"], a0, a1)
    _, C = block_means(s["years"], s["C"], a0, a1)
    w0, w1 = RUNS[lab].window
    _, Ew = block_means(s["years"], s["E"], w0, w1 + 1)
    _, Cw = block_means(s["years"], s["C"], w0, w1 + 1)
    pair[lab] = dict(H=H, E=E, C=C, sdE=np.nanstd(Ew, ddof=1), sdC=np.nanstd(Cw, ddof=1))
lo = max(np.nanmin(pair["1235"]["H"]), np.nanmin(pair["1235_new_IC"]["H"]))
hi = min(np.nanmax(pair["1235"]["H"]), np.nanmax(pair["1235_new_IC"]["H"]))
bins = np.arange(lo, hi + 1e-9, 0.02)
print(f"\n1235 pair: common H range {lo:.3f}–{hi:.3f} K, bins of 0.02 K")
sdE = np.sqrt(0.5 * (pair["1235"]["sdE"] ** 2 + pair["1235_new_IC"]["sdE"] ** 2))
sdC = np.sqrt(0.5 * (pair["1235"]["sdC"] ** 2 + pair["1235_new_IC"]["sdC"] ** 2))
print(f"  pooled sd of 100-yr block means in the windows: E {sdE:.3f}°, C {sdC:.4f}")
for b0 in bins[:-1]:
    vals = {}
    for lab in pair:
        k = (pair[lab]["H"] >= b0) & (pair[lab]["H"] < b0 + 0.02)
        vals[lab] = (k.sum(), np.nanmean(pair[lab]["E"][k]) if k.any() else np.nan,
                     np.nanmean(pair[lab]["C"][k]) if k.any() else np.nan)
    n1, E1, C1 = vals["1235"]; n2, E2, C2 = vals["1235_new_IC"]
    if n1 and n2:
        print(f"  H {b0:.3f}–{b0+0.02:.3f}: n {n1:2d}/{n2:2d}  E {E1:.2f} vs {E2:.2f} (Δ {E2-E1:+.2f}, "
              f"{abs(E2-E1)/sdE:.1f} sd)  C {C1:.3f} vs {C2:.3f} (Δ {C2-C1:+.3f}, {abs(C2-C1)/sdC:.1f} sd)")

# ---- (d) 1230: the drift and the jump ------------------------------------------------------
s = S["1230"]; y = s["years"]
ref = mean_between(y, s["E"], 7200, 7699)
tj = None
for yy in range(7500, 9000):
    if mean_between(y, s["E"], yy, yy + 49) < ref - 1.0:
        tj = yy; break
print(f"\n1230: jump onset (first y>7500 with E over [y, y+49] > 1° equatorward of the 7200–7699 mean) = {tj}")
def blk(a): return {k: mean_between(y, s[k], a, a + 99) for k in ("H", "E", "C", "Hu")}
b_start, b_mid, b_end = blk(5100), blk(tj - 500), blk(tj - 100)
for name, b in (("5100–5199", b_start), (f"{tj-500}–{tj-401}", b_mid), (f"{tj-100}–{tj-1}", b_end)):
    print(f"  {name}: H {b['H']:.3f}  Hu {b['Hu']:.3f}  E {b['E']:.2f}  C {b['C']:.3f}")
for name, a, b in (("early drift", b_start, b_mid), ("last 500 yr", b_mid, b_end)):
    dH, dE, dC = b["H"] - a["H"], b["E"] - a["E"], b["C"] - a["C"]
    print(f"  {name}: ΔH {dH:+.3f} K  ΔE {dE:+.2f}°  ΔC {dC:+.3f}  ratio E {-dE/dH if abs(dH)>0.03 else float('nan'):.2f} °/K"
          f"  ratio C {dC/-dH if abs(dH)>0.03 else float('nan'):.3f} /K")
json.dump(dict(rows=rows, tj=tj), open(TMP + "analysis_a.json", "w"), default=float)

# ---- figure: (H, E) and (H, C) for all runs -----------------------------------------------
cmap = plt.get_cmap("viridis"); norm = plt.Normalize(1224, 1268)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for lab in LABELS:
    s = S[lab]; c = cmap(norm(s["mu"])); _, y0 = PARENT[lab]
    w0, w1 = RUNS[lab].window
    for ax, key in zip(axes, ("E", "C")):
        tr = (s["years"] > y0) & (s["years"] < w0)
        if lab in ("1230", "1225"):
            tr = s["years"] > y0
        ax.plot(s["H"][tr], s[key][tr], lw=0.3, color=c, alpha=0.6)
        e, Hb = block_means(s["years"], s["H"], w0, w1 + 1)
        _, Kb = block_means(s["years"], s[key], w0, w1 + 1)
        ax.plot(Hb, Kb, "o", ms=3.5, color=c, mec="white", mew=0.4,
                label=f"{s['mu']:g}" if key == "E" else None)
axes[0].invert_yaxis()
axes[0].set_ylabel("equivalent-area S edge (°S)"); axes[1].set_ylabel("Indo-Pacific band cover (30.5+36°S)")
for ax in axes:
    ax.set_xlabel("SH ocean θ 700–2025 m (K)"); ax.grid(alpha=0.3, lw=0.4); ax.invert_xaxis()
axes[0].legend(fontsize=7, ncol=2, title="μ (dots: 100-yr means in window)", title_fontsize=7)
fig.suptitle("Ice against the slow ocean index: transients (annual, thin) and stationary windows (100-yr means)")
fig.tight_layout(); fig.savefig(f"{OUT}/A_ice_vs_ocean.png", dpi=110); plt.close(fig)
