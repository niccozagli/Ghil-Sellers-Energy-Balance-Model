"""Analysis A, part 2: uncertainties of the fast/slow readouts, ice at equal H across μ,
and the 1230 drift in 100-yr blocks.

Signs: E sensitivities in ° equatorward per W m⁻² of μ *decrease* (ΔE/Δμ with E in °S);
C sensitivities in cover per W m⁻² decrease; slow ratios per K of H *cooling*.
Null distribution of the fast-stage estimator: the same estimator
[mean(y+20..y+50) − mean(y−99..y)] applied to the parent's own stationary record
(1240: 9000–17379; 1265: 4000–11999), y every 10 yr.
"""

import numpy as np

from common import PARENT, band_cover, block_means, equivalent_edge, load, lsg_layer_mean, mean_between
from gsebm.plasim_global import RUNS, run_mu

S = {}
for lab in ["1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75",
            "1232p5", "1230", "1225", "1228p5"]:
    d = load(lab)
    S[lab] = dict(years=d["years"], H=lsg_layer_mean(d, 1, -90, 0), E=equivalent_edge(d),
                  C=band_cover(d), mu=d["mu"])

null_range = {"1240": (9000, 17379), "1265": (4000, 11999)}
null = {}
for p, (a, b) in null_range.items():
    s = S[p]
    est = {k: [] for k in ("E", "C", "H")}
    for y in range(a + 100, b - 50, 10):
        for k in est:
            est[k].append(mean_between(s["years"], s[k], y + 20, y + 50) - mean_between(s["years"], s[k], y - 99, y))
    null[p] = {k: np.nanstd(v, ddof=1) for k, v in est.items()}
    print(f"null sd of fast-stage estimator, parent {p}: E {null[p]['E']:.3f}°  C {null[p]['C']:.4f}  H {null[p]['H']:.4f} K")

print("\nrun          Δμ    fastE (°/W)     fastC (/W)        slowΔH   slow E (°/K)   slow C (/K)    slow share E")
for lab in ["1250", "1245", "1235", "1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5"]:
    parent, y0 = PARENT[lab]
    p, s = S[parent], S[lab]
    dmu = s["mu"] - run_mu(parent)
    par = {k: mean_between(p["years"], p[k], y0 - 99, y0) for k in ("H", "E", "C")}
    fast = {k: mean_between(s["years"], s[k], y0 + 20, y0 + 50) for k in ("H", "E", "C")}
    w0, w1 = RUNS[lab].window
    win = {k: mean_between(s["years"], s[k], w0, w1) for k in ("H", "E", "C")}
    # sd of a 31-yr mean in the run's own window (non-overlapping blocks) for the slow-stage noise
    _, b31E = block_means(s["years"], s["E"], w0, w1 + 1, 31)
    _, b31C = block_means(s["years"], s["C"], w0, w1 + 1, 31)
    fE, fC = (fast["E"] - par["E"]) / dmu, -(fast["C"] - par["C"]) / dmu
    sfE, sfC = null[parent]["E"] / abs(dmu), null[parent]["C"] / abs(dmu)
    dH = win["H"] - fast["H"]
    slE, slC = (win["E"] - fast["E"]) / dH, -(win["C"] - fast["C"]) / dH
    sslE, sslC = np.nanstd(b31E, ddof=1) / abs(dH), np.nanstd(b31C, ddof=1) / abs(dH)
    share = (win["E"] - fast["E"]) / (win["E"] - par["E"])
    print(f"{lab:12s}{dmu:6.2f}  {fE:6.3f} ± {sfE:5.3f}  {fC:7.4f} ± {sfC:6.4f}  {dH:7.3f}  "
          f"{slE:6.2f} ± {sslE:4.2f}  {slC:6.3f} ± {sslC:5.3f}  {share:5.2f}")

# ---- ice at equal H across μ: transients (after year 300) vs stationary equilibria --------------
print("\nIce at equal H: run X's 100-yr blocks (after year 300) within ±0.01 K of run Y's window-mean H.")
print("ΔE = E_X − E_Y (°S; negative = X further equatorward); per W m⁻² = ΔE/(μ_X − μ_Y).")
eq = {}
for lab in ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]:
    s = S[lab]; w0, w1 = RUNS[lab].window
    eq[lab] = {k: mean_between(s["years"], s[k], w0, w1) for k in ("H", "E", "C")}
for X in ["1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5", "1235", "1230", "1225"]:
    s = S[X]; _, y0 = PARENT[X]
    stop = 7800 if X == "1230" else (5600 if X == "1225" else int(s["years"][-1]) + 1)
    yb, Hb = block_means(s["years"], s["H"], y0 + 300, stop)
    _, Eb = block_means(s["years"], s["E"], y0 + 300, stop)
    _, Cb = block_means(s["years"], s["C"], y0 + 300, stop)
    for Y, e in eq.items():
        if Y == X or run_mu(Y) == s["mu"]:
            continue
        k = np.abs(Hb - e["H"]) <= 0.01
        if k.sum() >= 1:
            dE = np.nanmean(Eb[k]) - e["E"]; dC = np.nanmean(Cb[k]) - e["C"]
            dmu = s["mu"] - run_mu(Y)
            print(f"  X {X:11s} at H of Y {Y:11s} (H {e['H']:.3f}, n {k.sum():2d}, years {int(yb[k][0])}–{int(yb[k][-1])+99}):"
                  f" ΔE {dE:+.2f}°  ΔC {dC:+.3f}  Δμ {dmu:+.2f}  → {dE/dmu:+.3f} °/W  {-dC/dmu:+.4f} /W")

# ---- 1230 in 100-yr blocks ---------------------------------------------------------------------
s = S["1230"]
print("\n1230, 100-yr block means: year, H, E, C, and change per 100 yr")
yb, Hb = block_means(s["years"], s["H"], 4600, 8600)
_, Eb = block_means(s["years"], s["E"], 4600, 8600)
_, Cb = block_means(s["years"], s["C"], 4600, 8600)
for i in range(len(yb)):
    dE = Eb[i] - Eb[i - 1] if i else np.nan; dH = Hb[i] - Hb[i - 1] if i else np.nan
    print(f"  {int(yb[i])}: H {Hb[i]:.3f} ({dH:+.3f})  E {Eb[i]:.2f} ({dE:+.2f})  C {Cb[i]:.3f}")
