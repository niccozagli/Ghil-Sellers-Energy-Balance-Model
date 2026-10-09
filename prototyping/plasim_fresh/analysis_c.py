"""Analysis C: which natural variability carries the albedo, and is it aligned with the forced change?

SW_s = TOA absorbed shortwave south of 20°S in sector s (PW; all cells in the sector's longitudes).
Anomalies about each window mean, no detrending. Variances of annual values and of non-overlapping
10-yr and 50-yr block means. Shares = var_s / Σ var. Forced change at run X = SW(next higher μ on the
warm branch) − SW(next lower μ); one-sided at the ends (1245: 1245 − 1242.5; 1232.5: 1233.75 − 1232.5).
Pre-registered: the "disconnect" is rejected if, for every run ≤ 1245, the 50-yr shares match the
forced shares within ±15 percentage points.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import band_cover, load
from gsebm.plasim_global import RUNS, run_mu

CELLS = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/cells/"
OUT = "../../figures/plasim_fresh"
R = 6.371e6
SECTORS = ("Atlantic", "Indian", "Pacific")
ALL = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC",
       "1233p75", "1232p5", "1230", "1228p5"]
WARM = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]


def sector_sw(lab):
    z = np.load(CELLS + f"{lab}.npz")
    lat, lon = z["lat"], z["lon"]
    d = load(lab)
    gw = d["gw"] / d["gw"].sum()
    row_area = 4 * np.pi * R**2 * gw
    masks = [((lon >= 295) | (lon < 20)), ((lon >= 20) & (lon < 115)), ((lon >= 115) & (lon < 295))]
    frac = np.array([m.sum() / 64 for m in masks])
    rows = lat < -20
    sw = np.stack([(z["rst_sector"][:, rows, s] * row_area[rows] * frac[s]).sum(axis=1) / 1e15
                   for s in range(3)], axis=1)  # (t, 3) PW
    return z["years"], sw


def blocks(x, n):
    m = (len(x) // n) * n
    return np.nanmean(x[:m].reshape(-1, n, *x.shape[1:]), axis=1)


data, means = {}, {}
for lab in ALL:
    years, sw = sector_sw(lab)
    w0, w1 = RUNS[lab].window
    k = (years >= w0) & (years <= w1)
    x = sw[k]
    means[lab] = np.nanmean(x, axis=0)
    a = x - means[lab]
    if lab == "1235_new_IC":
        a = a[-1500:]
    data[lab] = a

print("Variance shares (%) of SW south of 20°S by sector [Atl / Ind / Pac], and total sd (PW)")
print(f"{'run':12s} {'annual':>20s} {'10-yr':>20s} {'50-yr':>20s}   sd_1  sd_10  sd_50   forced shares   forced ΔSW (PW)")
order = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
verdict = []
for lab in ALL:
    a = data[lab]
    out = []
    sds = []
    for n in (1, 10, 50):
        b = a if n == 1 else blocks(a, n)
        v = np.nanvar(b, axis=0, ddof=1)
        out.append(100 * v / v.sum())
        sds.append(np.nanstd(np.nansum(b, axis=1), ddof=1))
    fs, fd = "", ""
    base = "1235" if lab == "1235_new_IC" else lab
    if base in order:
        i = order.index(base)
        hi_lab = order[i - 1] if i > 0 else base
        lo_lab = order[i + 1] if i < len(order) - 1 else base
        dsw = means[hi_lab] - means[lo_lab]
        share = 100 * dsw / dsw.sum()
        fs = "/".join(f"{x:4.0f}" for x in share)
        fd = "/".join(f"{x:+.3f}" for x in dsw)
        diff = np.abs(out[2] - share)
        verdict.append((lab, diff.max()))
    print(f"{lab:12s} " + "  ".join("/".join(f"{x:4.0f}" for x in o) for o in out)
          + f"   {sds[0]:.3f}  {sds[1]:.3f}  {sds[2]:.3f}   {fs}   {fd}")
dsw = means["1245"] - means["1232p5"]
print(f"\nwhole-branch forced change 1245 → 1232.5: ΔSW {dsw.round(3)} PW, shares {np.round(100*dsw/dsw.sum())} %")
print("max |50-yr share − forced share| per run (pp):", [(l, round(float(v))) for l, v in verdict])

# periodograms (raw, no smoothing) for selected runs
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
for ax, lab in zip(axes, ["1265", "1245", "1240", "1232p5"]):
    a = np.nan_to_num(data[lab])
    f = np.fft.rfftfreq(a.shape[0], 1.0)[1:]
    for s in range(3):
        p = np.abs(np.fft.rfft(a[:, s]))[1:] ** 2 / a.shape[0]
        ax.loglog(f, p, lw=0.4, color=cols[s], label=SECTORS[s], alpha=0.8)
    ax.set_title(f"μ = {run_mu(lab):g}", fontsize=9); ax.set_xlabel("frequency (1/yr)")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    for per in (50, 130, 1000):
        ax.axvline(1 / per, color="0.6", lw=0.6, ls=":")
axes[0].set_ylabel("raw periodogram of SW south of 20°S (PW² yr)"); axes[0].legend(fontsize=7)
fig.suptitle("Raw periodograms (unsmoothed) by sector; dotted: 50, 130, 1000 yr")
fig.tight_layout(); fig.savefig(f"{OUT}/C_periodograms.png", dpi=105); plt.close(fig)

# band cover variance by sector (annual and 50-yr), for description
print("\nActive-band cover (30.5+36°S; 1265: 41.5°S row) sd by sector, annual / 50-yr blocks [Atl / Ind / Pac]")
for lab in ALL:
    d = load(lab); w0, w1 = RUNS[lab].window; k = (d["years"] >= w0) & (d["years"] <= w1)
    lats = (-41.53,) if run_mu(lab) >= 1250 else (-30.46, -36.0)
    c = np.stack([band_cover(d, sectors=(s,), lats=lats)[k] for s in (1, 2, 3)], axis=1)
    if lab == "1235_new_IC":
        c = c[-1500:]
    a = c - np.nanmean(c, axis=0)
    s1 = np.nanstd(a, axis=0, ddof=1); s50 = np.nanstd(blocks(a, 50), axis=0, ddof=1)
    print(f"  {lab:12s} rows {lats}: " + "/".join(f"{x:.3f}" for x in s1) + "   " + "/".join(f"{x:.3f}" for x in s50))
