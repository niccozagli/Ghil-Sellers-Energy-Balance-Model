"""Check 2 (criteria in Claude/plasim_mechanism/plan_checks_2_5.md). Annual values."""

import matplotlib.pyplot as plt
import numpy as np

from common import band_cover, load, lsg_layer_mean, mean_between
from gsebm.plasim_global import RUNS

OUT = "../../figures/plasim_fresh"
ONSET = {"1230": 7824, "1225": 7590}
ATL_ROWS = {"1230": (-30.46, -36.0), "1225": (-19.38, -24.92)}

# ---- 2a: excursion size --------------------------------------------------------------------------
d = load("1230"); y = d["years"]; A = band_cover(d, sectors=(1,))
ref = (y >= 7200) & (y <= 7699)
mref, sref = np.nanmean(A[ref]), np.nanstd(A[ref], ddof=1)
peak = np.nanmax(A[(y >= 7814) & (y <= 7840)])
zj = (peak - mref) / sref
creep = (y >= 5100) & (y <= 7823)
zcreep = (A[creep] - mref) / sref
print(f"2a. 1230 pause reference mean {mref:.3f}, sd {sref:.3f}; jump peak {peak:.3f} → z_j = {zj:.2f}")
print(f"    largest excursion in 1230's creep 5100–7823 (same reference): z = {np.nanmax(zcreep):.2f}")
for lab in ["1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]:
    e = load(lab); w0, w1 = RUNS[lab].window; k = (e["years"] >= w0) & (e["years"] <= w1)
    a = band_cover(e, sectors=(1,))[k]
    if lab == "1235_new_IC":
        a = a[-1500:]
    z = (a - np.nanmean(a)) / np.nanstd(a, ddof=1)
    n = np.sum(z >= zj)
    print(f"    {lab:12s}: years with z ≥ z_j: {n} in {len(a)} yr ({4000 * n / len(a):.1f} per 4000 yr); largest z {np.nanmax(z):.2f}")

# ---- 2b: freezing events vs Atlantic maxima; reversals ----------------------------------------
def atl_maxima(years, a, period):
    k = np.flatnonzero((years >= period[0] - 20) & (years <= period[1] + 20))
    out = []
    for i in k:
        lo, hi = max(i - 20, 0), min(i + 21, len(a))
        if np.isfinite(a[i]) and a[i] == np.nanmax(a[lo:hi]) and period[0] <= years[i] <= period[1]:
            out.append(int(years[i]))
    return np.array(out)


def ip_cells(e):
    lat_s = e["lat"][e["lat"] < 0]
    ocean = e["ocean"][e["lat"] < 0]
    rows = [int(np.argmin(np.abs(lat_s - r))) for r in (-19.38, -24.92, -30.46, -36.0)]
    ip = (e["lon"] >= 20) & (e["lon"] < 295)
    cells = [(r, c) for r in rows for c in np.flatnonzero(ocean[r] & ip)]
    return cells, e["sic_map_south"].astype(float)


def events(years, sic, cells, first, last):
    """Freezing events: previous-20-yr mean < 0.9, then ≥ 0.98 for 20 consecutive years, all within [first, last]."""
    ev = []
    for (r, c) in cells:
        x = sic[:, r, c]
        for i in range(20, len(years) - 19):
            t = years[i]
            if t < first + 20 or t + 19 > last:
                continue
            if np.nanmean(x[i - 20:i]) < 0.9 and np.all(x[i:i + 20] >= 0.98) and x[i - 1] < 0.98:
                ev.append((int(t), r, c, i))
    return ev


def reverts(years, sic, ev, last):
    """Share of events whose cell later has cover < 0.5 for ≥ 10 consecutive years before `last`."""
    n_rev, risk = 0, []
    for (t, r, c, i) in ev:
        x = sic[:, r, c]; j_end = np.flatnonzero(years <= last)[-1]
        risk.append(years[j_end] - t)
        low = x[i + 20:j_end + 1] < 0.5
        run = 0; found = False
        for v in low:
            run = run + 1 if v else 0
            if run >= 10:
                found = True; break
        n_rev += found
    return n_rev, len(ev), (np.median(risk) if risk else np.nan)


print("\n2b(i). Freezing events (Indian/Pacific cells, 19.4–36°S) near Atlantic maxima")
res = {}
for lab in ("1230", "1225"):
    e = load(lab); y = e["years"]
    cells, sic = ip_cells(e)
    first = 5100 if lab == "1230" else 5800
    last = ONSET[lab] - 1
    A = band_cover(e, sectors=(1,), lats=ATL_ROWS[lab])
    mx = atl_maxima(y, A, (first, last))
    ev = events(y, sic, cells, first, last)
    yrs = np.arange(first, last + 1)
    out = {}
    for W in (5, 10, 15):
        near_ev = np.mean([np.min(np.abs(mx - t)) <= W for (t, *_) in ev]) if ev else np.nan
        chance = np.mean([np.min(np.abs(mx - t)) <= W for t in yrs])
        out[W] = (near_ev, chance)
    res[lab] = dict(n=len(ev), out=out, mx=mx, ev=ev, A=A, y=y)
    print(f"  {lab}: Atlantic rows {ATL_ROWS[lab]}, {len(mx)} maxima in {first}–{last} "
          f"(mean spacing {np.mean(np.diff(mx)):.0f} yr); {len(ev)} freezing events")
    for W, (f, ch) in out.items():
        print(f"     ±{W:2d} yr: event fraction {f:.2f}  chance {ch:.2f}  difference {f - ch:+.2f}")
    n_rev, n, risk = reverts(y, sic, ev, last)
    res[lab]["rev"] = (n_rev, n, risk)
    print(f"  2b(ii) {lab}: {n_rev} of {n} frozen cells revert (<0.5 for ≥10 yr) before the onset; median time at risk {risk:.0f} yr")

e = load("1232p5"); y = e["years"]; w0, w1 = RUNS["1232p5"].window
cells, sic = ip_cells(e)
ev = events(y, sic, cells, w0, w1)
n_rev, n, risk = reverts(y, sic, ev, w1)
print(f"  2b(ii) 1232.5 window: {n_rev} of {n} frozen cells revert; median time at risk {risk:.0f} yr")
res["1232p5"] = (n_rev, n, risk)

# ---- 2c: stall and onset excursion ---------------------------------------------------------------
print("\n2c. Ocean stall before the onset and the Atlantic excursion at the onset")
for lab in ("1230", "1225"):
    e = load(lab); y = e["years"]; t0 = ONSET[lab]
    H = lsg_layer_mean(e, 1, -90, 0)
    late = (mean_between(y, H, t0 - 50, t0 - 1) - mean_between(y, H, t0 - 500, t0 - 451)) / 450
    early = (mean_between(y, H, t0 - 550, t0 - 501) - mean_between(y, H, t0 - 1500, t0 - 1451)) / 950
    stall = (-late) < (-early) / 3
    A = band_cover(e, sectors=(1,), lats=ATL_ROWS[lab])
    at = np.nanmax(A[(y >= t0 - 10) & (y <= t0 + 10)]); before = np.nanmax(A[(y >= t0 - 1000) & (y <= t0 - 11)])
    print(f"  {lab}: cooling rate last 500 yr {-late*1000:.3f} K/kyr vs preceding 1000 yr {-early*1000:.3f} K/kyr "
          f"→ stall {stall};  Atlantic max at onset ±10 yr {at:.3f} vs previous 1000 yr {before:.3f} → largest {at > before}")

# ---- figure ----------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
for ax, lab in zip(axes, ("1230", "1225")):
    r = res[lab]; y = r["y"]; first = 5100 if lab == "1230" else 5800
    k = (y >= first) & (y <= ONSET[lab] + 100)
    ax.plot(y[k], r["A"][k], lw=0.6, color="#2a78d6", label=f"Atlantic band cover, rows {ATL_ROWS[lab]}")
    for t in r["mx"]:
        ax.axvline(t, color="#2a78d6", lw=0.4, alpha=0.4)
    tev = [t for (t, *_) in r["ev"]]
    ax.plot(tev, np.full(len(tev), np.nanmin(r["A"][k]) - 0.02), "v", color="#eb6834", ms=6,
            label="Indian/Pacific cell freezes over (event)")
    ax.axvline(ONSET[lab], color="k", ls=":", lw=1.2, label="jump onset")
    ax.set_title(f"μ = {lab}", fontsize=9); ax.grid(alpha=0.3, lw=0.4); ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel("model year (thin blue lines: Atlantic maxima)")
fig.suptitle("Check 2b: do Indian/Pacific cells freeze over at Atlantic maxima during the creep?")
fig.tight_layout(); fig.savefig(f"{OUT}/check2_ratchet.png", dpi=110); plt.close(fig)
