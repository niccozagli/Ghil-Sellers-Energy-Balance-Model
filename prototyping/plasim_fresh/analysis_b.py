"""Analysis B: cell-level albedo law in the active rows (annual values, window years).

c = annual ice cover (fraction of year covered); rst = TOA absorbed SW; rss = surface net SW.
Primary (pre-registered): pooled cell-years per row × sector, OLS slope of rst (and rss) on c over
c in [0.1, 0.5] and [0.6, 1.0]; c-bin means of width 0.1. Ts-conditioned: same slopes within 2-K bins
of the cell's annual Ts (≥30 cell-years and c sd ≥ 0.1 per bin and range), count-weighted mean.
Check (added): within-cell slopes (values demeaned per cell × run inside each c-range), to remove
differences in cloud climatology between cells. Halves of each window as a reproducibility check.
"""

import matplotlib.pyplot as plt
import numpy as np

from gsebm.plasim_global import run_mu

CELLS = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/cells/"
OUT = "../../figures/plasim_fresh"
RUNS_B = ["1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5",
          "1230", "1228p5"]
SECT = {"Atlantic": lambda lon: (lon >= 295) | (lon < 20), "Indian": lambda lon: (lon >= 20) & (lon < 115),
        "Pacific": lambda lon: (lon >= 115) & (lon < 295)}
RANGES = {"low": (0.1, 0.5), "high": (0.6, 1.0)}


def gather(row_lat, sectors, half=None):
    out = {k: [] for k in ("c", "rst", "rss", "ts", "cell", "run", "mu")}
    for i, lab in enumerate(RUNS_B):
        z = np.load(CELLS + f"{lab}.npz")
        r = int(np.argmin(np.abs(z["row_lat"] - row_lat)))
        lon = z["lon"]
        m = z["ocean"][r] & np.any([SECT[s](lon) for s in sectors], axis=0)
        n = z["cell_sea_ice_concentration"].shape[0]
        t = slice(None) if half is None else (slice(0, n // 2) if half == 0 else slice(n // 2, n))
        c = z["cell_sea_ice_concentration"][t, r][:, m].astype(float)
        out["c"].append(c.ravel())
        out["rst"].append(z["cell_rst"][t, r][:, m].astype(float).ravel())
        out["rss"].append(z["cell_rss"][t, r][:, m].astype(float).ravel())
        out["ts"].append(z["cell_surface_temperature"][t, r][:, m].astype(float).ravel())
        out["cell"].append(np.tile(np.flatnonzero(m), c.shape[0]))
        out["run"].append(np.full(c.size, i))
        out["mu"].append(np.full(c.size, run_mu(lab)))
    return {k: np.concatenate(v) for k, v in out.items()}


def ols(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.std(x[ok]) < 0.1 * (1 if x.max() <= 1.01 else 0):
        return np.nan
    return np.polyfit(x[ok], y[ok], 1)[0]


def slopes(g, field):
    res = {}
    for name, (lo, hi) in RANGES.items():
        k = (g["c"] >= lo) & (g["c"] <= hi) & np.isfinite(g[field])
        pooled = ols(g["c"][k], g[field][k]) if k.sum() >= 30 and np.std(g["c"][k]) >= 0.05 else np.nan
        # Ts-conditioned
        num, den = 0.0, 0
        for t0 in np.arange(np.floor(np.nanmin(g["ts"][k]) if k.any() else 0), np.nanmax(g["ts"][k]) if k.any() else 0, 2.0):
            kk = k & (g["ts"] >= t0) & (g["ts"] < t0 + 2)
            if kk.sum() >= 30 and np.std(g["c"][kk]) >= 0.1:
                num += kk.sum() * np.polyfit(g["c"][kk], g[field][kk], 1)[0]; den += kk.sum()
        cond = num / den if den else np.nan
        # within-cell (cell × run demeaned)
        key = g["cell"][k] * 100 + g["run"][k]
        xc, yc = g["c"][k].copy(), g[field][k].copy()
        for u in np.unique(key):
            j = key == u
            xc[j] -= xc[j].mean(); yc[j] -= yc[j].mean()
        within = np.polyfit(xc, yc, 1)[0] if k.sum() >= 30 and np.std(xc) >= 0.03 else np.nan
        res[name] = dict(n=int(k.sum()), pooled=pooled, cond=cond, within=within, cond_n=den)
    return res


cases = [(-36.0, ("Indian", "Pacific")), (-36.0, ("Atlantic",)), (-30.46, ("Atlantic",)),
         (-30.46, ("Indian", "Pacific")), (-41.53, ("Indian", "Pacific")), (-41.53, ("Atlantic",)),
         (-24.92, ("Atlantic", "Pacific"))]
print("Slopes of absorbed SW on annual cover, W m⁻² per unit cover (negative = brighter with more ice).")
print("pooled / Ts-conditioned / within-cell; n = cell-years in range. Ratio = high/low (Ts-conditioned).")
for row, sec in cases:
    g = gather(row, sec)
    for field in ("rst", "rss"):
        s = slopes(g, field)
        lo, hi = s["low"], s["high"]
        ratio = hi["cond"] / lo["cond"] if np.isfinite(hi["cond"]) and np.isfinite(lo["cond"]) else np.nan
        print(f"{abs(row):4.1f}°S {'+'.join(sec):16s} {field}: low n {lo['n']:6d} {lo['pooled']:7.1f}/{lo['cond']:7.1f}/{lo['within']:7.1f}"
              f" | high n {hi['n']:6d} {hi['pooled']:7.1f}/{hi['cond']:7.1f}/{hi['within']:7.1f} | ratio {ratio:5.2f}")
    if row == -36.0 and sec == ("Indian", "Pacific"):
        for h in (0, 1):
            gh = gather(row, sec, half=h)
            for field in ("rst", "rss"):
                s = slopes(gh, field)
                print(f"     half {h+1} {field}: low {s['low']['pooled']:7.1f}/{s['low']['cond']:7.1f}/{s['low']['within']:7.1f}"
                      f"  high {s['high']['pooled']:7.1f}/{s['high']['cond']:7.1f}/{s['high']['within']:7.1f}"
                      f"  ratio(cond) {s['high']['cond']/s['low']['cond']:5.2f}")

# brightening of always-covered cells: rst and rss vs Ts for c > 0.98
print("\nAlways-covered cell-years (c > 0.98): slope of absorbed SW on Ts (W m⁻² per K; positive = darker when warmer)")
for row, sec in [(-41.53, ("Atlantic", "Indian", "Pacific")), (-36.0, ("Atlantic", "Indian", "Pacific")),
                 (-30.46, ("Atlantic", "Indian", "Pacific"))]:
    g = gather(row, sec)
    k = (g["c"] > 0.98) & np.isfinite(g["ts"]) & np.isfinite(g["rst"]) & np.isfinite(g["rss"])
    for field in ("rst", "rss"):
        key = g["cell"][k] * 100 + g["run"][k]
        x, y = g["ts"][k].copy(), g[field][k].copy()
        for u in np.unique(key):
            j = key == u; x[j] -= x[j].mean(); y[j] -= y[j].mean()
        print(f"  {abs(row):4.1f}°S {field}: n {k.sum():6d}  within-cell slope {np.polyfit(x, y, 1)[0]:6.2f}"
              f"  (Ts range {np.nanmin(g['ts'][k]):.1f}–{np.nanmax(g['ts'][k]):.1f} K)")

# figure: c-bin means for IP 36°S and Atlantic 30.5°S, per run
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, (row, sec) in enumerate([(-36.0, ("Indian", "Pacific")), (-36.0, ("Atlantic",)), (-30.46, ("Atlantic",))]):
    g = gather(row, sec)
    for rr, field in enumerate(("rst", "rss")):
        ax = axes[rr, col]
        ax.scatter(g["c"][::7], g[field][::7], s=1, color="0.75", alpha=0.3, rasterized=True)
        edges = np.linspace(0, 1, 11)
        for i, lab in enumerate(RUNS_B):
            k = g["run"] == i
            if (k & (g["c"] > 0.05) & (g["c"] < 0.95)).sum() < 50:
                continue
            idx = np.clip(np.digitize(g["c"][k], edges) - 1, 0, 9)
            means = [np.nanmean(g[field][k][idx == b]) if (idx == b).sum() >= 20 else np.nan for b in range(10)]
            ax.plot(0.5 * (edges[1:] + edges[:-1]), means, "-o", ms=3, lw=1,
                    color=plt.get_cmap("plasma")((run_mu(lab) - 1226) / 42), label=f"{run_mu(lab):g}")
        ax.set_title(f"{abs(row):.1f}°S {'+'.join(sec)}: {field} vs annual cover", fontsize=9)
        ax.set_xlabel("annual ice cover (fraction of year)"); ax.set_ylabel(f"{field} (W m⁻²)")
        ax.grid(alpha=0.3, lw=0.4)
axes[0, 0].legend(fontsize=6, ncol=2, title="μ (c-bin means)", title_fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/B_albedo_law.png", dpi=105); plt.close(fig)
