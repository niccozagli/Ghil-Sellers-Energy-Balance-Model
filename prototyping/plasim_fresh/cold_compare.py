"""Cold state against warm state (criteria: Claude/plasim_mechanism/plan_cold.md). Basic methods."""

import json

import numpy as np

from common import load, lsg_layer_mean, sector_ice_area, equivalent_edge

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
WARM = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
COLD = {"1230": (11600, 16369), "1228p5": (15000, 16899)}
LAGS = (1, 2, 5, 10, 20)


def acf(x, lags):
    x = np.nan_to_num(x - np.nanmean(x)); v = np.sum(x * x) / len(x)
    return np.array([np.sum(x[:-k] * x[k:]) / len(x) / v for k in lags])


def efold(x):
    x = np.nan_to_num(x - np.nanmean(x)); v = np.sum(x * x) / len(x)
    for k in range(1, 200):
        if np.sum(x[:-k] * x[k:]) / len(x) / v < np.exp(-1):
            return k
    return np.nan


def detr(x):
    t = np.arange(len(x)); ok = np.isfinite(x); c = np.polyfit(t[ok], x[ok], 1); return x - np.polyval(c, t)


def seg_se(x, fn, L=500):
    n = len(x) // L
    vals = [fn(x[i * L:(i + 1) * L]) for i in range(n)]
    return (np.std(vals, axis=0, ddof=1) / np.sqrt(n)) if n >= 3 else np.full(np.shape(fn(x[:L])), np.nan), n


def window(lab):
    if lab in COLD:
        return COLD[lab]
    return STAT[lab]["0.95"], STAT[lab]["end"]


series = {}
for lab in WARM + list(COLD):
    d = load(lab); y = d["years"]; a, b = window(lab); k = (y >= a) & (y <= b)
    series[lab] = dict(atl=sector_ice_area(d, (1,))[k], ip=sector_ice_area(d, (2, 3))[k],
                       tot=sector_ice_area(d, (1, 2, 3))[k], mu=d["mu"])

print("1. Fast memory: autocorrelation at lags", LAGS, "(± 2 SE from 500-yr segments), e-folding lag, sd")
R1 = {}
for lab, s in series.items():
    out = {}
    for q in ("atl", "ip", "tot"):
        a = acf(s[q], LAGS); se, n = seg_se(s[q], lambda z: acf(z, LAGS))
        out[q] = dict(acf=a.tolist(), se=np.asarray(se).tolist(), efold=efold(s[q]), sd=float(np.nanstd(s[q])), nseg=n)
    R1[lab] = out
    print(f"  {lab:12s} " + "  ".join(f"{q}: ACF {np.round(out[q]['acf'], 2)} e-fold {out[q]['efold']} sd {out[q]['sd']:.2f}"
                                      for q in ("atl", "ip", "tot")))
for q in ("ip", "tot"):
    a30, a28 = np.array(R1["1230"][q]["acf"]), np.array(R1["1228p5"][q]["acf"])
    s30, s28 = np.array(R1["1230"][q]["se"]), np.array(R1["1228p5"][q]["se"])
    diff = a28 - a30; tol = 2 * np.sqrt(np.nan_to_num(s30) ** 2 + np.nan_to_num(s28) ** 2)
    print(f"  slowing along the cold branch ({q}, lags 1–10): 1228.5 − 1230 = {np.round(diff[:4], 3)}  2 SE = {np.round(tol[:4], 3)}"
          f" → {bool(np.all(diff[:4] > tol[:4]))}")
    wmin = min(R1[l][q]["efold"] for l in WARM[1:])
    print(f"  cold faster than warm ({q}): e-fold 1230 {R1['1230'][q]['efold']}, 1228.5 {R1['1228p5'][q]['efold']}; warm minimum {wmin}"
          f" → {R1['1230'][q]['efold'] < wmin and R1['1228p5'][q]['efold'] < wmin}")

print("\n2. Ice–ocean link (100-yr block means; from the sharpened analysis B):")
for lab in WARM + list(COLD):
    b = SH["B"][lab]["base"]
    print(f"  {lab:12s} slope on W {b['slope']:7.2f} ± {b['se']:.2f}  r {b['r']:+.2f}   (on H: {SH['B'][lab]['onH']['slope']:7.2f}, r {SH['B'][lab]['onH']['r']:+.2f})")
print(f"  'ocean no longer paces the ice' (|r| < 0.5 in both cold runs): "
      f"{all(abs(SH['B'][l]['base']['r']) < 0.5 for l in COLD)}")

print("\n3. Sunlight cost of ice: within-cell slope of absorbed sunlight on annual cover (W m⁻² per unit cover)")
CELLS = TMP + "cells/"


def cell_slope(labs, row_lat, sectors, field):
    xs, ys = [], []
    for lab in labs:
        z = np.load(CELLS + f"{lab}.npz"); r = int(np.argmin(np.abs(z["row_lat"] - row_lat))); lon = z["lon"]
        sect = {1: (lon >= 295) | (lon < 20), 2: (lon >= 20) & (lon < 115), 3: (lon >= 115) & (lon < 295)}
        m = z["ocean"][r] & np.any([sect[s] for s in sectors], axis=0)
        c = z["cell_sea_ice_concentration"][:, r][:, m].astype(float); f = z[f"cell_{field}"][:, r][:, m].astype(float)
        keep = (np.nanmean(c, axis=0) > 0.02) & (np.nanmean(c, axis=0) < 0.98)   # cells with part-year ice
        c, f = c[:, keep], f[:, keep]
        xs.append((c - np.nanmean(c, axis=0)).ravel()); ys.append((f - np.nanmean(f, axis=0)).ravel())
    x, y = np.concatenate(xs), np.concatenate(ys); ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]), int(ok.sum())


warm_labs = ["1240", "1237p5", "1235", "1233p75", "1232p5"]
R3 = {}
for field in ("rst", "rss"):
    w, nw = cell_slope(warm_labs, -36.0, (2, 3), field)
    c25, n25 = cell_slope(["1230", "1228p5"], -24.92, (1, 3), field)
    c19, n19 = cell_slope(["1230", "1228p5"], -19.38, (3,), field)
    R3[field] = dict(warm36=w, cold25=c25, cold19=c19)
    print(f"  {field}: warm Indian/Pacific 36°S {w:7.1f} (n {nw})   cold 24.9°S Atl+Pac {c25:7.1f} (n {n25})   cold 19.4°S Pac {c19:7.1f} (n {n19})"
          f"  → costlier by >20%: 24.9 {abs(c25) > 1.2 * abs(w)}, 19.4 {abs(c19) > 1.2 * abs(w)}")

print("\n4. Sensitivity between settled states (Southern ice added per W m⁻² of μ; global cooling per W m⁻²)")
def settled(lab):
    d = load(lab); y = d["years"]; a, b = window(lab); k = (y >= a) & (y <= b)
    w = d["gw"] / d["gw"].sum(); ice = sector_ice_area(d, (1, 2, 3))[k]; ts = (d["ts"] @ w)[k]
    blk = [np.nanmean(ice[i:i + 100]) for i in range(0, len(ice) - 99, 100)]
    return np.nanmean(ice), np.nanstd(blk, ddof=1) / np.sqrt(len(blk)), np.nanmean(ts)
s30, s28 = settled("1230"), settled("1228p5")
dice = (s28[0] - s30[0]) / 1.5; dice_se = np.sqrt(s28[1] ** 2 + s30[1] ** 2) / 1.5
print(f"  cold 1230 → 1228.5: ice {dice:.2f} ± {dice_se:.2f} ×10¹² m² per W m⁻²; cooling {(s30[2] - s28[2]) / 1.5:.2f} K per W m⁻²")
print("  warm-branch values (earlier): ice 0.6–1.1 per W m⁻² down to 1235; 2.3 and 1.9 in the last two steps; cooling 0.21–0.28, then 0.46 and 0.40")
print(f"  'cold branch near its end' (≥ 1.9): {dice >= 1.9}")

print("\n5. The 1225 creep towards the snowball, 300-yr segments (detrended within each segment / raw)")
d = load("1225"); y = d["years"]
ip, at, tot = sector_ice_area(d, (2, 3)), sector_ice_area(d, (1,)), sector_ice_area(d, (1, 2, 3))
H = lsg_layer_mean(d, 1, -90, 0); E = equivalent_edge(d)
R5 = []
for a in range(5700, 7500, 300):
    k = (y >= a) & (y < a + 300)
    row = dict(start=a, E=float(np.nanmean(E[k])), H=float(np.nanmean(H[k])), tot=float(np.nanmean(tot[k])))
    for q, x in (("ip", ip[k]), ("atl", at[k])):
        row[q + "_acf_det"] = acf(detr(x), (1, 5, 10)).tolist(); row[q + "_acf_raw"] = acf(x, (1, 5, 10)).tolist()
        row[q + "_sd_det"] = float(np.nanstd(detr(x)))
    R5.append(row)
    print(f"  {a}–{a + 299}: edge {row['E']:.1f}°S  H {row['H']:.3f}  ice {row['tot']:.0f}  "
          f"IP ACF(1,5,10) det {np.round(row['ip_acf_det'], 2)} raw {np.round(row['ip_acf_raw'], 2)} sd {row['ip_sd_det']:.2f}   "
          f"Atl ACF det {np.round(row['atl_acf_det'], 2)} sd {row['atl_sd_det']:.2f}")
json.dump(dict(R1=R1, R3=R3, R5=R5, cold_sens=dict(ice=dice, ice_se=dice_se, cool=(s30[2] - s28[2]) / 1.5)),
          open(TMP + "cold_compare.json", "w"), default=float)
