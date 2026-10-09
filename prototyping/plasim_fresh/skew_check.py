"""Exploratory (not pre-registered): skewness of slow fluctuations across μ, with robustness variants."""

import itertools
import json

import numpy as np

from common import global_ts, load, sector_ice_area

STAT = json.load(open("/Users/niccolo/.claude/jobs/3fe717ba/tmp/stationary.json"))
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}
RUNS = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]
out = {}
for lab in RUNS:
    d = load(lab); y = d["years"]
    a, b = WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))
    k = (y >= a) & (y <= b)
    series = {"Ts": global_ts(d)[k], "lessIPice": -sector_ice_area(d, (2, 3))[k]}
    row = {}
    for (name, x), L, dt, half in itertools.product(series.items(), (1, 50, 100, 200), (False, True), (None, 0, 1)):
        x = x.copy(); n = len(x)
        if half is not None:
            x = x[:n // 2] if half == 0 else x[n // 2:]
        if dt:
            t = np.arange(len(x)); ok = np.isfinite(x); x = x - np.polyval(np.polyfit(t[ok], x[ok], 1), t)
        m = len(x) // L
        bm = np.nanmean(x[:m * L].reshape(m, L), axis=1) if L > 1 else x
        z = (bm - np.nanmean(bm)) / np.nanstd(bm)
        row[f"{name}_L{L}_d{int(dt)}_h{half}"] = float(np.nanmean(z ** 3))
    out[lab] = row
    print(f"{lab:12s} Ts skew: annual {row['Ts_L1_d0_hNone']:+.2f}  50y {row['Ts_L50_d0_hNone']:+.2f}  100y {row['Ts_L100_d0_hNone']:+.2f}  "
          f"200y {row['Ts_L200_d0_hNone']:+.2f}  100y detr {row['Ts_L100_d1_hNone']:+.2f}  halves {row['Ts_L100_d0_h0']:+.2f}/{row['Ts_L100_d0_h1']:+.2f}"
          f"   | less IP ice 100y {row['lessIPice_L100_d0_hNone']:+.2f} (halves {row['lessIPice_L100_d0_h0']:+.2f}/{row['lessIPice_L100_d0_h1']:+.2f})")
keys = [k for k in out["1245"] if "_L1_" not in k]
neg_1232 = sum(out["1232p5"][k] < 0 for k in keys); neg_1233 = sum(out["1233p75"][k] < 0 for k in keys)
pos_ref = sum(all(out[l][k] > 0 for l in ["1245", "1242p5", "1240", "1237p5", "1235"]) for k in keys)
print(f"\nvariants (block 50/100/200, detrend, halves; Ts and less-IP-ice): {len(keys)}")
print(f"  1232.5 negative in {neg_1232}; 1233.75 negative in {neg_1233}; all of 1245–1235 positive in {pos_ref}")
json.dump(out, open("/Users/niccolo/.claude/jobs/3fe717ba/tmp/skew.json", "w"))
