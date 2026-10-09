"""Objective start of the stationary part of each run (rule fixed before any analysis).

The slow variable H (SH 700–2025 m θ) must have settled after the μ step. H_100 = 100-yr block means aligned to
the step; H_before = the parent's last 100 yr (the run's own first 10 yr for runs from 1367; the 100 yr after the
jump for 1230); H_final = mean of the last 2000 yr of valid record.
band(q) = max((1 − q)·|H_final − H_before|, 3 sd of the 100-yr means in the last 2000 yr).
start(q) = the first 100-yr block from which the next 10 blocks (1000 yr) all lie within band(q) of H_final.
end = last valid year. Primary q = 0.95; robustness q = 0.90, 0.99.
1228.5 (record begins 500 yr after a −1.5 step from the cold state): whole record.
Later excursions (events, wandering) are kept as natural variability.
"""

import json

import numpy as np

from common import PARENT, load, lsg_layer_mean

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
LABS = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC",
        "1233p75", "1232p5", "1230", "1228p5"]
H = {}
for lab in set(LABS) | {"1265", "1240"}:
    d = load(lab); H[lab] = (d["years"], lsg_layer_mean(d, 1, -90, 0))
out = {}
for lab in LABS:
    y, h = H[lab]; end = int(y[np.isfinite(h)][-1])
    if lab == "1228p5":
        out[lab] = {str(q): int(y[0]) for q in (0.9, 0.95, 0.99)}; out[lab]["end"] = end
        print(f"{lab:12s} whole record {int(y[0])}–{end}"); continue
    parent, y0 = PARENT[lab]
    if lab == "1230":
        y0 = 8199; hb = np.nanmean(h[(y >= 8200) & (y < 8300)])
    elif parent == "1367":
        hb = np.nanmean(h[(y > y0) & (y <= y0 + 10)])
    else:
        yp, hp = H[parent]; hb = np.nanmean(hp[(yp > y0 - 100) & (yp <= y0)])
    ref_len = 2000
    hf = np.nanmean(h[(y > end - ref_len) & (y <= end)])
    starts = np.arange(y0 + 1, end - 98, 100)
    hb100 = np.array([np.nanmean(h[(y >= a) & (y < a + 100)]) for a in starts])
    ref = [np.nanmean(h[(y >= a) & (y < a + 100)]) for a in range(end - ref_len + 1, end - 98, 100)]
    sd100 = np.nanstd(ref, ddof=1)
    res = {}
    for q in (0.9, 0.95, 0.99):
        band = max((1 - q) * abs(hf - hb), 3 * sd100)
        ok = np.abs(hb100 - hf) <= band
        k = [i for i in range(len(starts)) if np.all(ok[i:i + 10])]
        res[str(q)] = int(starts[k[0]]) if k else None
        res["band" + str(q)] = float(band)
    res["end"] = end; out[lab] = res
    print(f"{lab:12s} H before {hb:.3f}  final {hf:.3f}  start q=0.90: {res['0.9']}  q=0.95: {res['0.95']}  "
          f"q=0.99: {res['0.99']}  end {end}  → {end - res['0.95'] + 1} yr at q=0.95 (band ±{res['band0.95']:.3f} K)")
json.dump(out, open(TMP + "stationary.json", "w"))
