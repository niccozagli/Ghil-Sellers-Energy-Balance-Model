"""Added sensitivity check for 1B S1: halfway times from 50-yr block means, first block with progress ≥ 0.5
(no persistence requirement), for the slow part of E and of each ocean layer."""

import json

import numpy as np

from common import PARENT

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
S = np.load(TMP + "check1_series.npy", allow_pickle=True).item()
B1 = json.load(open(TMP + "check1.json"))["B1"]
out = {}
for lab in ["1288", "1265", "1245", "1235", "1233p75", "1232p5"]:
    s = S[lab]; st = s["starts"]; y0 = PARENT[lab][1]
    row = {}
    for q in ("E", "W", "Hu", "H", "Hd"):
        o = B1[lab][q]
        x = s["ser"][q]; p = (x - o["fast"]) / (o["final"] - o["fast"])
        # 50-yr blocks starting at y0+1
        b0 = np.arange(y0 + 1, st.max() - 40, 50)
        pb = np.array([np.nanmean(p[(st >= a) & (st < a + 50)]) for a in b0])
        k = np.flatnonzero(pb >= 0.5)
        row[q] = float(b0[k[0]] + 25 - y0) if k.size else np.nan
    out[lab] = row
    te = row["E"]
    tags = []
    for q in ("W", "Hu", "H", "Hd"):
        tq = row[q]; tl = max(50.0, 0.2 * max(te, tq))
        tags.append(f"{q}:{'pace' if abs(te - tq) <= tl else ('ahead' if te < tq else 'lags')}")
    print(f"{lab:8s} t50_slow (50-yr blocks): " + "  ".join(f"{q} {row[q]:5.0f}" for q in row) + "   " + " ".join(tags))
json.dump(out, open(TMP + "check1_timing_sens.json", "w"))
