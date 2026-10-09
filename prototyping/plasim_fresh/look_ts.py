"""Descriptive look at surface temperature fluctuations (no criteria): global, Southern Hemisphere, and the
band of T21 rows around each run's ice edge. Settled periods; 1225 cold stage in 300-yr segments (trend removed)."""

import json

import numpy as np

from common import equivalent_edge, load

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json"))
COLD = {"1230": (11600, 15499), "1228p5": (15000, 16899)}
LABS = ["1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]


def acf(x, lags):
    x = np.nan_to_num(x - np.nanmean(x)); v = np.sum(x * x) / len(x)
    return [np.sum(x[:-k] * x[k:]) / len(x) / v for k in lags]


def detr(x):
    t = np.arange(len(x)); c = np.polyfit(t, x, 1); return x - np.polyval(c, t)


def series(d, edge):
    w = d["gw"] / d["gw"].sum(); lat = d["lat"]; s = lat < 0
    band = (lat < 0) & (np.abs(np.abs(lat) - edge) <= 6)
    g = d["ts"] @ w
    sh = d["ts"][:, s] @ (w[s] / w[s].sum())
    eb = d["ts"][:, band] @ (w[band] / w[band].sum())
    return g, sh, eb


print(f"{'run':12s}{'edge':>6s} | {'global Ts: sd  ACF1 ACF5 ACF10':>32s} | {'SH Ts: sd  ACF1 ACF5 ACF10':>28s} | {'edge-band Ts: sd ACF1 ACF5 ACF10':>34s}")
for lab in LABS:
    d = load(lab); y = d["years"]
    a, b = COLD.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))
    k = (y >= a) & (y <= b); edge = float(np.nanmean(equivalent_edge(d)[k]))
    out = []
    for x in series(d, edge):
        x = x[k]; out.append((np.nanstd(x), *acf(x, (1, 5, 10))))
    print(f"{lab:12s}{edge:6.1f} | " + " | ".join(f"{o[0]:6.3f} {o[1]:5.2f} {o[2]:5.2f} {o[3]:5.2f}" for o in out))
print("\n1225 cold stage (trend removed within each 300-yr segment)")
d = load("1225"); y = d["years"]
for a0 in range(5700, 7500, 300):
    k = (y >= a0) & (y < a0 + 300); edge = float(np.nanmean(equivalent_edge(d)[k]))
    out = []
    for x in series(d, edge):
        x = detr(x[k]); out.append((np.std(x), *acf(x, (1, 5, 10))))
    print(f"  {a0}–{a0 + 299} edge {edge:5.1f} | " + " | ".join(f"{o[0]:6.3f} {o[1]:5.2f} {o[2]:5.2f} {o[3]:5.2f}" for o in out))
