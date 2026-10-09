"""Analysis D: persistence of the Indo-Pacific band cover (rows 30.5 + 36°S) towards the transition.

Annual anomalies about the window (or half-window) mean, no detrending. Autocorrelation at lags
1, 5, 10, 20, 50 yr (sample ACF, biased normalization); variance. 1235_new_IC: last 1500 yr.
Pre-registered: slowing is claimed only if the 10–50-yr autocorrelation rises monotonically from 1245
to 1232.5, by more than the 1235-pair difference, in both halves.
"""

import numpy as np

from common import band_cover, load
from gsebm.plasim_global import RUNS

ORDER = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
LAGS = (1, 5, 10, 20, 50)


def acf(x, lags):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x = np.where(ok, x - np.nanmean(x), 0.0)
    v = np.sum(x * x) / ok.sum()
    out = []
    for L in lags:
        pair_ok = ok[:-L] & ok[L:]
        out.append(np.sum(x[:-L] * x[L:] * pair_ok) / pair_ok.sum() / v)
    return np.array(out)


series = {}
for lab in ORDER + ["1235_new_IC"]:
    d = load(lab); w0, w1 = RUNS[lab].window
    k = (d["years"] >= w0) & (d["years"] <= w1)
    c = band_cover(d)[k]
    if lab == "1235_new_IC":
        c = c[-1500:]
    series[lab] = c

res = {}
for part in ("full", "h1", "h2"):
    res[part] = {}
    for lab, c in series.items():
        n = len(c)
        x = c if part == "full" else (c[: n // 2] if part == "h1" else c[n // 2:])
        res[part][lab] = (acf(x, LAGS), np.nanvar(x, ddof=1))

for part in ("full", "h1", "h2"):
    print(f"\n[{part}] IP band cover: ACF at lags {LAGS}, sd")
    for lab in ORDER + ["1235_new_IC"]:
        a, v = res[part][lab]
        print(f"  {lab:12s} " + " ".join(f"{x:6.2f}" for x in a) + f"   sd {np.sqrt(v):.4f}")

print("\nPre-registered test per lag (10, 20, 50): monotonic rise over", ORDER, "and rise > pair difference")
for part in ("full", "h1", "h2"):
    for li, L in enumerate(LAGS):
        if L < 10:
            continue
        vals = np.array([res[part][lab][0][li] for lab in ORDER])
        mono = bool(np.all(np.diff(vals) > 0))
        rise = vals[-1] - vals[0]
        pair = abs(res[part]["1235"][0][li] - res[part]["1235_new_IC"][0][li])
        print(f"  [{part}] lag {L:2d}: values {np.round(vals, 2)}  monotonic {mono}  rise {rise:+.2f}  pair diff {pair:.2f}")
