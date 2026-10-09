"""Added after seeing the pre-registered result (labelled as such): recovery measured on running means
(5 and 11 yr, centred), and the rank of the jump swing's retreat among the creep swings."""

import json

import numpy as np

from common import load
from swings import RUNS, REF, END2, SEGS, STAT, SH, rows, peaks, measures, TMP

sw = json.load(open(TMP + "swings.json"))


def rm(x, n):
    k = np.ones(n) / n
    out = np.full(len(x), np.nan); v = np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, mode="valid")
    out[n // 2:n // 2 + len(v)] = v
    return out


print("Recovery τ½/P on running means (peaks and recovery both on the smoothed Atlantic 30.5°S row):")
tab = {}
for n in (5, 11):
    for rec in (0.5, 1 / 3):
        line = []
        for lab in RUNS:
            d = load(lab); y_ = d["years"]; k = (y_ >= STAT[lab]["0.95"]) & (y_ <= STAT[lab]["end"])
            x, y = rows(d); xs = rm(x[k], n); ok = np.isfinite(xs)
            P = int(round(SH["A"][lab]["full"]["per_acf"]))
            mres = measures(xs[ok], y[k][ok], P, 0.5, rec, 20)
            tab[(n, rec, lab)] = (mres["tau"], mres["tau_se"])
            line.append(f"{lab}:{mres['tau']:.3f}±{mres['tau_se']:.3f}")
        rv = [tab[(n, rec, l)] for l in REF]; rmn = np.mean([v[0] for v in rv]); rse = np.sqrt(np.sum([v[1] ** 2 for v in rv])) / len(rv)
        ok = all(tab[(n, rec, l)][0] - rmn > 2 * np.sqrt(rse ** 2 + tab[(n, rec, l)][1] ** 2) for l in END2)
        print(f"  running mean {n:2d} yr, recovery to {rec:.2f}: ref {rmn:.3f}; " + "  ".join(line[-3:]) + f"  → stickier at the end: {ok}")

d = load("1230"); yy = d["years"]; x30, y30 = rows(d); kc = (yy >= 5100) & (yy <= 7823); Pc = sw["Pc"]
cy = yy[kc]; pk = peaks(x30[kc], int(round(Pc / 2)), np.nanmean(x30[kc]))
cr = [(int(cy[i]), x30[kc][i]) for i in pk if cy[i] + Pc <= 7823 and cy[i] - 500 >= 4600]


def retreat(t, h):
    m = np.nanmean(x30[(yy >= t - 500) & (yy < t)]); kk = (yy > t) & (yy <= t + Pc)
    return (np.nanmin(x30[kk]) - m) / (h - m)


rt = np.array([retreat(t, h) for t, h in cr]); rj = retreat(sw["tj"], sw["hj"])
print(f"\nJump swing retreat {rj:.2f}; creep swings: n = {len(rt)}, shallower than the jump's: {np.sum(rt > rj)} "
      f"(rank {np.sum(rt > rj) + 1} of {len(rt) + 1}, 1 = shallowest)")
