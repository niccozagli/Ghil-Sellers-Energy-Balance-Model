"""What are the slow natural fluctuations? (criteria: Claude/plasim_mechanism/plan_slow.md). Basic methods."""

import itertools
import json

import numpy as np

from common import global_ts, load, sector_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
RUNS = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]
WARMC = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
SEQ = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}
FIELDS = ["sic", "ts", "rst", "th0_100", "th300_600", "t1025_2000"]
T21F = {"sic", "ts", "rst"}


def window(lab):
    return WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))


ANN = {}
for lab in RUNS:
    d = load(lab); y = d["years"]
    ANN[lab] = dict(y=y, Ts=global_ts(d), negIP=-sector_ice_area(d, (2, 3)), d=d)


def maps(lab, L):
    if L == 200:
        z = dict(np.load(TMP + f"slowmaps/{lab}.npz")); n = len(z["block_start"]) // 2
        for k in FIELDS:
            z[k] = 0.5 * (z[k][0:2 * n:2] + z[k][1:2 * n:2])
        z["block_start"] = z["block_start"][0:2 * n:2]; z["L"] = 200
        return z
    return dict(np.load(TMP + (f"slowmaps/{lab}.npz" if L == 100 else f"slowmaps50/{lab}.npz")))


def index_blocks(lab, starts, L, key):
    a = ANN[lab]
    return np.array([np.nanmean(a[key][(a["y"] >= s) & (a["y"] < s + L)]) for s in starts])


def weights_and_mask(z, key):
    if key in T21F:
        lat = z["t21_lat"]; w = np.cos(np.radians(lat))[:, None] * np.ones((1, 64))
        m = (lat[:, None] <= -10) & (lat[:, None] >= -60) & np.ones((1, 64), bool)
        if key == "sic":
            m = m & (z["lsm"] < 0.5)
    else:
        lat = z["lsg_lat2d"]; w = np.cos(np.radians(np.clip(lat, -89, 89)))
        m = (lat <= -10) & (lat >= -60)
    return w, m


def detrend_blocks(x):
    t = np.arange(x.shape[0], dtype=float); t -= t.mean()
    xm = np.nanmean(x, axis=0)
    slope = np.nansum((x - xm) * t.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0) / np.sum(t * t)
    return x - slope * t.reshape((-1,) + (1,) * (x.ndim - 1))


def regress_map(F, I, detrend):
    F = F.astype(float).copy(); I = I.astype(float).copy()
    if detrend:
        F = detrend_blocks(F); I = detrend_blocks(I[:, None])[:, 0]
    Ic = I - I.mean(); Fc = F - np.nanmean(F, axis=0)
    return np.nansum(Fc * Ic.reshape((-1,) + (1,) * (F.ndim - 1)), axis=0) / np.sum(Ic * Ic)


def wcorr(a, b, w, m):
    ok = m & np.isfinite(a) & np.isfinite(b)
    a, b, w = a[ok], b[ok], w[ok]
    am, bm = np.sum(w * a) / w.sum(), np.sum(w * b) / w.sum()
    return float(np.sum(w * (a - am) * (b - bm)) / np.sqrt(np.sum(w * (a - am) ** 2) * np.sum(w * (b - bm) ** 2)))


def wslope(n, f, w, m):
    ok = m & np.isfinite(n) & np.isfinite(f)
    return float(np.sum(w[ok] * n[ok] * f[ok]) / np.sum(w[ok] * f[ok] ** 2))


SETTLED = {lab: {k: np.nanmean(maps(lab, 100)[k], axis=0) for k in FIELDS} for lab in RUNS}
TSM = {lab: float(np.nanmean(ANN[lab]["Ts"][(ANN[lab]["y"] >= window(lab)[0]) & (ANN[lab]["y"] <= window(lab)[1])])) for lab in RUNS}


def forced_map(lab, key):
    base = "1235" if lab == "1235_new_IC" else lab
    if base in SEQ:
        i = SEQ.index(base); a = SEQ[max(i - 1, 0)]; b = SEQ[min(i + 1, len(SEQ) - 1)]
    else:
        a, b = "1230", "1228p5"
    return (SETTLED[a][key] - SETTLED[b][key]) / (TSM[a] - TSM[b])


# ---------------- S1 ----------------
print("S1. Pattern correlation of the slow natural fluctuation (regression on the slow index) with the forced change")
S1 = {}
variants = list(itertools.product((50, 100, 200), ("Ts", "negIP"), (False, True)))
for lab in RUNS:
    S1[lab] = {}
    for L, idx, dt in variants:
        z = maps(lab, L); I = index_blocks(lab, z["block_start"], L, idx)
        res = {}
        for key in FIELDS:
            nat = regress_map(z[key], I, dt); frc = forced_map(lab, key); w, m = weights_and_mask(z, key)
            res[key] = dict(r=wcorr(nat, frc, w, m), amp=wslope(nat, frc, w, m) if idx == "Ts" else np.nan)
        S1[lab][f"L{L}_{idx}_d{int(dt)}"] = res
    b = S1[lab]["L100_Ts_d0"]
    print(f"  {lab:12s} " + "  ".join(f"{k} r {b[k]['r']:+.2f} amp {b[k]['amp']:.2f}" for k in FIELDS))


def verdict_s1():
    out = {}
    for vk in S1["1245"]:
        upper = sum(all(S1[l][vk][k]["r"] >= 0.7 for k in ("sic", "ts", "th0_100", "th300_600")) for l in WARMC)
        deep = sum(S1[l][vk]["t1025_2000"]["r"] < 0.5 for l in WARMC)
        out[vk] = (upper, deep)
    return out


vs = verdict_s1()
n = len(WARMC)
print(f"  'miniature' (all four upper fields r ≥ 0.7) in ≥ {2*n/3:.1f} of {n} runs, per variant: "
      + ", ".join(f"{k}:{v[0]}" for k, v in vs.items()))
print(f"  'different at depth' (r < 0.5 at 1025–2000 m) per variant: " + ", ".join(f"{k}:{v[1]}" for k, v in vs.items()))
print(f"  → miniature holds in all variants: {all(v[0] >= 2*n/3 for v in vs.values())};  different at depth in all variants: "
      f"{all(v[1] >= 2*n/3 for v in vs.values())}")
for key in FIELDS:
    rs = [S1[l]["L100_Ts_d0"][key]["r"] for l in WARMC]
    rmin = [min(S1[l][vk][key]["r"] for vk in S1[l]) for l in WARMC]; rmax = [max(S1[l][vk][key]["r"] for vk in S1[l]) for l in WARMC]
    print(f"    {key:11s} r (L100, Ts): median {np.median(rs):+.2f}; range over runs and variants {min(rmin):+.2f} … {max(rmax):+.2f}")

# ---------------- S2 ----------------
print("\nS2. Lead–lag with global Ts (lag of max correlation; positive = the index leads), running means")
REG = {lab: dict(np.load(TMP + f"regional/{lab}.npz")) for lab in RUNS}


def W_series(lab):
    d = ANN[lab]["d"]; edge = SH["EDGE"][lab]
    rows = (d["lsg_lat"] >= -edge) & (d["lsg_lat"] <= -edge + 10); wv = d["layer_volume"][rows, 0]
    return np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()


def runmean(x, n):
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    v = np.convolve(x, np.ones(n) / n, mode="valid")
    out = np.full(len(x), np.nan); out[n // 2:n // 2 + len(v)] = v
    return out


def lagcorr(x, y, lags):
    out = []
    for k in lags:
        if k >= 0:
            a, b = x[:len(x) - k], y[k:]
        else:
            a, b = x[-k:], y[:len(y) + k]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(out)


LAGS = np.arange(-200, 201, 5)
S2 = {}
for lab in RUNS:
    a0, a1 = window(lab); a = ANN[lab]; k = (a["y"] >= a0) & (a["y"] <= a1)
    r = REG[lab]; kr = (r["years"] >= a0) & (r["years"] <= a1)
    assert np.array_equal(r["years"][kr], a["y"][k])
    idx = {"EP 0–100": r["EP_0_100"][kr], "EP 300–600": r["EP_300_600"][kr], "EA 0–100": r["EA_0_100"][kr],
           "EA 300–600": r["EA_300_600"][kr], "edge water 0–700": W_series(lab)[k], "less Indian/Pacific ice": a["negIP"][k]}
    S2[lab] = {}
    for n_rm, dt in itertools.product((30, 50, 100), (False, True)):
        ts = a["Ts"][k].copy()
        if dt:
            t = np.arange(len(ts)); ts = ts - np.polyval(np.polyfit(t, ts, 1), t)
        tsr = runmean(ts, n_rm)
        res = {}
        for name, x in idx.items():
            x = x.copy()
            if dt:
                ok = np.isfinite(x); t = np.arange(len(x)); x = x - np.polyval(np.polyfit(t[ok], x[ok], 1), t)
            c = lagcorr(runmean(x, n_rm), tsr, LAGS)
            j = int(np.nanargmax(np.abs(c)))
            res[name] = (int(LAGS[j]), float(c[j]), float(c[LAGS == 0][0]))
        S2[lab][f"rm{n_rm}_d{int(dt)}"] = res
    print(f"  {lab:12s} " + "  ".join(f"{nm}: lag {v[0]:+4d} (r {v[1]:+.2f})" for nm, v in S2[lab]["rm50_d0"].items()))
for name in S2["1245"]["rm50_d0"]:
    cats = []
    for vk in S2["1245"]:
        lags = [S2[l][vk][name][0] for l in RUNS if l not in ("1250",)]
        nr = len(lags)
        lead = sum(l >= 20 for l in lags); tog = sum(abs(l) <= 10 for l in lags); fol = sum(l <= -20 for l in lags)
        cats.append("leads" if lead >= 2 * nr / 3 else "together" if tog >= 2 * nr / 3 else "follows" if fol >= 2 * nr / 3 else "mixed")
    print(f"  {name:24s}: verdicts over variants {cats}")

# ---------------- S3 ----------------
print("\nS3. Size and time scale of the slow fluctuation (block means of global Ts / Indian/Pacific ice area)")
S3 = {}
for lab in RUNS:
    a0, a1 = window(lab); a = ANN[lab]; S3[lab] = {}
    for L, dt in itertools.product((50, 100, 200), (False, True)):
        starts = np.arange(a0, a1 - L + 2, L)
        for key in ("Ts", "negIP"):
            b = index_blocks(lab, starts, L, key)
            if dt:
                t = np.arange(len(b)); b = b - np.polyval(np.polyfit(t, b, 1), t)
            bc = b - b.mean(); v = np.sum(bc * bc)
            ac1 = np.sum(bc[:-1] * bc[1:]) / v; ac2 = np.sum(bc[:-2] * bc[2:]) / v
            S3[lab][f"{key}_L{L}_d{int(dt)}"] = (float(np.std(b, ddof=1)), float(ac1), float(ac2), len(b))
    s = S3[lab]
    print(f"  {lab:12s} Ts: sd100 {s['Ts_L100_d0'][0]:.3f} K, ac(100 yr) {s['Ts_L100_d0'][1]:+.2f}, ac(200 yr) {s['Ts_L100_d0'][2]:+.2f}   "
          f"IP ice: sd100 {s['negIP_L100_d0'][0]:.2f}, ac(100) {s['negIP_L100_d0'][1]:+.2f}")
REF3 = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC"]
for stat_i, nm in ((0, "sd"), (1, "autocorrelation at one block")):
    ok_all = True
    for vk in S3["1245"]:
        mx = max(S3[l][vk][stat_i] for l in REF3)
        ok_all &= all(S3[l][vk][stat_i] > mx for l in ("1233p75", "1232p5"))
    print(f"  'grows towards the end' ({nm}) in all variants: {ok_all}")

# ---------------- S4 ----------------
print("\nS4. Events: 100-yr blocks with |z| ≥ 2.5 in global Ts")
S4 = []
for lab in RUNS:
    z = maps(lab, 100); I = index_blocks(lab, z["block_start"], 100, "Ts"); zz = (I - I.mean()) / I.std()
    for j in np.flatnonzero(np.abs(zz) >= 2.5):
        dTs = I[j] - I.mean(); ev = {"run": lab, "start": int(z["block_start"][j]), "z": float(zz[j]), "dTs": float(dTs)}
        reg = {k: regress_map(z[k], I, False) for k in FIELDS}
        for key in FIELDS:
            comp = (z[key][j] - np.nanmean(z[key], axis=0)) / dTs; w, m = weights_and_mask(z, key)
            ev[key] = dict(r_forced=wcorr(comp, forced_map(lab, key), w, m), r_reg=wcorr(comp, reg[key], w, m))
        S4.append(ev)
        print(f"  {lab:12s} block {ev['start']}–{ev['start'] + 99}: z {ev['z']:+.1f} (ΔTs {dTs:+.2f} K)  "
              + "  ".join(f"{k}: r_forced {ev[k]['r_forced']:+.2f} r_reg {ev[k]['r_reg']:+.2f}" for k in FIELDS))
json.dump(dict(S1=S1, S2=S2, S3=S3, S4=S4), open(TMP + "slow_analysis.json", "w"), default=float)
