"""Do slow natural fluctuations become more correlated towards the transition? (criteria: plan_coherence.md)"""

import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from common import load, lsg_layer_mean, sector_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
RUNS = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5", "1230", "1228p5"]
ORDER = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
REF = ["1245", "1242p5", "1240", "1237p5"]; END2 = ["1233p75", "1232p5"]
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}


def window(lab):
    return WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))


def blocks(x, L):
    n = len(x) // L
    return np.nanmean(x[:n * L].reshape(n, L), axis=1)


def detr(B):
    t = np.arange(B.shape[0], dtype=float); t -= t.mean()
    return B - np.outer(t, np.nansum((B - np.nanmean(B, 0)) * t[:, None], 0) / np.sum(t * t))


def corr_stats(B):
    """B: (blocks, variables). Mean off-diagonal correlation and λ1/N of the correlation matrix."""
    ok = np.all(np.isfinite(B), axis=0) & (np.nanstd(B, axis=0) > 0)
    B = B[:, ok]
    C = np.corrcoef(B.T); n = C.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    lam = np.linalg.eigvalsh(C)
    return float(np.mean(off)), float(lam[-1] / n)


# ---- annual indices (C1) ----
IDX = {}
for lab in RUNS:
    d = load(lab); y = d["years"]; a, b = window(lab); k = (y >= a) & (y <= b)
    reg = dict(np.load(TMP + f"regional/{lab}.npz")); kr = (reg["years"] >= a) & (reg["years"] <= b)
    edge = SH["EDGE"][lab]; rows = (d["lsg_lat"] >= -edge) & (d["lsg_lat"] <= -edge + 10); wv = d["layer_volume"][rows, 0]
    W = np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()
    w = d["gw"] / d["gw"].sum(); s = d["lat"] < 0
    IDX[lab] = np.column_stack([
        -sector_ice_area(d, (1,))[k], -sector_ice_area(d, (2,))[k], -sector_ice_area(d, (3,))[k], W[k],
        reg["EP_0_100"][kr], reg["EP_300_600"][kr], reg["EA_0_100"][kr], reg["EA_300_600"][kr],
        lsg_layer_mean(d, 1, -90, 0)[k], d["ts"][k][:, s] @ (w[s] / w[s].sum())])


# ---- spatial fields (C2) from block-mean maps ----
def spatial_fields(lab, L):
    z = dict(np.load(TMP + (f"slowmaps50/{lab}.npz" if L == 50 else f"slowmaps/{lab}.npz")))
    lat, lsm = z["t21_lat"], z["lsm"]
    ts = z["ts"][:, (lat <= -20) & (lat >= -50), :].reshape(z["ts"].shape[0], -1)
    rows_ice = (lat <= -24) & (lat >= -42)
    sic = (z["sic"][:, rows_ice, :] * (lsm[rows_ice] < 0.5)).reshape(z["sic"].shape[0], -1)
    sic = sic[:, np.nanstd(sic, axis=0) > 1e-4]
    llat = z["lsg_lat2d"]; m = (llat <= -10) & (llat >= -40)
    o = z["th0_100"][:, m]
    good = (np.isfinite(ts).mean(axis=1) > 0.5) & (np.isfinite(o).mean(axis=1) > 0.5)
    ts, sic, o = ts[good], sic[good], o[good]
    o = o[:, np.all(np.isfinite(o), axis=0)]
    return {"Ts 20–50°S": ts, "ice cover": sic, "ocean 0–100 m": o}


def measure(lab, L, dt, seg=None):
    out = {}
    X = IDX[lab]
    if seg is not None:
        X = X[seg[0]:seg[1]]
    B = np.column_stack([blocks(X[:, j], L) for j in range(X.shape[1])])
    if dt:
        B = detr(B)
    out["C1 mean corr"], out["C1 λ1/N"] = corr_stats(B)
    F = spatial_fields(lab, L)
    for nm, M in F.items():
        if seg is not None:
            M = M[seg[0] // L:seg[1] // L]
        if dt:
            M = detr(M)
        out[f"C2 {nm}"] = corr_stats(M)[0]
    return out


if __name__ != "__main__":
    pass
RES = {}
for L, dt in itertools.product((50, 100), (False, True)):
    key = f"L{L}_d{int(dt)}"; RES[key] = {}
    for lab in RUNS:
        full = measure(lab, L, dt)
        n = IDX[lab].shape[0] // 1000
        segs = [measure(lab, L, dt, (i * 1000, (i + 1) * 1000)) for i in range(n)] if n >= 3 else []
        RES[key][lab] = {m: (full[m], (np.std([s[m] for s in segs], ddof=1) / np.sqrt(len(segs))) if segs else np.nan) for m in full}

prim = RES["L50_d0"]
measures = list(prim["1245"].keys())
print("Primary (50-yr blocks, no detrending): value ± SE (1000-yr segments)")
print(f"{'run':12s}" + "".join(f"{m:>20s}" for m in measures))
for lab in RUNS:
    print(f"{lab:12s}" + "".join(f"{prim[lab][m][0]:13.2f} ±{prim[lab][m][1]:4.2f}" for m in measures))
print("\nCriterion 'more correlated towards the end' (late > ref by 2 SE and Spearman ≥ 0.8), per variant:")
for m in measures:
    oks = []
    for vk, R in RES.items():
        vals = [R[l][m][0] for l in ORDER]; rho = spearmanr(np.arange(len(ORDER)), vals).correlation
        ref = np.mean([R[l][m][0] for l in REF]); rse = np.sqrt(np.nansum([R[l][m][1] ** 2 for l in REF])) / len(REF)
        late = [R[l][m] for l in END2]
        ok = rho >= 0.8 and all(v - ref > 2 * np.sqrt(rse ** 2 + np.nan_to_num(s) ** 2) for v, s in late)
        oks.append(f"{vk}: ρ {rho:+.2f} {'PASS' if ok else 'fail'}")
    print(f"  {m:18s} " + " | ".join(oks))
json.dump(RES, open(TMP + "coherence.json", "w"), default=float)
