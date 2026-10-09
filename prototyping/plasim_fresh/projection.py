"""Projection of slow natural fluctuations onto the full forced response (criteria: plan_projection.md)."""

import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from common import global_ts, load

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json"))
WARM = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
ORDER = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
COLD = ["1230", "1228p5"]
SEQ = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}
SURF = ["sic", "ts", "rst", "th0_100"]; MID = ["th300_600"]; DEEP = ["t1025_2000"]
FIELDS = SURF + MID + DEEP
T21F = {"sic", "ts", "rst"}


def window(lab):
    return WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))


def maps(lab, L):
    return dict(np.load(TMP + (f"slowmaps/{lab}.npz" if L == 100 else f"slowmaps50/{lab}.npz")))


def wmask(z, key):
    if key in T21F:
        lat = z["t21_lat"]; w = np.cos(np.radians(lat))[:, None] * np.ones((1, 64))
        m = (lat[:, None] <= -10) & (lat[:, None] >= -60) & np.ones((1, 64), bool)
        if key == "sic":
            m = m & (z["lsm"] < 0.5)
    else:
        lat = z["lsg_lat2d"]; w = np.cos(np.radians(np.clip(lat, -89, 89))); m = (lat <= -10) & (lat >= -60)
    return w, m


TS = {}
for lab in WARM + COLD + ["1250"]:
    d = load(lab); y = d["years"]; a, b = window(lab)
    TS[lab] = float(np.nanmean(global_ts(d)[(y >= a) & (y <= b)]))
MEAN = {lab: {k: np.nanmean(maps(lab, 100)[k], axis=0) for k in FIELDS} for lab in WARM + COLD + ["1250"]}
Z0 = maps("1240", 100); WM = {k: wmask(Z0, k) for k in FIELDS}


def common_forced(k):
    return (MEAN["1232p5"][k] - MEAN["1245"][k]) / (TS["1232p5"] - TS["1245"])


def local_forced(lab, k):
    base = "1235" if lab == "1235_new_IC" else lab
    if base in SEQ:
        i = SEQ.index(base); a = SEQ[max(i - 1, 0)]; b = SEQ[min(i + 1, len(SEQ) - 1)]
    else:
        a, b = "1230", "1228p5"
    return (MEAN[a][k] - MEAN[b][k]) / (TS[a] - TS[b])


def vec(field, k):
    w, m = WM[k]; ok = m & np.isfinite(field)
    return field[ok] * np.sqrt(w[ok]), ok


def numbers(lab, block_sel, L, forced_kind, scaling, detrend):
    z = maps(lab, L)
    good = np.ones(len(z["block_start"]), bool)
    for k in FIELDS:
        good &= np.array([np.isfinite(z[k][i]).any() for i in range(len(good))])
    block_sel = block_sel & good
    F, A = {}, {}
    for k in FIELDS:
        f = common_forced(k) if forced_kind == "common" else local_forced(lab, k)
        x = z[k][block_sel].astype(float)
        if detrend:
            t = np.arange(x.shape[0], dtype=float); t -= t.mean()
            x = x - (np.nansum((x - np.nanmean(x, 0)) * t[:, None, None], 0) / np.sum(t * t)) * t[:, None, None]
        x = x - np.nanmean(x, axis=0)
        w, m = WM[k]; ok = m & np.isfinite(f) & np.all(np.isfinite(x), axis=0)
        F[k] = f[ok] * np.sqrt(w[ok]); A[k] = x[:, ok] * np.sqrt(w[ok])
    nb = next(iter(A.values())).shape[0]
    if scaling == "forced":
        s = {k: 1 / np.linalg.norm(F[k]) for k in FIELDS}
    else:
        s = {k: 1 / np.sqrt(np.mean(np.sum(A[k] ** 2, axis=1))) for k in FIELDS}
    Fc = np.concatenate([s[k] * F[k] for k in FIELDS]); Ac = np.concatenate([s[k] * A[k] for k in FIELDS], axis=1)
    proj = Ac @ Fc / np.linalg.norm(Fc)
    f_full = float(np.sum(proj ** 2) / np.sum(Ac ** 2))
    fk = {k: float(np.sum((A[k] @ F[k]) ** 2) / (np.sum(F[k] ** 2) * np.sum(A[k] ** 2))) for k in FIELDS}
    p = {k: A[k] @ F[k] / np.sum(F[k] ** 2) for k in FIELDS}
    ps = np.mean([p[k] for k in SURF], axis=0); pd = p["t1025_2000"]
    g = float(np.sum((ps - ps.mean()) * (pd - pd.mean())) / np.sum((ps - ps.mean()) ** 2))
    r = float(np.corrcoef(ps, pd)[0, 1])
    sv = np.linalg.svd(Ac - Ac.mean(0), compute_uv=False); eof1 = float(sv[0] ** 2 / np.sum(sv ** 2))
    return dict(f_full=f_full, fk=fk, g=g, r=r, eof1=eof1, nb=nb)


def run_all(lab, L, fk_, sc, dt):
    z = maps(lab, L); nb = len(z["block_start"])
    full = numbers(lab, np.ones(nb, bool), L, fk_, sc, dt)
    per = 1000 // L; nseg = nb // per; segs = []
    for i in range(nseg):
        sel = np.zeros(nb, bool); sel[i * per:(i + 1) * per] = True
        segs.append(numbers(lab, sel, L, fk_, sc, dt))
    out = dict(full)
    for key in ("f_full", "g", "r"):
        vals = [s[key] for s in segs]
        out[key + "_seg_mean"] = float(np.mean(vals)) if vals else np.nan
        out[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) >= 3 else np.nan
    out["nseg"] = nseg
    return out


VARIANTS = list(itertools.product((100, 50), ("common", "local"), ("forced", "variance"), (False, True)))
RES = {}
for L, fk_, sc, dt in VARIANTS:
    key = f"L{L}_{fk_}_{sc}_d{int(dt)}"
    RES[key] = {lab: run_all(lab, L, fk_, sc, dt) for lab in WARM + COLD}

prim = RES["L100_common_forced_d0"]
print("Primary variant (100-yr blocks, common forced direction 1245→1232.5, forced-norm scaling, no detrending)")
print(f"{'run':12s}{'blocks':>7s}{'f_full':>15s}{'g (deep gain)':>18s}{'r':>14s}{'EOF1':>7s}   per-field f_k")
for lab in WARM + COLD:
    v = prim[lab]
    print(f"{lab:12s}{v['nb']:7d}  {v['f_full']:.3f} ±{v['f_full_se']:.3f}  {v['g']:+.3f} ±{v['g_se']:.3f}  {v['r']:+.2f} ±{v['r_se']:.2f}"
          f"  {v['eof1']:.2f}   " + " ".join(f"{k}:{v['fk'][k]:.2f}" for k in FIELDS))

print("\nCriteria (Spearman ρ with decreasing μ over", ORDER, "; late − early > 2 SE):")
summary = {}
for qty in ("f_full", "g"):
    passes = {}
    for vk, res in RES.items():
        vals = [res[l][qty] for l in ORDER]
        rho = spearmanr(np.arange(len(ORDER)), vals).correlation
        late = np.mean([res[l][qty] for l in ("1233p75", "1232p5")]); early = np.mean([res[l][qty] for l in ("1245", "1242p5")])
        se = np.sqrt(np.nansum([res[l][qty + "_se"] ** 2 for l in ("1233p75", "1232p5", "1245", "1242p5")])) / 2
        ok = (rho >= 0.8) and (late - early > 2 * se)
        passes[vk] = (ok, rho, late, early, se)
    summary[qty] = passes
    prim_ok = passes["L100_common_forced_d0"][0]
    n_ok = sum(v[0] for v in passes.values())
    verdict = "RISES STEADILY" if n_ok == len(passes) else ("rises, not robust" if prim_ok else "does not rise (primary fails)")
    print(f"  {qty}: primary ρ {passes['L100_common_forced_d0'][1]:+.2f}, late {passes['L100_common_forced_d0'][2]:.3f} vs early "
          f"{passes['L100_common_forced_d0'][3]:.3f} (2 SE {2*passes['L100_common_forced_d0'][4]:.3f}) → pass {prim_ok}; "
          f"all variants: {n_ok}/{len(passes)} → {verdict}")
    for vk, v in passes.items():
        if not v[0]:
            print(f"      fails in {vk}: ρ {v[1]:+.2f}, late {v[2]:.3f}, early {v[3]:.3f}, 2SE {2*v[4]:.3f}")
json.dump(dict(RES=RES, summary={q: {k: [bool(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4])] for k, v in p.items()}
                                  for q, p in summary.items()}), open(TMP + "projection.json", "w"), default=float)
