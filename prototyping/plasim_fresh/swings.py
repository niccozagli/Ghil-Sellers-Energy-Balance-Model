"""Do Atlantic swings tip the next row? (criteria: Claude/plasim_mechanism/plan_swings.md). Basic methods."""

import itertools
import json

import numpy as np

from common import band_cover, load

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
RUNS = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
REF = ["1245", "1242p5", "1240", "1237p5"]; END2 = ["1233p75", "1232p5"]
SEGS = [(5100, 5599), (5600, 6099), (6100, 6599), (6600, 7099), (7100, 7599), (7600, 7823)]


def rows(d):
    r30 = int(np.argmin(np.abs(d["lat"] + 30.46))); oc = d["ocean_counts"]
    x = d["sic"][:, r30, 1]
    y = (d["sic"][:, r30, 2] * oc[r30, 2] + d["sic"][:, r30, 3] * oc[r30, 3]) / (oc[r30, 2] + oc[r30, 3])
    return x, y


def acf_period(a):
    a = np.nan_to_num(a - np.nanmean(a)); v = np.sum(a * a) / len(a)
    r = np.array([1.0] + [np.sum(a[:-k] * a[k:]) / len(a) / v for k in range(1, 201)])
    kmin = 10 + int(np.argmin(r[10:151])); return kmin + 1 + int(np.argmax(r[kmin + 1:min(3 * kmin, 200) + 1]))


def peaks(x, w, m):
    out = []
    for i in range(w, len(x) - w):
        if np.isfinite(x[i]) and x[i] > m and x[i] == np.nanmax(x[i - w:i + w + 1]):
            if not out or i - out[-1] > w:
                out.append(i)
    return out


def measures(x, y, P, wfrac=0.5, rec=0.5, W=20, m=None):
    m = np.nanmean(x) if m is None else m
    w = max(int(round(P * wfrac)), 5)
    pk = peaks(x, w, m)
    pk = [i for i in pk if i - W >= 0 and i + max(W, P) < len(x)]
    h = np.array([x[i] for i in pk])
    tau = []
    for i in pk:
        thr = m + (x[i] - m) * rec
        s = np.flatnonzero(x[i + 1:i + P + 1] <= thr)
        tau.append((s[0] + 1) if s.size else P)
    tau = np.array(tau) / P
    R = np.array([np.nanmean(y[i + 1:i + W + 1]) - np.nanmean(y[i - W:i]) for i in pk])
    Rn = R / (h - m)
    se = lambda v: np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v))) if len(v) > 2 else np.nan
    return dict(n=len(pk), mean=float(m), h50=float(np.median(h)), h90=float(np.percentile(h, 90)), hmax=float(h.max()),
                tau=float(np.mean(tau)), tau_se=float(se(tau)), R=float(np.mean(R)), R_se=float(se(R)),
                Rn=float(np.nanmean(Rn)), Rn_se=float(se(Rn)))


VARIANTS = list(itertools.product((0.5, 1 / 3), (0.5, 1 / 3), (10, 20, 30)))  # peak window frac, recovery level, W
res = {}
for lab in RUNS:
    d = load(lab); y_ = d["years"]; k = (y_ >= STAT[lab]["0.95"]) & (y_ <= STAT[lab]["end"])
    x, y = rows(d); x, y = x[k], y[k]
    P = int(round(SH["A"][lab]["full"]["per_acf"]))
    res[lab] = {"P": P, "base": measures(x, y, P),
                "var": {f"w{wf:.2f}_r{rc:.2f}_W{W}": measures(x, y, P, wf, rc, W) for wf, rc, W in VARIANTS}}
d = load("1230"); yy = d["years"]; x30, y30 = rows(d)
kc = (yy >= 5100) & (yy <= 7823)
Pc = acf_period(band_cover(d, sectors=(1,))[kc])
for a, b in SEGS:
    k = (yy >= a) & (yy <= b)
    res[f"1230:{a}"] = {"P": Pc, "base": measures(x30[k], y30[k], Pc),
                        "var": {f"w{wf:.2f}_r{rc:.2f}_W{W}": measures(x30[k], y30[k], Pc, wf, rc, W) for wf, rc, W in VARIANTS}}
print(f"1230 creep period (autocorrelation, 5100–7823): {Pc} yr")
print(f"{'run':12s}{'P':>4s}{'n':>5s}{'mean':>7s}{'h50':>6s}{'h90':>6s}{'hmax':>6s}{'τ½/P':>12s}{'R (W=20)':>16s}{'R per swing':>16s}")
for lab, r in res.items():
    b = r["base"]
    print(f"{lab:12s}{r['P']:4d}{b['n']:5d}{b['mean']:7.3f}{b['h50']:6.2f}{b['h90']:6.2f}{b['hmax']:6.2f}"
          f"{b['tau']:7.3f}±{b['tau_se']:.3f}{b['R']:9.4f}±{b['R_se']:.4f}{b['Rn']:9.3f}±{b['Rn_se']:.3f}")

# ---- criteria ----
print("\nP2 (h90 of last two above every run 1245–1237.5, both peak windows):")
for wf in (0.5, 1 / 3):
    key = f"w{wf:.2f}_r0.50_W20"
    ref_max = max(res[l]["var"][key]["h90"] for l in REF)
    print(f"   window {wf:.2f}P: ref max {ref_max:.3f}; 1233.75 {res['1233p75']['var'][key]['h90']:.3f}, 1232.5 {res['1232p5']['var'][key]['h90']:.3f} → "
          f"{all(res[l]['var'][key]['h90'] > ref_max for l in END2)}")


def crit(field):
    ok_all, lines = True, []
    for vk in res["1245"]["var"]:
        rv = [res[l]["var"][vk] for l in REF]
        rm = np.mean([v[field] for v in rv]); rse = np.sqrt(np.nansum([v[field + "_se"] ** 2 for v in rv])) / len(rv)
        ok = all(res[l]["var"][vk][field] - rm > 2 * np.sqrt(rse ** 2 + res[l]["var"][vk][field + "_se"] ** 2) for l in END2)
        ok_all &= ok
        lines.append((vk, rm, [res[l]["var"][vk][field] for l in END2], ok))
    return ok_all, lines


for field, name in (("tau", "P3 stickier swings (τ½/P)"), ("R", "P4a Indian/Pacific response R"), ("Rn", "P4b response per unit swing")):
    ok, lines = crit(field)
    n_ok = sum(l[3] for l in lines)
    print(f"\n{name}: met in {n_ok} of {len(lines)} variants → {'HOLDS' if ok else 'does not hold'}")
    for vk, rm, v, o in lines[:12]:
        print(f"   {vk}: ref {rm:.3f}; 1233.75 {v[0]:.3f}, 1232.5 {v[1]:.3f}  {o}")

print("\nP5: last full creep segment (7100–7599) vs 1232.5")
c, w = res["1230:7100"]["base"], res["1232p5"]["base"]
for f in ("mean", "h90", "tau", "R"):
    print(f"   {f}: creep {c[f]:.3f}  1232.5 {w[f]:.3f}  → {c[f] > w[f]}")

# ---- P6: the jump swing ----
m500 = lambda t: np.nanmean(x30[(yy >= t - 500) & (yy < t)])
creep_pk = peaks(x30[kc], int(round(Pc / 2)), np.nanmean(x30[kc]))
cy = yy[kc]


def retreat(t, h):
    m = m500(t); kk = (yy > t) & (yy <= t + Pc)
    return (np.nanmin(x30[kk]) - m) / (h - m)


def resp(t, W=20):
    return np.nanmean(y30[(yy > t) & (yy <= t + W)]) - np.nanmean(y30[(yy >= t - W) & (yy < t)])


cr = [(int(cy[i]), x30[kc][i]) for i in creep_pk if cy[i] + Pc <= 7823 and cy[i] - 500 >= 4600]
kj = (yy >= 7815) & (yy <= 7850); tj = int(yy[kj][np.argmax(x30[kj])]); hj = float(np.nanmax(x30[kj]))
rt = [retreat(t, h) for t, h in cr]; rj = retreat(tj, hj)
print(f"\nP6: jump swing peak {hj:.3f} in {tj}; largest creep peak {max(h for _, h in cr):.3f}")
print(f"   retreat fraction (0 = back to the 500-yr mean, 1 = no retreat): jump {rj:.2f}; creep median {np.median(rt):.2f} "
      f"(range {min(rt):.2f}–{max(rt):.2f}) → shallower than median: {rj > np.median(rt)}")
for W in (10, 20, 30, 60, 100):
    rc = [resp(t, W) for t, _ in cr]
    print(f"   Indian/Pacific response W={W:3d}: jump {resp(tj, W):+.4f}; creep 90th pct {np.percentile(rc, 90):+.4f} "
          f"(median {np.median(rc):+.4f}) → above 90th: {resp(tj, W) > np.percentile(rc, 90)}")
json.dump(dict(res=res, Pc=Pc, tj=tj, hj=hj), open(TMP + "swings.json", "w"), default=float)
