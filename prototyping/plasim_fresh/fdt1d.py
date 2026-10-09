"""One-dimensional FDT check (criteria: Claude/plasim_mechanism/plan_fdt.md)."""

import json

import numpy as np

from common import band_cover, load, lsg_layer_mean, sector_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json")); SH = json.load(open(TMP + "sharpen.json"))
SEQ = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
RUNS = SEQ[:6] + ["1235_new_IC"] + SEQ[6:] + ["1230", "1228p5"]
WIN = {"1230": (11600, 16369), "1228p5": (15000, 16899)}
REF = ["1245", "1242p5", "1240", "1237p5", "1235"]; END2 = ["1233p75", "1232p5"]
KS = (20, 50, 100, 200)


def get(lab, win=None):
    d = load(lab); y = d["years"]
    a, b = win or WIN.get(lab, (STAT[lab]["0.95"], STAT[lab]["end"]))
    k = (y >= a) & (y <= b); w = d["gw"] / d["gw"].sum()
    return dict(mu=d["mu"], Ts=(d["ts"] @ w)[k], ice=sector_ice_area(d, (1, 2, 3))[k],
                ip=sector_ice_area(d, (2, 3))[k], H=lsg_layer_mean(d, 1, -90, 0)[k])


def tau_int(x, K):
    x = np.nan_to_num(x - np.nanmean(x)); v = np.sum(x * x) / len(x)
    return 1 + 2 * sum(np.sum(x[:-k] * x[k:]) / len(x) / v for k in range(1, K + 1))


def seg_se(x, fn, L=1000):
    n = len(x) // L
    v = [fn(x[i * L:(i + 1) * L]) for i in range(n)]
    return np.std(v, ddof=1) / np.sqrt(n) if n >= 3 else np.nan


D = {lab: get(lab) for lab in RUNS}
D["1230_noexc"] = get("1230", (11600, 15499))


def sens(lab, q):
    base = "1235" if lab == "1235_new_IC" else lab
    if base in SEQ:
        i = SEQ.index(base); a = SEQ[max(i - 1, 0)]; b = SEQ[min(i + 1, len(SEQ) - 1)]
    else:
        a, b = "1230", "1228p5"
    return (np.nanmean(D[b][q]) - np.nanmean(D[a][q])) / (D[b]["mu"] - D[a]["mu"])


res = {}
for q, unit in (("Ts", "K"), ("ice", "10¹² m²"), ("ip", "10¹² m²"), ("H", "K")):
    print(f"\n=== {q}: S = d(mean)/dμ ({unit} per W m⁻²), τ_int (yr) for K = {KS}, sd of annual values")
    res[q] = {}
    for lab in RUNS + ["1230_noexc"]:
        x = D[lab][q]; S = sens(lab.split("_noexc")[0], q)
        taus = {K: tau_int(x, K) for K in KS}; ses = {K: seg_se(x, lambda z, K=K: tau_int(z, K)) for K in (50, 100)}
        res[q][lab] = dict(S=S, tau=taus, se=ses, sd=float(np.nanstd(x)))
        print(f"  {lab:12s} S {S:+8.3f}   τ " + " ".join(f"{taus[K]:7.1f}" for K in KS)
              + f"   (±{ses[50]:.1f}, ±{ses[100]:.1f})   sd {np.nanstd(x):.3f}")
    for K in (20, 50, 100, 200):
        ratio = {lab: res[q][lab]["S"] / res[q][lab]["tau"][K] for lab in REF + END2}
        ref = np.mean([ratio[l] for l in REF])
        rel = {l: ratio[l] / ref for l in END2}
        verdict = ("consistent" if all(abs(v - 1) <= 0.3 for v in rel.values()) else
                   "FAILS (sensitivity outgrows τ)" if all(v > 1.3 for v in rel.values()) else "mixed")
        print(f"  K={K:3d}: S/τ relative to the 1245–1235 mean: 1233.75 {rel['1233p75']:.2f}, 1232.5 {rel['1232p5']:.2f} → {verdict}")
        res[q][f"rel_K{K}"] = rel

print("\nAtlantic oscillation as a damped resonance λ = −γ + iω (from period P and regularity ACF(P))")
osc = {}
for lab in ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]:
    P = SH["A"][lab]["full"]["per_acf"]; r = SH["A"][lab]["full"]["reg"]
    osc[lab] = dict(P=P, omega=2 * np.pi / P, gamma=-np.log(r) / P)
    print(f"  {lab:12s} P {P:5.1f} yr  ω {2*np.pi/P:.4f} /yr  γ {-np.log(r)/P:.4f} /yr  (decay time {P/-np.log(r):5.0f} yr)")
d = load("1230"); y = d["years"]; k = (y >= 5100) & (y <= 7823)
a = band_cover(d, sectors=(1,))[k]; a = np.nan_to_num(a - np.nanmean(a)); v = np.sum(a * a) / len(a)
r = np.array([1.0] + [np.sum(a[:-j] * a[j:]) / len(a) / v for j in range(1, 201)])
kmin = 10 + int(np.argmin(r[10:151])); P = kmin + 1 + int(np.argmax(r[kmin + 1:min(3 * kmin, 200) + 1]))
osc["1230creep"] = dict(P=float(P), omega=2 * np.pi / P, gamma=-np.log(max(r[P], 1e-3)) / P)
print(f"  1230 creep   P {P:5.1f} yr  ω {2*np.pi/P:.4f} /yr  γ {osc['1230creep']['gamma']:.4f} /yr  (creep is not stationary)")
json.dump(dict(res=res, osc=osc), open(TMP + "fdt1d.json", "w"), default=float)
