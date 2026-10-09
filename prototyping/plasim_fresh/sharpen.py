"""Sharpened fingerprint analysis (criteria: Claude/plasim_mechanism/plan_sharpen.md). Basic methods only."""

import itertools
import json

import numpy as np

from common import band_cover, equivalent_edge, load, lsg_layer_mean, sector_ice_area, sector_sw

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
STAT = json.load(open(TMP + "stationary.json"))
WARM = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
ALL = WARM[:9] + ["1235_new_IC"] + WARM[9:] + ["1230", "1228p5"]
ATL_RUNS = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC", "1233p75", "1232p5"]
REF = ["1245", "1242p5", "1240", "1237p5"]
END2 = ["1233p75", "1232p5"]

# ---------------- data ----------------
D = {}
for lab in ALL:
    d = load(lab); y = d["years"]
    ys, sw = sector_sw(lab)
    assert np.array_equal(ys, y)
    ser = dict(atl=band_cover(d, sectors=(1,)), ipb=band_cover(d), ice_ip=sector_ice_area(d, (2, 3)),
               a30=d["sic"][:, int(np.argmin(np.abs(d["lat"] + 30.46))), 1], sw_tot=sw.sum(axis=1), sw_pac=sw[:, 2],
               H=lsg_layer_mean(d, 1, -90, 0), E=equivalent_edge(d))
    D[lab] = dict(d=d, y=y, ser=ser, mu=d["mu"])


def mask(lab, q="0.95", half=False):
    s, e = STAT[lab][q], STAT[lab]["end"]
    if half:
        s = s + (e - s + 1) // 2
    y = D[lab]["y"]
    return (y >= s) & (y <= e)


def W_series(lab, edge):
    d = D[lab]["d"]
    rows = (d["lsg_lat"] >= -edge) & (d["lsg_lat"] <= -edge + 10)
    wv = d["layer_volume"][rows, 0]
    return np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()


EDGE = {lab: float(np.nanmean(D[lab]["ser"]["E"][mask(lab)])) for lab in ALL}
for lab in ALL:
    D[lab]["ser"]["W"] = W_series(lab, EDGE[lab])


def prep(x, detrend):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    t = np.arange(len(x))
    if detrend and ok.sum() > 10:
        c = np.polyfit(t[ok], x[ok], 1); x = x - np.polyval(c, t)
    x = x - np.nanmean(x)
    return np.where(ok, x, 0.0)


def acf(x, maxlag):
    v = np.sum(x * x) / len(x)
    return np.array([1.0] + [np.sum(x[:-k] * x[k:]) / len(x) / v for k in range(1, maxlag + 1)])


def segs(m, length=1000):
    idx = np.flatnonzero(m)
    n = len(idx) // length
    return [idx[i * length:(i + 1) * length] for i in range(n)]


def seg_stat(lab, fn, q="0.95", half=False):
    m = mask(lab, q, half)
    vals = [fn(np.zeros(len(m), bool) | np.isin(np.arange(len(m)), s)) for s in segs(m)]
    vals = [v for v in vals if np.isfinite(v)]
    return (np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)), len(vals)) if len(vals) >= 3 else (np.nan, np.nan, len(vals))


# ---------------- A: Atlantic oscillation ----------------
def osc_measures(m, lab, detrend=False):
    a = prep(D[lab]["ser"]["atl"][m], detrend)
    f = np.fft.rfftfreq(len(a)); p = np.abs(np.fft.rfft(a)) ** 2
    k = (f > 1 / 200) & (f < 1 / 25)
    per_pg = 1 / f[k][np.argmax(p[k])]
    r = acf(a, 200)
    kmin = 10 + int(np.argmin(r[10:151]))
    kmax = kmin + 1 + int(np.argmax(r[kmin + 1:min(3 * kmin, 200) + 1]))
    return dict(per_pg=per_pg, per_acf=float(kmax), reg=float(r[kmax]), sd=float(np.std(a)))


def run_mean(x, n):
    return np.convolve(x, np.ones(n) / n, mode="valid")


def amp_rm(m, lab, n, detrend=False):
    return float(np.std(run_mean(prep(D[lab]["ser"]["atl"][m], detrend), n)))


def reach(m, lab, pct):
    return float(np.nanpercentile(D[lab]["ser"]["a30"][m], pct))


def leak(m, lab, P, both=False, rm=1, detrend=False):
    a = prep(D[lab]["ser"]["atl"][m], detrend); b = prep(D[lab]["ser"]["ipb"][m], detrend)
    if rm > 1:
        a, b = run_mean(a, rm), run_mean(b, rm)
    lags = range(-P, P + 1) if both else range(0, P + 1)
    best = -2
    for k in lags:
        c = np.corrcoef(a[:len(a) - k], b[k:])[0, 1] if k >= 0 else np.corrcoef(a[-k:], b[:len(b) + k])[0, 1]
        best = max(best, c)
    return float(best)


A = {}
print("A. Atlantic oscillation (stationary q=0.95; ± = segment SE, n segments)")
for lab in ATL_RUNS:
    m = mask(lab)
    full = osc_measures(m, lab)
    P = int(round(full["per_acf"]))
    res = dict(full=full, P=P)
    for key in ("per_pg", "per_acf", "reg", "sd"):
        res[key + "_seg"] = seg_stat(lab, lambda mm, key=key: osc_measures(mm, lab)[key])
    res["amp10"] = amp_rm(m, lab, 10); res["amp5"] = amp_rm(m, lab, 5); res["amp20"] = amp_rm(m, lab, 20)
    for pct in (95, 99, 99.9):
        res[f"reach{pct}"] = reach(m, lab, pct)
        res[f"reach{pct}_seg"] = seg_stat(lab, lambda mm, pct=pct: reach(mm, lab, pct))
    for name, kw in (("leak", {}), ("leak_both", {"both": True}), ("leak_rm10", {"rm": 10}), ("leak_rm10_both", {"rm": 10, "both": True})):
        res[name] = leak(m, lab, P, **kw)
        res[name + "_seg"] = seg_stat(lab, lambda mm, kw=kw: leak(mm, lab, P, **kw))
    # variants of the period/regularity: q, half, detrend
    var = {}
    for q, half, dt in itertools.product(("0.9", "0.95", "0.99"), (False, True), (False, True)):
        var[f"q{q}_h{int(half)}_d{int(dt)}"] = osc_measures(mask(lab, q, half), lab, dt)
    res["variants"] = var
    A[lab] = res
    print(f"  {lab:12s} period pg {full['per_pg']:5.1f} ({res['per_pg_seg'][0]:5.1f}±{res['per_pg_seg'][1]:4.1f})  "
          f"acf {full['per_acf']:4.0f} ({res['per_acf_seg'][0]:5.1f}±{res['per_acf_seg'][1]:4.1f})  "
          f"regularity {full['reg']:.2f} ({res['reg_seg'][0]:.2f}±{res['reg_seg'][1]:.2f})  sd {full['sd']:.3f}  "
          f"reach99 {res['reach99']:.2f}±{res['reach99_seg'][1]:.2f}  leak {res['leak']:.2f}±{res['leak_seg'][1]:.2f} "
          f"(rm10 {res['leak_rm10']:.2f}, ±P {res['leak_both']:.2f})  n_seg {res['reg_seg'][2]}")

# criteria A
def mono(vals, ses):
    ok = True
    for i in range(len(vals) - 1):
        if vals[i + 1] < vals[i] and (vals[i] - vals[i + 1]) > 2 * np.sqrt(np.nan_to_num(ses[i]) ** 2 + np.nan_to_num(ses[i + 1]) ** 2):
            ok = False
    return ok


seq = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
for key in ("per_pg", "per_acf"):
    vals = [A[l]["full"][key] for l in seq]; ses = [A[l][key + "_seg"][1] for l in seq]
    print(f"  A1 {key}: {np.round(vals, 1)} → monotonic within 2 SE: {mono(vals, ses)}")
    for vk in A["1245"]["variants"]:
        v2 = [A[l]["variants"][vk][key] for l in seq]
        if not mono(v2, ses):
            print(f"     variant {vk} breaks monotonicity: {np.round(v2, 1)}")


def group_test(key, late, early, use_full=True):
    lv = [A[l]["full"][key] if key in A[l]["full"] else A[l][key] for l in late]
    ev = [A[l]["full"][key] if key in A[l]["full"] else A[l][key] for l in early]
    ls = [A[l][key + "_seg"][1] for l in late]; es = [A[l][key + "_seg"][1] for l in early]
    se = np.sqrt(np.nanmean(np.square(ls)) / len(late) + np.nanmean(np.square(es)) / len(early))
    return np.mean(lv), np.mean(ev), se


lm, em, se = group_test("reg", ["1235", "1235_new_IC", "1233p75", "1232p5"], ["1245", "1242p5", "1240"])
print(f"  A2 regularity late {lm:.3f} vs early {em:.3f}, SE {se:.3f} → less regular: {em - lm > 2 * se}")
for vk in A["1245"]["variants"]:
    l2 = np.mean([A[l]["variants"][vk]["reg"] for l in ["1235", "1235_new_IC", "1233p75", "1232p5"]])
    e2 = np.mean([A[l]["variants"][vk]["reg"] for l in ["1245", "1242p5", "1240"]])
    if not (e2 - l2 > 2 * se):
        print(f"     A2 variant {vk}: late {l2:.3f} early {e2:.3f} → criterion not met")
for key in ("reach95", "reach99", "reach99.9", "leak", "leak_both", "leak_rm10", "leak_rm10_both"):
    ref = np.mean([A[l][key] for l in REF]); ref_se = np.sqrt(np.nanmean([A[l][key + "_seg"][1] ** 2 for l in REF]) / len(REF))
    ok = all(A[l][key] - ref > 2 * np.sqrt(ref_se ** 2 + np.nan_to_num(A[l][key + "_seg"][1]) ** 2) for l in END2)
    print(f"  A4/A5 {key:15s}: ref mean {ref:.3f}; 1233.75 {A['1233p75'][key]:.3f}, 1232.5 {A['1232p5'][key]:.3f} → exceeds by >2 SE: {ok}")

# ---------------- B: slow gain ----------------
def block(x, m, L):
    idx = np.flatnonzero(m); n = len(idx) // L
    return np.array([np.nanmean(x[idx[i * L:(i + 1) * L]]) for i in range(n)])


def nat_slope(lab, xkey="W", L=100, q="0.95", half=False, detrend=False):
    m = mask(lab, q, half); y = D[lab]["y"][m]
    X = D[lab]["ser"][xkey][m].copy(); Y = D[lab]["ser"]["ice_ip"][m].copy()
    if detrend:
        for Z in (X, Y):
            ok = np.isfinite(Z); c = np.polyfit(y[ok], Z[ok], 1); Z -= np.polyval(c, y)
    mm = np.ones(len(X), bool)
    xb, yb = block(X, mm, L), block(Y, mm, L)
    ok = np.isfinite(xb) & np.isfinite(yb); xb, yb = xb[ok] - xb[ok].mean(), yb[ok] - yb[ok].mean()
    n = len(xb)
    if n < 6:
        return dict(slope=np.nan, se=np.nan, r=np.nan, n=n)
    b = (xb @ yb) / (xb @ xb); res = yb - b * xb
    r1 = np.corrcoef(res[:-1], res[1:])[0, 1]; neff = n * (1 - r1) / (1 + r1) if r1 > 0 else n
    se = np.sqrt(res @ res / (n - 2) / (xb @ xb)) * np.sqrt(n / max(neff, 2.5))
    return dict(slope=float(b), se=float(se), r=float(np.corrcoef(xb, yb)[0, 1]), n=n, neff=float(neff))


def stat_mean(lab, key, edge_lab=None):
    m = mask(lab)
    if key == "W":
        return float(np.nanmean(W_series(lab, EDGE[edge_lab or lab])[m]))
    return float(np.nanmean(D[lab]["ser"][key][m]))


def forced(lab):
    base = "1235" if lab == "1235_new_IC" else lab
    if base in WARM:
        i = WARM.index(base); a = WARM[max(i - 1, 0)]; b = WARM[min(i + 1, len(WARM) - 1)]
    else:
        a, b = "1230", "1228p5"
    dW = stat_mean(a, "W", lab) - stat_mean(b, "W", lab)
    dI = stat_mean(a, "ice_ip") - stat_mean(b, "ice_ip")
    return dI / dW, (a, b)


B = {}
print("\nB. Slow gain: slope of Indo-Pacific ice area (10¹² m²) on W (K), L = 100, q = 0.95")
for lab in ALL:
    f, pair = forced(lab)
    base = nat_slope(lab); dtv = nat_slope(lab, detrend=True); onH = nat_slope(lab, "H")
    var = {}
    for L, q, half, dt in itertools.product((50, 100, 200), ("0.9", "0.95", "0.99"), (False, True), (False, True)):
        var[f"L{L}_q{q}_h{int(half)}_d{int(dt)}"] = nat_slope(lab, L=L, q=q, half=half, detrend=dt)
    def b1(s):
        return abs(s["slope"] - f) <= 2 * s["se"] and s["r"] >= 0.5
    B[lab] = dict(forced=f, pair=pair, base=base, detr=dtv, onH=onH, variants=var, B1=bool(b1(base) and b1(dtv)))
    print(f"  {lab:12s} forced {f:7.2f} ({pair[0]}–{pair[1]})  natural {base['slope']:7.2f} ± {base['se']:5.2f} (r {base['r']:+.2f}, n {base['n']}, "
          f"n_eff {base.get('neff', np.nan):.0f})  detrended {dtv['slope']:7.2f} ± {dtv['se']:5.2f} (r {dtv['r']:+.2f})  → B1 {B[lab]['B1']}")

print("  B2: late (1233.75, 1232.5) vs reference (1245–1237.5) in every variant")
n_ok, n_var = 0, 0
for vk in B["1245"]["variants"]:
    ref = [B[l]["variants"][vk] for l in REF]
    if any(np.isnan(r["slope"]) for r in ref):
        continue
    # gain = −slope (more ice per K of cooling); compare magnitudes
    rm = np.mean([-r["slope"] for r in ref]); rse = np.sqrt(np.sum([r["se"] ** 2 for r in ref])) / len(ref)
    ok = all(np.isfinite(B[l]["variants"][vk]["slope"]) and
             -B[l]["variants"][vk]["slope"] - rm > 2 * np.sqrt(rse ** 2 + B[l]["variants"][vk]["se"] ** 2) for l in END2)
    ok1 = [(-B[l]["variants"][vk]["slope"] - rm > 2 * np.sqrt(rse ** 2 + B[l]["variants"][vk]["se"] ** 2)) for l in END2]
    n_var += 1; n_ok += ok
    B.setdefault("_B2", {})[vk] = dict(ref_gain=rm, gains=[-B[l]["variants"][vk]["slope"] for l in END2], each=ok1)
print(f"     criterion met in {n_ok} of {n_var} variants")
each = np.array([v["each"] for v in B["_B2"].values()])
print(f"     1233.75 alone above the reference in {each[:, 0].sum()} of {len(each)} variants; 1232.5 alone in {each[:, 1].sum()}")
print("     example (L100, q0.95, full, not detrended):", B["_B2"]["L100_q0.95_h0_d0"])
print("\nA4 added (descriptive): Atlantic 30.5°S row cover — mean, 99th percentile, excess of the 99th percentile over the mean")
for lab in ATL_RUNS:
    m = mask(lab); x = D[lab]["ser"]["a30"][m]
    print(f"  {lab:12s} mean {np.nanmean(x):.3f}  p99 {np.nanpercentile(x, 99):.3f}  excess {np.nanpercentile(x, 99) - np.nanmean(x):.3f}")

# ---------------- C: slow albedo fluctuations and decadal memory ----------------
def c1(m, lab, key, L, detrend):
    x = prep(D[lab]["ser"][key][m], detrend); n = len(x) // L
    return float(np.std(x[:n * L].reshape(n, L).mean(axis=1), ddof=1))


def c2(m, lab, lag, detrend):
    x = prep(D[lab]["ser"]["ice_ip"][m], detrend)
    return float(acf(x, lag)[lag])


C = {}
print("\nC. Slow albedo fluctuations (sd of 50-yr means of SW, PW) and decadal memory (lag-10 ACF of Indo-Pacific ice area)")
for lab in ALL:
    m = mask(lab); res = {}
    for key in ("sw_tot", "sw_pac"):
        for L in (50, 100):
            for dt in (False, True):
                nm = f"{key}_L{L}_d{int(dt)}"
                res[nm] = c1(m, lab, key, L, dt); res[nm + "_seg"] = seg_stat(lab, lambda mm, key=key, L=L, dt=dt: c1(mm, lab, key, L, dt))
    for lag in (5, 10, 20):
        for dt in (False, True):
            nm = f"acf{lag}_d{int(dt)}"
            res[nm] = c2(m, lab, lag, dt); res[nm + "_seg"] = seg_stat(lab, lambda mm, lag=lag, dt=dt: c2(mm, lab, lag, dt))
    for q, half in (("0.9", False), ("0.99", False), ("0.95", True)):
        mm = mask(lab, q, half)
        res[f"sw_tot_L50_d0_q{q}_h{int(half)}"] = c1(mm, lab, "sw_tot", 50, False)
        res[f"acf10_d0_q{q}_h{int(half)}"] = c2(mm, lab, 10, False)
    C[lab] = res
    print(f"  {lab:12s} sd50 total {res['sw_tot_L50_d0']:.3f}±{res['sw_tot_L50_d0_seg'][1]:.3f}  Pacific {res['sw_pac_L50_d0']:.3f}  "
          f"(detrended {res['sw_tot_L50_d1']:.3f}; 100-yr {res['sw_tot_L100_d0']:.3f})   ACF10 {res['acf10_d0']:.2f}±{res['acf10_d0_seg'][1]:.2f} "
          f"(det {res['acf10_d1']:.2f}; lag5 {res['acf5_d0']:.2f}; lag20 {res['acf20_d0']:.2f})")
REFC = ["1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC"]
for base in ("sw_tot_L50", "sw_tot_L100", "sw_pac_L50", "sw_pac_L100", "acf5", "acf10", "acf20"):
    for dt in (0, 1):
        nm = f"{base}_d{dt}"
        mx = max(C[l][nm] for l in REFC); lmx = max(REFC, key=lambda l: C[l][nm]); mse = C[lmx][nm + "_seg"][1]
        ok = all(C[l][nm] - mx > 2 * np.sqrt(np.nan_to_num(mse) ** 2 + np.nan_to_num(C[l][nm + "_seg"][1]) ** 2) for l in END2)
        print(f"  C rise at end, {nm:14s}: max of reference {mx:.3f} ({lmx}); 1233.75 {C['1233p75'][nm]:.3f}, 1232.5 {C['1232p5'][nm]:.3f} → {ok}")
for nm in ("sw_tot_L50_d0_q0.9_h0", "sw_tot_L50_d0_q0.99_h0", "sw_tot_L50_d0_q0.95_h1", "acf10_d0_q0.9_h0", "acf10_d0_q0.99_h0", "acf10_d0_q0.95_h1"):
    mx = max(C[l][nm] for l in REFC)
    print(f"  C variant {nm}: max of reference {mx:.3f}; 1233.75 {C['1233p75'][nm]:.3f}, 1232.5 {C['1232p5'][nm]:.3f} → both above max: "
          f"{all(C[l][nm] > mx for l in END2)}")

json.dump(dict(A=A, B=B, C=C, EDGE=EDGE), open(TMP + "sharpen.json", "w"), default=float)
