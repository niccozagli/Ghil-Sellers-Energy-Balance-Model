import json, numpy as np
from common import load, global_ts, south_ice_area, lsg_layer_mean, row_ocean_area
ST = json.load(open('/Users/niccolo/.claude/jobs/3fe717ba/tmp/stationary.json'))
runs = ["1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
mu = {r: float(r.replace("p", ".")) for r in runs}
out = {}
for r in runs:
    d = load(r); y = d["years"]; a, b = ST[r]["0.95"], ST[r]["end"]
    k = (y >= a) & (y <= b)
    lat = d["lat"]; area = row_ocean_area(d, 0)
    bands = {"41.5": [41.53], "36": [36.0], "30.5": [30.46], "eq25": [24.92, 19.38, 13.84]}
    f = {"Ts": global_ts(d), "ice": south_ice_area(d) / 1e12, "H": lsg_layer_mean(d, 1, -90, 0)}
    for bn, ls in bands.items():
        f["ice_" + bn] = sum(np.nan_to_num(d["sic"][:, int(np.argmin(np.abs(lat + l))), 0]) * area[int(np.argmin(np.abs(lat + l)))] for l in ls) / 1e12
    res = {"L": int(k.sum())}
    for q, v in f.items():
        v = v[k]; yy = y[k]
        m = float(np.nanmean(v))
        se = {}
        for seg in (500, 1000):
            n = len(v) // seg
            sm = [np.nanmean(v[i * seg:(i + 1) * seg]) for i in range(n)]
            se[seg] = float(np.std(sm, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
            se[f"sd{seg}"] = float(np.std(sm, ddof=1)) if n >= 2 else np.nan
        res[q] = {"mean": m, **{str(s): x for s, x in se.items()}}
    out[r] = res
json.dump(out, open('/Users/niccolo/.claude/jobs/3fe717ba/tmp/sens_precision.json', 'w'), indent=1)
print("run  L   Ts mean  SE500 SE1000 sd1000 | ice mean SE1000 sd1000 | H SE1000")
for r in runs:
    o = out[r]
    print(f"{r:7s} {o['L']:5d} {o['Ts']['mean']:.3f} {o['Ts']['500']:.3f} {o['Ts']['1000']:.3f} {o['Ts']['sd1000']:.3f} | {o['ice']['mean']:.2f} {o['ice']['1000']:.2f} {o['ice']['sd1000']:.2f} | {o['H']['1000']:.4f}")
print("\nsensitivity per W m-2 between neighbours (SE from 1000-yr segments; 500-yr in brackets)")
for r1, r2 in zip(runs[:-1], runs[1:]):
    dm = mu[r1] - mu[r2]
    line = f"{r1}->{r2} (dmu {dm}):"
    for q in ("Ts", "ice", "H"):
        s = (out[r1][q]["mean"] - out[r2][q]["mean"]) / dm
        se = np.hypot(out[r1][q]["1000"], out[r2][q]["1000"]) / dm
        se5 = np.hypot(out[r1][q]["500"], out[r2][q]["500"]) / dm
        line += f"  {q} {s:.3f}±{se:.3f} [{se5:.3f}]"
    print(line)
print("\nice added per W m-2 by row band (10^12 m2)")
for r1, r2 in zip(runs[:-1], runs[1:]):
    dm = mu[r1] - mu[r2]
    print(f"{r1}->{r2}: " + "  ".join(f"{bn} {(out[r2]['ice_'+bn]['mean']-out[r1]['ice_'+bn]['mean'])/dm:.2f}" for bn in ("41.5", "36", "30.5", "eq25")))
