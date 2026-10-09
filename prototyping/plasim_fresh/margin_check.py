"""Margin re-check (doubt on observation 4): μ forcing needed per unit of ice change, for several ice measures.

Window means. Cap |lat| ≥ 20°S. Insolation term = Σ ΔI (1 − mean α) A over cap rows (check 4's definition).
Ice measures: 0.5 / 0.3 / 0.7 thresholds on the window-mean zonal ocean cover (latitude, °), the
equivalent-area edge (°) and Southern ice area (10¹² m²; margin then in PW per 10¹² m²).
"""

import numpy as np

from common import load, row_ocean_area
from gsebm.plasim_global import RUNS, ice_edge_latitude

R = 6.371e6
SEQ = ["1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]


def state(lab):
    d = load(lab); w0, w1 = RUNS[lab].window
    k = (d["years"] >= w0) & (d["years"] <= w1)
    m = lambda x: np.nanmean(x[k], axis=0)
    rst, rsut = m(d["rst"]), m(d["rsut"])
    I = rst - rsut
    lat = d["lat"]; area = d["gw"] / d["gw"].sum() * 4 * np.pi * R**2
    ice = m(d["sic"][:, :, 0])
    out = dict(lat=lat, area=area, I=I, alpha=1 - rst / I)
    for th in (0.3, 0.5, 0.7):
        out[f"edge{th}"] = float(ice_edge_latitude(ice[None], lat, south=True, threshold=th)[0])
    south = lat < 0
    out["area_ice"] = float(np.nansum(ice[south] * row_ocean_area(d)[south])) / 1e12
    # equivalent edge from the window-mean area (same construction as common.equivalent_edge)
    rows = np.flatnonzero(south); rows = rows[np.argsort(lat[rows])]
    a = row_ocean_area(d)[rows] / 1e12; absl = np.abs(lat[rows])
    mids = 0.5 * (absl[:-1] + absl[1:]); pole = np.concatenate([[90.0], mids]); eq = np.concatenate([mids, [0.0]])
    cum = np.concatenate([[0.0], np.cumsum(a)])
    j = int(np.clip(np.searchsorted(cum, out["area_ice"], side="right") - 1, 0, len(a) - 1))
    out["eqedge"] = pole[j] + (out["area_ice"] - cum[j]) / a[j] * (eq[j] - pole[j])
    return out


st = {lab: state(lab) for lab in SEQ + ["1235_new_IC"]}


def margin(a, b, key):
    A, B = st[a], st[b]; cap = A["lat"] <= -20
    ab = 0.5 * (A["alpha"] + B["alpha"])[cap]
    ins = ((B["I"] - A["I"])[cap] * (1 - ab) * A["area"][cap]).sum() / 1e15
    d = B[key] - A[key]
    if key == "area_ice":
        d = -d  # more ice = positive "advance"
    return ins / d, d


keys = ["edge0.5", "edge0.3", "edge0.7", "eqedge", "area_ice"]
print("Window-mean ice measures:")
for lab in SEQ + ["1235_new_IC"]:
    print(f"  {lab:12s} " + "  ".join(f"{k} {st[lab][k]:7.2f}" for k in keys))
segs = {"early 1265→1245": ("1265", "1245"), "mid 1245→1235": ("1245", "1235"), "late 1235→1232.5": ("1235", "1232p5"),
        "late (new IC) 1235→1232.5": ("1235_new_IC", "1232p5")}
print("\nMargin = insolation term in the cap per unit of ice change (PW per degree; for area: PW per 10¹² m²)")
for name, (a, b) in segs.items():
    print(f"  {name:28s} " + "  ".join(f"{k} {margin(a, b, k)[0]:7.4f}" for k in keys))
print("\nPer step:")
for a, b in zip(SEQ[:-1], SEQ[1:]):
    print(f"  {a:>6s}→{b:<7s} " + "  ".join(f"{k} {margin(a, b, k)[0]:7.4f}" for k in keys))
