"""Re-check of the equilibrium energetics quoted in the review (window means)."""

import numpy as np

from common import equivalent_edge, load, row_ocean_area
from gsebm.plasim_global import RUNS, ice_edge_latitude, ocean_northward_transport

LABS = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC",
        "1233p75", "1232p5", "1230", "1228p5", "1225"]
st = {}
for lab in LABS:
    d = load(lab)
    w0, w1 = RUNS[lab].window
    k = (d["years"] >= w0) & (d["years"] <= w1)
    w = d["gw"] / d["gw"].sum()

    def m(x):
        return np.nanmean(x[k], axis=0)

    rst, rsut, rlut, ts = m(d["rst"]), m(d["rsut"]), m(d["rlut"]), m(d["ts"])
    insolation = rst - rsut
    ice = m(d["sic"][:, :, 0])
    south = d["lat"] < 0
    area_s = np.nansum(ice[south] * row_ocean_area(d)[south]) / 1e12
    area_n = np.nansum(ice[~south] * row_ocean_area(d)[~south]) / 1e12
    flux = np.nan_to_num(m(d["coupling_flux"]))
    order = np.argsort(d["lsg_lat"])
    oht = ocean_northward_transport(flux[order], d["lsg_area"][order])  # north face of each row
    lsg_lat = d["lsg_lat"][order]
    face = lsg_lat + 1.25
    j20 = int(np.argmin(np.abs(face + 20)))
    jrel = int(np.argmin(np.where(lsg_lat < 0, flux[order], np.inf)))
    st[lab] = dict(
        mu=d["mu"], Ts=ts @ w, I=insolation @ w, alpha=1 - (rst @ w) / (insolation @ w),
        olr=-(rlut @ w), net=(rst + rlut) @ w, eS=float(np.nanmean(equivalent_edge(d)[k])),
        e05=float(ice_edge_latitude(ice[None], d["lat"], True)[0]),
        eN=float(ice_edge_latitude(ice[None], d["lat"], False)[0]),
        aS=area_s, aN=area_n, oht20=-oht[j20], rel_lat=lsg_lat[jrel], rel_val=flux[order][jrel],
    )

print(f"{'run':12s}{'μ':>8s}{'Ts':>8s}{'albedo':>8s}{'OLR':>7s}{'TOAnet':>7s}{'eqS':>7s}{'e0.5S':>7s}"
      f"{'e0.5N':>7s}{'iceS':>7s}{'iceN':>7s}{'OHT20S':>8s}{'relLat':>7s}{'relW':>7s}")
for lab, s in st.items():
    print(f"{lab:12s}{s['mu']:8.2f}{s['Ts']:8.2f}{s['alpha']:8.3f}{s['olr']:7.1f}{s['net']:7.2f}{s['eS']:7.1f}"
          f"{s['e05']:7.1f}{s['eN']:7.1f}{s['aS']:7.1f}{s['aN']:7.1f}{s['oht20']:8.2f}{s['rel_lat']:7.1f}"
          f"{s['rel_val']:7.1f}")
print("(ice areas in 10^12 m^2; OHT20S = poleward ocean heat transport across ~20°S, PW;"
      " relLat/relW = latitude and value of the strongest SH ocean heat release, W m^-2)")

print("\nBudget per step: ΔT per W m⁻² of μ; direct and albedo parts of ΔASR (W m⁻², global); gain; λ = ΔOLR/ΔT")
pairs = [("1312", "1288"), ("1288", "1265"), ("1265", "1250"), ("1250", "1245"), ("1245", "1242p5"),
         ("1242p5", "1240"), ("1240", "1237p5"), ("1237p5", "1235"), ("1235", "1233p75"),
         ("1233p75", "1232p5"), ("1232p5", "1230"), ("1228p5", "1225"), ("1245", "1235"),
         ("1235", "1232p5"), ("1265", "1232p5")]
for a, b in pairs:
    A, B = st[a], st[b]
    dmu = B["mu"] - A["mu"]
    dT = B["Ts"] - A["Ts"]
    am, Im = 0.5 * (A["alpha"] + B["alpha"]), 0.5 * (A["I"] + B["I"])
    direct = (B["I"] - A["I"]) * (1 - am)
    alb = -Im * (B["alpha"] - A["alpha"])
    dolr = B["olr"] - A["olr"]
    print(f"  {a:>7s}→{b:<7s} ΔT {dT:6.2f} K  ΔT/Δμ {dT/dmu:5.2f}  direct {direct:6.2f}  albedo {alb:6.2f}"
          f"  gain {(direct+alb)/direct:5.2f}  ΔOLR {dolr:6.2f}  λ {dolr/dT:5.2f}"
          f"  ΔiceS/Δμ {(B['aS']-A['aS'])/-dmu:5.2f}  ΔiceN/Δμ {(B['aN']-A['aN'])/-dmu:5.2f}")
