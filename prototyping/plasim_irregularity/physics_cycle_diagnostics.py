import pickle, sys
from pathlib import Path
import numpy as np
from gsebm import plasim_koopman_irregularity as ir
J = Path(sys.argv[1]); MUS = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
B = 36
out = {}
for mu in MUS:
    p = pickle.load(open(J / "phys" / f"phys_{mu}.pkl", "rb"))
    psi, years, P = p["psi1"], p["years"], p["period"]; ph = ir.wrapped_phase(psi); idx = ir.phase_bin_index(ph, B)
    sic, area, lat, basin = p["sic"], p["area"], p["lat"], p["basin"]
    ice = (sic * area[None]).sum((1, 2)) / 1e12              # 10^6 km²
    rows_ice = (sic * area[None]).sum(2) / 1e12               # per row
    cov = ice.mean()
    C = ir._group_means(ice[:, None], idx, B)[:, 0]
    kmax, kmin = int(np.argmax(C)), int(np.argmin(C))
    # edge latitude: per lon, northernmost basin cell with sic>=0.5; mean over lons with ice
    edge = np.full(years.size, np.nan)
    covered = sic >= 0.5
    lat_grid = np.broadcast_to(lat[:, None], basin.shape)
    northmost = np.where(covered, lat_grid[None], -90).max(1)  # (years, lon)
    has = covered.any(1)
    edge = np.array([northmost[t][has[t] & basin.any(0)].mean() if (has[t] & basin.any(0)).any() else np.nan for t in range(years.size)])
    Ce = ir._group_means(np.nan_to_num(edge, nan=np.nanmin(edge))[:, None], idx, B)[:, 0]
    Crow = ir._group_means(rows_ice, idx, B)
    rows_area = area.sum(1) / 1e12
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_max = Crow[kmax] / rows_area; frac_min = Crow[kmin] / rows_area
    # stage durations (years per cycle): time ∝ number of annual samples in phase range
    counts = np.bincount(idx, minlength=B).astype(float); share = counts / counts.sum()
    order = [(kmax + i) % B for i in range(((kmin - kmax) % B) + 1)]
    decline = share[order[:-1]].sum() * P if kmin != kmax else np.nan   # ice max -> ice min
    growth = P - decline
    # heat budget composite over the cycle
    bud = p["budget"]
    comp = {k: ir._group_means(bud[k][:, None], idx, B)[:, 0] for k in ("storage", "surface", "advection", "residual", "heat_content", "box_temperature")}
    # stage-integrated heat change (J) = sum over bins of mean storage(W) * years spent * s
    sec = 365.25 * 86400
    dur = share * P * sec  # seconds per bin per cycle
    stage = {}
    for name, bins in (("decline", order[:-1]), ("growth", [k for k in range(B) if k not in order[:-1]])):
        stage[name] = {k: float(np.sum(comp[k][bins] * dur[bins]) * 1e12 / 1e21) for k in ("storage", "surface", "advection", "residual")}  # 10^21 J
    Hrange = comp["heat_content"].max() - comp["heat_content"].min()
    kHmax = int(np.argmax(comp["heat_content"])); kHmin = int(np.argmin(comp["heat_content"]))
    # cycle vs residual relation (gyre HC vs ice)
    H = bud["heat_content"]
    cyc_I, cyc_H = C[idx], comp["heat_content"][idx]
    res_I = p["ice_resid_clean"]; res_H = ir.cycle_residuals(H[:, None], psi, None, None)["amplitude"][:, 0]
    lags = np.arange(-25, 26)
    cc = ir.lagged_correlation(cyc_H, cyc_I, lags); rc = ir.lagged_correlation(res_H, res_I, lags)
    # per-cycle amplitude vs mean |psi|
    edges = ir.cycle_crossings(psi, years)
    amp, rng_I, rng_H, L = [], [], [], []
    for a, b2 in zip(edges[:-1], edges[1:]):
        s = (years >= a) & (years < b2)
        if s.sum() < 5: continue
        amp.append(np.abs(psi[s]).mean()); rng_I.append(np.ptp(ice[s])); rng_H.append(np.ptp(H[s])); L.append(b2 - a)
    out[mu] = dict(P=P, mean_ice=cov, ice_max=C[kmax], ice_min=C[kmin], ice_range=C[kmax]-C[kmin],
                   edge_max=Ce[kmax], edge_min=Ce[kmin], edge_mean=np.nanmean(edge),
                   decline=decline, growth=growth, phase_min=kmin/B,
                   frac_max=frac_max, frac_min=frac_min, lat=lat,
                   Hrange=Hrange/1e21, phase_Hmax=kHmax/B, phase_Hmin=kHmin/B,
                   stage=stage, comp=comp,
                   cyc_peak=(cc[np.argmax(np.abs(cc))], lags[np.argmax(np.abs(cc))]),
                   res_peak=(rc[np.argmax(np.abs(rc))], lags[np.argmax(np.abs(rc))]),
                   amp_vs_iceRange=np.corrcoef(amp, rng_I)[0,1], amp_vs_HRange=np.corrcoef(amp, rng_H)[0,1],
                   amp_vs_L=np.corrcoef(amp, L)[0,1], iceRange_vs_L=np.corrcoef(rng_I, L)[0,1],
                   Tbox=np.mean(bud["box_temperature"]), cc=cc, rc=rc, lags=lags)
pickle.dump(out, open(J / "phys" / "summary.pkl", "wb"))
f = lambda k, fmt: print(f"{k:28s}" + "".join(format(out[m][k], fmt).rjust(9) for m in MUS))
print(" " * 28 + "".join(m.rjust(9) for m in MUS))
for k, fmt in (("P",".1f"),("mean_ice",".2f"),("ice_max",".2f"),("ice_min",".2f"),("ice_range",".2f"),("edge_mean",".1f"),("edge_max",".1f"),("edge_min",".1f"),
               ("phase_min",".2f"),("decline",".1f"),("growth",".1f"),("Hrange",".2f"),("phase_Hmax",".2f"),("phase_Hmin",".2f"),("Tbox",".2f"),
               ("amp_vs_iceRange",".2f"),("amp_vs_HRange",".2f"),("amp_vs_L",".2f"),("iceRange_vs_L",".2f")):
    f(k, fmt)
print("cycle HC–ice peak corr/lag:", {m: (round(out[m]["cyc_peak"][0],2), int(out[m]["cyc_peak"][1])) for m in MUS})
print("resid HC–ice peak corr/lag:", {m: (round(out[m]["res_peak"][0],2), int(out[m]["res_peak"][1])) for m in MUS})
for m in MUS:
    st = out[m]["stage"]
    print(m, "decline:", {k: round(v,2) for k,v in st["decline"].items()}, "growth:", {k: round(v,2) for k,v in st["growth"].items()})
lat = out["1240"]["lat"]; rows = (lat < -15) & (lat > -50)
print("rows", np.round(lat[rows],1))
for m in MUS:
    print(m, "cover at ice max", np.round(out[m]["frac_max"][rows],2), "at ice min", np.round(out[m]["frac_min"][rows],2))
