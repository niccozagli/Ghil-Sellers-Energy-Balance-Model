import pickle, sys
from pathlib import Path
import numpy as np
from gsebm import plasim_koopman_irregularity as ir
J = Path(sys.argv[1]); MUS = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
S = pickle.load(open(J / "phys" / "summary.pkl", "rb")); B = 36; sec = 365.25 * 86400
print("mu   | charging (HC rising) per-cycle anomaly integrals 1e21 J: surf adv resid | discharging: surf adv resid | storage amp TW | k_cycle k_resid (TW per 1e6 km2)")
for mu in MUS:
    p = pickle.load(open(J / "phys" / f"phys_{mu}.pkl", "rb")); s = S[mu]
    psi = p["psi1"]; ph = ir.wrapped_phase(psi); idx = ir.phase_bin_index(ph, B)
    counts = np.bincount(idx, minlength=B); dur = counts / counts.sum() * s["P"] * sec
    comp = {k: s["comp"][k] - np.average(s["comp"][k], weights=counts) for k in ("storage", "surface", "advection", "residual")}
    rising = comp["storage"] > 0
    ch = [np.sum(comp[k][rising] * dur[rising]) * 1e12 / 1e21 for k in ("surface", "advection", "residual")]
    dc = [np.sum(comp[k][~rising] * dur[~rising]) * 1e12 / 1e21 for k in ("surface", "advection", "residual")]
    amp = (comp["storage"].max() - comp["storage"].min()) / 2
    # ice cycle and residual in 1e6 km2
    sic, area = p["sic"], p["area"]; ice = (sic * area[None]).sum((1, 2)) / 1e12
    Ci = ir._group_means(ice[:, None], idx, B)[:, 0]; Ci -= np.average(Ci, weights=counts)
    k_cycle = np.sum(counts * comp["storage"] * Ci) / np.sum(counts * Ci**2)
    res_st = ir.cycle_residuals(p["budget"]["storage"][:, None], psi, None, None)["amplitude"][:, 0]
    res_i = p["ice_resid_clean"]
    # storage residual vs ice residual, best lag 0..4
    ks = [np.sum(res_st[L:] * res_i[:res_i.size - L]) / np.sum(res_i[:res_i.size - L]**2) for L in range(0, 5)]
    print(f"{mu:7s}| {ch[0]:6.2f} {ch[1]:6.2f} {ch[2]:6.2f} | {dc[0]:6.2f} {dc[1]:6.2f} {dc[2]:6.2f} | {amp:5.1f} | {k_cycle:6.1f} {min(ks):6.1f}")
