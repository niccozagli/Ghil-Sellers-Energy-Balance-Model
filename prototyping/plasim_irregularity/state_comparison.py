"""Per μ: compare Koopman states (Ts+θ, ice+θ, ice+Ts+θ); coupling and slow-mode search."""
import pickle, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from gsebm import plasim_koopman_irregularity as ir
SRC = Path("/Volumes/Nicco/Plasim/extracted"); LOCAL = Path("data/Plasim")
OUT = Path(sys.argv[1]); STAGE = OUT / "stage"; PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
STATES = {"Ts+θ": ("surface", "ocean"), "ice+θ": ("ice", "ocean"), "ice+Ts+θ": ("ice", "surface", "ocean")}
for mu in ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]:
    target = OUT / f"state_{mu}.pkl"
    if target.exists(): continue
    name = PREFIX + mu; t = time.time()
    if (LOCAL / name).is_dir(): root = LOCAL
    else:
        shutil.rmtree(STAGE, ignore_errors=True); (STAGE / name).mkdir(parents=True)
        for f in (SRC / name).glob("*.nc"): subprocess.run(["cp", str(f), str(STAGE / name)], check=True)
        root = STAGE
    entry = {}; base = None
    for label, blocks in STATES.items():
        r = ir.irregularity_analysis(root, mu, state_blocks=blocks)
        c = ir.coupling_analysis(r, surrogates=100)
        m = ir.coupled_mode_search(r, c)
        if base is None: base = r["psi1"]
        lam = r["rates"][r["leading_index"]]
        entry[label] = {
            "period": r["period"], "leading_rate": lam, "rank": r["rank"],
            "psi1_agreement": float(abs(np.mean(r["psi1"] * np.conj(base)))),
            "ice_fractions": r["ice_residuals"]["fractions"],
            "band": c["clock removed"]["band_peak"], "threshold": c["clock removed"]["band_threshold"],
            "ice_leading": c["clock removed"]["ice_leading"],
            "clock_share": c["ice variance explained by clock error"],
            "slow": {k: m[k] for k in ("rates", "ice_r2", "gyre_r2", "joint_r2", "patterns")},
            "headline": ir.headline_metrics(r),
        }
        print(mu, label, f"P={r['period']:.1f} agree={entry[label]['psi1_agreement']:.2f} band={entry[label]['band']} joint={m['joint_r2']}", flush=True)
    pickle.dump(entry, open(target, "wb")); shutil.rmtree(STAGE, ignore_errors=True)
print("ALL DONE", flush=True)
