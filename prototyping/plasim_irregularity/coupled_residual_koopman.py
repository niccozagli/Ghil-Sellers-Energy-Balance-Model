import pickle, shutil, subprocess, sys
from pathlib import Path
from gsebm import plasim_koopman_irregularity as ir
SRC = Path("/Volumes/Nicco/Plasim/extracted"); LOCAL = Path("data/Plasim")
OUT = Path(sys.argv[1]); STAGE = OUT / "stage"; PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
for mu in ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]:
    target = OUT / f"cmode_{mu}.pkl"
    if target.exists(): continue
    name = PREFIX + mu
    if (LOCAL / name).is_dir(): root = LOCAL
    else:
        shutil.rmtree(STAGE, ignore_errors=True); (STAGE / name).mkdir(parents=True)
        for f in (SRC / name).glob("*.nc"): subprocess.run(["cp", str(f), str(STAGE / name)], check=True)
        root = STAGE
    r = ir.irregularity_analysis(root, mu); c = ir.coupling_analysis(r, surrogates=20)
    out = {lag: ir.coupled_residual_koopman(r, c, lag=lag) for lag in (1, 3)}
    out["eof15"] = ir.coupled_residual_koopman(r, c, eof_count=15)
    pickle.dump(out, open(target, "wb")); shutil.rmtree(STAGE, ignore_errors=True)
    for key, k in out.items():
        m = k["coupled"]; lead = k["modes"][0] if k["modes"] else None
        print(mu, key, "coupled τ=%.1f P=%.0f R² ice %.2f gyre %.2f" % (m["decay_time"], m["period"], m["ice_r2"], m["gyre_r2"]),
              "| slowest τ=%.1f R² ice %.2f gyre %.2f" % (lead["decay_time"], lead["ice_r2"], lead["gyre_r2"]), flush=True)
print("ALL DONE", flush=True)
