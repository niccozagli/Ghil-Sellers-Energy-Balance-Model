"""Per μ, one at a time: coupling robustness sweep and gyre heat budget."""
import pickle, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from gsebm import plasim_koopman_irregularity as ir

SRC = Path("/Volumes/Nicco/Plasim/extracted"); LOCAL = Path("data/Plasim")
OUT = Path(sys.argv[1]); STAGE = OUT / "stage"; PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
MUS = ["1232p5", "1233p75", "1235", "1237p5", "1240", "1242p5", "1245"]
for mu in MUS:
    target = OUT / f"rob_{mu}.pkl"
    if target.exists(): continue
    name = PREFIX + mu; t = time.time()
    if (LOCAL / name).is_dir():
        root = LOCAL
    else:
        shutil.rmtree(STAGE, ignore_errors=True); (STAGE / name).mkdir(parents=True)
        for f in (SRC / name).glob("*.nc"):
            subprocess.run(["cp", str(f), str(STAGE / name)], check=True)
        root = STAGE; print(f"{mu}: copied {time.time()-t:.0f}s", flush=True)
    base = ir.irregularity_analysis(root, mu)
    coupling = ir.coupling_analysis(base)
    years = base["fields"]["years"]
    budget = ir.gyre_heat_budget(root, mu, years)
    regression = {kind: ir.budget_regression(budget, coupling[kind]["ice"], base["psi1"])
                  for kind in ("raw", "clock removed")}
    # Budget terms regressed on the gyre heat content residual itself (what builds a gyre anomaly).
    content = ir.cycle_residuals(budget["heat_content"][:, None], base["psi1"], None, None)["amplitude"][:, 0]
    regression["gyre heat content"] = ir.budget_regression(budget, content, base["psi1"])
    robustness = ir.coupling_robustness(root, mu, base=base, progress=lambda m: print(m, flush=True))
    pickle.dump({"mu": mu, "budget": {k: v for k, v in budget.items()}, "budget_regression": regression,
                 "robustness": robustness, "coupling_base": {k: coupling[k]["band_peak"] for k in ("raw", "clock removed")}},
                open(target, "wb"))
    shutil.rmtree(STAGE, ignore_errors=True)
    print(f"{mu}: done {time.time()-t:.0f}s", flush=True)
print("ALL DONE", flush=True)
