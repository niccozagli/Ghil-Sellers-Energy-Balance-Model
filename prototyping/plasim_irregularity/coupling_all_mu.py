"""Per μ (one at a time, local copy when needed): irregularity + coupling analysis
and lagged regression maps on the ice residual. Saves compact results."""
import pickle, shutil, subprocess, sys, time
from pathlib import Path
import h5py, numpy as np
from gsebm import plasim_koopman_irregularity as ir
from gsebm.plasim_koopman_single import load_map_fields

SRC = Path("/Volumes/Nicco/Plasim/extracted"); LOCAL = Path("data/Plasim")
OUT = Path(sys.argv[1]); STAGE = OUT / "stage"; PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
MUS = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
MAP_LAGS = np.array([-4, 0, 2, 4, 6, 8])

def say(*a):
    print(*a, flush=True)

for mu in MUS:
    target = OUT / f"coupling_{mu}.pkl"
    if target.exists(): continue
    name = PREFIX + mu; t = time.time()
    if (LOCAL / name).is_dir():
        root = LOCAL
    else:
        shutil.rmtree(STAGE, ignore_errors=True); (STAGE / name).mkdir(parents=True)
        for f in (SRC / name).glob("*.nc"):
            subprocess.run(["cp", str(f), str(STAGE / name)], check=True)
        root = STAGE; say(f"{mu}: copied {time.time()-t:.0f}s")
    r = ir.irregularity_analysis(root, mu)
    c = ir.coupling_analysis(r)
    years = r["fields"]["years"]; psi = r["psi1"]
    ice_raw = c["raw"]["ice"]; ice_clean = c["clock removed"]["ice"]
    maps = load_map_fields(root, mu, years)
    regressions = {}
    for field_name, field in maps["fields"].items():
        residual = ir.cycle_residuals(field["scale"] * field["anomaly"], psi, None, None)["amplitude"]
        regressions[field_name] = {
            "raw": ir.lagged_regression_maps(ice_raw, residual, MAP_LAGS).astype(np.float32),
            "clock removed": ir.lagged_regression_maps(ice_clean, residual, MAP_LAGS).astype(np.float32),
            "label": field["label"], "unit": field["unit"], "grid": field["grid"], "wet": field["wet"],
        }
        del residual
    flux = sum(ir.load_absolute_map(root, mu, years, n) for n in ("rss", "rls", "hfss", "hfls"))
    flux -= flux.mean(axis=0)
    residual = ir.cycle_residuals(flux, psi, None, None)["amplitude"]
    regressions["net_surface_flux"] = {
        "raw": ir.lagged_regression_maps(ice_raw, residual, MAP_LAGS).astype(np.float32),
        "clock removed": ir.lagged_regression_maps(ice_clean, residual, MAP_LAGS).astype(np.float32),
        "label": "Net surface heat flux (down +)", "unit": "W m⁻²", "grid": "t21",
        "wet": np.ones(flux.shape[1:], bool),
    }
    del flux, residual, maps["fields"]
    entry = {
        "mu": mu, "window": ir.ANALYSIS_WINDOWS[mu], "period": r["period"], "years": years,
        "psi1": psi, "coupling": c, "map_lags": MAP_LAGS, "regressions": regressions,
        "grid": {k: maps[k] for k in ("t21_lat", "t21_lon", "lsm", "lsg_lat", "lsg_lon", "t21_basin", "lsg_basin")},
        "ocean_lat": r["fields"]["ocean_lat"], "surface_lat": r["fields"]["surface_lat"],
        "ocean_residual": r["blocks"]["ocean"]["residuals"]["amplitude"].astype(np.float32),
        "ocean_weights": r["fields"]["ocean_weights"],
        "transverse_pattern": r["residual_koopman"].get("patterns", {}).get("ocean"),
        "eof1_ocean": r["eofs"]["patterns"][1][0],
    }
    pickle.dump(entry, open(target, "wb"))
    shutil.rmtree(STAGE, ignore_errors=True)
    raw, clean = c["raw"], c["clock removed"]
    say(f"{mu}: band raw {raw['band_peak']} thr {raw['band_threshold']:.2f} | clean {clean['band_peak']} thr {clean['band_threshold']:.2f} | "
        f"peak raw {raw['peak']} clean {clean['peak']} | clock expl ice {c['ice variance explained by clock error']:.2f} | {time.time()-t:.0f}s")
say("ALL DONE")
