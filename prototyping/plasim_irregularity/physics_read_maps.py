"""Physical diagnostics per μ from saved ψ1 and budget + sea-ice maps (read once)."""
import pickle, sys
from pathlib import Path
import h5py, numpy as np
from gsebm import plasim_koopman_irregularity as ir
J = Path(sys.argv[1]); OUT = J / "phys"
PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
MUS = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
for mu in MUS:
    target = OUT / f"phys_{mu}.pkl"
    if target.exists(): continue
    c = pickle.load(open(J / "coup" / f"coupling_{mu}.pkl", "rb"))
    b = pickle.load(open(J / "rob" / f"rob_{mu}.pkl", "rb"))["budget"]
    psi, years = c["psi1"], c["years"]
    root = Path("data/Plasim") if (Path("data/Plasim") / (PREFIX + mu)).is_dir() else Path("/Volumes/Nicco/Plasim/extracted")
    d = root / (PREFIX + mu)
    with h5py.File(d / f"{PREFIX}{mu}_spinup_raw_maps.nc", "r") as f, h5py.File(d / f"{PREFIX}{mu}_spinup_basin_masks.nc", "r") as m:
        y = f["year"][:]; first = int(np.flatnonzero(y == years[0])[0])
        sic = np.asarray(f["sea_ice_concentration"][first:first + years.size], dtype=np.float32)
        ts = np.asarray(f["surface_temperature"][first:first + years.size], dtype=np.float32)
        lat = f["t21_lat"][:]; w = f["t21_gaussian_weight"][:]; basin = m["t21_south_atlantic"][:].astype(bool)
    area = 6.371e6**2 * 2 * np.pi / 64 * w[:, None] * basin
    pickle.dump({"mu": mu, "psi1": psi, "years": years, "period": c["period"], "budget": b,
                 "sic": sic * basin[None], "ts_rows": np.array([(ts[:, j][:, basin[j]].mean(1) if basin[j].any() else np.full(years.size, np.nan)) for j in range(lat.size)]).T,
                 "lat": lat, "area": area, "basin": basin, "ice_resid_clean": c["coupling"]["clock removed"]["ice"],
                 "ocean_lat": c["ocean_lat"]}, open(target, "wb"))
    print(mu, "done", flush=True)
print("ALL DONE")
