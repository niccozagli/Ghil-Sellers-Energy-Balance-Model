"""Copy each archive locally, check the analysis variables, pick a clean window,
run the irregularity analysis, save results, delete the copy. One μ at a time."""
import json, pickle, shutil, subprocess, sys, time
from pathlib import Path
import h5py, numpy as np
from gsebm import plasim_koopman_irregularity as ir

SRC = Path("/Volumes/Nicco/Plasim/extracted"); LOCAL = Path("data/Plasim")
OUT = Path(sys.argv[1]); STAGE = OUT / "stage"
PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"
MUS = ["1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
NEEDED = ("surface_temperature", "sea_ice_concentration", "temperature_upper")

def clean_years(archive):
    with h5py.File(archive, "r") as f:
        years = f["year"][:]; n = years.size
        good = years == years[0] + np.arange(n)
        for name in NEEDED:
            ds = f[name]
            for a in range(0, n, 10):
                b = min(a + 10, n)
                try:
                    x = ds[a:b]
                except Exception:
                    good[a:b] = False; continue
                x = x.reshape(b - a, -1)
                finite = np.isfinite(x)
                # wet/ocean cells are finite in every good year; compare with the typical count
                good[a:b] &= finite.sum(1) > 0
                if name != "temperature_upper":
                    good[a:b] &= finite.all(1)
    return years, good

def choose_window(years, good, length=4000, minimum=2000):
    first = years[0]; idx = np.flatnonzero(good)
    # longest clean stretch, preferring the latest; cap at `length` from its end
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]; stops = np.r_[idx[breaks], idx[-1]]
    best = max(zip(starts, stops), key=lambda s: (min(s[1] - s[0] + 1, length), s[1]))
    a, b = best; a = max(a, b - length + 1)
    if b - a + 1 < minimum:
        return None
    return int(first + a), int(first + b)

log = open(OUT / "log.txt", "a")
def say(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)

for mu in MUS:
    target = OUT / f"result_{mu}.pkl"
    if target.exists(): continue
    name = PREFIX + mu; t = time.time()
    local = LOCAL / name
    if local.is_dir():
        root = LOCAL
    else:
        shutil.rmtree(STAGE, ignore_errors=True); (STAGE / name).mkdir(parents=True)
        for f in (SRC / name).glob("*.nc"):
            subprocess.run(["cp", str(f), str(STAGE / name)], check=True)
        root = STAGE
        say(f"{mu}: copied in {time.time()-t:.0f}s")
    years, good = clean_years(root / name / f"{name}_spinup_raw_maps.nc")
    bad = np.flatnonzero(~good)
    bad_ranges = []
    if bad.size:
        br = np.flatnonzero(np.diff(bad) > 1)
        for s, e in zip(np.r_[0, br + 1], np.r_[br, bad.size - 1]):
            bad_ranges.append((int(years[0] + bad[s]), int(years[0] + bad[e])))
    window = choose_window(years, good)
    say(f"{mu}: years {years[0]}-{years[-1]}, bad year ranges {bad_ranges}, window {window}")
    entry = {"mu": mu, "years": (int(years[0]), int(years[-1])), "bad_ranges": bad_ranges, "window": window}
    if window:
        try:
            r = ir.irregularity_analysis(root, mu, window=window)
            entry |= {
                "summary": r["summary"], "block_summaries": r["block_summaries"],
                "headline": ir.headline_metrics(r), "period": r["period"],
                "psi1": r["psi1"], "fields_years": r["fields"]["years"],
                "band_residuals": r["band_residuals"], "floquet": r["floquet"],
                "residual_koopman": {k: r["residual_koopman"][k] for k in ("rates", "clock_share", "residual_share", "transverse", "lag")},
                "episodes": ir.weak_episodes(r["psi1"], r["fields"]["years"])["episodes"],
            }
            say(f"{mu}: done, period {r['period']:.1f} yr, {time.time()-t:.0f}s")
        except Exception as error:
            entry["error"] = repr(error); say(f"{mu}: analysis failed: {error!r}")
    pickle.dump(entry, open(target, "wb"))
    shutil.rmtree(STAGE, ignore_errors=True)
say("ALL DONE")
