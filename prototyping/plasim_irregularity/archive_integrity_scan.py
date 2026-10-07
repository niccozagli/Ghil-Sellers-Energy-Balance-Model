"""Read-only integrity scan of PlaSim raw-map archives (scratch tool)."""
import json, sys, time
from pathlib import Path
import h5py, numpy as np

root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Volumes/Nicco/Plasim/extracted"); out = Path(sys.argv[1])
for directory in sorted(root.iterdir()):
    if not directory.is_dir(): continue
    name = directory.name; target = out / f"integrity_{name}.json"
    if target.exists(): continue
    archive = directory / f"{name}_spinup_raw_maps.nc"; masks = directory / f"{name}_spinup_basin_masks.nc"
    report = {"experiment": name, "problems": [], "variables": {}}
    t = time.time()
    try:
        with h5py.File(masks, "r") as m: report["mask_variables"] = len(m.keys())
    except Exception as e: report["problems"].append(f"mask file unreadable: {e!r}")
    try:
        f = h5py.File(archive, "r")
    except Exception as e:
        report["problems"].append(f"archive unreadable: {e!r}"); target.write_text(json.dumps(report, indent=1)); continue
    with f:
        attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else str(v)) for k, v in f.attrs.items()}
        report["attrs"] = {k: attrs[k] for k in ("first_year", "last_year", "committed_last_year", "committed_blocks", "experiment_state") if k in attrs}
        years = f["year"][:]; n = years.size
        expected = years[0] + np.arange(n) if n else years
        bad_label = np.flatnonzero(years != expected)
        report["years"] = [int(years[0]), int(years[-1]), int(n)]
        if bad_label.size:
            report["problems"].append(f"{bad_label.size} year labels differ from first+index, e.g. index {bad_label[:5].tolist()} -> {years[bad_label[:5]].tolist()}")
        bad_years = {}  # year index -> set of reasons
        for var, ds in f.items():
            if not isinstance(ds, h5py.Dataset) or ds.ndim == 0 or ds.shape[0] != n or var == "year": continue
            step = ds.chunks[0] if ds.chunks else 10
            nan_counts = np.full(n, -1); fills = np.zeros(n, int); zeros = np.zeros(n, bool); unreadable = []
            for a in range(0, n, step):
                b = min(a + step, n)
                try:
                    x = ds[a:b]
                except Exception as e:
                    unreadable.append([int(expected[a]), int(expected[b - 1])]); continue
                x = x.reshape(b - a, -1).astype(float)
                nan_counts[a:b] = np.isnan(x).sum(1)
                fills[a:b] = (np.abs(np.nan_to_num(x)) > 9e36).sum(1)
                zeros[a:b] = np.all(np.nan_to_num(x) == 0, axis=1)
            readable = nan_counts >= 0
            typical = int(np.median(nan_counts[readable])) if readable.any() else -1
            odd_nan = np.flatnonzero(readable & (nan_counts != typical))
            odd_fill = np.flatnonzero(fills > 0)
            all_zero = np.flatnonzero(zeros & readable) if not np.all(zeros[readable]) else np.array([], int)
            info = {}
            if unreadable: info["unreadable_year_ranges"] = unreadable
            if odd_nan.size: info["nan_count_differs_years"] = int(odd_nan.size)
            if odd_fill.size: info["fill_value_years"] = int(odd_fill.size)
            if all_zero.size: info["all_zero_years"] = int(all_zero.size)
            if info: report["variables"][var] = info
            for idx in odd_nan: bad_years.setdefault(int(idx), set()).add("nan")
            for idx in odd_fill: bad_years.setdefault(int(idx), set()).add("fill")
            for idx in all_zero: bad_years.setdefault(int(idx), set()).add("zero")
            for a0, b0 in unreadable:
                for yy in range(a0, b0 + 1): bad_years.setdefault(int(yy - expected[0]), set()).add("unreadable")
        idx = np.array(sorted(bad_years), int)
        ranges = []
        if idx.size:
            starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]; stops = np.r_[starts[1:], idx.size]
            for s, e in zip(starts, stops):
                reasons = sorted(set().union(*(bad_years[i] for i in idx[s:e])))
                ranges.append([int(expected[idx[s]]), int(expected[idx[e - 1]]), reasons])
        report["bad_year_ranges"] = ranges
    report["seconds"] = round(time.time() - t, 1)
    target.write_text(json.dumps(report, indent=1))
    print(name, report["years"], "bad ranges:", ranges[:20], "problems:", report["problems"], flush=True)
