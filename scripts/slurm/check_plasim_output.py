"""Read-only check of PlaSim spinup output blocks and restart files on the cluster.

For each run label it lists output files whose time length differs from the
year range in their name (or that cannot be opened), and the newest files in
``restart/spinup``. Run from the repository clone on a login node:

    .venv/bin/python scripts/slurm/check_plasim_output.py --experiments-root <experiments> 1231p5 1230p5_bis
    .venv/bin/python scripts/slurm/check_plasim_output.py --experiments-root <experiments> 1231p5 --all
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import xarray as xr

BLOCK = re.compile(r"\.(\d+)-(\d+)\.nc$")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("runs", nargs="+", help="Labels after _MU_, e.g. 1231p5 1235_cold.")
parser.add_argument("--experiments-root", type=Path, required=True, help="Directory of experiment folders.")
parser.add_argument("--prefix", default="CONTROL_360ppm_T21L10_10000Y_MU_")
parser.add_argument("--last-decades", type=int, default=60, help="Number of newest blocks to check.")
parser.add_argument("--all", action="store_true", help="Check every block.")
parser.add_argument("--restarts", type=int, default=12, help="Number of newest restart files to list.")
args = parser.parse_args()

for run in args.runs:
    experiment = args.experiments_root / f"{args.prefix}{run}"
    files = sorted((experiment / "output" / "spinup").glob("*_*.*-*.nc"),
                   key=lambda p: int(BLOCK.search(p.name).group(1)))
    if not args.all:
        files = files[-4 * args.last_decades:]
    bad = []
    for path in files:
        first, last = map(int, BLOCK.search(path.name).groups())
        try:
            with xr.open_dataset(path, decode_times=False) as dataset:
                n = dataset.sizes.get("time", -1)
        except Exception as error:
            n = f"UNREADABLE ({type(error).__name__})"
        if n != last - first + 1:
            bad.append(f"    {path.name}: time={n}")
    print(f"== {run}: checked {len(files)} output files, {len(bad)} bad", flush=True)
    for line in bad:
        print(line, flush=True)

    restart = experiment / "restart" / "spinup"
    entries = sorted(restart.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if restart.is_dir() else []
    print(f"  restart/spinup: {len(entries)} files, newest:")
    for path in entries[:args.restarts]:
        stat = path.stat()
        print(f"    {path.name:60s} {stat.st_size:>10d}  {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}")
    print(flush=True)
