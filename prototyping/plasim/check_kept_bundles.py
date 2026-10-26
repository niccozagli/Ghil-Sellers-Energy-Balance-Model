"""Check the newest kept restart bundles and output blocks for write damage.

Question: after the 17 October 2026 scratch failure, are the newest restart
bundles and output decades that were kept in each run intact, so that the
run can restart from them?

Inputs: for each RUN:YEAR (YEAR = label of the newest kept bundle), in
<experiments>/CONTROL_360ppm_T21L10_10000Y_MU_<RUN>/:
restart/spinup/*_{REST,LSGREST}.{YEAR-10,YEAR}.12 and
output/spinup/*_{PLA,OCE,ICE,LSG}.{YEAR-19}-{YEAR-10}.nc, {YEAR-9}-{YEAR}.nc,
each compared with the same files 100 years earlier (YEAR-100).

Method: restarts are read record by record (Fortran sequential, little
endian); reported are truncation, the record-length layout against the
reference, and the number of records that are entirely zero bytes. Output
blocks: the floating-point variables with two or more dimensions; reported
are missing or extra variables against the reference and variables whose
shape or NaN count differ, or whose count of exact zeros differs by more
than 1% of their size. Read-only.

Run on the cluster from the repository clone:

    .venv/bin/python prototyping/plasim/check_kept_bundles.py \\
        ~/plasim-workspace/experiments 1230p5:28539 1231p5_bis:29669
"""
import argparse
from pathlib import Path
import struct

import h5py
import numpy as np

PREFIX = "CONTROL_360ppm_T21L10_10000Y_MU_"


def records(path: Path) -> tuple[list[tuple[int, bool]], str]:
    """Return (length, all-zero) per record and 'ok' or 'BROKEN'."""
    out = []
    with path.open("rb") as file:
        while True:
            head = file.read(4)
            if len(head) < 4:
                return out, "ok"
            (size,) = struct.unpack("<i", head)
            data = file.read(size)
            if len(data) != size or file.read(4) != head:
                return out, "BROKEN"
            out.append((size, data.count(0) == size))


def variables(path: Path) -> dict[str, tuple[tuple[int, ...], int, int]]:
    """Return shape, zero count and NaN count of each float variable."""
    out = {}
    with h5py.File(path, "r") as file:
        for name, item in file.items():
            if not isinstance(item, h5py.Dataset):
                continue
            if item.dtype.kind != "f" or item.ndim < 2:
                continue
            data = item[()]
            zeros = int((data == 0).sum())
            nans = int(np.isnan(data).sum())
            out[name] = (data.shape, zeros, nans)
    return out


def check_restarts(restart: Path, stem: str, year: int) -> None:
    for kind in ("REST", "LSGREST"):
        ref, _ = records(restart / f"{stem}_{kind}.{year - 100}.12")
        ref_layout = [size for size, _ in ref]
        ref_zero = sum(zero for _, zero in ref)
        for label in (year - 10, year):
            rec, status = records(restart / f"{stem}_{kind}.{label}.12")
            same = [size for size, _ in rec] == ref_layout
            zero = sum(zero for _, zero in rec)
            print(f"  {kind}.{label}: {status}, {len(rec)} records, "
                  f"layout {'same as' if same else 'DIFFERS from'} "
                  f"{year - 100}, all-zero records {zero} (ref {ref_zero})")


def differing(cur: dict, ref: dict) -> list[str]:
    out = []
    for name in sorted(set(cur) & set(ref)):
        shape, zeros, nans = cur[name]
        rshape, rzeros, rnans = ref[name]
        limit = 0.01 * np.prod(shape)
        if shape != rshape or nans != rnans or abs(zeros - rzeros) > limit:
            out.append(f"{name} zeros {zeros}/{rzeros} nan {nans}/{rnans}")
    return out


def check_outputs(output: Path, stem: str, year: int) -> None:
    for stream in ("PLA", "OCE", "ICE", "LSG"):
        ref_name = f"{stem}_{stream}.{year - 109}-{year - 100}.nc"
        ref = variables(output / ref_name)
        for last in (year - 10, year):
            cur = variables(output / f"{stem}_{stream}.{last - 9}-{last}.nc")
            missing = sorted(set(ref) - set(cur))
            extra = sorted(set(cur) - set(ref))
            odd = differing(cur, ref)
            print(f"  {stream} {last - 9}-{last}: {len(cur)} variables, "
                  f"missing {missing}, extra {extra}, differing {odd[:6]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", type=Path)
    parser.add_argument("runs", nargs="+", help="RUN:YEAR, e.g. 1230p5:28539")
    args = parser.parse_args()
    for spec in args.runs:
        run, year = spec.split(":")
        stem = f"{PREFIX}{run}"
        experiment = args.experiments / stem
        print(f"== {run} (newest kept bundle {year})", flush=True)
        check_restarts(experiment / "restart" / "spinup", stem, int(year))
        check_outputs(experiment / "output" / "spinup", stem, int(year))


if __name__ == "__main__":
    main()
