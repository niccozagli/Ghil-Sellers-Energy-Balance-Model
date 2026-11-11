"""Years in which the incoming solar of a PlaSim-LSG run differs from its usual value.

Question: the run check of MU_1230 gives an incoming-solar range of
0.024 W m-2, against 0.0007-0.0035 in the other ALICE runs checked on
2026-11-11, although incoming solar should be the same in every year
(Plasim_Indicators.md §5). Which years differ, and by how much?

Input: one run's zonal product, <run>_<phase>_indicators_zonal.nc (on JASMIN,
in extracted/indicators/<run>/), all its years.

Method: incoming solar S = global area mean (Gaussian weights) of
rst_zonal - rsut_zonal per year, as in run_checks of gsebm.plasim_indicators.
Deviation of each year from the run's median S. Printed: the median, the
range, the number of years with |deviation| above a threshold (default
0.003 W m-2), and those years (label, model year, S, deviation), at most 50.

Run from the repository root (on a JASMIN sci server, after module load jaspy):

    PYTHONPATH=src python prototyping/plasim/incoming_solar_years.py <zonal file> [--threshold 0.003]
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from gsebm.plasim_indicators import area_mean


def incoming_solar(zonal: xr.Dataset) -> xr.DataArray:
    """Global-mean incoming solar per year (W m-2): area mean of rst - rsut."""
    return area_mean(zonal["rst_zonal"] - zonal["rsut_zonal"], zonal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zonal", type=Path, help="the run's ..._indicators_zonal.nc")
    parser.add_argument("--threshold", type=float, default=0.003, help="|deviation| in W m-2")
    args = parser.parse_args()

    zonal = xr.open_dataset(args.zonal)
    solar = incoming_solar(zonal).load()
    median = float(solar.median())
    deviation = solar - median
    odd = np.flatnonzero(np.abs(deviation.values) > args.threshold)
    labels, model_years = zonal["year"].values, zonal["model_year"].values
    print(f"{args.zonal.name}: {solar.sizes['year']} years, labels {labels[0]}-{labels[-1]}")
    print(f"median {median:.4f} W m-2, range {float(solar.max() - solar.min()):.4f}, "
          f"{len(odd)} year(s) with |deviation| > {args.threshold}")
    for index in odd[:50]:
        print(f"  label {labels[index]}  model year {model_years[index]}  "
              f"S {float(solar[index]):.4f}  deviation {float(deviation[index]):+.4f}")


if __name__ == "__main__":
    main()
