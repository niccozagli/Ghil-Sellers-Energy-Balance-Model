"""Stationarity tests of one PlaSim-LSG run over a window of years.

Question: is a run stationary over the window [start, end of run], judged
indicator by indicator? The working definition and its history are in
PLASIM_INFO/STATIONARITY.md; this script only computes the numbers.

Input: the run's series file in data/Plasim/indicators/<run>/ (written on
JASMIN by scripts/slurm/reduce_plasim_runs.py; definitions in
Plasim_Indicators.md §4 and §6). Annual values, no smoothing or detrending
beyond the linear fits named below. Time is in years since the run's first
year of output.

Method, per indicator, over the window of length L:
- block slopes: the window is cut into consecutive blocks of B years
  (several B); a straight line is fitted in each block. Reported: the mean
  slope (per 1000 years), t = mean / (sd / sqrt(K)) over the K blocks, and
  how many slopes are positive. Block lengths giving fewer than 3 blocks are
  skipped.
- drift bound: the largest drift (per 1000 years) the window cannot exclude,
  max over B of |mean slope| + 2 sd / sqrt(K).
- split half: difference of the means of the two halves, in standard errors
  corrected for autocorrelation (z).
- tau: integral autocorrelation time of the residuals about the fit,
  tau = (1 + 2 sum rho_k) / 2 years, summed up to the first rho_k <= 0;
  L / tau is the number of independent stretches in the window.

Run from the repository root:

    PYTHONPATH=src uv run python prototyping/plasim/stationarity_tests.py \
        CONTROL_360ppm_T21L10_10000Y_MU_1245 --start 3000 [--blocks 1000 1500 2000]
"""
import argparse

import numpy as np

from gsebm.paths import get_data_dir
from gsebm.plasim_indicators import load_indicators, product_phase, series_path

# Name in the output -> variable of the series file (Plasim_Indicators.md §6).
SERIES = {
    "tas global": "tas_global", "TOA net": "toa_net", "surface net": "surface_net",
    "ice area NH": "ice_area_nh", "ice area SH": "ice_area_sh",
    "T 0-700 m": "t_0_700", "T 700-2025 m": "t_700_2025", "T 2025 m-bottom": "t_2025_bottom",
    "T deep NH": "t_deep_nh", "T deep SH": "t_deep_sh",
    "S 0-700 m": "s_0_700", "S 700-2025 m": "s_700_2025", "S 2025 m-bottom": "s_2025_bottom",
    "S Atlantic 0-700 m": "s_atlantic_0_700", "AMOC 46-66N": "amoc_46_66n", "ACC": "acc", "sea level": "sea_level",
}


def indicators(run: str) -> dict[str, np.ndarray]:
    """Annual global, hemispheric and depth-band indicators of a run, from its series file."""
    folder = get_data_dir(create=False) / "Plasim" / "indicators" / run
    series = load_indicators(series_path(folder, run, product_phase(folder)))
    return {name: np.asarray(series[variable].values, dtype=float) for name, variable in SERIES.items()}


def linear_fit(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Slope per year of a straight-line fit, and the residuals about it."""
    years = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(years, x, 1)
    return slope, x - (intercept + slope * years)


def variance_factor(residuals: np.ndarray) -> float:
    """g = 1 + 2 sum rho_k up to the first rho_k <= 0, so var(mean) = sigma^2 g / n."""
    r = residuals - residuals.mean()
    n = len(r)
    spectrum = np.fft.rfft(r, 2 * n)
    rho = np.fft.irfft(spectrum * np.conj(spectrum))[:n] / (r @ r)
    g = 1.0
    for k in range(1, n // 4):
        if rho[k] <= 0:
            break
        g += 2 * rho[k]
    return g


def window_tests(x: np.ndarray) -> dict[str, float]:
    """Split-half and autocorrelation numbers of one window."""
    n = len(x)
    _, residuals = linear_fit(x)
    sigma = residuals.std(ddof=2)
    g = variance_factor(residuals)
    first, second = x[: n // 2], x[n // 2:]
    standard_error = sigma * np.sqrt(g / len(first) + g / len(second))
    return {
        "L": n,
        "tau": g / 2,
        "z": (second.mean() - first.mean()) / standard_error,
    }


def block_slopes(x: np.ndarray, block: int) -> dict[str, float]:
    """Slopes (per 1000 years) of consecutive blocks of ``block`` years."""
    count = len(x) // block
    slopes = np.array([linear_fit(x[i * block:(i + 1) * block])[0] * 1000 for i in range(count)])
    standard_error = slopes.std(ddof=1) / np.sqrt(count)
    return {
        "mean": slopes.mean(),
        "standard error": standard_error,
        "t": slopes.mean() / standard_error,
        "positive": int((slopes > 0).sum()),
        "K": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="experiment name, e.g. CONTROL_360ppm_T21L10_10000Y_MU_1245")
    parser.add_argument("--start", type=int, default=0, help="first year of the window, in years since start")
    parser.add_argument("--blocks", type=int, nargs="+", default=[1000, 1500, 2000],
                        help="block lengths in years")
    args = parser.parse_args()

    series = indicators(args.run)
    length = len(next(iter(series.values())))
    print(f"{args.run}: window {args.start}-{length - 1} years since start ({length - args.start} years)")
    print("drift bound per 1000 yr | block slopes: mean per 1000 yr, t over blocks, positive slopes / blocks")
    for name, x in series.items():
        window = x[args.start:]
        tests = window_tests(window)
        blocks, bound = [], 0.0
        for block in args.blocks:
            if len(window) // block < 3:
                continue
            b = block_slopes(window, block)
            bound = max(bound, abs(b["mean"]) + 2 * b["standard error"])
            blocks.append(f"B{block}: {b['mean']:+.4f} t={b['t']:+.2f} {b['positive']}/{b['K']}")
        print(f"{name:18s} tau={tests['tau']:6.1f} L/tau={tests['L'] / tests['tau']:5.0f} "
              f"z={tests['z']:+.2f} bound={bound:.4f} | "
              + " | ".join(blocks))


if __name__ == "__main__":
    main()
