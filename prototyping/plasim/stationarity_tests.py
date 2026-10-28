"""Stationarity tests of one PlaSim-LSG run over a window of years.

Question: is a run stationary over the window [start, end of run], judged
indicator by indicator? The working definition and its history are in
PLASIM_INFO/STATIONARITY.md; this script only computes the numbers.

Input: the run's indicator product in data/Plasim/indicators/<run>/ (zonal
and sections files; definitions in Plasim_Indicators.md §4). Annual values,
no smoothing or detrending beyond the linear fits named below. Time is in
years since the run's first year of output.

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
from gsebm.plasim_indicators import area_mean, area_total, load_indicators, product_stem, volume_mean

BANDS = {"0-700 m": slice(0, 10), "700-2025 m": slice(10, 17), "2025 m-bottom": slice(17, 22)}


def load_run(run: str):
    """Zonal and sections products of a run (phase stationary for the base run)."""
    phase = "spinup" if "_MU_" in run else "stationary"
    folder = get_data_dir(create=False) / "Plasim" / "indicators" / run
    zonal = load_indicators(folder / f"{product_stem(run, phase)}_zonal.nc")
    sections = load_indicators(folder / f"{product_stem(run, phase)}_sections.nc")
    return zonal, sections


def indicators(zonal, sections) -> dict[str, np.ndarray]:
    """Annual global and hemispheric indicators of Plasim_Indicators.md §4."""
    wet_area = zonal["lsg_row_wet_area"]
    lat = sections["lsg_lat"]
    t, s = sections["t_section"], sections["s_section"]
    series = {
        "tas global": area_mean(zonal["tas_zonal"], zonal),
        "TOA net": area_mean(zonal["rst_zonal"] + zonal["rlut_zonal"], zonal),
        "surface net": area_mean(zonal["surface_net_flux_zonal"], zonal),
        "ice area NH": area_total(zonal["sic_ocean_zonal"], zonal, "nh") / 1e12,
        "ice area SH": area_total(zonal["sic_ocean_zonal"], zonal, "sh") / 1e12,
        **{f"T {band}": volume_mean(t, sections, levels) for band, levels in BANDS.items()},
        "T deep NH": volume_mean(t, sections, BANDS["2025 m-bottom"], "nh"),
        "T deep SH": volume_mean(t, sections, BANDS["2025 m-bottom"], "sh"),
        **{f"S {band}": volume_mean(s, sections, levels) for band, levels in BANDS.items()},
        "S Atlantic 0-700 m": volume_mean(sections["s_atlantic_section"], sections, BANDS["0-700 m"],
                                          atlantic=True),
        "AMOC 46-66N": sections["psi_atlantic"].where((lat >= 46) & (lat <= 66)).max(("lsg_level", "lsg_lat")),
        "ACC": zonal["acc"],
        "sea level": (zonal["zeta_zonal"] * wet_area).sum("lsg_lat") / wet_area.sum(),
    }
    return {name: np.asarray(values.values, dtype=float) for name, values in series.items()}


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

    series = indicators(*load_run(args.run))
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
