"""Plain look at residual drift in the last warm runs: 500-yr block means (no fitting).

Quantities: SH 700–2025 m θ, Southern ice area, global Ts, from the cached annual series.
Run from the repository root with PYTHONPATH=src.
"""

import sys

import numpy as np

sys.path.insert(0, "prototyping/plasim_fresh")
from common import global_ts, load, lsg_layer_mean, south_ice_area  # noqa: E402

for lab in ["1235", "1233p75", "1232p5"]:
    d = load(lab)
    y = d["years"]
    H = lsg_layer_mean(d, 1, -90, 0)
    ice = south_ice_area(d) / 1e12
    ts = global_ts(d)
    ok = np.isfinite(H)
    end = int(y[ok][-1])
    print(f"{lab}: years {int(y[0])}–{end}")
    for a in range(int(y[0]), end + 1, 500):
        m = (y >= a) & (y < a + 500) & ok
        if m.sum() < 200:
            continue
        print(f"  {a}-{a + 499} ({m.sum():3d} yr): H {H[m].mean():.3f} K  "
              f"S ice {ice[m].mean():6.1f}e12 m²  Ts {ts[m].mean():.2f} K")
