"""Plain look: annual ice cover of the Atlantic 30.5°S row (and the Indian/Pacific 30.5°S row) in several runs."""

import matplotlib.pyplot as plt
import numpy as np

from common import load

OUT = "../../figures/plasim_fresh"
cases = [("1240", (14000, 14800)), ("1235", (12000, 12800)), ("1232p5", (22000, 22800)), ("1230", (5200, 6000)),
         ("1230", (7000, 7800)), ("1230", (7700, 8500))]
fig, axes = plt.subplots(len(cases), 1, figsize=(13, 13), sharey=True)
for ax, (lab, (a, b)) in zip(axes, cases):
    d = load(lab); y = d["years"]; k = (y >= a) & (y <= b)
    r30 = int(np.argmin(np.abs(d["lat"] + 30.46))); r25 = int(np.argmin(np.abs(d["lat"] + 24.92)))
    ax.plot(y[k], d["sic"][k, r30, 1], color="#2a78d6", lw=0.9, label="Atlantic 30.5°S row")
    ax.plot(y[k], d["sic"][k, r25, 1], color="#2a78d6", lw=0.6, ls="--", label="Atlantic 24.9°S row")
    ip = (d["sic"][k, r30, 2] * d["ocean_counts"][r30, 2] + d["sic"][k, r30, 3] * d["ocean_counts"][r30, 3]) / (d["ocean_counts"][r30, 2] + d["ocean_counts"][r30, 3])
    ax.plot(y[k], ip, color="#1baf7a", lw=0.9, label="Indian/Pacific 30.5°S row")
    ax.set_title(f"μ = {d['mu']:g}, years {a}–{b}", fontsize=9); ax.grid(alpha=0.3, lw=0.4)
    ax.set_ylabel("fraction of year\nice-covered", fontsize=8)
axes[0].legend(fontsize=7, ncol=3, frameon=False)
axes[-1].set_xlabel("model year")
fig.tight_layout(); fig.savefig(f"{OUT}/look_row305.png", dpi=100); plt.close(fig)
print("ok")
