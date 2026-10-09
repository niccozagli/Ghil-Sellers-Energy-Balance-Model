"""Plain look: annual records after each μ step, by parent run. No smoothing."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from common import PARENT, STEP_YEAR, equivalent_edge, global_ts, load, lsg_layer_mean, south_edge

OUT = "../../figures/plasim_fresh"
GROUPS = {
    "from_1265": ["1250", "1245", "1240", "1235", "1230", "1225"],
    "from_1240": ["1240", "1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5"],
    "from_1367_and_1230": ["1312", "1288", "1265", "1228p5"],
}
norm = colors.Normalize(1222, 1316)
cmap = plt.get_cmap("viridis")

for name, labels in GROUPS.items():
    fig, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
    for lab in labels:
        d = load(lab)
        parent, year0 = PARENT[lab][0], STEP_YEAR[lab]
        if name == "from_1240" and lab == "1240":
            year0 = 14999  # 1240 continued: zero step
        t = d["years"] - year0
        keep = t > -200 if lab == "1240" and name == "from_1240" else np.ones_like(t, bool)
        c = cmap(norm(d["mu"]))
        kw = dict(lw=0.4, color=c, label=f"{d['mu']:g}")
        axes[0].plot(t[keep], global_ts(d)[keep], **kw)
        axes[1].plot(t[keep], south_edge(d)[keep], **kw)
        axes[2].plot(t[keep], equivalent_edge(d)[keep], **kw)
        axes[3].plot(t[keep], lsg_layer_mean(d, 0, -90, 0)[keep], **kw)
        axes[4].plot(t[keep], lsg_layer_mean(d, 1, -90, 0)[keep], **kw)
        axes[5].plot(t[keep], lsg_layer_mean(d, 2, -90, 90)[keep], **kw)
    titles = ["global Ts (K)", "S edge, 0.5 of zonal ocean cover (°S)",
              "S equivalent-area edge (°S)", "SH ocean θ 0–700 m (K)",
              "SH ocean θ 700–2025 m (K)", "global ocean θ > 2025 m (K)"]
    for ax, ti in zip(axes, titles):
        ax.set_ylabel(ti, fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
    if name != "from_1367_and_1230":
        axes[0].set_ylim(250, 272)
        axes[1].set_ylim(18, 45)
        axes[2].set_ylim(18, 45)
    axes[1].invert_yaxis(); axes[2].invert_yaxis()
    axes[0].legend(ncol=6, fontsize=7, title="μ (annual values)", title_fontsize=7)
    axes[-1].set_xlabel("years since the μ step")
    fig.suptitle(f"Annual records, runs {name.replace('_', ' ')} (plain annual values)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/transients_{name}.png", dpi=110)
    plt.close(fig)
print("done")
