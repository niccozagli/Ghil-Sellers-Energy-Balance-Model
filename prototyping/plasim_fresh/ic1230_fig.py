"""Plain look: the μ = 1230 run from its start (the 1265 state at year 4499) to the jump, against settled states."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import equivalent_edge, global_ts, load, lsg_layer_mean, south_ice_area

TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
ST = json.load(open(TMP + "stationary.json"))
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fields(d):
    row = lambda x: d["sic"][:, int(np.argmin(np.abs(d["lat"] + x))), 0]
    return {"ice": south_ice_area(d) / 1e12, "edge": equivalent_edge(d), "Ts": global_ts(d),
            "H": lsg_layer_mean(d, 1, -90, 0), "A": lsg_layer_mean(d, 2, -90, 0),
            "r36": row(36.0), "r30": row(30.46), "r25": row(24.92)}


settled = {}
for lab in ["1265", "1235", "1235_new_IC", "1233p75", "1232p5", "1230"]:
    d = load(lab)
    f = fields(d)
    k = (d["years"] >= ST[lab]["0.95"]) & (d["years"] <= ST[lab]["end"])
    settled[lab] = {q: float(np.nanmean(v[k])) for q, v in f.items()}
d0 = load("1265")
ic = {q: float(v[d0["years"] == 4499][0]) for q, v in fields(d0).items()}
d = load("1230")
f = fields(d)
y = d["years"]
k = y <= 8400
json.dump({"settled": settled, "ic": ic}, open(TMP + "ic1230.json", "w"), indent=1)

refs = [("1235", "settled 1235", "0.55", "--"), ("1233p75", "settled 1233.75", AQUA, "--"),
        ("1232p5", "settled 1232.5", ORANGE, "--"), ("1230", "cold state 1230", BLUE, ":")]
panels = [("ice", "Southern ice area (10¹² m²)"), ("Ts", "global surface temperature (K)"),
          ("H", "Southern ocean 700–2025 m (K)"), ("A", "Southern ocean below 2025 m (K)")]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
for ax, (q, lab) in zip(axes[:4], panels):
    ax.plot(y[k], f[q][k], color=GREY, lw=0.5)
    for r, name, c, ls in refs:
        ax.axhline(settled[r][q], color=c, ls=ls, lw=1.2, label=name)
    ax.plot([4499], [ic[q]], "kD", ms=6, label="start: 1265 state, yr 4499")
    ax.set_ylabel(lab)
for ax in axes:
    ax.axvspan(7200, 7824, color=AQUA, alpha=0.12, lw=0)
    ax.axvline(7824, color="k", ls=":", lw=1)
    ax.set_xlabel("model year (μ = 1230 run, annual values)")
    ax.grid(alpha=0.3, lw=0.4)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
axes[0].set_title("Ice")
axes[1].set_title("Global temperature")
axes[2].set_title("Deep ocean")
axes[3].set_title("Abyss")
ax = axes[4]
for x, c in ((36.0, ORANGE), (30.46, BLUE), (24.92, AQUA)):
    key = {36.0: "r36", 30.46: "r30", 24.92: "r25"}[x]
    ax.plot(y[k], f[key][k], color=c, lw=0.5, label=f"{abs(x):.1f}°S row")
    ax.axhline(settled["1232p5"][key], color=c, ls="--", lw=1.2)
ax.set_ylabel("annual ice cover of the row (all sectors)")
ax.set_title("Rows (dashed: settled 1232.5)")
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("../../figures/plasim_fresh/ic1230.png", dpi=110)
plt.close(fig)
print("ok")
