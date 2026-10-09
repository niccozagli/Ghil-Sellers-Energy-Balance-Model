"""Check 5 (criteria in Claude/plasim_mechanism/plan_checks_2_5.md)."""

import matplotlib.pyplot as plt
import numpy as np

from common import equivalent_edge, load

C5 = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/check5/"
OUT = "../../figures/plasim_fresh"


def m50(y, x, a):
    k = (y >= a) & (y <= a + 49)
    return np.nanmean(x[k])


def onset(y, x, ref, start, thresh=None, direction=None):
    """First year > start whose 50-yr mean departs from the reference mean by more than thresh
    (default 3 sd of the reference period's 50-yr block means)."""
    k = (y >= ref[0]) & (y <= ref[1])
    mref = np.nanmean(x[k])
    blocks = [m50(y, x, a) for a in range(ref[0], ref[1] - 48, 50)]
    sd = np.nanstd(blocks, ddof=1)
    th = thresh if thresh is not None else 3 * sd
    for yy in range(start + 1, int(y.max()) - 49):
        dev = m50(y, x, yy) - mref
        if (direction is None and abs(dev) > th) or (direction is not None and direction * dev > th):
            return yy, mref, sd
    return None, mref, sd


# validation against check 6 (window means)
for name, lab, w in (("1232.5 window", "1232p5_20430_24429", None), ("1230 cold window", "1230_12369_16368", None)):
    z = np.load(C5 + lab + ".npz")
    print(f"validation {name}: Fp {np.mean(z['Fp']):+.3f}  U {np.mean(z['U']):+.3f}  G {np.mean(z['G']):+.3f} PW")

cases = {"1230": ("1230_5000_8600", (7200, 7699), 7500, (5000, 8600)),
         "1225": ("1225_5500_7900", (7000, 7499), 7450, (5500, 7900))}
stat = {"1230": "1232p5_20430_24429", "1225": "1228p5_15400_16898"}
fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex="col")
for j, (lab, (fn, ref, start, span)) in enumerate(cases.items()):
    z = np.load(C5 + fn + ".npz"); y = z["years"]
    d = load(lab); ye = d["years"]; E = equivalent_edge(d)
    Ei = np.interp(y, ye, E)
    D = Ei - z["cap_eqlat"]  # positive = capped front poleward of the ice edge? (°S: larger = further south)
    D = z["cap_eqlat"] - Ei   # degrees by which the capped front lies poleward of the edge
    t_e, mE, _ = onset(y, -Ei, ref, start, thresh=1.0, direction=+1)
    print(f"\n{lab}: edge onset {t_e} (reference {ref}, edge mean {-mE:.2f}°S)")
    for key, x in (("capped area S", z["cap_south"]), ("capped area global", z["cap_global"]), ("coupling gap G", z["G"]),
                   ("capped-front distance D", D)):
        t, mref, sd = onset(y, x, ref, start)
        print(f"  {key:24s}: reference mean {mref:8.3f}  sd(50-yr) {sd:.4f}  onset {t}  "
              f"(lead over edge: {t_e - t if t else 'n/a'} yr)")
    # D criterion (annual values; 50-yr means also shown)
    pre = (y >= t_e - 300) & (y < t_e)
    kref = (y >= ref[0]) & (y <= ref[1])
    zs = np.load(C5 + stat[lab] + ".npz"); ds = load({"1230": "1232p5", "1225": "1228p5"}[lab])
    Ds = zs["cap_eqlat"] - np.interp(zs["years"], ds["years"], equivalent_edge(ds))
    print(f"  D annual: min in reference {D[kref].min():.2f}°, min in stationary run {Ds.min():.2f}°, "
          f"min in 300 yr before onset {D[pre].min():.2f}°; 50-yr means before onset "
          f"{[round(m50(y, D, a), 2) for a in range(t_e - 300, t_e, 50)]}")
    print(f"  stationary run mean: capped S {zs['cap_south'].mean():.2f}  global {zs['cap_global'].mean():.2f}  "
          f"G {zs['G'].mean():+.3f}  D {Ds.mean():.2f}")
    k = (y >= span[0]) & (y <= span[1])
    for i, (x, lbl) in enumerate(((Ei, "equivalent edge (°S)"), (z["cap_south"], "S capped area (10¹² m²)"),
                                   (D, "capped front poleward of edge (°)"), (z["G"], "coupling gap G (PW)"))):
        ax = axes[i, j]
        ax.plot(y[k], x[k], lw=0.5, color="0.6")
        yb = np.arange(span[0], span[1] - 49, 50)
        ax.plot(yb + 25, [m50(y, x, a) for a in yb], "o-", ms=2.5, lw=1, color="#2a78d6")
        ax.axvline(t_e, color="#eb6834", ls=":", lw=1.2)
        ax.axvspan(*ref, color="0.92", zorder=0)
        ax.set_ylabel(lbl, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
        if i == 0:
            ax.invert_yaxis(); ax.set_title(f"μ = {lab}: edge onset {t_e} (orange); reference shaded", fontsize=9)
    axes[-1, j].set_xlabel("model year")
fig.suptitle("Check 5: the 9 m cap and the coupling gap around the jumps (grey annual, blue 50-yr means)")
fig.tight_layout(); fig.savefig(f"{OUT}/check5_artefacts.png", dpi=110); plt.close(fig)
