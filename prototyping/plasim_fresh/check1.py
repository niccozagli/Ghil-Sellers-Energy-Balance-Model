"""Check 1 (criteria in Claude/plasim_mechanism/plan_check1.md): 1A, 1B, 1C for every μ."""

import json

import numpy as np

from common import PARENT, equivalent_edge, load, lsg_layer_mean
from gsebm.plasim_global import RUNS, run_mu

O10 = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/ocean10/"
TMP = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/"
TF = 271.25
ALL = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1235_new_IC",
       "1233p75", "1232p5", "1230", "1228p5", "1225"]


def wmean(x, w, axis):
    ok = np.isfinite(x) & (w > 0)
    return np.nansum(np.where(ok, x * w, 0), axis=axis) / np.sum(np.where(ok, w, 0), axis=axis)


def ocean_indices(z, E_set):
    """10-yr-block series of B, Lk, K, T_edge−Tf, cell strengths, salinity contrast for a fixed settled edge."""
    lat, vlat, db = z["lsg_lat"], z["lsg_vector_lat"], z["depth_bounds"]
    vol = z["wet_volume"]
    thick = db[:, 1] - db[:, 0]
    V = z["meridional_volume_transport"].astype(float)
    psi500 = np.nansum(V[:, :, :8], axis=2) / 1e6           # Sv, northward transport above 500 m
    psi = np.nancumsum(V, axis=2) / 1e6                       # full-depth streamfunction at interfaces
    nb = V.shape[0]
    B = np.full(nb, np.nan); pole = np.full(nb, np.nan); equ = np.full(nb, np.nan)
    sel = np.flatnonzero((vlat > -72) & (vlat < -10))
    for i in range(nb):
        p = psi500[i]
        cands = [j for j in sel[:-1] if p[j] > 0 and p[j + 1] <= 0]
        if cands:
            j = min(cands, key=lambda j: abs(vlat[j] + E_set))
            B[i] = vlat[j] + (vlat[j + 1] - vlat[j]) * p[j] / (p[j] - p[j + 1])
            sp = (vlat < B[i]) & (vlat > -76); se = (vlat > B[i]) & (vlat < -10)
            pole[i] = np.nanmax(psi[i][sp]); equ[i] = np.nanmin(psi[i][se])
    conv = z["convective_adjustment"].astype(float)
    lev1000 = db[:, 1] <= 1025
    cidx = wmean(conv[:, :, lev1000], thick[lev1000][None, None, :], axis=2)  # (nb, 68)
    srow = (lat > -70) & (lat < -10)
    Lk = np.array([lat[srow][np.nanargmax(c[srow])] if np.any(np.isfinite(c[srow])) else np.nan for c in cidx])
    erow = np.abs(lat + E_set) <= 2.5
    K = np.nanmean(cidx[:, erow], axis=1)
    th = z["potential_temperature"].astype(float)
    lev = (db[:, 0] >= 100) & (db[:, 1] <= 700)
    w = vol[erow][:, lev]
    Tedge = wmean(th[:, erow][:, :, lev], w[None], axis=(1, 2)) - TF
    sal = z["salinity"].astype(float)
    s_top = wmean(sal[:, erow, 1], vol[erow, 1][None], axis=1)
    deep = (db[:, 0] >= 700) & (db[:, 1] <= 1025)
    s_deep = wmean(sal[:, erow][:, :, deep], vol[erow][:, deep][None], axis=(1, 2))
    return dict(B=-B, Lk=-Lk, K=K, Tedge=Tedge, pole=pole, equ=equ, dS=s_deep - s_top)


def blocks10(years, x, starts):
    return np.array([np.nanmean(x[(years >= s) & (years <= s + 9)]) for s in starts])


S = {}
for lab in ALL:
    d = load(lab); z = np.load(O10 + lab + ".npz")
    w0, w1 = RUNS[lab].window
    k = (d["years"] >= w0) & (d["years"] <= w1)
    E = equivalent_edge(d)
    E_set = float(np.nanmean(E[k])) if lab != "1225" else np.nan
    starts = z["block_start"]
    ser = dict(E=blocks10(d["years"], E, starts),
               H=blocks10(d["years"], lsg_layer_mean(d, 1, -90, 0), starts),
               Hu=blocks10(d["years"], lsg_layer_mean(d, 0, -90, 0), starts),
               Hd=blocks10(d["years"], lsg_layer_mean(d, 2, -90, 90), starts))
    if np.isfinite(E_set):
        rows = (d["lsg_lat"] >= -E_set) & (d["lsg_lat"] <= -E_set + 10)
        wv = d["layer_volume"][rows, 0]
        Wa = np.nansum(d["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()
        ser["W"] = blocks10(d["years"], Wa, starts)
        ser.update(ocean_indices(z, E_set))
        fl = d["coupling_flux"]; band = np.abs(d["lsg_lat"] + E_set) <= 5
        rel = np.nansum(np.nan_to_num(fl[:, band]) * d["lsg_area"][band], axis=1) / 1e15
        ser["release"] = blocks10(d["years"], rel, starts)
    S[lab] = dict(mu=d["mu"], E_set=E_set, starts=starts, win=(w0, w1), ser=ser)

# ---------------- 1A ----------------
print("1A. Window means (zonal). B = overturning boundary, Lk = latitude of max 0–1000 m convection (°S).")
print(f"{'run':12s}{'E':>6s}{'B':>7s}{'B−E':>6s}{'Lk':>6s}{'Lk−E':>6s}{'K':>6s}{'pole Sv':>8s}{'equ Sv':>8s}"
      f"{'Tedge−Tf':>9s}{'H−Tf':>6s}{'rel PW':>7s}{'ΔS':>6s}")
A1 = {}
for lab in ALL[:-1]:
    s = S[lab]; st = s["starts"]; w0, w1 = s["win"]
    kw = (st >= w0) & (st + 9 <= w1)
    m = {k: float(np.nanmean(v[kw])) for k, v in s["ser"].items()}
    A1[lab] = m
    print(f"{lab:12s}{s['E_set']:6.1f}{m['B']:7.1f}{m['B']-s['E_set']:6.1f}{m['Lk']:6.1f}{m['Lk']-s['E_set']:6.1f}"
          f"{m['K']:6.3f}{m['pole']:8.1f}{m['equ']:8.1f}{m['Tedge']:9.2f}{m['H']-TF:6.2f}{m['release']:7.2f}{m['dS']:6.2f}")
c1 = [lab for lab in ALL[1:-1] if abs(A1[lab]["B"] - S[lab]["E_set"]) > 2.5]
c2 = [lab for lab in ALL[1:-1] if abs(A1[lab]["Lk"] - S[lab]["E_set"]) > 2.5]
print(f"C1 misses (|B−E|>2.5°, 1288 down): {c1} → {'holds' if not c1 else ('falsified' if len(c1) >= 2 else 'one miss')}")
print(f"C2 misses (|Lk−E|>2.5°, 1288 down): {c2} → {'holds' if not c2 else ('falsified' if len(c2) >= 2 else 'one miss')}")

# ---------------- 1B ----------------
STEP = ["1312", "1288", "1265", "1250", "1245", "1235", "1242p5", "1237p5", "1235_new_IC", "1233p75", "1232p5"]
QB = ["E", "B", "K", "W", "Hu", "H", "Hd"]


def t50(starts, p, y_step):
    for i in range(len(p) - 4):
        if starts[i] > y_step and np.all(p[i:i + 5] >= 0.5):
            return float(starts[i] + 5 - y_step)
    return np.nan


B1 = {}
print("\n1B. Halfway times (years after the step); slow = progress from the years-20–50 level. 'nr' = not resolvable.")
for lab in STEP:
    s = S[lab]; st = s["starts"]; parent, y0 = PARENT[lab]
    w0, w1 = s["win"]; kw = (st >= w0) & (st + 9 <= w1)
    kf = (st >= y0 + 21) & (st <= y0 + 41)
    if parent == "1367":
        kp_src, kp = s, (st >= y0 + 1) & (st <= y0 + 10)
    else:
        P = S[parent]; kp_src, kp = P, (P["starts"] >= y0 - 99) & (P["starts"] <= y0 - 9)
    out = {}
    for q in QB:
        x = s["ser"][q]
        if parent == "1367":
            xp = np.nanmean(x[kp])
        else:
            # parent value: same quantity computed with this run's settled edge where it is edge-relative
            if q in ("E", "H", "Hu", "Hd"):
                xp = np.nanmean(kp_src["ser"][q][kp])
            else:
                zP = np.load(O10 + parent + ".npz"); dP = load(parent)
                if q == "W":
                    rows = (dP["lsg_lat"] >= -s["E_set"]) & (dP["lsg_lat"] <= -s["E_set"] + 10)
                    wv = dP["layer_volume"][rows, 0]
                    Wa = np.nansum(dP["theta_layers"][:, rows, 0] * wv, axis=1) / wv.sum()
                    xp = np.nanmean(Wa[(dP["years"] >= y0 - 99) & (dP["years"] <= y0)])
                else:
                    oi = ocean_indices(zP, s["E_set"])
                    xp = np.nanmean(oi[q][(zP["block_start"] >= y0 - 99) & (zP["block_start"] <= y0 - 9)])
        xf, xF = np.nanmean(x[kf]), np.nanmean(x[kw])
        b50 = [np.nanmean(x[(st >= a) & (st < a + 50)]) for a in range(w0, w1 - 48, 50)]
        sd50 = np.nanstd(b50, ddof=1)
        res_tot = sd50 / abs(xF - xp) <= 0.15 if xF != xp else False
        res_slow = sd50 / abs(xF - xf) <= 0.15 if xF != xf else False
        p_tot = (x - xp) / (xF - xp); p_slow = (x - xf) / (xF - xf)
        out[q] = dict(parent=xp, fast=xf, final=xF, sd50=sd50, res_tot=bool(res_tot), res_slow=bool(res_slow),
                      t50=t50(st, p_tot, y0) if res_tot else np.nan, t50s=t50(st, p_slow, y0) if res_slow else np.nan,
                      fast_share=(xf - xp) / (xF - xp) if xF != xp else np.nan)
    B1[lab] = out
    row = "  ".join(f"{q} {out[q]['t50']:>5.0f}/{out[q]['t50s']:>5.0f}" if out[q]['res_tot'] or out[q]['res_slow']
                    else f"{q}   nr/   nr" for q in QB)
    print(f"  {lab:12s} Δμ {s['mu']-run_mu(PARENT[lab][0]):6.1f}  " + row.replace("nan", "  nr"))


def tol(a, b):
    return max(50.0, 0.2 * max(a, b))


print("\n  S1 (ice slow part vs layer: keeps pace / ahead / lags) and S2 (B, K vs E and H):")
verd = {}
for lab in STEP:
    o = B1[lab]; te = o["E"]["t50s"]
    s1 = {}
    for q in ("W", "Hu", "H", "Hd"):
        tq = o[q]["t50s"]
        if np.isnan(te) or np.isnan(tq):
            s1[q] = "nr"
        else:
            s1[q] = "pace" if abs(te - tq) <= tol(te, tq) else ("ahead" if te < tq else "lags")
    s2 = {}
    for q in ("B", "K"):
        tq, tE, tH = o[q]["t50"], o["E"]["t50"], o["H"]["t50"]
        if np.isnan(tq):
            s2[q] = "nr"; continue
        wi = (not np.isnan(tE)) and abs(tq - tE) <= tol(tq, tE)
        wo = (not np.isnan(tH)) and abs(tq - tH) <= tol(tq, tH)
        s2[q] = "with ice & ocean" if wi and wo else ("with ice" if wi else ("with slow ocean" if wo else "neither"))
    verd[lab] = dict(S1=s1, S2=s2)
    print(f"    {lab:12s} S1 {s1}   S2 {s2}")
res = [l for l in STEP if any(v != "nr" for v in verd[l]["S1"].values())]
ocean_paced = [l for l in res if any(v in ("pace", "lags") for v in verd[l]["S1"].values())]
ice_paced = [l for l in res if all(v == "ahead" for v in verd[l]["S1"].values() if v != "nr")]
print(f"  Verdict S1: resolvable runs {len(res)}; ocean-paced {len(ocean_paced)} {ocean_paced}; ice-paced {len(ice_paced)} {ice_paced}")
resB = [l for l in STEP if verd[l]["S2"]["B"] != "nr"]
withice = [l for l in resB if "with ice" in verd[l]["S2"]["B"]]
withocean = [l for l in resB if "slow ocean" in verd[l]["S2"]["B"] or "ocean" in verd[l]["S2"]["B"]]
print(f"  Verdict S2 (B): resolvable {len(resB)}; with ice {len(withice)} {withice}; with slow ocean {len(withocean)} {withocean}")

# ---------------- 1C ----------------
print("\n1C. Slopes of E on H (° per K): forced (neighbours), slow stage, natural (100-yr blocks in window)")
SEQ = ["1312", "1288", "1265", "1250", "1245", "1242p5", "1240", "1237p5", "1235", "1233p75", "1232p5"]
eqv = {lab: (np.nanmean(S[lab]["ser"]["E"][(S[lab]["starts"] >= S[lab]["win"][0]) & (S[lab]["starts"] + 9 <= S[lab]["win"][1])]),
             np.nanmean(S[lab]["ser"]["H"][(S[lab]["starts"] >= S[lab]["win"][0]) & (S[lab]["starts"] + 9 <= S[lab]["win"][1])]))
       for lab in ALL[:-1]}
C1R = {}
for lab in ALL[:-1]:
    base = "1235" if lab == "1235_new_IC" else lab
    if base in SEQ:
        i = SEQ.index(base); a = SEQ[max(i - 1, 0)]; b = SEQ[min(i + 1, len(SEQ) - 1)]
    else:
        a, b = "1230", "1228p5"
    forced = (eqv[a][0] - eqv[b][0]) / (eqv[a][1] - eqv[b][1])
    slow = np.nan
    if lab in B1 and B1[lab]["E"]["res_slow"]:
        o = B1[lab]; slow = (o["E"]["final"] - o["E"]["fast"]) / (o["H"]["final"] - o["H"]["fast"])
    d = load(lab); w0, w1 = S[lab]["win"]
    if lab == "1235_new_IC":
        w0 = w1 - 1499
    Eb = np.array([np.nanmean(equivalent_edge(d)[(d["years"] >= a0) & (d["years"] < a0 + 100)]) for a0 in range(w0, w1 - 98, 100)])
    Hb = np.array([np.nanmean(lsg_layer_mean(d, 1, -90, 0)[(d["years"] >= a0) & (d["years"] < a0 + 100)]) for a0 in range(w0, w1 - 98, 100)])
    ok = np.isfinite(Eb) & np.isfinite(Hb)
    X = Hb[ok] - Hb[ok].mean(); Y = Eb[ok] - Eb[ok].mean()
    nat = (X @ Y) / (X @ X); resid = Y - nat * X
    se = np.sqrt(resid @ resid / (ok.sum() - 2) / (X @ X)); r = np.corrcoef(X, Y)[0, 1]
    if se > 0.5 * abs(forced):
        v = "not resolvable"
    elif abs(r) < 0.25:
        v = "no slow link"
    elif abs(nat - forced) <= 2 * se and r >= 0.5:
        v = "one relation"
    elif abs(nat - forced) > 2 * se:
        v = "different"
    else:
        v = "agrees, weak r"
    C1R[lab] = dict(forced=forced, slow=slow, natural=nat, se=se, r=r, n=int(ok.sum()), verdict=v,
                    H_range=float(np.ptp(Hb[ok])))
    print(f"  {lab:12s} forced {forced:6.2f} ({a}–{b})  slow {slow:6.2f}  natural {nat:6.2f} ± {se:4.2f} (r {r:+.2f}, n {ok.sum()}, "
          f"H range {np.ptp(Hb[ok]):.3f} K) → {v}")

json.dump(dict(A1={k: v for k, v in A1.items()}, E_set={k: S[k]["E_set"] for k in S},
               B1={k: {q: {kk: (float(vv) if not isinstance(vv, bool) else vv) for kk, vv in o.items()} for q, o in v.items()} for k, v in B1.items()},
               verd=verd, C1=C1R), open(TMP + "check1.json", "w"), default=float)
np.save(TMP + "check1_series.npy", {k: dict(mu=v["mu"], E_set=v["E_set"], starts=v["starts"], win=v["win"], ser=v["ser"]) for k, v in S.items()}, allow_pickle=True)
