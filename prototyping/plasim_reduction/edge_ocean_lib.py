"""Analyses 1 and 2: the ice edge against ocean heat convergence, and the edge's energy margin.

Window means (RUNS windows) of every warm run, plus the cold states 1230 and 1228.5.
Analysis 1: per hemisphere, profiles of ocean heat release R = -(surface heat flux into the ocean) on LSG rows
(W m-2 of ocean), and poleward ocean / atmosphere / total heat transport (PW); ice edge marked.
  ocean transport  = plasim_global.ocean_northward_transport (cumulative uptake, steady state)
  total transport  = plasim_global.implied_northward_transport (cumulative ASR - OLR)
  atmosphere       = total - ocean (ocean interpolated to the T21 faces)
Analysis 2: cap budget between neighbouring runs. Cap = poleward of phi_c = (pair-mean edge) - 10 deg, fixed
within the pair. Terms integrated over the cap (PW): dASR = insolation term dI(1-mean a) + albedo term -mean(I) da,
dOLR, ocean release in the cap (= poleward ocean transport at phi_c), atmospheric import (= total import - ocean).
Each change divided by the edge shift (deg; negative = edge moved equatorward).
"""
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import read_global_series, equilibrium_state, time_mean, run_mu, EARTH_RADIUS_M

ROOT = Path('data/Plasim')
WARM = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']
EXTRA = ['1235_new_IC', '1230', '1228p5']
A = 4 * np.pi * EARTH_RADIUS_M**2
E = {}
for lab in WARM + EXTRA:
    s = read_global_series(ROOT, lab); eq = equilibrium_state(s)
    o = np.argsort(s.lat); oo = np.argsort(s.lsg_lat)
    E[lab] = dict(eq=eq, lat=s.lat[o], w=s.weight[o], ins=time_mean(s.insolation)[o], asr=time_mean(s.absorbed_shortwave)[o],
                  olr=time_mean(s.outgoing_longwave)[o], lsg_lat=s.lsg_lat[oo], area=s.lsg_row_area[oo],
                  uptake=np.nan_to_num(time_mean(s.ocean_heat_uptake))[oo])
    print(lab, 'S edge %.1f N edge %.1f' % (eq.south_edge, eq.north_edge), flush=True)

def faces_t21(w):  # latitudes of the north faces of the T21 rows (south to north), w normalized to sum 1
    return np.degrees(np.arcsin(np.clip(-1 + 2 * np.cumsum(w), -1, 1)))
def faces_lsg(lat):
    d = np.diff(lat).mean(); return lat + d / 2

def transports(e):
    tot = e['eq'].total_transport; ft = faces_t21(e['w'])
    oce = e['eq'].ocean_transport; fo = faces_lsg(e['lsg_lat'])
    oce_t = np.interp(ft, fo, oce)
    return ft, tot, oce_t, tot - oce_t, fo, oce


def cap_terms(e, phic, south):
    sgn = -1 if south else 1
    rowsT = (np.sign(e['lat']) == sgn) & (np.abs(e['lat']) >= phic)
    area_T = A * e['w'] / e['w'].sum()
    ft, tot, oce_t, atm, fo, oce = transports(e)
    oce_c = float(np.interp(phic, np.sort(np.abs(fo[np.sign(fo) == sgn])), (sgn * oce[np.sign(fo) == sgn])[np.argsort(np.abs(fo[np.sign(fo) == sgn]))]))
    tot_c = float(np.interp(phic, np.sort(np.abs(ft[np.sign(ft) == sgn])), (sgn * tot[np.sign(ft) == sgn])[np.argsort(np.abs(ft[np.sign(ft) == sgn]))]))
    return dict(rows=rowsT, area=area_T, ins=e['ins'], asr=e['asr'], olr=e['olr'], ocean_in=oce_c / 1, total_in=tot_c, atm_in=tot_c - oce_c)

def pair(a, b, south=True):
    ea, eb = E[a], E[b]
    edge = lambda e: e['eq'].south_edge if south else e['eq'].north_edge
    phic = 0.5 * (edge(ea) + edge(eb)) - 10
    ca, cb = cap_terms(ea, phic, south), cap_terms(eb, phic, south)
    r, ar = ca['rows'], ca['area']
    alb_a, alb_b = 1 - ea['asr'] / ea['ins'], 1 - eb['asr'] / eb['ins']
    Ibar, abar = 0.5 * (ea['ins'] + eb['ins']), 0.5 * (alb_a + alb_b)
    PW = lambda f: float((f[r] * ar[r]).sum() / 1e15)
    d = dict(insolation=PW((eb['ins'] - ea['ins']) * (1 - abar)), albedo=PW(-Ibar * (alb_b - alb_a)), olr=-PW(eb['olr'] - ea['olr']),
             ocean=cb['ocean_in'] - ca['ocean_in'], atm=cb['atm_in'] - ca['atm_in'])
    d['residual'] = d['insolation'] + d['albedo'] + d['olr'] + d['ocean'] + d['atm']
    dedge = edge(eb) - edge(ea)
    return phic, dedge, run_mu(b) - run_mu(a), d, ca, cb

