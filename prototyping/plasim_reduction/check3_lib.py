"""Check 3: surface (ice) vs atmosphere (cloud) part of the albedo term per degree of edge shift; OLR attribution.

Single-layer shortwave model (Donohoe & Battisti 2011) per T21 cell on window-mean maps (time-mean fluxes; an
approximation). I = rst - rsut, reflected fraction A = (I - rst)/I, surface downwelling D = rss/(1-as)/I, alpha = as:
  rho = (A - alpha D^2)/(1 - alpha^2 D^2), tau = D (1 - rho alpha), P = rho + tau^2 alpha/(1 - rho alpha).
Surface part of dP between two runs = mean over both atmospheres of P(alpha2) - P(alpha1); atmosphere part = rest.
Albedo term = -mean(I) dP, integrated over the cap |lat| >= 20 S (cell areas) in PW, relative to 1245.
OLR attribution: cap-mean Ts change per degree of edge, and cap OLR change per K of cap-mean Ts, early vs late.
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
R = 6.371e6; REF = '1245'
RUNS_ = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5', '1230', '1228p5']
def load(lab, part='full'):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: m[f'{k}__{part}']
    I = g('rst') - g('rsut'); rst = g('rst'); a_s = np.clip(g('as'), 0, 0.95)
    A = np.clip((I - rst) / I, 0, 1); D = np.clip(g('rss') / (1 - a_s) / I, 0, 1)
    rho = (A - a_s * D**2) / (1 - a_s**2 * D**2); tau = D * (1 - rho * a_s)
    lat = m['t21_lat']; w = m['t21_gaussian_weight']; nlon = m['t21_lon'].size
    area = (w / w.sum() * 4 * np.pi * R**2 / nlon)[:, None] * np.ones((1, nlon))
    ocean = m['lsm'] < 0.5; sic = g('sea_ice_concentration')
    zonal_ice = np.where(ocean.sum(1) > 0, (np.nan_to_num(sic) * ocean).sum(1) / np.maximum(ocean.sum(1), 1), np.nan)
    edge = float(ice_edge_latitude(zonal_ice[None], lat, south=True)[0])
    return dict(I=I, A=A, a=a_s, rho=rho, tau=tau, lat=lat, area=area, olr=-g('rlut'), ts=g('surface_temperature'), edge=edge, ocean=ocean, sic=sic)
P = lambda rho, tau, a: rho + tau**2 * a / (1 - rho * a)
