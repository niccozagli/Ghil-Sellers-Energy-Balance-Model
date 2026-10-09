import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np, xarray as xr
from gsebm import RunSettings, StochasticRunSettings, LimitCycleForcing, DAY, YEAR
from gsebm.parameters import default_model_parameters
from gsebm.sde import solve_temperature_sde
from gsebm.ebm_stability import stability

OUT = Path('data/ebm_limit_cycle'); NOISE = 4e-4; Q_STD = 2.5e-7
def one(args):
    mu, real = args
    name = OUT / f"limit_cycle_warm_mu{str(mu).replace('.', 'p')}_{{{real}}}.nc"
    if name.exists(): return str(name)
    st = stability(mu, np.full(205, 300.0)); x = st.x; south = x < 0
    x0 = float(x[south][np.argmax(np.abs(st.modes[0].real[south]))])
    osc = LimitCycleForcing(center_x=x0, width_x=0.067, period_seconds=20 * YEAR, reference_temperature=281.3, frequency_sensitivity=0.038, amplitude_relaxation_seconds=5 * YEAR, amplitude_std=0.1, phase_diffusion=1 / (3000 * YEAR), slip_rate=1 / (600 * YEAR), slip_size=float(np.pi), q_amplitude=Q_STD * np.sqrt(2))
    stoch = StochasticRunSettings(dt=DAY, noise_amplitude=NOISE, noise_grid_step_degrees=3.0,
                                  noise_length_scale_degrees=3.0, noise_seed=500 + 10 * real + int(mu * 1000) % 100, save_every=30)
    sol = solve_temperature_sde(params=replace(default_model_parameters(), mu=mu), settings=RunSettings(final_time=5000 * YEAR),
                                stochastic_settings=stoch, initial_condition_kind='scalar', initial_scalar_value=290.0,
                                oscillator=osc, oscillator_seed=900 + real)
    ds = xr.Dataset({'temperature': (('time', 'latitude'), sol.temperature), 'cycle_amplitude': (('time',), sol.forcing[:, 0]), 'cycle_phase': (('time',), sol.forcing[:, 1])},
                    coords={'time': sol.t, 'latitude': sol.x})
    ds.attrs.update({'title': 'Stochastic EBM with a climate-coupled limit-cycle heating', 'param_mu': mu, 'initial_temperature': 290.0,
                     **{f'stochastic_{k}': (str(v) if v is None else v) for k, v in asdict(stoch).items()},
                     **{f'oscillator_{k}': v for k, v in asdict(osc).items()}, 'oscillator_seed': 900 + real})
    ds.to_netcdf(name, engine='scipy'); return str(name)
if __name__ == '__main__':
    with ProcessPoolExecutor(7) as ex:
        for n in ex.map(one, itertools.product((1.0, 0.99, 0.98, 0.975), (1, 2))): print(n, flush=True)
