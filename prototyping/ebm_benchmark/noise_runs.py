import itertools, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import replace
from gsebm import DAY, YEAR, RunSettings, StochasticRunSettings, default_model_parameters, run_stochastic_state, save_stochastic_state_dataset

OUT = Path('data/ebm_noise')
def one(args):
    mu, noise, real = args
    name = f"stochastic_warm_mu{str(mu).replace('.', 'p')}_noise{noise:.0e}_{{{real}}}.nc"
    if (OUT / name).exists(): return name
    sol = run_stochastic_state(
        params=replace(default_model_parameters(), mu=mu),
        settings=RunSettings(final_time=5000 * YEAR),
        stochastic_settings=StochasticRunSettings(dt=DAY, noise_amplitude=noise, noise_grid_step_degrees=3.0,
                                                  noise_length_scale_degrees=3.0, noise_seed=1000 * real + int(noise * 1e5) + int(mu * 1e3),
                                                  save_every=30),
        initial_temperature=290.0)
    save_stochastic_state_dataset(sol, output_dir=OUT, filename=name, initial_temperature=290.0)
    return name
if __name__ == '__main__':
    jobs = list(itertools.product((1.0, 0.98, 0.97), (1e-4, 2e-4), (1, 2, 3)))
    with ProcessPoolExecutor(7) as ex:
        for n in ex.map(one, jobs): print(n, flush=True)
