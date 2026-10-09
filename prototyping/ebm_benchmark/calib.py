import numpy as np
from gsebm import RunSettings, StochasticRunSettings, OscillatorForcing, DAY, YEAR
from gsebm.sde import solve_temperature_sde
from gsebm.ebm_stability import stability
from dataclasses import replace
from gsebm.parameters import default_model_parameters
guess = np.full(205, 300.0)
for mu in (1.0, 0.98, 0.972):
    st = stability(mu, guess); guess = st.temperature; x = st.x
    south = x < 0; v1 = st.modes[0].real
    x0 = x[south][np.argmax(np.abs(v1[south]))]
    osc = OscillatorForcing(center_x=float(x0), width_x=0.067, period_seconds=20 * YEAR, quality=5.0, q_std=1e-6)
    sol = solve_temperature_sde(params=replace(default_model_parameters(), mu=mu), settings=RunSettings(final_time=600 * YEAR),
        stochastic_settings=StochasticRunSettings(dt=DAY, noise_amplitude=0.0, save_every=30),
        custom_initial_temperature=st.temperature, initial_condition_kind='custom', initial_x=st.x, oscillator=osc, oscillator_seed=1)
    T = sol.temperature[sol.t > 100 * YEAR]; k = np.argmin(abs(x - x0))
    w = np.gradient(x); gm = T @ w / w.sum()
    print(f'mu={mu} x0={x0:.3f} (lat {np.degrees(np.arcsin(x0)):.1f}) T std at x0 = {T[:, k].std():.2f} K, global-mean std {gm.std():.2f} K, q std {sol.forcing.std():.2e}')
