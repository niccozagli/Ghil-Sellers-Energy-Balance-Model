import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from gsebm.diagnostics import meridional_heat_transfer_rate_watts_per_square_meter
    from gsebm.ivp import build_ivp_operator
    from gsebm.parameters import (
        RunSettings,
        StochasticRunSettings,
        default_model_parameters,
    )
    from gsebm.physics import surface_albedo
    from gsebm.sde import build_spatial_noise_process, solve_temperature_sde
    from gsebm.time import DAY, YEAR

    return (
        DAY,
        RunSettings,
        StochasticRunSettings,
        YEAR,
        build_ivp_operator,
        build_spatial_noise_process,
        default_model_parameters,
        meridional_heat_transfer_rate_watts_per_square_meter,
        mo,
        np,
        plt,
        solve_temperature_sde,
        surface_albedo,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # GSEBM Stochastic IVP Explorer

    This prototype runs the stochastic temperature solver from two uniform
    initial conditions:

    - warm start: `290 K`
    - cold start: `240 K`

    The drift is the same semi-discrete PDE operator as in the
    deterministic model. The stochastic term is additive, white in time,
    and spatially smoothed with Gaussian kernels centered every `5°`
    latitude.

    The stochastic solver now uses an IMEX step:

    - diffusion is treated implicitly
    - radiative reaction is treated explicitly
    - additive noise is applied explicitly

    so the explorer can use the standard IVP latitude grid directly.
    The noise basis is zero at the two pole points.

    Edit the values in the next cell to test whether a given stochastic
    run looks numerically reasonable.
    """)
    return


@app.cell
def _(DAY, YEAR):
    final_time = 50.0 * YEAR  # [s]
    dt =  DAY  # [s]
    save_every = 30  # [step]
    noise_amplitude = 5e-4  # [K s^-1/2]
    noise_grid_step_degrees = 3.0  # [deg]
    noise_length_scale_degrees = 3.0  # [deg]

    warm_initial_temperature = 290.0  # [K]
    cold_initial_temperature = 240.0  # [K]
    return (
        cold_initial_temperature,
        dt,
        final_time,
        noise_amplitude,
        noise_grid_step_degrees,
        noise_length_scale_degrees,
        save_every,
        warm_initial_temperature,
    )


@app.cell
def _(
    RunSettings,
    StochasticRunSettings,
    dt,
    final_time,
    noise_amplitude,
    noise_grid_step_degrees,
    noise_length_scale_degrees,
    save_every,
):
    run_settings = RunSettings(final_time=final_time)
    stochastic_settings = StochasticRunSettings(
        dt=dt,
        noise_amplitude=noise_amplitude,
        noise_grid_step_degrees=noise_grid_step_degrees,
        noise_length_scale_degrees=noise_length_scale_degrees,
        noise_seed=None, #every time a different seed
        save_every=save_every,
    )
    run_settings
    stochastic_settings
    return run_settings, stochastic_settings


@app.cell
def _(default_model_parameters):
    from dataclasses import replace
    params = default_model_parameters()

    # params = replace(params,mu=0.965)
    return (params,)


@app.cell
def _(
    build_ivp_operator,
    build_spatial_noise_process,
    cold_initial_temperature,
    params,
    run_settings,
    solve_temperature_sde,
    stochastic_settings,
    warm_initial_temperature,
):
    operator = build_ivp_operator(settings=run_settings, params=params)
    noise_process = build_spatial_noise_process(
        operator.x,
        coarse_step_degrees=stochastic_settings.noise_grid_step_degrees,
        length_scale_degrees=stochastic_settings.noise_length_scale_degrees,
    )

    warm_solution = solve_temperature_sde(
        params=params,
        settings=run_settings,
        stochastic_settings=stochastic_settings,
        initial_condition_kind="scalar",
        initial_scalar_value=warm_initial_temperature,
        noise_process=noise_process,
    )
    cold_solution = solve_temperature_sde(
        params=params,
        settings=run_settings,
        stochastic_settings=stochastic_settings,
        initial_condition_kind="scalar",
        initial_scalar_value=cold_initial_temperature,
        noise_process=noise_process,
    )
    return cold_solution, noise_process, operator, warm_solution


@app.cell
def _(YEAR, cold_solution, warm_solution):
    averaging_window_years = 35.0
    averaging_window_seconds = averaging_window_years * YEAR
    _warm_mask = warm_solution.t >= (warm_solution.t[-1] - averaging_window_seconds)
    _cold_mask = cold_solution.t >= (cold_solution.t[-1] - averaging_window_seconds)
    warm_asymptotic_temperature = warm_solution.temperature[_warm_mask].mean(axis=0)
    cold_asymptotic_temperature = cold_solution.temperature[_cold_mask].mean(axis=0)
    return (
        averaging_window_years,
        cold_asymptotic_temperature,
        warm_asymptotic_temperature,
    )


@app.cell
def _(YEAR, cold_solution, plt, warm_solution):
    time_in_years_warm = warm_solution.t / YEAR
    time_in_years_cold = cold_solution.t / YEAR

    mean_temperature_fig, mean_temperature_ax = plt.subplots(figsize=(8, 4))
    mean_temperature_ax.plot(
        time_in_years_warm,
        warm_solution.temperature.mean(axis=1),
        color="red",
        label="Warm Start",
    )
    mean_temperature_ax.plot(
        time_in_years_cold,
        cold_solution.temperature.mean(axis=1),
        color="blue",
        label="Cold Start",
    )
    mean_temperature_ax.set_xlabel("Time [years]")
    mean_temperature_ax.set_ylabel("Mean temperature [K]")
    mean_temperature_ax.set_title("Stochastic mean temperature over time")
    mean_temperature_ax.legend()
    mean_temperature_ax.grid(True, alpha=0.3)
    mean_temperature_fig
    return


@app.cell
def _(
    averaging_window_years,
    cold_asymptotic_temperature,
    cold_solution,
    meridional_heat_transfer_rate_watts_per_square_meter,
    np,
    operator,
    plt,
    surface_albedo,
    warm_asymptotic_temperature,
    warm_solution,
):
    warm_albedo = surface_albedo(
        warm_asymptotic_temperature,
        operator.empirical_fields.b_parameter,
        operator.empirical_fields.surface_height_offset,
        operator.params,
    )
    cold_albedo = surface_albedo(
        cold_asymptotic_temperature,
        operator.empirical_fields.b_parameter,
        operator.empirical_fields.surface_height_offset,
        operator.params,
    )

    warm_flux = meridional_heat_transfer_rate_watts_per_square_meter(
        warm_solution.x,
        warm_asymptotic_temperature,
        np.gradient(warm_asymptotic_temperature, warm_solution.x),
        operator.empirical_fields.sensible_heat_flux_coefficient,
        operator.empirical_fields.latent_heat_flux_coefficient,
        operator.params,
    )
    cold_flux = meridional_heat_transfer_rate_watts_per_square_meter(
        cold_solution.x,
        cold_asymptotic_temperature,
        np.gradient(cold_asymptotic_temperature, cold_solution.x),
        operator.empirical_fields.sensible_heat_flux_coefficient,
        operator.empirical_fields.latent_heat_flux_coefficient,
        operator.params,
    )

    profile_fig, (temperature_ax, albedo_ax, flux_ax) = plt.subplots(
        3,
        1,
        figsize=(8, 9),
        sharex=True,
    )
    temperature_ax.plot(
        warm_solution.x,
        warm_asymptotic_temperature,
        color="red",
        label="Warm Start",
    )
    temperature_ax.plot(
        cold_solution.x,
        cold_asymptotic_temperature,
        color="blue",
        label="Cold Start",
    )
    temperature_ax.set_ylabel("Temperature [K]")
    temperature_ax.set_title(f"Asymptotic temperature profiles ({averaging_window_years:.0f}-year mean)")
    temperature_ax.legend()
    temperature_ax.grid(True, alpha=0.3)

    albedo_ax.plot(
        warm_solution.x,
        warm_albedo,
        color="red",
        label="Warm Start",
    )
    albedo_ax.plot(
        cold_solution.x,
        cold_albedo,
        color="blue",
        label="Cold Start",
    )
    albedo_ax.set_ylabel("Albedo [-]")
    albedo_ax.set_title(f"Asymptotic albedo profiles ({averaging_window_years:.0f}-year mean)")
    albedo_ax.legend()
    albedo_ax.grid(True, alpha=0.3)

    flux_ax.plot(
        warm_solution.x,
        warm_flux,
        color="red",
        label="Warm Start",
    )
    flux_ax.plot(
        cold_solution.x,
        cold_flux,
        color="blue",
        label="Cold Start",
    )
    flux_ax.set_ylim(bottom=0)
    flux_ax.set_xlabel("Latitude x [-]")
    flux_ax.set_ylabel(r"Heat flux $j$ [W m$^{-2}$]")
    flux_ax.set_title(f"Asymptotic meridional heat-transfer rate ({averaging_window_years:.0f}-year mean)")
    flux_ax.legend()
    flux_ax.grid(True, alpha=0.3)
    flux_ax.set_xlim(left=0,right=1)
    profile_fig.tight_layout()
    profile_fig
    return


@app.cell
def _(noise_process, plt):
    basis_fig, basis_ax = plt.subplots(figsize=(8, 4))
    sample_columns = noise_process.normalized_basis[:, ::6]
    basis_ax.plot(noise_process.x, sample_columns)
    basis_ax.set_xlabel("Latitude x [-]")
    basis_ax.set_ylabel("Basis value [-]")
    basis_ax.set_title("Representative Gaussian noise kernels on the fine grid")
    basis_ax.grid(True, alpha=0.3)
    basis_fig
    return


@app.cell
def _(YEAR, np, plt, warm_solution):
    _fig, _ax = plt.subplots()

    x0_index = np.where( warm_solution.x < 10**-2)[0][-1] 
    x25_index = np.where( warm_solution.x - 0.25 < 10**-2)[0][-1] 
    x50_index = np.where( warm_solution.x - 0.50 < 10**-2)[0][-1] 
    x1_index = np.where( np.abs( warm_solution.x - 1) < 10**-2)[0][-1] 

    _ax.plot(warm_solution.t / YEAR, warm_solution.temperature[:,x0_index],
    label=f"{warm_solution.x[x0_index]:.2f}")

    _ax.plot(warm_solution.t / YEAR, warm_solution.temperature[:,x25_index],
    label=f"{warm_solution.x[x25_index]:.2f}")

    _ax.plot(warm_solution.t / YEAR, warm_solution.temperature[:,x50_index],
    label=f"{warm_solution.x[x50_index]:.2f}")

    _ax.plot(warm_solution.t / YEAR, warm_solution.temperature[:,x1_index],
    label=f"{warm_solution.x[x1_index]:.2f}")

    _ax.legend()
    return x0_index, x1_index, x25_index, x50_index


@app.cell
def _(YEAR, cold_solution, plt, x0_index, x1_index, x25_index, x50_index):
    _fig, _ax = plt.subplots()


    _ax.plot(cold_solution.t / YEAR, cold_solution.temperature[:,x0_index],
    label=f"{cold_solution.x[x0_index]:.2f}")

    _ax.plot(cold_solution.t / YEAR, cold_solution.temperature[:,x25_index],
    label=f"{cold_solution.x[x25_index]:.2f}")

    _ax.plot(cold_solution.t / YEAR, cold_solution.temperature[:,x50_index],
    label=f"{cold_solution.x[x50_index]:.2f}")

    _ax.plot(cold_solution.t / YEAR, cold_solution.temperature[:,x1_index],
    label=f"{cold_solution.x[x1_index]:.2f}")

    _ax.legend()
    return


@app.cell
def _():
    return


@app.cell
def _(
    YEAR,
    averaging_window_years,
    cold_asymptotic_temperature,
    cold_solution,
    np,
    plt,
    warm_asymptotic_temperature,
    warm_solution,
):
    _warm_mask = warm_solution.t >= (warm_solution.t[-1] - averaging_window_years * YEAR)
    _cold_mask = cold_solution.t >= (cold_solution.t[-1] - averaging_window_years * YEAR)

    warm_latitude_degrees = 90.0 * warm_solution.x
    cold_latitude_degrees = 90.0 * cold_solution.x
    warm_time_years = warm_solution.t[_warm_mask] / YEAR
    cold_time_years = cold_solution.t[_cold_mask] / YEAR


    warm_anomaly = warm_solution.temperature[_warm_mask] - warm_asymptotic_temperature[np.newaxis, :]
    cold_anomaly = cold_solution.temperature[_cold_mask] - cold_asymptotic_temperature[np.newaxis, :]

    anomaly_fig = plt.figure(figsize=(10, 8))
    warm_ax = anomaly_fig.add_subplot(2, 1, 1, projection="3d")
    cold_ax = anomaly_fig.add_subplot(2, 1, 2, projection="3d")

    warm_latitude_mesh, warm_time_mesh = np.meshgrid(warm_latitude_degrees, warm_time_years)
    cold_latitude_mesh, cold_time_mesh = np.meshgrid(cold_latitude_degrees, cold_time_years)

    warm_surface = warm_ax.plot_surface(
        warm_latitude_mesh,
        warm_time_mesh,
        warm_anomaly,
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=False,
    )
    cold_surface = cold_ax.plot_surface(
        cold_latitude_mesh,
        cold_time_mesh,
        cold_anomaly,
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=False,
    )

    warm_ax.set_title(f"Warm-start asymptotic anomalies ({averaging_window_years:.0f}-year window)")
    warm_ax.set_xlabel("Latitude [deg]")
    warm_ax.set_ylabel("Time [years]")
    warm_ax.set_zlabel("Temperature anomaly [K]")
    anomaly_fig.colorbar(warm_surface, ax=warm_ax, shrink=0.6, pad=0.08)

    cold_ax.set_title(f"Cold-start asymptotic anomalies ({averaging_window_years:.0f}-year window)")
    cold_ax.set_xlabel("Latitude [deg]")
    cold_ax.set_ylabel("Time [years]")
    cold_ax.set_zlabel("Temperature anomaly [K]")
    anomaly_fig.colorbar(cold_surface, ax=cold_ax, shrink=0.6, pad=0.08)

    anomaly_fig.tight_layout()
    anomaly_fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
