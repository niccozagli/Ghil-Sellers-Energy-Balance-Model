"""Tests for stochastic temperature integration."""

from functools import lru_cache
import unittest

import numpy as np
from scipy.linalg import solve_banded

from gsebm.bvp import solve_temperature_bvp
from gsebm.ivp import build_ivp_grid, build_ivp_operator, solve_temperature_ivp
from gsebm.parameters import RunSettings, StochasticRunSettings
from gsebm.sde import (
    build_noise_latitude_grid,
    build_spatial_noise_process,
    solve_temperature_sde,
)
from gsebm.time import DAY, HOUR, YEAR


@lru_cache(maxsize=1)
def _warm_bvp_reference() -> tuple[np.ndarray, np.ndarray]:
    solution = solve_temperature_bvp(
        initial_guess_kind="scalar",
        initial_scalar_value=300.0,
        tolerance=1e-2,
    )
    return solution.x.copy(), solution.temperature.copy()


def _interpolate_warm_reference(x_grid: np.ndarray) -> np.ndarray:
    reference_x, reference_temperature = _warm_bvp_reference()
    return np.interp(x_grid, reference_x, reference_temperature)


class SDETest(unittest.TestCase):
    def test_build_noise_latitude_grid_matches_5_degree_resolution(self) -> None:
        latitude = build_noise_latitude_grid(5.0)
        self.assertEqual(latitude.shape, (37,))
        self.assertEqual(float(latitude[0]), -90.0)
        self.assertEqual(float(latitude[-1]), 90.0)
        np.testing.assert_allclose(np.diff(latitude), 5.0)

    def test_spatial_noise_process_has_zero_boundary_rows_and_unit_interior_variance(self) -> None:
        x = np.linspace(-1.0, 1.0, 9)
        process = build_spatial_noise_process(
            x,
            coarse_step_degrees=5.0,
            length_scale_degrees=5.0,
        )
        row_squared_norm = np.sum(process.normalized_basis**2, axis=1)
        self.assertEqual(row_squared_norm[0], 0.0)
        self.assertEqual(row_squared_norm[-1], 0.0)
        np.testing.assert_allclose(row_squared_norm[1:-1], 1.0)

    def test_split_identity_matches_rhs(self) -> None:
        operator = build_ivp_operator()
        rng = np.random.default_rng(7)
        warm_profile = _interpolate_warm_reference(operator.x)
        profiles = [
            np.full_like(operator.x, 280.0),
            warm_profile,
            240.0 + 70.0 * rng.random(operator.x.size),
            250.0 + 40.0 * np.cos(np.pi * operator.x / 2.0),
        ]

        for profile in profiles:
            split = operator.frozen_diffusion_operator(profile).apply(profile) + operator.reaction_tendency(profile)
            rhs = operator.rhs(0.0, profile)
            np.testing.assert_allclose(split, rhs, atol=1e-12, rtol=1e-12)

    def test_zero_noise_matches_one_imex_step(self) -> None:
        run_settings = RunSettings(final_time=1.0e3, time_output_count=5)
        stochastic_settings = StochasticRunSettings(dt=1.0e3, noise_amplitude=0.0)
        operator = build_ivp_operator(settings=run_settings)
        initial_temperature = np.full_like(operator.x, 280.0)
        diffusion_operator = operator.frozen_diffusion_operator(initial_temperature)
        expected = solve_banded(
            (1, 1),
            diffusion_operator.to_banded_matrix(diagonal_shift=1.0, scale=-1.0e3),
            initial_temperature + 1.0e3 * operator.reaction_tendency(initial_temperature),
            check_finite=False,
        )

        solution = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
        )

        self.assertEqual(solution.step_count, 1)
        np.testing.assert_allclose(solution.temperature[-1], expected)

    def test_seeded_noise_is_reproducible(self) -> None:
        run_settings = RunSettings(final_time=3.0e3, time_output_count=5)
        stochastic_settings = StochasticRunSettings(
            dt=1.0e3,
            noise_amplitude=1.0e-4,
            noise_seed=7,
        )

        solution_1 = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
        )
        solution_2 = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
        )

        np.testing.assert_allclose(solution_1.t, solution_2.t)
        np.testing.assert_allclose(solution_1.temperature, solution_2.temperature)

    def test_save_every_subsamples_output_times(self) -> None:
        run_settings = RunSettings(final_time=3.0 * DAY, time_output_count=5)
        stochastic_settings = StochasticRunSettings(
            dt=DAY,
            noise_amplitude=0.0,
            save_every=2,
        )

        solution = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
        )

        np.testing.assert_allclose(solution.t, np.array([0.0, 2.0 * DAY, 3.0 * DAY]))
        self.assertEqual(solution.temperature.shape[0], 3)

    def test_imex_scheme_remains_stable_for_large_day_scale_step_on_refined_grid(self) -> None:
        x_grid = build_ivp_grid()
        warm_profile = _interpolate_warm_reference(x_grid)
        run_settings = RunSettings(final_time=30.0 * DAY, time_output_count=5)
        stochastic_settings = StochasticRunSettings(
            dt=DAY,
            noise_amplitude=0.0,
        )

        solution = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            x_grid=x_grid,
            initial_condition_kind="custom",
            custom_initial_temperature=warm_profile,
            initial_x=x_grid,
        )

        self.assertTrue(np.isfinite(solution.temperature).all())
        self.assertLess(np.max(np.abs(solution.temperature[-1] - warm_profile)), 1.0)

    def test_zero_noise_matches_bdf_on_perturbed_warm_state(self) -> None:
        x_grid = build_ivp_grid()
        warm_profile = _interpolate_warm_reference(x_grid)
        perturbed_profile = warm_profile + 0.5 * np.cos(np.pi * x_grid)
        run_settings = RunSettings(final_time=90.0 * DAY, time_output_count=5)
        stochastic_settings = StochasticRunSettings(
            dt=6.0 * HOUR,
            noise_amplitude=0.0,
            save_every=30,
        )

        sde_solution = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=stochastic_settings,
            x_grid=x_grid,
            initial_condition_kind="custom",
            custom_initial_temperature=perturbed_profile,
            initial_x=x_grid,
        )
        bdf_solution = solve_temperature_ivp(
            settings=run_settings,
            x_grid=x_grid,
            initial_condition_kind="custom",
            custom_initial_temperature=perturbed_profile,
            initial_x=x_grid,
            method="BDF",
            rtol=1e-8,
            atol=1e-8,
        )

        np.testing.assert_allclose(sde_solution.temperature[-1], bdf_solution.temperature[-1], atol=0.1)
        self.assertLess(np.max(np.abs(sde_solution.temperature[-1] - warm_profile)), 1.0)

    def test_weak_noise_ensemble_mean_tracks_deterministic_and_variance_grows(self) -> None:
        x_grid = np.linspace(-0.95, 0.95, 41)
        warm_profile = _interpolate_warm_reference(x_grid)
        run_settings = RunSettings(final_time=10.0 * DAY, time_output_count=5)
        deterministic_settings = StochasticRunSettings(
            dt=DAY,
            noise_amplitude=0.0,
            save_every=1,
        )
        noisy_settings = StochasticRunSettings(
            dt=DAY,
            noise_amplitude=2.0e-6,
            save_every=1,
        )

        deterministic_solution = solve_temperature_sde(
            settings=run_settings,
            stochastic_settings=deterministic_settings,
            x_grid=x_grid,
            initial_condition_kind="custom",
            custom_initial_temperature=warm_profile,
            initial_x=x_grid,
        )

        ensemble = []
        for seed in range(32):
            solution = solve_temperature_sde(
                settings=run_settings,
                stochastic_settings=StochasticRunSettings(
                    dt=noisy_settings.dt,
                    noise_amplitude=noisy_settings.noise_amplitude,
                    noise_grid_step_degrees=noisy_settings.noise_grid_step_degrees,
                    noise_length_scale_degrees=noisy_settings.noise_length_scale_degrees,
                    noise_seed=seed,
                    save_every=noisy_settings.save_every,
                ),
                x_grid=x_grid,
                initial_condition_kind="custom",
                custom_initial_temperature=warm_profile,
                initial_x=x_grid,
            )
            ensemble.append(solution.temperature)

        ensemble_temperature = np.asarray(ensemble)
        ensemble_mean = np.mean(ensemble_temperature, axis=0)
        mean_variance = np.mean(np.var(ensemble_temperature, axis=0), axis=1)
        time = deterministic_solution.t
        linear_fit = np.polyfit(time[1:], mean_variance[1:], 1)
        fitted_variance = linear_fit[0] * time + linear_fit[1]
        residual = np.sum((mean_variance[1:] - fitted_variance[1:]) ** 2)
        total = np.sum((mean_variance[1:] - np.mean(mean_variance[1:])) ** 2)
        r_squared = 1.0 - residual / total

        np.testing.assert_allclose(ensemble_mean[-1], deterministic_solution.temperature[-1], atol=0.1)
        self.assertTrue(np.all(np.diff(mean_variance) >= -1e-12))
        self.assertGreater(linear_fit[0], 0.0)
        self.assertGreater(r_squared, 0.9)


if __name__ == "__main__":
    unittest.main()
