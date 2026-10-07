"""Tests for the irregularity diagnostics around the Koopman cycle."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_koopman_irregularity as irregularity


def noisy_clock(
    rng: np.random.Generator, years: int = 6000, period: float = 50.0,
    phase_noise: float = 0.0,
) -> np.ndarray:
    """psi1 with unit radius and a phase random walk of the given step std."""
    theta = 2 * np.pi * np.arange(years) / period
    theta = theta + np.cumsum(phase_noise * rng.standard_normal(years))
    return np.exp(1j * theta)


class ResidualTests(unittest.TestCase):
    def test_composite_removes_a_sharp_switch_that_two_modes_cannot(self) -> None:
        rng = np.random.default_rng(0)
        psi = noisy_clock(rng, phase_noise=0.05)
        phase = irregularity.wrapped_phase(psi)
        # A cell covered for exactly the first 0.4 of each cycle.
        step = (phase < 0.4 * 2 * np.pi).astype(float)
        step -= step.mean()
        harmonic = psi**2
        a = np.linalg.lstsq(
            np.column_stack((np.ones(psi.size), psi, harmonic, psi.conj(), harmonic.conj())),
            step, rcond=None,
        )[0][1:3]

        residuals = irregularity.cycle_residuals(step[:, None], psi, harmonic, a[:, None])

        self.assertGreater(residuals["fractions"]["koopman"], 0.05)
        # Only the bin straddling the switch is not constant.
        self.assertLess(residuals["fractions"]["composite"], 0.03)
        self.assertLessEqual(
            residuals["fractions"]["amplitude"], residuals["fractions"]["composite"] + 1e-12,
        )

    def test_phase_localized_noise_is_found_at_its_phase(self) -> None:
        rng = np.random.default_rng(1)
        psi = noisy_clock(rng, phase_noise=0.02)
        phase = irregularity.wrapped_phase(psi)
        burst = np.abs(np.angle(np.exp(1j * (phase - np.pi)))) < np.pi / 6
        values = np.cos(phase) + np.where(burst, 0.5, 0.05) * rng.standard_normal(psi.size)

        residual = irregularity.cycle_residuals(values, psi, None, None)["amplitude"]
        stats = irregularity.phase_conditioned_stats(residual, phase, 18)

        peak = stats["centres"][np.argmax(stats["variance"])]
        self.assertLess(abs(peak - np.pi), np.pi / 6)
        self.assertGreater(np.max(stats["variance"]) / np.min(stats["variance"]), 30)

    def test_lag_one_by_phase_recovers_an_ar1(self) -> None:
        rng = np.random.default_rng(2)
        psi = noisy_clock(rng, years=20000)
        series = np.zeros(psi.size)
        for t in range(1, psi.size):
            series[t] = 0.7 * series[t - 1] + rng.standard_normal()

        slopes = irregularity.lag_one_by_phase(series, irregularity.wrapped_phase(psi), 6)

        np.testing.assert_allclose(slopes, 0.7, atol=0.05)


class ClockTests(unittest.TestCase):
    def test_phase_diffusion_matches_the_random_walk(self) -> None:
        rng = np.random.default_rng(3)
        step = 0.1
        psi = noisy_clock(rng, years=20000, phase_noise=step)

        diffusion = irregularity.phase_diffusion(psi)

        self.assertAlmostEqual(diffusion["diffusion"], step**2 / 2, delta=0.15 * step**2 / 2)

    def test_cycle_lengths_of_a_regular_clock(self) -> None:
        psi = noisy_clock(np.random.default_rng(4), years=1000, period=40.0)

        lengths = irregularity.cycle_lengths(psi, np.arange(1000))

        np.testing.assert_allclose(lengths, 40.0, atol=1e-9)
        self.assertEqual(lengths.size, 23)  # the pass at year 0 is not a crossing


class EpisodeTests(unittest.TestCase):
    def test_weak_episode_is_found_and_ends_are_not(self) -> None:
        years = np.arange(3000)
        psi = np.exp(2j * np.pi * years / 50)
        psi[1000:1400] *= 0.3

        episodes = irregularity.weak_episodes(psi, years)

        self.assertEqual(episodes["episodes"].shape, (1, 2))
        start, stop = episodes["episodes"][0]
        self.assertLess(abs(start - 1000), 40)
        self.assertLess(abs(stop - 1400), 40)


class FlipTests(unittest.TestCase):
    def test_flip_jitter_is_zero_for_a_regular_switch(self) -> None:
        psi = noisy_clock(np.random.default_rng(5), years=2000)
        phase = irregularity.wrapped_phase(psi)
        cover = (phase < np.pi).astype(float)[:, None]

        flips = irregularity.ice_flip_phases(cover, psi)

        self.assertLess(flips["onset_std"][0], 0.07)
        self.assertLess(flips["retreat_std"][0], 0.07)
        self.assertAlmostEqual(flips["onset_per_cycle"][0], 1.0, delta=0.05)
        self.assertLess(abs(np.angle(np.exp(1j * (flips["retreat_phase"][0] - np.pi)))), 0.1)

    def test_flip_jitter_grows_with_onset_noise(self) -> None:
        rng = np.random.default_rng(6)
        psi = noisy_clock(rng, years=20000)
        phase = irregularity.wrapped_phase(psi)
        cycle = np.floor(np.unwrap(np.angle(psi)) / (2 * np.pi)).astype(int)
        jitter = 0.3
        threshold = np.pi + jitter * rng.standard_normal(cycle.max() + 1)
        cover = (phase < threshold[cycle]).astype(float)[:, None]

        flips = irregularity.ice_flip_phases(cover, psi)

        self.assertAlmostEqual(flips["retreat_std"][0], jitter, delta=0.07)
        self.assertLess(flips["onset_std"][0], 0.07)


class FloquetTests(unittest.TestCase):
    def test_constant_contraction_gives_known_multipliers(self) -> None:
        rng = np.random.default_rng(7)
        psi = noisy_clock(rng, years=30000, period=50.0)
        phase = irregularity.wrapped_phase(psi)
        rates = np.array([0.8, 0.5])
        pcs = np.zeros((psi.size, 2))
        for t in range(1, psi.size):
            pcs[t] = rates * pcs[t - 1] + rng.standard_normal(2)

        model = irregularity.fit_phase_linear_model(pcs, phase, harmonics=1)
        result = irregularity.floquet_multipliers(model, 2 * np.pi / 50)

        np.testing.assert_allclose(np.abs(result["multipliers"]), rates**50, rtol=0.6, atol=1e-6)
        np.testing.assert_allclose(result["rate"], np.log(rates), atol=0.02)
        np.testing.assert_allclose(result["local_growth"], 0.8, atol=0.03)

    def test_phase_dependent_growth_is_localized(self) -> None:
        rng = np.random.default_rng(8)
        psi = noisy_clock(rng, years=30000, period=40.0)
        phase = irregularity.wrapped_phase(psi)
        series = np.zeros((psi.size, 1))
        for t in range(1, psi.size):
            series[t] = (0.6 + 0.3 * np.cos(phase[t - 1])) * series[t - 1] + rng.standard_normal()

        model = irregularity.fit_phase_linear_model(series, phase, harmonics=1)
        result = irregularity.floquet_multipliers(model, 2 * np.pi / 40)

        expected = np.abs(0.6 + 0.3 * np.cos(result["phase_grid"]))
        np.testing.assert_allclose(result["local_growth"], expected, atol=0.04)


class ResidualKoopmanTests(unittest.TestCase):
    def test_transverse_rate_of_a_phase_driven_ar1(self) -> None:
        rng = np.random.default_rng(10)
        psi = noisy_clock(rng, years=1500, period=40.0, phase_noise=0.02)
        phase = irregularity.wrapped_phase(psi)
        pcs = np.zeros((psi.size, 2))
        for t in range(1, psi.size):
            pcs[t] = np.array([0.85, 0.4]) * pcs[t - 1] + rng.standard_normal(2)

        koopman = irregularity.fit_residual_koopman(pcs, phase, lag=1)
        slowest = irregularity.transverse_modes(koopman)[0]

        self.assertAlmostEqual(koopman["rates"][slowest].real, np.log(0.85), delta=0.06)
        self.assertLess(koopman["clock_share"][slowest], 0.2)
        clock = np.flatnonzero(koopman["clock_share"] > 0.8)
        self.assertTrue(np.any(np.abs(koopman["rates"][clock].imag - 2 * np.pi / 40) < 0.01))

    def test_lagged_correlation_finds_the_delay(self) -> None:
        rng = np.random.default_rng(11)
        first = rng.standard_normal(3000)
        second = np.r_[np.zeros(7), -first[:-7]] + 0.1 * rng.standard_normal(3000)

        correlation = irregularity.lagged_correlation(first, second, np.arange(-10, 11))

        self.assertEqual(int(np.argmin(correlation)) - 10, 7)


class ClockErrorTests(unittest.TestCase):
    def test_clock_error_correlation_is_removed(self) -> None:
        rng = np.random.default_rng(12)
        years = 6000
        theta = 2 * np.pi * np.arange(years) / 50
        delta = np.zeros(years)
        for t in range(1, years):
            delta[t] = 0.9 * delta[t - 1] + 0.05 * rng.standard_normal()
        true_phase = theta + delta
        psi = np.exp(1j * theta)  # the clock does not see delta
        # Many rows with different phase lags, all driven by the true phase.
        offsets = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        rows = np.cos(true_phase[:, None] - offsets[None]) + 0.05 * rng.standard_normal((years, 12))
        ice = np.cos(true_phase - 1.0) + 0.05 * rng.standard_normal(years)
        residuals = irregularity.cycle_residuals(rows, psi, None, None)["composite"]
        ice_residual = irregularity.cycle_residuals(ice[:, None], psi, None, None)["composite"]
        slope, shape = irregularity.cycle_tangent_and_shape(rows, psi)
        ice_slope, ice_shape = irregularity.cycle_tangent_and_shape(ice[:, None], psi)
        before = abs(np.corrcoef(ice_residual[:, 0], residuals[:, 3])[0, 1])

        estimate, alpha = irregularity.estimate_clock_error(
            [residuals], [slope], [shape], [np.ones(12)],
        )
        ice_clean = irregularity.remove_clock_error(ice_residual, ice_slope, ice_shape, estimate, alpha)
        rows_clean = irregularity.remove_clock_error(residuals, slope, shape, estimate, alpha)
        after = abs(np.corrcoef(ice_clean[:, 0], rows_clean[:, 3])[0, 1])

        self.assertGreater(np.corrcoef(estimate, delta)[0, 1], 0.85)
        self.assertGreater(before, 0.5)
        self.assertLess(after, 0.1)

    def test_surrogate_threshold_exceeds_independent_correlation(self) -> None:
        rng = np.random.default_rng(13)
        x = np.zeros(3000)
        y = np.zeros(3000)
        for t in range(1, 3000):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal()
            y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        lags = np.arange(-10, 11)

        threshold = irregularity.surrogate_threshold(x, y, lags, count=100)

        self.assertGreater(threshold, 0.05)
        self.assertLess(threshold, 0.3)

    def test_lagged_regression_recovers_slope(self) -> None:
        rng = np.random.default_rng(14)
        x = rng.standard_normal(5000)
        field = np.zeros((5000, 2))
        field[3:, 0] = 2.0 * x[:-3]

        maps = irregularity.lagged_regression_maps(x, field, np.array([0, 3]))

        self.assertAlmostEqual(maps[1, 0], 2.0 * x.std(), delta=0.05)
        self.assertAlmostEqual(maps[0, 0], 0.0, delta=0.1)


class BudgetRegressionTests(unittest.TestCase):
    def test_storage_response_to_a_lagged_flux_pulse(self) -> None:
        rng = np.random.default_rng(15)
        years = 4000
        psi = noisy_clock(rng, years=years, period=50.0)
        ice = rng.standard_normal(years)
        surface = np.zeros(years)
        surface[2:] = -3.0 * ice[:-2]
        advection = rng.standard_normal(years)
        storage = surface + advection
        budget = {
            "storage": storage, "surface": surface, "advection": advection,
            "residual": storage - surface - advection,
            "heat_content": np.cumsum(storage) * 365.25 * 86400 * 1e12,
        }

        result = irregularity.budget_regression(budget, ice, psi, np.arange(-2, 5))

        lag_two = list(result["lags"]).index(2)
        self.assertAlmostEqual(result["surface"][lag_two], -3.0, delta=0.15)
        self.assertAlmostEqual(result["storage"][lag_two], -3.0, delta=0.2)
        np.testing.assert_allclose(result["residual"], 0.0, atol=1e-9)


class EofTests(unittest.TestCase):
    def test_eofs_reconstruct_their_blocks(self) -> None:
        rng = np.random.default_rng(9)
        first, second = rng.standard_normal((300, 4)), rng.standard_normal((300, 3))
        weights = [np.full(4, 0.25), np.full(3, 1 / 3)]

        eofs = irregularity.residual_eofs([first, second], weights, count=7)

        np.testing.assert_allclose(
            eofs["pcs"] @ eofs["patterns"][0], first - first.mean(axis=0), atol=1e-10,
        )
        self.assertAlmostEqual(float(eofs["fractions"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
