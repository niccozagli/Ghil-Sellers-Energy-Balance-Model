"""Tests for the IVP discretization and time integration."""

import unittest

import numpy as np

from gsebm.ivp import (
    build_initial_temperature,
    build_ivp_grid,
    build_ivp_operator,
    build_ivp_time_grid,
    solve_temperature_ivp,
)
from gsebm.parameters import RunSettings
from gsebm.time import YEAR


class IVPTest(unittest.TestCase):
    def test_build_ivp_grid_matches_default_size(self) -> None:
        grid = build_ivp_grid()
        self.assertEqual(grid.shape, (205,))
        self.assertTrue(np.all(np.diff(grid) > 0.0))

    def test_build_ivp_time_grid_matches_default_size(self) -> None:
        times = build_ivp_time_grid()
        self.assertEqual(times.shape, (101,))
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 35.0 * YEAR)

    def test_default_initial_temperature_uses_uniform_profile(self) -> None:
        x = np.linspace(-0.5, 0.5, 7)
        initial = build_initial_temperature(x, kind="scalar", scalar_value=280.0)
        np.testing.assert_allclose(initial, np.full_like(x, 280.0))

    def test_observational_initial_temperature_is_interpolated(self) -> None:
        x = np.linspace(-1.0, 1.0, 9)
        initial = build_initial_temperature(x, kind="default")
        self.assertEqual(initial.shape, x.shape)
        self.assertTrue(np.isfinite(initial).all())
        self.assertGreater(float(np.max(initial)), float(np.min(initial)))

    def test_custom_initial_temperature_is_interpolated(self) -> None:
        x = np.linspace(-1.0, 1.0, 5)
        initial = build_initial_temperature(
            x,
            kind="custom",
            custom_temperature=np.array([250.0, 280.0, 300.0]),
            initial_x=np.array([-1.0, 0.0, 1.0]),
        )
        self.assertEqual(initial.shape, x.shape)
        self.assertAlmostEqual(float(initial[0]), 250.0)
        self.assertAlmostEqual(float(initial[-1]), 300.0)

    def test_invalid_initial_condition_kind_is_rejected(self) -> None:
        x = np.linspace(-1.0, 1.0, 5)
        with self.assertRaises(ValueError):
            build_initial_temperature(x, kind="invalid")  # type: ignore[arg-type]

    def test_transport_term_vanishes_for_constant_temperature(self) -> None:
        operator = build_ivp_operator()
        temperature = np.full_like(operator.x, 280.0)
        transport = operator.transport_term(temperature)
        np.testing.assert_allclose(transport, 0.0, atol=1e-12)

    def test_rhs_is_finite(self) -> None:
        operator = build_ivp_operator()
        temperature = np.full_like(operator.x, 280.0)
        rhs = operator.rhs(0.0, temperature)
        self.assertEqual(rhs.shape, temperature.shape)
        self.assertTrue(np.isfinite(rhs).all())

    def test_short_ivp_solve_returns_temperature_history(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solution = solve_temperature_ivp(
            settings=settings,
            method="BDF",
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
        )
        self.assertEqual(solution.t.shape, (5,))
        self.assertEqual(solution.temperature.shape, (5, solution.x.size))
        self.assertTrue(np.isfinite(solution.temperature).all())


if __name__ == "__main__":
    unittest.main()
