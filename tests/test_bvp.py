"""Tests for the steady-state BVP components."""

import unittest

import numpy as np

from gsebm.bvp import (
    build_bvp_fields,
    build_bvp_grid,
    build_bvp_initial_guess,
    build_bvp_problem,
    solve_temperature_bvp,
)
from gsebm.ivp import solve_temperature_ivp
from gsebm.parameters import RunSettings


class BVPTest(unittest.TestCase):
    def test_build_bvp_grid_matches_ivp_default_size(self) -> None:
        grid = build_bvp_grid()
        self.assertEqual(grid.shape, (205,))
        self.assertTrue(np.all(np.diff(grid) > 0.0))

    def test_build_bvp_fields_are_callable(self) -> None:
        fields = build_bvp_fields()
        x = np.linspace(-0.9, 0.9, 7)
        self.assertEqual(np.asarray(fields.solar_irradiance(x)).shape, x.shape)
        self.assertEqual(np.asarray(fields.sensible_heat_flux_derivative(x)).shape, x.shape)

    def test_scalar_initial_guess_has_zero_derivative(self) -> None:
        x = np.linspace(-1.0, 1.0, 9)
        guess = build_bvp_initial_guess(x, kind="scalar", scalar_value=280.0)
        np.testing.assert_allclose(guess.values[0], 280.0)
        np.testing.assert_allclose(guess.values[1], 0.0)

    def test_custom_initial_guess_is_interpolated(self) -> None:
        x = np.linspace(-1.0, 1.0, 9)
        guess = build_bvp_initial_guess(
            x,
            kind="custom",
            custom_x=np.array([-1.0, 0.0, 1.0]),
            custom_temperature=np.array([250.0, 280.0, 260.0]),
        )
        self.assertEqual(guess.values.shape, (2, x.size))
        self.assertTrue(np.isfinite(guess.values).all())

    def test_problem_ode_is_finite(self) -> None:
        problem = build_bvp_problem()
        x = build_bvp_grid()
        y = np.vstack((np.full_like(x, 280.0), np.zeros_like(x)))
        dydx = problem.ode(x, y)
        self.assertEqual(dydx.shape, y.shape)
        self.assertTrue(np.isfinite(dydx).all())

    def test_boundary_conditions_match_zero_slope_requirement(self) -> None:
        residual = build_bvp_problem().boundary_conditions(
            np.array([280.0, 0.0]),
            np.array([280.0, 0.0]),
        )
        np.testing.assert_allclose(residual, 0.0)

    def test_ivp_based_initial_guess_uses_final_ivp_state(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        ivp_solution = solve_temperature_ivp(
            settings=settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
            method="BDF",
        )
        guess = build_bvp_initial_guess(
            ivp_solution.x,
            kind="ivp",
            ivp_solution=ivp_solution,
            settings=settings,
        )
        self.assertEqual(guess.values.shape, (2, ivp_solution.x.size))
        self.assertTrue(np.isfinite(guess.values).all())

    def test_short_bvp_solve_returns_profile(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        ivp_solution = solve_temperature_ivp(
            settings=settings,
            initial_condition_kind="scalar",
            initial_scalar_value=280.0,
            method="BDF",
        )
        solution = solve_temperature_bvp(
            settings=settings,
            ivp_solution=ivp_solution,
            initial_guess_kind="ivp",
            tolerance=1e-2,
        )
        self.assertEqual(solution.temperature.shape, solution.x.shape)
        self.assertEqual(solution.temperature_x.shape, solution.x.shape)
        self.assertTrue(np.isfinite(solution.temperature).all())


if __name__ == "__main__":
    unittest.main()
