"""Tests for high-level model workflows."""

import tempfile
import unittest

import numpy as np
import xarray as xr

from gsebm.parameters import RunSettings
from gsebm.run import (
    edge_mu_bifurcation_dataset,
    edge_state_dataset,
    run_edge_mu_bifurcation,
    run_edge_state,
    run_warm_cold_mu_bifurcation,
    run_warm_cold_state,
    save_edge_mu_bifurcation_dataset,
    save_edge_state_dataset,
    save_warm_cold_mu_bifurcation_dataset,
    save_warm_cold_state_dataset,
    warm_cold_mu_bifurcation_dataset,
    warm_cold_state_dataset,
)


class RunWorkflowTest(unittest.TestCase):
    def test_run_warm_cold_state_returns_two_ivp_branches(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_state(
            settings=settings,
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        self.assertEqual(solutions.warm_state.temperature.shape, (5, solutions.warm_state.x.size))
        self.assertEqual(solutions.cold_state.temperature.shape, (5, solutions.cold_state.x.size))
        self.assertTrue(np.isfinite(solutions.warm_state.temperature).all())
        self.assertTrue(np.isfinite(solutions.cold_state.temperature).all())

    def test_run_edge_state_returns_bvp_branch(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solution = run_edge_state(
            settings=settings,
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
        )

        self.assertEqual(solution.edge_state.temperature.shape, solution.edge_state.x.shape)
        self.assertTrue(np.isfinite(solution.edge_state.temperature).all())

    def test_warm_cold_state_dataset_has_expected_dimensions(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_state(
            settings=settings,
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )
        dataset = warm_cold_state_dataset(
            solutions,
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        self.assertEqual(dataset["warm_state_temperature"].dims, ("time", "latitude"))
        self.assertEqual(dataset["cold_state_temperature"].dims, ("time", "latitude"))
        self.assertIn("param_mu", dataset.attrs)
        self.assertIn("setting_final_time", dataset.attrs)
        self.assertEqual(float(dataset.attrs["warm_initial_temperature"]), 300.0)

    def test_save_warm_cold_state_dataset_writes_netcdf(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_state(
            settings=settings,
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_warm_cold_state_dataset(
                solutions,
                filename="warm_cold_state.nc",
                warm_initial_temperature=300.0,
                cold_initial_temperature=250.0,
                ivp_method="BDF",
                output_dir=temp_dir,
            )
            self.assertTrue(path.exists())
            dataset = xr.open_dataset(path, engine="scipy")
            try:
                self.assertIn("warm_state_temperature", dataset.data_vars)
                self.assertIn("cold_state_temperature", dataset.data_vars)
            finally:
                dataset.close()

    def test_edge_state_dataset_has_expected_dimensions(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solution = run_edge_state(
            settings=settings,
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
        )
        dataset = edge_state_dataset(
            solution,
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
            bvp_max_nodes=10000,
        )

        self.assertEqual(dataset["edge_state_temperature"].dims, ("latitude",))
        self.assertEqual(dataset["edge_state_temperature_derivative"].dims, ("latitude",))
        self.assertIn("param_mu", dataset.attrs)
        self.assertIn("setting_final_time", dataset.attrs)
        self.assertEqual(float(dataset.attrs["edge_initial_temperature"]), 260.0)

    def test_save_edge_state_dataset_writes_netcdf(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solution = run_edge_state(
            settings=settings,
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_edge_state_dataset(
                solution,
                filename="edge_state.nc",
                edge_initial_temperature=260.0,
                bvp_tolerance=1e-2,
                bvp_max_nodes=10000,
                output_dir=temp_dir,
            )
            self.assertTrue(path.exists())
            dataset = xr.open_dataset(path, engine="scipy")
            try:
                self.assertIn("edge_state_temperature", dataset.data_vars)
                self.assertIn("edge_state_temperature_derivative", dataset.data_vars)
            finally:
                dataset.close()

    def test_run_warm_cold_mu_bifurcation_returns_expected_branch_counts(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        self.assertEqual(solutions.mu_values, (0.95, 1.0, 1.05))
        self.assertEqual(len(solutions.warm_states), 3)
        self.assertEqual(len(solutions.cold_states), 3)

    def test_run_edge_mu_bifurcation_returns_expected_branch_counts(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_edge_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
            bvp_max_nodes=20000,
        )

        self.assertEqual(solutions.mu_values, (0.95, 1.0, 1.05))
        self.assertEqual(len(solutions.edge_states), 3)
        self.assertTrue(all(edge_state is not None for edge_state in solutions.edge_states))

    def test_warm_cold_mu_bifurcation_dataset_has_expected_dimensions(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )
        dataset = warm_cold_mu_bifurcation_dataset(
            solutions,
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        self.assertEqual(dataset["warm_state_temperature"].dims, ("mu", "time", "latitude"))
        self.assertEqual(dataset["cold_state_temperature"].dims, ("mu", "time", "latitude"))
        self.assertNotIn("edge_state_temperature", dataset.data_vars)
        self.assertNotIn("param_mu", dataset.attrs)
        self.assertIn("setting_final_time", dataset.attrs)

    def test_edge_mu_bifurcation_dataset_has_expected_dimensions(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_edge_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
            bvp_max_nodes=20000,
        )
        dataset = edge_mu_bifurcation_dataset(
            solutions,
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
            bvp_max_nodes=20000,
        )

        self.assertEqual(dataset["edge_state_temperature"].dims, ("mu", "latitude"))
        self.assertEqual(dataset["edge_state_temperature_derivative"].dims, ("mu", "latitude"))
        self.assertNotIn("param_mu", dataset.attrs)
        self.assertIn("setting_final_time", dataset.attrs)

    def test_save_warm_cold_mu_bifurcation_dataset_writes_netcdf(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            warm_initial_temperature=300.0,
            cold_initial_temperature=250.0,
            ivp_method="BDF",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_warm_cold_mu_bifurcation_dataset(
                solutions,
                filename="warm_cold_mu_bifurcation.nc",
                warm_initial_temperature=300.0,
                cold_initial_temperature=250.0,
                ivp_method="BDF",
                output_dir=temp_dir,
            )
            self.assertTrue(path.exists())
            dataset = xr.open_dataset(path, engine="scipy")
            try:
                self.assertIn("mu", dataset.coords)
                self.assertIn("warm_state_temperature", dataset.data_vars)
                self.assertNotIn("edge_state_temperature", dataset.data_vars)
            finally:
                dataset.close()

    def test_save_edge_mu_bifurcation_dataset_writes_netcdf(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_edge_mu_bifurcation(
            settings=settings,
            mu_values=(0.95, 1.0, 1.05),
            edge_initial_temperature=260.0,
            bvp_tolerance=1e-2,
            bvp_max_nodes=20000,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_edge_mu_bifurcation_dataset(
                solutions,
                filename="edge_mu_bifurcation.nc",
                edge_initial_temperature=260.0,
                bvp_tolerance=1e-2,
                bvp_max_nodes=20000,
                output_dir=temp_dir,
            )
            self.assertTrue(path.exists())
            dataset = xr.open_dataset(path, engine="scipy")
            try:
                self.assertIn("mu", dataset.coords)
                self.assertIn("edge_state_temperature", dataset.data_vars)
            finally:
                dataset.close()


if __name__ == "__main__":
    unittest.main()
