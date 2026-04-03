"""Tests for high-level model workflows."""

import tempfile
import unittest

import numpy as np
import xarray as xr

from gsebm.parameters import RunSettings
from gsebm.run import (
    edge_state_dataset,
    run_edge_state,
    run_warm_cold_state,
    save_edge_state_dataset,
    save_warm_cold_state_dataset,
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


if __name__ == "__main__":
    unittest.main()
