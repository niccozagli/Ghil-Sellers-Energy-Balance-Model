"""Tests for diagnostic quantities derived from model states."""

import tempfile
import unittest

import numpy as np
import xarray as xr

from gsebm.diagnostics import (
    CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER,
    build_bvp_problem_from_dataset,
    build_ivp_operator_from_dataset,
    edge_state_albedo_from_dataset,
    edge_state_heat_transfer_from_dataset,
    meridional_heat_transfer_rate_watts_per_square_meter,
    model_parameters_from_dataset_attrs,
    pde_meridional_flux,
    run_settings_from_dataset_attrs,
    warm_cold_state_albedo_from_dataset,
    warm_cold_state_heat_transfer_from_dataset,
)
from gsebm.parameters import RunSettings, default_model_parameters
from gsebm.physics import total_diffusivity
from gsebm.run import (
    run_edge_state,
    run_warm_cold_state,
    save_edge_state_dataset,
    save_warm_cold_state_dataset,
)


class DiagnosticsTest(unittest.TestCase):
    def test_pde_meridional_flux_matches_ivp_flux_definition(self) -> None:
        params = default_model_parameters()
        latitude = np.array([0.0, 0.5])
        temperature = np.array([280.0, 270.0])
        temperature_x = np.array([2.0, -1.0])
        k1 = np.array([3.0e-5, 2.0e-5])
        k2 = np.array([0.0, 1.0e-2])

        expected = (
            np.cos(0.5 * np.pi * latitude)
            * total_diffusivity(temperature, k1, k2, params)
            * temperature_x
            * (2.0 / np.pi) ** 2
        )
        np.testing.assert_allclose(
            pde_meridional_flux(latitude, temperature, temperature_x, k1, k2, params),
            expected,
        )

    def test_meridional_heat_transfer_rate_matches_conversion_formula(self) -> None:
        params = default_model_parameters()
        latitude = np.array([0.0])
        temperature = np.array([280.0])
        temperature_x = np.array([2.0])
        k1 = np.array([3.0e-5])
        k2 = np.array([0.0])

        flux = pde_meridional_flux(latitude, temperature, temperature_x, k1, k2, params)
        expected = (
            -flux
            * np.pi
            / 2.0
            * CALORIES_PER_SQUARE_CENTIMETER_SECOND_TO_WATTS_PER_SQUARE_METER
        )
        np.testing.assert_allclose(
            meridional_heat_transfer_rate_watts_per_square_meter(
                latitude, temperature, temperature_x, k1, k2, params
            ),
            expected,
        )

    def test_dataset_attribute_reconstruction_matches_original_configuration(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        solutions = run_warm_cold_state(settings=settings, ivp_method="BDF")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_warm_cold_state_dataset(
                solutions,
                filename="warm_cold_state.nc",
                warm_initial_temperature=300.0,
                cold_initial_temperature=250.0,
                ivp_method="BDF",
                output_dir=temp_dir,
            )
            dataset = xr.open_dataset(path, engine="scipy")
            try:
                params = model_parameters_from_dataset_attrs(dataset)
                recovered_settings = run_settings_from_dataset_attrs(dataset)
                self.assertEqual(params.mu, solutions.params.mu)
                self.assertEqual(recovered_settings.final_time, solutions.settings.final_time)
            finally:
                dataset.close()

    def test_dataset_reconstruction_builds_solver_objects(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        warm_cold_solutions = run_warm_cold_state(settings=settings, ivp_method="BDF")
        edge_solution = run_edge_state(settings=settings, edge_initial_temperature=260.0, bvp_tolerance=1e-2)

        with tempfile.TemporaryDirectory() as temp_dir:
            warm_cold_path = save_warm_cold_state_dataset(
                warm_cold_solutions,
                filename="warm_cold_state.nc",
                warm_initial_temperature=300.0,
                cold_initial_temperature=250.0,
                ivp_method="BDF",
                output_dir=temp_dir,
            )
            edge_path = save_edge_state_dataset(
                edge_solution,
                filename="edge_state.nc",
                edge_initial_temperature=260.0,
                bvp_tolerance=1e-2,
                bvp_max_nodes=10000,
                output_dir=temp_dir,
            )
            warm_cold_dataset = xr.open_dataset(warm_cold_path, engine="scipy")
            edge_dataset = xr.open_dataset(edge_path, engine="scipy")
            try:
                operator = build_ivp_operator_from_dataset(warm_cold_dataset)
                problem = build_bvp_problem_from_dataset(edge_dataset)
                self.assertEqual(operator.x.shape, warm_cold_dataset["latitude"].shape)
                self.assertAlmostEqual(problem.params.mu, edge_solution.params.mu)
            finally:
                warm_cold_dataset.close()
                edge_dataset.close()

    def test_dataset_diagnostics_have_expected_shapes(self) -> None:
        settings = RunSettings(final_time=1e3, time_output_count=5)
        warm_cold_solutions = run_warm_cold_state(settings=settings, ivp_method="BDF")
        edge_solution = run_edge_state(settings=settings, edge_initial_temperature=260.0, bvp_tolerance=1e-2)

        with tempfile.TemporaryDirectory() as temp_dir:
            warm_cold_path = save_warm_cold_state_dataset(
                warm_cold_solutions,
                filename="warm_cold_state.nc",
                warm_initial_temperature=300.0,
                cold_initial_temperature=250.0,
                ivp_method="BDF",
                output_dir=temp_dir,
            )
            edge_path = save_edge_state_dataset(
                edge_solution,
                filename="edge_state.nc",
                edge_initial_temperature=260.0,
                bvp_tolerance=1e-2,
                bvp_max_nodes=10000,
                output_dir=temp_dir,
            )
            warm_cold_dataset = xr.open_dataset(warm_cold_path, engine="scipy")
            edge_dataset = xr.open_dataset(edge_path, engine="scipy")
            try:
                warm_cold_albedo = warm_cold_state_albedo_from_dataset(warm_cold_dataset)
                warm_cold_heat_transfer = warm_cold_state_heat_transfer_from_dataset(warm_cold_dataset)
                edge_albedo = edge_state_albedo_from_dataset(edge_dataset)
                edge_heat_transfer = edge_state_heat_transfer_from_dataset(edge_dataset)

                self.assertEqual(
                    warm_cold_albedo["warm_state_albedo"].shape,
                    warm_cold_dataset["warm_state_temperature"].shape,
                )
                self.assertEqual(
                    warm_cold_heat_transfer["cold_state_heat_transfer"].shape,
                    warm_cold_dataset["cold_state_temperature"].shape,
                )
                self.assertEqual(edge_albedo.shape, edge_dataset["edge_state_temperature"].shape)
                self.assertEqual(edge_heat_transfer.shape, edge_dataset["edge_state_temperature"].shape)
            finally:
                warm_cold_dataset.close()
                edge_dataset.close()


if __name__ == "__main__":
    unittest.main()
