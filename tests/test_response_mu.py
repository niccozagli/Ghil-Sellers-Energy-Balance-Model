"""Tests for the mu response Green's-function script."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from gsebm.parameters import ModelParameters, RunSettings, StochasticRunSettings


def _load_response_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "Response_mu.py"
    spec = importlib.util.spec_from_file_location("response_mu_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Response_mu.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sample_dataset(path: Path, *, attrs: dict[str, object] | None = None) -> None:
    latitude = np.linspace(-0.8, 0.8, 5)
    temperature = np.vstack(
        (
            280.0 + np.linspace(0.0, 1.0, latitude.size),
            281.0 + np.linspace(0.0, 1.0, latitude.size),
        )
    )
    dataset = xr.Dataset(
        data_vars={"temperature": (("sample", "latitude"), temperature)},
        coords={"sample": np.arange(temperature.shape[0]), "latitude": latitude},
        attrs={} if attrs is None else attrs,
    )
    dataset.to_netcdf(path, engine="scipy")


class ResponseMuScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_response_module()

    def test_load_initial_condition_samples_requires_temperature_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "samples.nc"
            dataset = xr.Dataset(
                data_vars={"not_temperature": (("sample", "latitude"), np.ones((1, 2)))},
                coords={"sample": [0], "latitude": [-0.5, 0.5]},
            )
            dataset.to_netcdf(path, engine="scipy")

            with self.assertRaisesRegex(ValueError, "temperature"):
                self.module.load_initial_condition_samples(path)

    def test_load_initial_condition_samples_requires_expected_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "samples.nc"
            dataset = xr.Dataset(
                data_vars={"temperature": (("time", "latitude"), np.ones((1, 2)))},
                coords={"time": [0.0], "latitude": [-0.5, 0.5]},
            )
            dataset.to_netcdf(path, engine="scipy")

            with self.assertRaisesRegex(ValueError, "dimensions"):
                self.module.load_initial_condition_samples(path)

    def test_response_dataset_green_function_matches_central_difference(self) -> None:
        params = ModelParameters()
        settings = RunSettings(final_time=2.0e3)
        stochastic_settings = StochasticRunSettings(dt=1.0e3, noise_seed=4)
        time = np.array([0.0, 1.0e3, 2.0e3])
        latitude = np.array([-0.5, 0.5])
        plus = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        minus = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]])
        stationary = np.array([1.0, 2.0])
        stationary_std = np.array([0.1, 0.2])

        dataset = self.module.response_dataset(
            time=time,
            latitude=latitude,
            plus_mean_temperature=plus,
            minus_mean_temperature=minus,
            stationary_temperature=stationary,
            stationary_temperature_std=stationary_std,
            params=params,
            settings=settings,
            stochastic_settings=stochastic_settings,
            epsilon=0.25,
            input_path="samples.nc",
            sample_count=3,
            base_seed=4,
        )

        expected = (plus - minus) / 0.5
        np.testing.assert_allclose(dataset["green_function_temperature"].values, expected)
        self.assertEqual(dataset["plus_mean_temperature"].dims, ("time", "latitude"))
        self.assertEqual(dataset["stationary_temperature"].dims, ("latitude",))
        self.assertEqual(dataset["stationary_temperature"].attrs["units"], "K")
        np.testing.assert_allclose(dataset["stationary_temperature"].values, stationary)
        np.testing.assert_allclose(
            dataset["stationary_temperature_std"].values, stationary_std
        )
        self.assertEqual(dataset.attrs["sample_count"], 3)
        self.assertIn("param_mu", dataset.attrs)
        self.assertIn("mu_plus_first_step", dataset.attrs)
        self.assertIn("mu_minus_first_step", dataset.attrs)
        self.assertEqual(dataset["green_function_temperature"].attrs["units"], "K s^-1")
        self.assertIn("green_function_units_note", dataset.attrs)

    def test_stationary_temperature_from_samples_returns_ensemble_mean_and_std(self) -> None:
        samples = self.module.InitialConditionSamples(
            latitude=np.array([-0.5, 0.5]),
            temperature=np.array([[280.0, 282.0], [282.0, 286.0]]),
            attrs={},
        )

        mean, std = self.module.stationary_temperature_from_samples(samples)

        np.testing.assert_allclose(mean, np.array([281.0, 284.0]))
        np.testing.assert_allclose(std, np.array([1.0, 2.0]))

    def test_estimate_saved_temperature_memory_bytes_counts_saved_times_and_latitudes(self) -> None:
        estimated = self.module.estimate_saved_temperature_memory_bytes(
            final_time=3.0,
            dt=1.0,
            save_every=2,
            latitude_count=5,
        )

        self.assertEqual(estimated, 3 * 5 * 8)

    def test_response_time_grid_records_first_step_before_regular_sampling(self) -> None:
        time = self.module.response_time_grid(
            final_time=61.0,
            dt=1.0,
            save_every=30,
        )

        np.testing.assert_allclose(time, np.array([1.0, 30.0, 60.0, 61.0]))

    def test_select_evenly_spaced_post_transient_samples_filters_and_samples(self) -> None:
        time = np.arange(6.0)
        temperature = np.arange(18.0).reshape(6, 3)

        sample_time, sample_temperature = self.module.select_evenly_spaced_post_transient_samples(
            time=time,
            temperature=temperature,
            transient_time=2.0,
            sample_count=3,
        )

        np.testing.assert_allclose(sample_time, np.array([2.0, 4.0, 5.0]))
        np.testing.assert_allclose(sample_temperature, temperature[[2, 4, 5]])

    def test_select_evenly_spaced_post_transient_samples_requires_enough_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough post-transient"):
            self.module.select_evenly_spaced_post_transient_samples(
                time=np.array([0.0, 1.0]),
                temperature=np.ones((2, 3)),
                transient_time=1.0,
                sample_count=2,
            )

    def test_sampled_initial_condition_dataset_can_be_saved_and_loaded(self) -> None:
        samples = self.module.InitialConditionSamples(
            latitude=np.array([-0.5, 0.5]),
            temperature=np.array([[280.0, 281.0], [282.0, 283.0]]),
            attrs={},
        )
        dataset = self.module.sampled_initial_condition_dataset(
            samples,
            sample_time=np.array([10.0, 20.0]),
            params=ModelParameters(mu=0.97),
            baseline_settings=RunSettings(final_time=30.0),
            stochastic_settings=StochasticRunSettings(dt=1.0, noise_seed=3),
            baseline_final_time=30.0,
            transient_time=10.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.module.save_sampled_initial_condition_dataset(
                dataset,
                Path(temp_dir) / "samples.nc",
            )
            loaded = xr.open_dataset(path, engine="scipy")
            try:
                self.assertEqual(loaded["temperature"].dims, ("sample", "latitude"))
                self.assertEqual(loaded["sample_time"].dims, ("sample",))
                self.assertEqual(float(loaded.attrs["param_mu"]), 0.97)
                self.assertEqual(loaded.attrs["sampling_strategy"], "even_spacing")
            finally:
                loaded.close()

    def test_compute_response_means_uses_pairwise_common_seeds(self) -> None:
        samples = self.module.InitialConditionSamples(
            latitude=np.array([-0.5, 0.5]),
            temperature=np.array([[280.0, 281.0], [282.0, 283.0]]),
            attrs={},
        )
        settings = RunSettings(final_time=2.0e3)
        stochastic_settings = StochasticRunSettings(dt=1.0e3, save_every=1)
        calls = []

        def fake_run_signed_response(job):
            calls.append((job.sample_index, job.sign, job.seed))
            time = self.module.response_time_grid(
                job.settings.final_time,
                job.stochastic_settings.dt,
                job.stochastic_settings.save_every,
            )
            temperature = np.full((time.size, job.latitude.size), float(job.sign))
            return self.module.ResponseResult(
                sample_index=job.sample_index,
                sign=job.sign,
                time=time,
                temperature=temperature,
            )

        with patch.object(self.module, "run_signed_response", side_effect=fake_run_signed_response):
            time, plus_mean, minus_mean = self.module.compute_response_means(
                samples,
                params=ModelParameters(),
                settings=settings,
                stochastic_settings=stochastic_settings,
                epsilon=0.015,
                base_seed=10,
                workers=1,
            )

        self.assertEqual(time.shape, (2,))
        np.testing.assert_allclose(plus_mean, 1.0)
        np.testing.assert_allclose(minus_mean, -1.0)
        self.assertEqual(
            sorted(calls),
            [(0, -1, 10), (0, 1, 10), (1, -1, 11), (1, 1, 11)],
        )

    def test_run_signed_response_produces_expected_shape_for_tiny_zero_noise_run(self) -> None:
        latitude = np.linspace(-0.8, 0.8, 5)
        job = self.module.ResponseJob(
            sample_index=0,
            sign=1,
            seed=3,
            latitude=latitude,
            initial_temperature=np.full(latitude.size, 280.0),
            params=ModelParameters(),
            settings=RunSettings(final_time=2.0e3),
            stochastic_settings=StochasticRunSettings(dt=1.0e3, noise_amplitude=0.0, save_every=1),
            epsilon=0.015,
        )

        result = self.module.run_signed_response(job)

        self.assertEqual(result.temperature.shape, (2, latitude.size))
        self.assertTrue(np.isfinite(result.temperature).all())
        np.testing.assert_allclose(result.time, np.array([1.0e3, 2.0e3]))

    def test_run_signed_response_uses_perturbed_mu_only_on_first_step(self) -> None:
        class FakeDiffusionOperator:
            def to_banded_matrix(self, *, diagonal_shift=0.0, scale=1.0):
                banded = np.zeros((3, 2), dtype=float)
                banded[1, :] = 1.0
                return banded

        class FakeOperator:
            def __init__(self, mu: float):
                self.mu = mu

            def reaction_tendency(self, temperature):
                return np.full_like(temperature, self.mu, dtype=float)

            def frozen_diffusion_operator(self, temperature):
                return FakeDiffusionOperator()

        def fake_build_ivp_operator(*, x_grid, params, settings):
            return FakeOperator(params.mu)

        job = self.module.ResponseJob(
            sample_index=0,
            sign=1,
            seed=3,
            latitude=np.array([-0.5, 0.5]),
            initial_temperature=np.zeros(2),
            params=ModelParameters(mu=0.5),
            settings=RunSettings(final_time=2.0),
            stochastic_settings=StochasticRunSettings(dt=1.0, noise_amplitude=0.0, save_every=1),
            epsilon=0.2,
        )

        with patch.object(self.module, "build_ivp_operator", side_effect=fake_build_ivp_operator):
            result = self.module.run_signed_response(job)

        expected = np.array(
            [
                [0.7, 0.7],
                [1.2, 1.2],
            ]
        )
        np.testing.assert_allclose(result.temperature, expected)

    def test_cli_reads_samples_and_saves_response_dataset(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "samples.nc"
            _write_sample_dataset(input_path, attrs={"param_mu": 0.97})

            with (
                patch.object(
                    self.module,
                    "compute_response_means",
                    return_value=(
                        np.array([0.0, 1.0]),
                        np.ones((2, 5)),
                        np.zeros((2, 5)),
                    ),
                ) as compute_mock,
                patch.object(
                    self.module,
                    "save_response_dataset",
                    return_value=Path(temp_dir) / "response.nc",
                ) as save_mock,
                patch.object(self.module, "generate_initial_condition_samples") as generate_mock,
            ):
                result = runner.invoke(
                    self.module.app,
                    [
                        "--input-path",
                        str(input_path),
                        "--filename",
                        "response.nc",
                        "--output-dir",
                        temp_dir,
                        "--epsilon",
                        "0.02",
                        "--final-time",
                        "1.0",
                        "--dt",
                        "1.0",
                        "--workers",
                        "1",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        generate_mock.assert_not_called()
        self.assertEqual(compute_mock.call_args.kwargs["params"].mu, 0.97)
        self.assertEqual(compute_mock.call_args.kwargs["epsilon"], 0.02)
        self.assertEqual(save_mock.call_args.kwargs["filename"], "response.nc")
        saved_dataset = save_mock.call_args.args[0]
        self.assertIn("stationary_temperature", saved_dataset)
        self.assertEqual(saved_dataset["stationary_temperature"].dims, ("latitude",))

    def test_cli_without_input_path_generates_samples_in_memory(self) -> None:
        runner = CliRunner()
        samples = self.module.InitialConditionSamples(
            latitude=np.linspace(-0.8, 0.8, 5),
            temperature=np.ones((2, 5)) * 280.0,
            attrs={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    self.module,
                    "generate_initial_condition_samples",
                    return_value=(samples, np.array([10.0, 20.0])),
                ) as generate_mock,
                patch.object(
                    self.module,
                    "compute_response_means",
                    return_value=(
                        np.array([0.0, 1.0]),
                        np.ones((2, 5)),
                        np.zeros((2, 5)),
                    ),
                ) as compute_mock,
                patch.object(
                    self.module,
                    "save_response_dataset",
                    return_value=Path(temp_dir) / "response.nc",
                ) as save_mock,
            ):
                result = runner.invoke(
                    self.module.app,
                    [
                        "--filename",
                        "response.nc",
                        "--output-dir",
                        temp_dir,
                        "--baseline-final-time",
                        "1.0",
                        "--transient-time",
                        "0.1",
                        "--sample-count",
                        "2",
                        "--final-time",
                        "1.0",
                        "--dt",
                        "1.0",
                        "--workers",
                        "1",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        generate_mock.assert_called_once()
        self.assertEqual(generate_mock.call_args.kwargs["sample_count"], 2)
        self.assertEqual(compute_mock.call_args.args[0].temperature.shape, (2, 5))
        self.assertEqual(save_mock.call_args.kwargs["filename"], "response.nc")

    def test_cli_without_input_path_can_save_compact_generated_samples(self) -> None:
        runner = CliRunner()
        samples = self.module.InitialConditionSamples(
            latitude=np.linspace(-0.8, 0.8, 5),
            temperature=np.ones((2, 5)) * 280.0,
            attrs={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "compact_samples.nc"
            with (
                patch.object(
                    self.module,
                    "generate_initial_condition_samples",
                    return_value=(samples, np.array([10.0, 20.0])),
                ),
                patch.object(
                    self.module,
                    "compute_response_means",
                    return_value=(
                        np.array([0.0, 1.0]),
                        np.ones((2, 5)),
                        np.zeros((2, 5)),
                    ),
                ),
                patch.object(
                    self.module,
                    "save_response_dataset",
                    return_value=Path(temp_dir) / "response.nc",
                ),
            ):
                result = runner.invoke(
                    self.module.app,
                    [
                        "--filename",
                        "response.nc",
                        "--output-dir",
                        temp_dir,
                        "--baseline-final-time",
                        "1.0",
                        "--transient-time",
                        "0.1",
                        "--sample-count",
                        "2",
                        "--final-time",
                        "1.0",
                        "--dt",
                        "1.0",
                        "--workers",
                        "1",
                        "--save-samples-path",
                        str(sample_path),
                    ],
                )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            loaded = xr.open_dataset(sample_path, engine="scipy")
            try:
                self.assertEqual(loaded["temperature"].dims, ("sample", "latitude"))
                self.assertIn("sample_time", loaded)
            finally:
                loaded.close()

    def test_cli_rejects_oversized_baseline_memory_estimate(self) -> None:
        runner = CliRunner()

        with patch.object(self.module, "generate_initial_condition_samples") as generate_mock:
            result = runner.invoke(
                self.module.app,
                [
                    "--baseline-final-time",
                    "1.0",
                    "--transient-time",
                    "0.1",
                    "--sample-count",
                    "2",
                    "--max-baseline-memory-mb",
                    "0.000001",
                    "--final-time",
                    "1.0",
                    "--workers",
                    "1",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Estimated baseline temperature array", result.output)
        generate_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
