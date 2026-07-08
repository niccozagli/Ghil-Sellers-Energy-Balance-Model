"""Estimate the Green's function for first-step perturbations of ``mu``."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
import typer
import xarray as xr

from gsebm import (
    DAY,
    YEAR,
    ModelParameters,
    RunSettings,
    StochasticRunSettings,
    build_ivp_grid,
    build_ivp_operator,
    build_spatial_noise_process,
    default_model_parameters,
    get_data_dir,
    model_parameters_from_dataset_attrs,
    run_stochastic_state,
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Compute the Green's function for an impulsive perturbation of mu. "
        "Provide an input NetCDF with temperature(sample, latitude), or omit it "
        "to generate sampled initial conditions internally."
    ),
)


@dataclass(frozen=True)
class InitialConditionSamples:
    """Sampled invariant-measure initial conditions."""

    latitude: np.ndarray
    temperature: np.ndarray
    attrs: dict[str, object]


@dataclass(frozen=True)
class ResponseJob:
    """One signed response experiment from one initial condition."""

    sample_index: int
    sign: int
    seed: int
    latitude: np.ndarray
    initial_temperature: np.ndarray
    params: ModelParameters
    settings: RunSettings
    stochastic_settings: StochasticRunSettings
    epsilon: float


@dataclass(frozen=True)
class ResponseResult:
    """Temperature trajectory for one signed response experiment."""

    sample_index: int
    sign: int
    time: np.ndarray
    temperature: np.ndarray


BYTES_PER_FLOAT64 = 8
BYTES_PER_MEGABYTE = 1024**2


def _flatten_prefixed_attributes(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def load_initial_condition_samples(path: str | Path) -> InitialConditionSamples:
    """Load sampled initial profiles from ``temperature(sample, latitude)``."""

    input_path = Path(path)
    dataset = xr.open_dataset(input_path, engine="scipy")
    try:
        if "temperature" not in dataset:
            raise ValueError("Input dataset must contain a 'temperature' variable.")
        if "latitude" not in dataset.coords:
            raise ValueError("Input dataset must contain a 'latitude' coordinate.")

        temperature_da = dataset["temperature"]
        if temperature_da.dims != ("sample", "latitude"):
            raise ValueError("Input 'temperature' must have dimensions ('sample', 'latitude').")

        latitude = np.asarray(dataset["latitude"].values, dtype=float)
        temperature = np.asarray(temperature_da.values, dtype=float)
        attrs = dict(dataset.attrs)
    finally:
        dataset.close()

    if latitude.ndim != 1:
        raise ValueError("'latitude' must be one-dimensional.")
    if temperature.ndim != 2:
        raise ValueError("'temperature' must be two-dimensional.")
    if temperature.shape[1] != latitude.size:
        raise ValueError("The latitude dimension of 'temperature' must match the latitude coordinate.")
    if temperature.shape[0] < 1:
        raise ValueError("Input dataset must contain at least one sample.")
    if not np.all(np.diff(latitude) > 0.0):
        raise ValueError("'latitude' must be strictly increasing.")
    if not np.all(np.isfinite(latitude)):
        raise ValueError("'latitude' contains non-finite values.")
    if not np.all(np.isfinite(temperature)):
        raise ValueError("'temperature' contains non-finite values.")

    return InitialConditionSamples(latitude=latitude, temperature=temperature, attrs=attrs)


def response_time_grid(final_time: float, dt: float, save_every: int) -> np.ndarray:
    """Return the saved response times for the fixed-step stochastic solver."""

    if final_time < 0.0:
        raise ValueError("final_time must be nonnegative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    current_time = 0.0
    step_count = 0
    saved_times = [0.0]
    while current_time < final_time:
        step = min(dt, final_time - current_time)
        current_time += step
        step_count += 1
        if step_count % save_every == 0 or current_time >= final_time:
            saved_times.append(current_time)
    return np.asarray(saved_times, dtype=float)


def estimate_saved_temperature_memory_bytes(
    *,
    final_time: float,
    dt: float,
    save_every: int,
    latitude_count: int,
) -> int:
    """Estimate memory used by a saved ``temperature(time, latitude)`` array."""

    saved_time_count = response_time_grid(final_time, dt, save_every).size
    return int(saved_time_count * latitude_count * BYTES_PER_FLOAT64)


def select_evenly_spaced_post_transient_samples(
    *,
    time: np.ndarray,
    temperature: np.ndarray,
    transient_time: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return evenly spaced post-transient temperatures and their times."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if transient_time < 0.0:
        raise ValueError("transient_time must be nonnegative.")
    time_array = np.asarray(time, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if temperature_array.ndim != 2:
        raise ValueError("temperature must be two-dimensional.")
    if temperature_array.shape[0] != time_array.size:
        raise ValueError("temperature and time must share the time dimension.")

    post_transient_indices = np.flatnonzero(time_array >= transient_time)
    if post_transient_indices.size < sample_count:
        raise ValueError(
            "Not enough post-transient saved states to select the requested sample_count."
        )

    positions = np.linspace(0, post_transient_indices.size - 1, sample_count)
    selected_indices = post_transient_indices[np.rint(positions).astype(int)]
    return time_array[selected_indices], temperature_array[selected_indices]


def stationary_temperature_from_samples(
    samples: InitialConditionSamples,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the invariant-measure mean and std over the sampled ICs (latitude,)."""

    temperature = np.asarray(samples.temperature, dtype=float)
    return temperature.mean(axis=0), temperature.std(axis=0)


def sampled_initial_condition_dataset(
    samples: InitialConditionSamples,
    *,
    sample_time: np.ndarray,
    params: ModelParameters,
    baseline_settings: RunSettings,
    stochastic_settings: StochasticRunSettings,
    baseline_final_time: float,
    transient_time: float,
) -> xr.Dataset:
    """Build a compact dataset of sampled response initial conditions."""

    attrs: dict[str, object] = {
        "title": "Sampled invariant-measure initial conditions for response experiments",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "baseline_final_time": float(baseline_final_time),
        "transient_time": float(transient_time),
        "sample_count": int(samples.temperature.shape[0]),
        "sampling_strategy": "even_spacing",
    }
    attrs.update(_flatten_prefixed_attributes("param", asdict(params)))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(baseline_settings)))
    stochastic_attrs = asdict(stochastic_settings)
    if stochastic_attrs["noise_seed"] is None:
        stochastic_attrs["noise_seed"] = "None"
    attrs.update(_flatten_prefixed_attributes("stochastic", stochastic_attrs))

    return xr.Dataset(
        data_vars={
            "temperature": (
                ("sample", "latitude"),
                samples.temperature,
                {"units": "K", "long_name": "sampled initial temperature"},
            ),
            "sample_time": (
                ("sample",),
                np.asarray(sample_time, dtype=float),
                {"units": "s", "long_name": "baseline trajectory sample time"},
            ),
        },
        coords={
            "sample": ("sample", np.arange(samples.temperature.shape[0], dtype=int)),
            "latitude": (
                "latitude",
                samples.latitude,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )


def save_sampled_initial_condition_dataset(
    dataset: xr.Dataset,
    path: str | Path,
) -> Path:
    """Write compact sampled initial conditions to NetCDF."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path


def generate_initial_condition_samples(
    *,
    params: ModelParameters,
    baseline_settings: RunSettings,
    stochastic_settings: StochasticRunSettings,
    initial_temperature: float,
    transient_time: float,
    sample_count: int,
) -> tuple[InitialConditionSamples, np.ndarray]:
    """Run the baseline stochastic trajectory and sample post-transient profiles."""

    solution = run_stochastic_state(
        params=params,
        settings=baseline_settings,
        stochastic_settings=stochastic_settings,
        initial_temperature=initial_temperature,
    )
    sample_time, sample_temperature = select_evenly_spaced_post_transient_samples(
        time=solution.state.t,
        temperature=solution.state.temperature,
        transient_time=transient_time,
        sample_count=sample_count,
    )
    samples = InitialConditionSamples(
        latitude=np.asarray(solution.state.x, dtype=float),
        temperature=np.asarray(sample_temperature, dtype=float),
        attrs={},
    )
    return samples, sample_time


def run_signed_response(job: ResponseJob) -> ResponseResult:
    """Run one response trajectory with a first-step perturbation of ``mu``."""

    base_operator = build_ivp_operator(
        x_grid=job.latitude,
        params=job.params,
        settings=job.settings,
    )
    perturbed_params = replace(
        job.params,
        mu=job.params.mu + job.sign * job.epsilon / job.stochastic_settings.dt,
    )
    perturbed_operator = build_ivp_operator(
        x_grid=job.latitude,
        params=perturbed_params,
        settings=job.settings,
    )
    noise_process = build_spatial_noise_process(
        job.latitude,
        coarse_step_degrees=job.stochastic_settings.noise_grid_step_degrees,
        length_scale_degrees=job.stochastic_settings.noise_length_scale_degrees,
    )
    rng = np.random.default_rng(job.seed)

    current_time = 0.0
    step_count = 0
    y = np.asarray(job.initial_temperature, dtype=float).copy()
    saved_times = [0.0]
    saved_temperatures = [y.copy()]

    while current_time < job.settings.final_time:
        dt = min(job.stochastic_settings.dt, job.settings.final_time - current_time)
        reaction_operator = perturbed_operator if step_count == 0 else base_operator
        y_star = y + reaction_operator.reaction_tendency(y) * dt
        if job.stochastic_settings.noise_amplitude != 0.0:
            y_star = y_star + job.stochastic_settings.noise_amplitude * noise_process.sample(rng) * np.sqrt(dt)

        diffusion_operator = base_operator.frozen_diffusion_operator(y)
        y = solve_banded(
            (1, 1),
            diffusion_operator.to_banded_matrix(diagonal_shift=1.0, scale=-dt),
            y_star,
            check_finite=False,
        )

        current_time += dt
        step_count += 1
        if not np.all(np.isfinite(y)):
            raise RuntimeError("Response integration produced non-finite temperatures.")
        if step_count % job.stochastic_settings.save_every == 0 or current_time >= job.settings.final_time:
            saved_times.append(current_time)
            saved_temperatures.append(y.copy())

    return ResponseResult(
        sample_index=job.sample_index,
        sign=job.sign,
        time=np.asarray(saved_times, dtype=float),
        temperature=np.asarray(saved_temperatures, dtype=float),
    )


def _run_response_job(job: ResponseJob) -> ResponseResult:
    return run_signed_response(job)


def compute_response_means(
    samples: InitialConditionSamples,
    *,
    params: ModelParameters,
    settings: RunSettings,
    stochastic_settings: StochasticRunSettings,
    epsilon: float,
    base_seed: int,
    workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return response times plus ensemble mean plus/minus temperatures."""

    jobs = [
        ResponseJob(
            sample_index=sample_index,
            sign=sign,
            seed=base_seed + sample_index,
            latitude=samples.latitude,
            initial_temperature=samples.temperature[sample_index],
            params=params,
            settings=settings,
            stochastic_settings=stochastic_settings,
            epsilon=epsilon,
        )
        for sample_index in range(samples.temperature.shape[0])
        for sign in (1, -1)
    ]

    expected_time = response_time_grid(
        settings.final_time,
        stochastic_settings.dt,
        stochastic_settings.save_every,
    )
    plus_sum = np.zeros((expected_time.size, samples.latitude.size), dtype=float)
    minus_sum = np.zeros_like(plus_sum)
    completed = 0
    total = len(jobs)

    if workers <= 1:
        results = (run_signed_response(job) for job in jobs)
        for result in results:
            _accumulate_response_result(result, expected_time, plus_sum, minus_sum)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_response_job, job) for job in jobs]
            for future in as_completed(futures):
                _accumulate_response_result(future.result(), expected_time, plus_sum, minus_sum)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

    sample_count = samples.temperature.shape[0]
    return expected_time, plus_sum / sample_count, minus_sum / sample_count


def _accumulate_response_result(
    result: ResponseResult,
    expected_time: np.ndarray,
    plus_sum: np.ndarray,
    minus_sum: np.ndarray,
) -> None:
    if result.temperature.shape != plus_sum.shape:
        raise ValueError("A response trajectory has an unexpected shape.")
    np.testing.assert_allclose(result.time, expected_time)
    if result.sign == 1:
        plus_sum += result.temperature
    elif result.sign == -1:
        minus_sum += result.temperature
    else:
        raise ValueError("Response sign must be 1 or -1.")


def response_dataset(
    *,
    time: np.ndarray,
    latitude: np.ndarray,
    plus_mean_temperature: np.ndarray,
    minus_mean_temperature: np.ndarray,
    stationary_temperature: np.ndarray,
    stationary_temperature_std: np.ndarray | None = None,
    params: ModelParameters,
    settings: RunSettings,
    stochastic_settings: StochasticRunSettings,
    epsilon: float,
    input_path: str | Path | None,
    sample_count: int,
    base_seed: int,
) -> xr.Dataset:
    """Build the response Green's-function dataset."""

    green_function = (plus_mean_temperature - minus_mean_temperature) / (2.0 * epsilon)
    attrs: dict[str, object] = {
        "title": "Green's function for first-step mu perturbations",
        "latitude_name": "normalized latitude",
        "latitude_bounds": "[-1, 1]",
        "epsilon": float(epsilon),
        "sample_count": int(sample_count),
        "stochastic_base_seed": int(base_seed),
        "response_seed_scheme": "base_seed + sample_index (common random numbers per sign)",
        "mu_plus_first_step": float(params.mu + epsilon / stochastic_settings.dt),
        "mu_minus_first_step": float(params.mu - epsilon / stochastic_settings.dt),
        "perturbation_note": "mu is perturbed only during the first IMEX step.",
        "stationary_estimate_note": "mean over sampled initial conditions",
        "green_function_units_note": (
            "epsilon is interpreted in seconds because the first-step perturbation "
            "is mu +/- epsilon / dt. The Green's function is K per unit mu-impulse "
            "(mu * s); mu is dimensionless so this reduces to K s^-1."
        ),
    }
    if input_path is not None:
        attrs["input_path"] = str(input_path)
    else:
        attrs["input_path"] = "generated_in_memory"
    attrs.update(_flatten_prefixed_attributes("param", asdict(params)))
    attrs.update(_flatten_prefixed_attributes("setting", asdict(settings)))
    stochastic_attrs = asdict(stochastic_settings)
    attrs.update(_flatten_prefixed_attributes("stochastic", stochastic_attrs))

    data_vars: dict[str, object] = {
        "plus_mean_temperature": (
            ("time", "latitude"),
            plus_mean_temperature,
            {"units": "K", "long_name": "plus-perturbation ensemble mean temperature"},
        ),
        "minus_mean_temperature": (
            ("time", "latitude"),
            minus_mean_temperature,
            {"units": "K", "long_name": "minus-perturbation ensemble mean temperature"},
        ),
        "stationary_temperature": (
            ("latitude",),
            np.asarray(stationary_temperature, dtype=float),
            {
                "units": "K",
                "long_name": (
                    "invariant-measure mean temperature estimated from the "
                    "initial-condition ensemble"
                ),
            },
        ),
        "green_function_temperature": (
            ("time", "latitude"),
            green_function,
            {
                "units": "K s^-1",
                "long_name": "central finite-difference Green's function for temperature response",
            },
        ),
    }
    if stationary_temperature_std is not None:
        data_vars["stationary_temperature_std"] = (
            ("latitude",),
            np.asarray(stationary_temperature_std, dtype=float),
            {"units": "K", "long_name": "std of the initial-condition ensemble"},
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time, {"units": "s", "long_name": "response time"}),
            "latitude": (
                "latitude",
                latitude,
                {"units": "1", "long_name": "normalized latitude"},
            ),
        },
        attrs=attrs,
    )


def save_response_dataset(
    dataset: xr.Dataset,
    *,
    filename: str = "response_mu.nc",
    output_dir: str | Path | None = None,
) -> Path:
    """Write the response dataset to NetCDF."""

    destination_dir = get_data_dir() if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    dataset.to_netcdf(output_path, engine="scipy")
    return output_path


@app.command()
def main(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="NetCDF file containing temperature(sample, latitude); omit to generate samples internally.",
    ),
    filename: str = typer.Option("response_mu.nc", help="Output filename for the NetCDF response dataset."),
    output_dir: str | None = typer.Option(
        None,
        help="Directory in which to write the NetCDF file; defaults to the repository data directory.",
    ),
    mu: float = typer.Option(1.0, help="Baseline relative solar strength at which initial conditions are sampled."),
    initial_temperature: float = typer.Option(290.0, help="Uniform baseline initial condition in kelvin."),
    baseline_final_time: float = typer.Option(5000.0, help="Baseline integration time in years."),
    transient_time: float = typer.Option(500.0, help="Transient time to discard from the baseline in years."),
    sample_count: int = typer.Option(1000, help="Number of post-transient baseline states to sample."),
    max_baseline_memory_mb: float = typer.Option(
        1000.0,
        help="Maximum estimated baseline temperature-array memory in MB when generating samples.",
    ),
    save_samples_path: Path | None = typer.Option(
        None,
        help="Optional NetCDF path for saving compact generated samples; the full baseline trajectory is never saved.",
    ),
    epsilon: float = typer.Option(0.015, help="Impulse amplitude used in mu +/- epsilon / dt."),
    final_time: float = typer.Option(25.0, help="Response integration time in years."),
    dt: float = typer.Option(1.0, help="Fixed IMEX timestep in days."),
    noise_amplitude: float = typer.Option(4.0e-4, help="Additive noise amplitude in K s^-1/2."),
    noise_grid_step_degrees: float = typer.Option(3.0, help="Coarse latitude spacing of the sampled noise in degrees."),
    noise_length_scale_degrees: float = typer.Option(3.0, help="Gaussian kernel width for spatial smoothing in degrees."),
    save_every: int = typer.Option(30, help="Save every N stochastic timesteps of length dt."),
    base_seed: int = typer.Option(0, help="Seed for sample 0; sample i uses base_seed + i."),
    workers: int = typer.Option(2, help="Number of worker processes; use 1 for serial execution."),
    progress_every: int = typer.Option(10, help="Print progress every N completed signed simulations."),
    quiet: bool = typer.Option(False, "--quiet", help="Disable progress messages."),
) -> None:
    """Compute and save the mu response Green's function."""

    if epsilon <= 0.0:
        raise typer.BadParameter("epsilon must be positive.")
    if workers < 1:
        raise typer.BadParameter("workers must be at least 1.")
    if progress_every < 1:
        raise typer.BadParameter("progress_every must be at least 1.")
    if sample_count < 1:
        raise typer.BadParameter("sample_count must be at least 1.")
    if baseline_final_time <= 0.0:
        raise typer.BadParameter("baseline_final_time must be positive.")
    if transient_time < 0.0:
        raise typer.BadParameter("transient_time must be nonnegative.")
    if transient_time >= baseline_final_time:
        raise typer.BadParameter("transient_time must be less than baseline_final_time.")
    if max_baseline_memory_mb <= 0.0:
        raise typer.BadParameter("max_baseline_memory_mb must be positive.")

    settings = RunSettings(final_time=final_time * YEAR)
    stochastic_settings = StochasticRunSettings(
        dt=dt * DAY,
        noise_amplitude=noise_amplitude,
        noise_grid_step_degrees=noise_grid_step_degrees,
        noise_length_scale_degrees=noise_length_scale_degrees,
        noise_seed=base_seed,
        save_every=save_every,
    )

    def log(message: str) -> None:
        if not quiet:
            typer.echo(message, err=True)

    if input_path is not None:
        samples = load_initial_condition_samples(input_path)
        sample_dataset = xr.open_dataset(input_path, engine="scipy")
        try:
            params = model_parameters_from_dataset_attrs(sample_dataset)
        finally:
            sample_dataset.close()
        log(f"loaded samples from {input_path}")
    else:
        params = replace(default_model_parameters(), mu=mu)
        baseline_settings = RunSettings(final_time=baseline_final_time * YEAR)
        baseline_latitude_count = build_ivp_grid(baseline_settings).size
        estimated_memory_bytes = estimate_saved_temperature_memory_bytes(
            final_time=baseline_settings.final_time,
            dt=stochastic_settings.dt,
            save_every=stochastic_settings.save_every,
            latitude_count=baseline_latitude_count,
        )
        estimated_memory_mb = estimated_memory_bytes / BYTES_PER_MEGABYTE
        if estimated_memory_mb > max_baseline_memory_mb:
            raise typer.BadParameter(
                (
                    f"Estimated baseline temperature array is {estimated_memory_mb:.1f} MB, "
                    f"above --max-baseline-memory-mb={max_baseline_memory_mb:.1f}. "
                    "Increase the limit or reduce baseline_final_time/save_every."
                )
            )

        log(
            (
                f"generating baseline samples in RAM: horizon {baseline_final_time:g} years, "
                f"transient {transient_time:g} years, requested samples {sample_count}"
            )
        )
        log(
            (
                f"estimated baseline temperature memory {estimated_memory_mb:.1f} MB "
                f"for {baseline_latitude_count} latitudes"
            )
        )
        samples, sample_time = generate_initial_condition_samples(
            params=params,
            baseline_settings=baseline_settings,
            stochastic_settings=stochastic_settings,
            initial_temperature=initial_temperature,
            transient_time=transient_time * YEAR,
            sample_count=sample_count,
        )
        log(
            (
                f"selected {samples.temperature.shape[0]} samples from baseline times "
                f"{sample_time[0] / YEAR:.2f} to {sample_time[-1] / YEAR:.2f} years"
            )
        )

        if save_samples_path is not None:
            sample_dataset = sampled_initial_condition_dataset(
                samples,
                sample_time=sample_time,
                params=params,
                baseline_settings=baseline_settings,
                stochastic_settings=stochastic_settings,
                baseline_final_time=baseline_settings.final_time,
                transient_time=transient_time * YEAR,
            )
            saved_samples_path = save_sampled_initial_condition_dataset(sample_dataset, save_samples_path)
            log(f"saved compact generated samples to {saved_samples_path}")
        else:
            log("kept generated samples in RAM; no sample file written")

    response_steps = int(np.ceil(settings.final_time / stochastic_settings.dt))
    total_jobs = 2 * samples.temperature.shape[0]
    start_time = perf_counter()

    def log_progress(completed: int, total: int) -> None:
        if quiet:
            return
        if completed != 1 and completed != total and completed % progress_every != 0:
            return
        elapsed = perf_counter() - start_time
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        remaining = (total - completed) / rate if rate > 0.0 else float("nan")
        percent = 100.0 * completed / total
        typer.echo(
            (
                f"completed {completed}/{total} signed simulations "
                f"({percent:.1f}%, elapsed {elapsed / 60.0:.1f} min, "
                f"ETA {remaining / 60.0:.1f} min)"
            ),
            err=True,
        )

    log(
        (
            f"loaded {samples.temperature.shape[0]} samples on {samples.latitude.size} latitudes; "
            f"running {total_jobs} signed simulations with {workers} worker(s)"
        )
    )
    log(
        (
            f"response horizon {final_time:g} years, dt {dt:g} day(s), "
            f"{response_steps} steps per simulation, save_every={save_every}"
        )
    )
    log(
        (
            f"first-step perturbations: mu+={params.mu + epsilon / stochastic_settings.dt:.12g}, "
            f"mu-={params.mu - epsilon / stochastic_settings.dt:.12g}"
        )
    )

    time, plus_mean, minus_mean = compute_response_means(
        samples,
        params=params,
        settings=settings,
        stochastic_settings=stochastic_settings,
        epsilon=epsilon,
        base_seed=base_seed,
        workers=workers,
        progress_callback=log_progress,
    )
    stationary_mean, stationary_std = stationary_temperature_from_samples(samples)
    log(
        (
            f"stationary reference from {samples.temperature.shape[0]} ICs: "
            f"pole {stationary_mean[0]:.2f} K, equator "
            f"{stationary_mean[stationary_mean.size // 2]:.2f} K"
        )
    )
    log("building response dataset")
    dataset = response_dataset(
        time=time,
        latitude=samples.latitude,
        plus_mean_temperature=plus_mean,
        minus_mean_temperature=minus_mean,
        stationary_temperature=stationary_mean,
        stationary_temperature_std=stationary_std,
        params=params,
        settings=settings,
        stochastic_settings=stochastic_settings,
        epsilon=epsilon,
        input_path=input_path,
        sample_count=samples.temperature.shape[0],
        base_seed=base_seed,
    )
    output_path = save_response_dataset(dataset, filename=filename, output_dir=output_dir)
    log(f"saved response dataset to {output_path}")
    typer.echo(output_path)


if __name__ == "__main__":
    app()
