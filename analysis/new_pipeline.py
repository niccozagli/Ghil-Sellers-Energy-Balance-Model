import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from scipy.stats import binned_statistic_2d

    from koopman_response.utils import cosine_trapezoid_weights
    from gsebm.paths import get_data_dir
    from gsebm.reduced_dynamics import (
        fit_edmd_from_basis,
        fit_variable_bandwidth_diffusion_map,
        modal_covariance,
        project_observable,
    )
    from gsebm.time import YEAR

    plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})

    def regional_mean(field, weights, latitude, left, right):
        """Cosine-area weighted average on a Northern latitude interval."""
        mask = (latitude >= left) & (latitude <= right)
        return np.average(field[:, mask], axis=1, weights=weights[mask])

    def unbiased_covariance(values, max_lag):
        """Positive-lag covariance using the unbiased N-lag normalization."""
        centered = np.asarray(values, dtype=float) - np.mean(values)
        return np.asarray([
            centered[: centered.size - lag] @ centered[lag:]
            / (centered.size - lag)
            for lag in range(max_lag + 1)
        ])

    def slow_stable_covariance(result, coefficients, lags, mode_count):
        """Modal covariance retaining the slowest stable EDMD eigenmodes.

        Complex conjugate partners are retained together so the reconstructed
        covariance remains real apart from round-off.
        """
        eigenvalues = result.eigenvalues.astype(complex)
        candidates = np.flatnonzero(
            (np.arange(eigenvalues.size) != result.stationary_index)
            & (np.abs(eigenvalues) < 1.0 - 1.0e-10)
            & np.isfinite(result.rates.real)
            & np.isfinite(result.rates.imag)
        )
        ordered = candidates[np.argsort(result.rates[candidates].real)[::-1]]
        selected = []
        for index in ordered:
            if index in selected:
                continue
            selected.append(int(index))
            if abs(result.rates[index].imag) > 1.0e-10:
                partner = candidates[np.argmin(
                    abs(result.rates[candidates] - np.conj(result.rates[index]))
                )]
                if partner not in selected:
                    selected.append(int(partner))
            if len(selected) >= mode_count:
                break
        selected = np.asarray(selected, dtype=int)
        left_factor = coefficients.conj().T @ result.gram
        inverse_vectors = np.linalg.pinv(result.right_eigenvectors)
        amplitudes = (
            (left_factor @ result.right_eigenvectors)
            * (inverse_vectors @ coefficients)
        )
        return np.real_if_close(np.asarray([
            np.sum(amplitudes[selected] * eigenvalues[selected] ** lag)
            for lag in lags
        ])), selected

    return (
        YEAR,
        binned_statistic_2d,
        cosine_trapezoid_weights,
        fit_edmd_from_basis,
        fit_variable_bandwidth_diffusion_map,
        get_data_dir,
        mo,
        modal_covariance,
        np,
        plt,
        project_observable,
        regional_mean,
        slow_stable_covariance,
        unbiased_covariance,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Diffusion maps + EDMD at $\mu=1$

    An inspection notebook for the diffusion-EDMD spectrum and its
    eigenfunctions in reduced Northern-Hemisphere temperature phase space.
    """)
    return


@app.cell
def _(YEAR, cosine_trapezoid_weights, get_data_dir, np, regional_mean, xr):
    # Empirical closure uses the complete post-transient warm trajectory, not
    # merely the finite window used to fit the diffusion-map/EDMD operators.
    with xr.open_dataset(
        get_data_dir() / "new_stochastic_warm_mu1.nc", engine="scipy"
    ) as full_dataset:
        full_time = np.asarray(full_dataset.time.values, dtype=float)
        full_dt = float(np.median(np.diff(full_time)))
        full_latitude = np.asarray(full_dataset.latitude.values, dtype=float)
        full_northern = full_latitude >= 0.0
        full_latitude = full_latitude[full_northern]
        full_temperature = np.asarray(full_dataset.temperature.values)[
            full_time >= 500.0 * YEAR
        ][:, full_northern]
    full_weights = cosine_trapezoid_weights(
        full_latitude, n_features=full_latitude.size
    )
    full_mean_temperature = regional_mean(
        full_temperature, full_weights, full_latitude, 0.0, 1.0
    )
    full_delta_temperature = (
        regional_mean(full_temperature, full_weights, full_latitude, 0.0, 1.0 / 3.0)
        - regional_mean(full_temperature, full_weights, full_latitude, 1.0 / 3.0, 1.0)
    )
    return full_delta_temperature, full_dt, full_mean_temperature


@app.cell
def _(YEAR, cosine_trapezoid_weights, get_data_dir, np, regional_mean, xr):
    # Configuration and stationary trajectory
    state_count = 12_002
    lag = 2  # two saved 30-day states = 60 days
    origin_count = 12_000
    warm_dataset = xr.open_dataset(
        get_data_dir() / "new_stochastic_warm_mu1.nc", engine="scipy"
    )
    time = np.asarray(warm_dataset.time.values, dtype=float)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-12, atol=0.0):
        raise ValueError("Saved trajectory must be regularly sampled.")
    if not np.isclose(dt, 30.0 * 86400.0):
        raise ValueError("This notebook requires saved 30-day states.")

    post_transient = np.flatnonzero(time >= 500.0 * YEAR)
    if post_transient.size < state_count:
        raise ValueError("The post-transient trajectory is too short.")
    start = int(post_transient[0] + (post_transient.size - state_count) // 2)
    window = np.arange(start, start + state_count)

    all_latitude = np.asarray(warm_dataset.latitude.values, dtype=float)
    northern = all_latitude >= 0.0
    latitude = all_latitude[northern]
    temperature = np.asarray(
        warm_dataset.temperature.isel(time=window).values
    )[:, northern]
    anomalies = temperature - temperature.mean(axis=0)
    weights = cosine_trapezoid_weights(latitude, n_features=latitude.size)
    mean_temperature = regional_mean(temperature, weights, latitude, 0.0, 1.0)
    delta_temperature = (
        regional_mean(temperature, weights, latitude, 0.0, 1.0 / 3.0)
        - regional_mean(temperature, weights, latitude, 1.0 / 3.0, 1.0)
    )
    origins = np.arange(origin_count)
    targets = origins + lag
    if not np.array_equal(targets, np.arange(lag, state_count)):
        raise AssertionError("Snapshot pairs are not the requested contiguous window.")
    return (
        anomalies,
        delta_temperature,
        dt,
        mean_temperature,
        origins,
        targets,
        weights,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Diffusion-map basis and EDMD fit

    The diffusion-map basis uses the Northern anomaly fields and the
    cosine-trapezoid spatial metric. EDMD is then fit to all 10,000 shared
    snapshot origins at the 60-day lag.
    """)
    return


@app.cell
def _(anomalies, fit_variable_bandwidth_diffusion_map, weights):
    diffusion_map = fit_variable_bandwidth_diffusion_map(
        anomalies,
        weights,
        n_eigenfunctions=200,
        neighbor_count=256,
        local_scale_neighbors=32,
        beta=-0.5,
    )
    if diffusion_map.graph_components != 1:
        raise ValueError("The diffusion-map graph is disconnected.")
    return (diffusion_map,)


@app.cell
def _(diffusion_map, fit_edmd_from_basis, np, origins, targets):
    edmd = fit_edmd_from_basis(
        diffusion_map.basis,
        lag=2,
        basis_size=200,
        common_origin_count=origins.size,
        rcond=1.0e-10,
    )
    if not np.array_equal(origins + 2, targets):
        raise AssertionError("EDMD does not use the requested snapshot pairs.")
    return (edmd,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Diffusion-map diagnostics

    The intrinsic dimension is inferred from the peak log-slope of the
    variable-bandwidth kernel-sum curve. These plots also expose the selected
    bandwidth, geometric spectrum, and pilot-density distribution.
    """)
    return


@app.cell
def _(diffusion_map, np, plt):
    diagnostic_figure, diagnostic_axes = plt.subplots(1, 3, figsize=(14, 3.7))
    diagnostic_axes[0].plot(
        diffusion_map.epsilon_candidates,
        diffusion_map.log_slopes,
        marker=".",
    )
    diagnostic_axes[0].axvline(
        diffusion_map.epsilon,
        color="tab:red",
        linestyle="--",
        label="selected",
    )
    diagnostic_axes[0].set(
        xscale="log",
        xlabel=r"kernel bandwidth $\epsilon$",
        ylabel=r"$d\log S/d\log\epsilon$",
        title=("bandwidth selection\n"
               rf"estimated dimension $d={diffusion_map.intrinsic_dimension:.2f}$"),
    )
    diagnostic_axes[0].legend()
    diagnostic_axes[1].plot(diffusion_map.eigenvalues, ".")
    diagnostic_axes[1].set(
        xlabel="diffusion-function index",
        ylabel="diffusion eigenvalue",
        title="diffusion spectrum",
    )
    diagnostic_axes[2].hist(np.log10(diffusion_map.density), bins=40)
    diagnostic_axes[2].set(
        xlabel=r"$\log_{10}$ pilot density",
        ylabel="sample count",
        title=(f"{diffusion_map.neighbor_count}-NN graph; "
               f"{diffusion_map.graph_components} component"),
    )
    for _axis in diagnostic_axes:
        _axis.grid(alpha=0.25)
    diagnostic_figure.tight_layout()
    print(
        f"Estimated intrinsic dimension: {diffusion_map.intrinsic_dimension:.3f}; "
        f"selected epsilon: {diffusion_map.epsilon:.4g}"
    )
    diagnostic_figure
    return


@app.cell
def _(dt, edmd, np):
    edmd_rates = edmd.rates / (2.0 * dt)
    if not np.isfinite(edmd_rates).all():
        raise ValueError("EDMD produced a non-finite spectrum.")
    stationary_rate = edmd_rates[edmd.stationary_index]
    if abs(stationary_rate) * 365.25 * 86400.0 > 0.05:
        raise ValueError("The EDMD stationary eigenvalue is not near zero.")
    return (edmd_rates,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Eigenvalues

    Rates are shown in inverse years. The stationary eigenvalue is omitted.
    Red markers identify the six slowest stable nonstationary modes displayed
    in the phase-space panels below.
    """)
    return


@app.cell
def _(YEAR, edmd, edmd_rates, np, plt):
    nonstationary = np.flatnonzero(
        (np.arange(edmd_rates.size) != edmd.stationary_index)
        & (edmd_rates.real < 0.0)
        & np.isfinite(edmd_rates.real)
        & np.isfinite(edmd_rates.imag)
    )
    if nonstationary.size < 6:
        raise ValueError("Fewer than six stable nonstationary EDMD modes are available.")
    display_indices = nonstationary[np.argsort(edmd_rates[nonstationary].real)[::-1]][:6]
    rates_per_year = edmd_rates * YEAR
    spectrum_figure, spectrum_axis = plt.subplots(figsize=(6, 5))
    spectrum_axis.plot(
        rates_per_year[nonstationary].real,
        rates_per_year[nonstationary].imag,
        ".",
        color="0.55",
        ms=5,
    )


    spectrum_axis.axhline(0.0, color="black", lw=0.6)
    spectrum_axis.axvline(0.0, color="black", lw=0.6)
    spectrum_axis.set(
        xlabel=r"Re rate (year$^{-1}$)",
        ylabel=r"Im rate (year$^{-1}$)",
        title="Diffusion maps + EDMD spectrum",
    )
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.set_xlim(left=-4)
    spectrum_axis.set_ylim(bottom=-1,top=1)
    spectrum_figure.tight_layout()
    for _number, _index in enumerate(display_indices, 1):
        _rate = rates_per_year[_index]
        _decay = -1.0 / _rate.real
        _period = np.inf if abs(_rate.imag) < 1.0e-12 else 2.0 * np.pi / abs(_rate.imag)
        print(f"mode {_number}: index {_index}; decay time {_decay:.3g} yr; period {_period:.3g} yr")
    spectrum_figure
    return (display_indices,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Eigenfunctions in reduced phase space

    Each panel shows the conditional mean of the real part of one normalized
    EDMD eigenfunction over the same $\left(\overline{T}_N,\Delta T_N\right)$
    bins. Bins containing fewer than four samples are masked.
    """)
    return


@app.cell
def _(
    binned_statistic_2d,
    delta_temperature,
    diffusion_map,
    display_indices,
    edmd,
    edmd_rates,
    mean_temperature,
    np,
    plt,
):
    # EDMD uses only the leading ``basis_size`` diffusion functions.  The
    # diffusion-map object may contain more functions for later sensitivity
    # experiments, so slice it to the trial space used by this fit.
    eigenfunctions = (
        diffusion_map.basis[:, : edmd.basis_size] @ edmd.right_eigenvectors
    )
    counts, mean_edges, delta_edges, _ = binned_statistic_2d(
        mean_temperature, delta_temperature, None, statistic="count", bins=60
    )
    support = counts >= 4
    phase_figure, phase_axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for _number, (_axis, _index) in enumerate(zip(phase_axes.flat, display_indices), 1):
        _values = eigenfunctions[:, _index]
        _values /= np.sqrt(np.mean(np.abs(_values) ** 2))
        conditional_mean, _, _, _ = binned_statistic_2d(
            mean_temperature,
            delta_temperature,
            _values.real,
            statistic="mean",
            bins=(mean_edges, delta_edges),
        )
        conditional_mean[~support] = np.nan
        limit = np.nanmax(np.abs(conditional_mean))
        image = _axis.pcolormesh(
            mean_edges,
            delta_edges,
            conditional_mean.T,
            shading="auto",
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
        )
        _rate = edmd_rates[_index] * 365.25 * 86400.0
        _axis.set_title(rf"mode {_number}: $\tau ={1/ _rate.real:.2g}$ yr")
        phase_figure.colorbar(image, ax=_axis)
    for _axis in phase_axes[-1]:
        _axis.set_xlabel(r"$\overline{T}_N$ (K)")
    for _axis in phase_axes[:, 0]:
        _axis.set_ylabel(r"$\Delta T_N$ (K)")
    phase_figure.tight_layout()
    phase_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Correlation reconstruction

    Scalar observables are projected into the EDMD trial space. Their modal
    covariance is evaluated only at integer multiples of the 60-day EDMD
    operator lag and compared with the unbiased empirical covariance from the
    full post-transient warm trajectory.
    """)
    return


@app.cell
def _(
    YEAR,
    delta_temperature,
    diffusion_map,
    dt,
    edmd,
    full_delta_temperature,
    full_dt,
    full_mean_temperature,
    mean_temperature,
    modal_covariance,
    np,
    origins,
    plt,
    project_observable,
    slow_stable_covariance,
    unbiased_covariance,
):
    if not np.isclose(full_dt, dt):
        raise ValueError("Fit and full-trajectory sample intervals differ.")
    max_lag = int(round(40.0 * YEAR / full_dt))
    edmd_steps = np.arange(max_lag // edmd.lag + 1)
    correlation_mode_count = 10
    correlation_figure, correlation_axes = plt.subplots(
        1, 2, figsize=(12, 4), sharex=True
    )
    for _axis, (_name, _fit_values, _full_values) in zip(
        correlation_axes,
        (
            (r"$\overline{T}_N$", mean_temperature, full_mean_temperature),
            (r"$\Delta T_N$", delta_temperature, full_delta_temperature),
        ),
    ):
        empirical = unbiased_covariance(_full_values, max_lag)
        coefficients = project_observable(
            edmd,
            diffusion_map.basis,
            _fit_values,
            origin_count=origins.size,
        )
        full_reconstruction = np.asarray(
            modal_covariance(edmd, coefficients, edmd_steps)
        ).real
        reconstructed, retained_indices = slow_stable_covariance(
            edmd, coefficients, edmd_steps, correlation_mode_count
        )
        reconstructed = np.asarray(reconstructed).real
        # Both curves have covariance units (K^2); the EDMD curve lives only
        # on its native 60-day grid, so no fractional operator powers appear.
        empirical_years = np.arange(max_lag + 1) * full_dt / YEAR
        edmd_years = edmd_steps * edmd.lag * dt / YEAR
        _axis.plot(empirical_years, empirical, label="empirical")
        _axis.plot(
            edmd_years, full_reconstruction, color="tab:orange", alpha=0.3,
            label="diffusion-EDMD (all modes)",
        )
        _axis.plot(
            edmd_years, reconstructed, "o-", ms=2.5,
            label=f"diffusion-EDMD ({retained_indices.size} slow stable modes)",
        )
        _axis.axhline(0.0, color="black", lw=0.6)
        _axis.set(title=_name, xlabel="lag (years)", ylabel=r"covariance (K$^2$)")
        _axis.grid(alpha=0.25)
    correlation_axes[0].legend()
    correlation_axes[0].set_xlim(left=-1,right=25)
    correlation_figure.tight_layout()
    correlation_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Basis-size adequacy

    This separates static representation error from dynamical EDMD error.
    The left column measures how well the first $m$ diffusion functions span
    each scalar observable. The right column shows EDMD covariance at selected
    lags as $m$ increases, with the full-trajectory empirical values shown as
    dashed horizontal references.
    """)
    return


@app.cell
def _(
    YEAR,
    delta_temperature,
    diffusion_map,
    dt,
    fit_edmd_from_basis,
    full_delta_temperature,
    full_dt,
    full_mean_temperature,
    mean_temperature,
    modal_covariance,
    np,
    origins,
    plt,
    project_observable,
    unbiased_covariance,
):
    _maximum_basis = diffusion_map.basis.shape[1]
    _candidate_sizes = np.asarray((5, 10, 20, 40, 80, 120, 160, 200))
    _basis_sizes = _candidate_sizes[_candidate_sizes <= _maximum_basis]
    if _basis_sizes.size == 0 or _basis_sizes[-1] != _maximum_basis:
        _basis_sizes = np.unique(np.append(_basis_sizes, _maximum_basis))

    _lag_years = np.asarray((0.0, 1.0, 5.0, 10.0))
    _operator_steps = np.rint(_lag_years * YEAR / (2.0 * dt)).astype(int)
    _empirical_sample_lags = np.rint(
        _lag_years * YEAR / full_dt
    ).astype(int)
    _observables = (
        (r"$\overline{T}_N$", mean_temperature, full_mean_temperature),
        (r"$\Delta T_N$", delta_temperature, full_delta_temperature),
    )
    _colors = plt.cm.viridis(np.linspace(0.1, 0.9, _lag_years.size))
    _figure, _axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for _row, (_label, _fit_values, _full_values) in enumerate(_observables):
        _fit_centered = _fit_values[: origins.size] - np.mean(
            _fit_values[: origins.size]
        )
        _total_variance = np.mean(_fit_centered**2)
        _empirical_sequence = unbiased_covariance(
            _full_values, int(_empirical_sample_lags.max())
        )
        _empirical_targets = _empirical_sequence[_empirical_sample_lags]
        _r_squared = []
        _covariance_by_size = []

        for _basis_size in _basis_sizes:
            _result = fit_edmd_from_basis(
                diffusion_map.basis,
                lag=2,
                basis_size=int(_basis_size),
                common_origin_count=origins.size,
                rcond=1.0e-10,
            )
            _coefficients = project_observable(
                _result,
                diffusion_map.basis,
                _fit_values,
                origin_count=origins.size,
            )
            _projection = (
                diffusion_map.basis[: origins.size, : _result.basis_size]
                @ _coefficients
            )
            _r_squared.append(
                1.0 - np.mean((_fit_centered - _projection.real) ** 2)
                / _total_variance
            )
            _covariance_by_size.append(
                np.asarray(
                    modal_covariance(_result, _coefficients, _operator_steps)
                ).real
            )

        _r_squared = np.asarray(_r_squared)
        _covariance_by_size = np.vstack(_covariance_by_size)
        _axes[_row, 0].plot(_basis_sizes, _r_squared, "o-")
        _axes[_row, 0].axhline(1.0, color="black", lw=0.7)
        _axes[_row, 0].set(
            ylabel=rf"{_label}: projection $R^2$",
            ylim=(min(0.0, 1.05 * _r_squared.min()), 1.02),
        )

        for _lag_index, (_lag_value, _color) in enumerate(
            zip(_lag_years, _colors)
        ):
            _axes[_row, 1].plot(
                _basis_sizes,
                _covariance_by_size[:, _lag_index],
                "o-",
                color=_color,
                label=rf"{_lag_value:g} yr",
            )
            _axes[_row, 1].axhline(
                _empirical_targets[_lag_index],
                color=_color,
                linestyle="--",
                alpha=0.65,
            )
        _axes[_row, 1].set(ylabel=rf"{_label}: covariance (K$^2$)")
        print(
            f"{_label}: R^2 at m={_basis_sizes[-1]} is "
            f"{_r_squared[-1]:.5f}; projected/empirical lag-zero covariance "
            f"is {_covariance_by_size[-1, 0] / _empirical_targets[0]:.5f}"
        )

    for _axis in _axes.flat:
        _axis.grid(alpha=0.25)
    _axes[1, 0].set_xlabel("diffusion/EDMD basis size")
    _axes[1, 1].set_xlabel("diffusion/EDMD basis size")
    _axes[0, 1].legend(title="lag", ncols=2)
    _figure.tight_layout()
    _figure
    return


if __name__ == "__main__":
    app.run()
