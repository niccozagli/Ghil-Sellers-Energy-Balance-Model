# Ghil-Sellers Energy Balance Model

This repository contains a Python port of the Ghil-Sellers energy balance
model, developed incrementally from an existing MATLAB implementation.

The Python package is managed with `uv` and follows a `src/` layout.

## Current Status

The repository is still in an early migration stage.

Implemented so far:

- project packaging and repository structure
- a preserved MATLAB reference implementation in `matlab/`
- Python parameter objects in `src/gsebm/parameters.py`
- empirical latitude-dependent data objects in `src/gsebm/empirical.py`
- interpolation and IVP preprocessing for empirical fields in
  `src/gsebm/empirical.py`
- shared local physics formulas in `src/gsebm/physics.py`
- meridional heat-transfer diagnostics in `src/gsebm/diagnostics.py`
- a method-of-lines IVP solver in `src/gsebm/ivp.py`
- a stochastic IMEX temperature solver in `src/gsebm/sde.py`
- a steady-state BVP solver in `src/gsebm/bvp.py`
- separate reusable warm/cold and edge-state workflows in `src/gsebm/run.py`
- repository path helpers in `src/gsebm/paths.py`
- runnable scripts in `scripts/run_warm_cold_state.py` and
  `scripts/run_edge_state.py`
- `mu` bifurcation sweep scripts in `scripts/run_warm_cold_mu_bifurcation.py`
  and `scripts/run_edge_mu_bifurcation.py`
- tests for package import, default parameter values, empirical data, and
  local physics, IVP, and BVP relationships

## Model Overview

The codebase represents a one-dimensional energy balance model for the
zonal-mean temperature `T(x, t)` as a function of normalized latitude `x`
and time `t`.

The governing balance combines:

- a heat-capacity term `c(x)`
- meridional heat transport through a temperature-dependent diffusivity
- outgoing longwave radiation
- absorbed shortwave radiation
- temperature-dependent albedo, which introduces ice-albedo feedback

At a high level, the model has the form:

```text
c(x) dT/dt = transport(T, x) - radiative_loss(T) + absorbed_solar(T, x)
```

The empirical latitude-dependent inputs used by the model are:

- `C(x)`: effective heat capacity
- `Q(x)`: incoming solar irradiance
- `b(x)`: empirical coefficient used in the albedo law
- `z(x)`: height-offset field used in the albedo law
- `k1(x)`: sensible heat transport coefficient
- `k2(x)`: latent heat transport coefficient

These inputs are given only on coarse latitude data grids and are later
interpolated before being evaluated on a solver grid. The empirical data
grids are not themselves the PDE discretization grid.

The model is treated as hemispherically symmetric in the data layer, so the
tabulated one-hemisphere values are mirrored across the equator to produce
pole-to-pole inputs on the normalized latitude interval `[-1, 1]`.

For the time-dependent solve, the current Python design does not intend to
evaluate spline objects inside every PDE right-hand-side call. Instead, the
empirical layer now supports:

- continuous interpolants for the latitude-dependent fields
- one-time sampling of those fields on a fixed IVP solver grid

This keeps the eventual IVP integration path focused on array operations.

The shared physics layer currently implemented in Python includes:

- the temperature-dependent surface albedo law
- the moisture factor `g(T)` used in the diffusivity law
- the combined diffusivity `k1(x) + g(T) k2(x)`
- outgoing longwave radiation with cloud feedback
- absorbed shortwave radiation
- net radiative energy transport

These functions operate on already-evaluated local quantities. They are
shared by the time-dependent IVP solver and the steady-state BVP solver.

The Python IVP path is now assembled separately from the local physics
layer. It uses:

- a fixed latitude grid with near-pole shifted points
- precomputed latitude-only empirical fields on that grid
- a divergence-form transport discretization with zero boundary face flux
- `scipy.integrate.solve_ivp` for time integration

The Python BVP path is assembled separately from the IVP because it needs
continuous coefficient functions and their latitude derivatives on an
adaptive mesh. It uses:

- continuous empirical interpolants rather than pre-sampled IVP arrays
- a first-order steady-state ODE system for `[T, dT/dx]`
- zero-slope boundary conditions at both poles
- `scipy.integrate.solve_bvp` for the stationary solve
- either a constant guess or an IVP-based perturbed initial guess

<details>
<summary>Extended Physics Description</summary>

The model evolves the zonal-mean near-surface air temperature as a function
of normalized latitude and time. Its structure is a balance between local
energy storage, meridional heat transport, and radiative forcing.

The heat-capacity term `c(x)` controls how strongly each latitude responds
to a given energy imbalance. Because `c(x)` varies with latitude, the same
forcing does not produce the same temperature tendency everywhere.

Meridional heat transport is represented as a diffusive process. The total
transport coefficient is split into a sensible component `k1(x)` and a
latent component `g(T) k2(x)`. The factor `g(T)` increases with
temperature-dependent moisture availability, so transport strength is not
purely a function of latitude.

Outgoing longwave radiation is based on a Stefan-Boltzmann `sigma T^4`
term, modified by a cloud-feedback factor. In this formulation, radiative
loss remains strongly increasing with temperature, but not as a purely
blackbody law.

Absorbed shortwave radiation is `mu Q(x) (1 - alpha)`, where `mu` is a
solar-strength parameter, `Q(x)` is the incoming solar distribution, and
`alpha` is the local albedo.

The albedo law is one of the key nonlinearities in the model. It depends on
the empirical `b(x)` field, a height-offset contribution through `z(x)`,
and temperature. This creates an ice-albedo feedback: colder conditions can
increase albedo, which reduces absorbed solar radiation and promotes further
cooling. The model therefore admits the possibility of multiple stationary
states for some parameter choices.

The local formulas currently implemented in the Python physics layer are:

```text
alpha(T, x) =
    clip(b(x) - c1 [um + min(T - c2 z(x) - um, 0)], alpha_min, alpha_max)

g(T) = c4 / T^2 * exp(-c5 / T)

k(T, x) = k1(x) + g(T) k2(x)

OLR(T) = sigma T^4 [1 - m1 tanh(c3 T^6)]

ASR(T, x) = mu Q(x) [1 - alpha(T, x)]

F(T, x) = ASR(T, x) - OLR(T)
```

where:

- `alpha(T, x)` is the local surface albedo
- `g(T)` is the moisture-dependent factor in the transport law
- `k(T, x)` is the total meridional transport coefficient
- `OLR(T)` is outgoing longwave radiation
- `ASR(T, x)` is absorbed shortwave radiation
- `F(T, x)` is the local net radiative energy transport

In the original formulation, the empirical latitude-dependent fields are not
given as analytic formulas. They are provided as coarse tables and then
interpolated before use in the PDE and steady-state equations. That is why
the Python port separates:

- empirical data
- interpolation and preprocessing
- local physics formulas
- solver-specific equation assembly

The current IVP implementation follows the same continuous model structure
as the reference code, but it does not reproduce MATLAB `pdepe`
internals. The time-dependent solver is a custom method-of-lines
implementation built around SciPy's ODE integrators.

</details>

<details>
<summary>Stochastic Solver</summary>

The Python stochastic path reuses the same semi-discrete latitude operator
as the deterministic IVP, but adds additive noise in the temperature
tendency. It uses:

- the same fixed IVP latitude grid and empirical preprocessing
- a split of the semi-discrete tendency into diffusion and radiative
  reaction parts
- a fixed-step IMEX update: implicit frozen diffusion, explicit reaction,
  explicit additive noise
- a smooth spatial noise field built from Gaussian kernels centered on a
  coarse latitude grid
- zero direct stochastic forcing at the two pole nodes

The stochastic solver is separate from the deterministic `solve_ivp`
integration path. After spatial discretization, it advances

```text
dT = [diffusion(T) + reaction(T)] dt + sigma xi(x) dW
```

with an IMEX step of the form

```text
y* = y_n + dt reaction(y_n) + sigma sqrt(dt) xi_n(x)
(I - dt L_n) y_{n+1} = y*
```

where:

- `L_n` is the frozen tridiagonal diffusion operator built from `y_n`
- `xi_n(x)` is a fresh smooth random latitude field at each timestep
- `sigma` is the noise amplitude in units `K s^-1/2`

The spatial noise is not white on the solver grid. Instead, the code draws
independent standard Gaussian coefficients on a coarse latitude grid, maps
them to the solver grid with Gaussian kernels, and normalizes the interior
rows to unit pointwise variance. This avoids gridpoint-by-gridpoint rough
forcing on the full IVP grid.

</details>

## Layout

- `matlab/`: reference implementation
- `analysis/`: marimo analysis apps for saved datasets
- `scripts/`: runnable research scripts
- `src/gsebm/`: Python package for the new implementation
- `tests/`: Python test suite

## Development Approach

The port proceeds in small steps. The immediate goal is to translate the
model into clear Python modules while keeping the implementation easy to
compare against the reference code.

The current sequence is:

1. parameters and default settings
2. empirical latitude-dependent data
3. shared physical relationships
4. interpolation and preprocessing for empirical fields
5. IVP and BVP model components
6. diagnostics, plotting, unstable-branch workflows, and comparison tools

## Running Tests

At the current stage, the lightweight test suite can be run with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Running The Scripts

The current non-prototype entry points are:

```bash
uv run python scripts/run_warm_cold_state.py
uv run python scripts/run_edge_state.py
uv run python scripts/run_warm_cold_mu_bifurcation.py
uv run python scripts/run_edge_mu_bifurcation.py
```

The stochastic solver does not yet have a dedicated research script under
`scripts/`. It is currently intended to be used directly from Python or
through the prototype notebook in `prototyping/ivp_explorer.py`.

A minimal direct Python example is:

```python
from gsebm import (
    RunSettings,
    StochasticRunSettings,
    YEAR,
    DAY,
    solve_temperature_sde,
)

run_settings = RunSettings(final_time=35.0 * YEAR)
stochastic_settings = StochasticRunSettings(
    dt=1.0 * DAY,
    noise_amplitude=1.0e-5,
    noise_grid_step_degrees=5.0,
    noise_length_scale_degrees=5.0,
    noise_seed=7,  # or None for a fresh realization each run
    save_every=30,
)

solution = solve_temperature_sde(
    settings=run_settings,
    stochastic_settings=stochastic_settings,
    initial_condition_kind="scalar",
    initial_scalar_value=280.0,
)
```

Key stochastic settings:

- `dt`: fixed timestep for the IMEX solver
- `noise_amplitude`: additive noise amplitude in `K s^-1/2`
- `noise_grid_step_degrees`: coarse latitude spacing for the sampled noise
- `noise_length_scale_degrees`: Gaussian kernel width for spatial smoothing
- `noise_seed`: RNG seed, or `None` for a non-reproducible realization
- `save_every`: save every `N` timesteps rather than every step

The warm/cold script solves the warm-state and cold-state IVP branches and
writes a NetCDF file to:

```text
<repo-root>/data/<filename>.nc
```

with coordinates:

- `time`
- `latitude`

and variables:

- `warm_state_temperature`
- `cold_state_temperature`

The dataset attributes store the run settings, model parameters, solver
method, and the warm/cold initial temperatures.

The edge-state script solves the edge-state BVP branch and writes a NetCDF
file to:

```text
<repo-root>/data/<filename>.nc
```

with coordinate:

- `latitude`

and variables:

- `edge_state_temperature`
- `edge_state_temperature_derivative`

The dataset attributes store the run settings, model parameters, the edge
initial temperature, and the BVP solver controls.

## Analysis

The saved NetCDF outputs can be explored with the marimo app in:

```bash
uv run marimo edit analysis/analysis.py
```
