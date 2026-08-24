# AGENTS.md

Guidance for AI agents working in this repository.

## Project Overview

This repository contains the Python implementation of the Ghil-Sellers Energy
Balance Model. It started as a port from a preserved MATLAB implementation,
but ongoing work should treat the Python package as the maintained codebase.
The package uses a `src/` layout and is managed with `uv`.

The model is a one-dimensional zonal-mean energy balance model on normalized
latitude `x in [-1, 1]`. It combines latitude-dependent heat capacity,
diffusive meridional heat transport, outgoing longwave radiation, absorbed
shortwave radiation, and temperature-dependent albedo.

The code is an active research codebase. Prefer small, well-tested changes
that preserve the model's documented numerical and physical behavior. The
MATLAB files can be useful historical context, but new work does not need to
mirror their structure or implementation choices.

## Repository Layout

- `src/gsebm/`: Python package.
- `tests/`: `unittest` test suite.
- `scripts/`: Typer command-line entry points for research runs.
- `analysis/`: marimo analysis notebooks/apps for saved datasets.
- `matlab/`: archived original MATLAB implementation. Use it only for
  historical context or when the user explicitly asks for MATLAB comparison.
- `data/`: generated NetCDF datasets, ignored by git.
- `figures/`: generated figures, ignored by git.
- `prototyping/`: exploratory code. Do not let prototypes silently become
  package behavior without tests.

## Core Python Modules

- `parameters.py`: frozen dataclasses for physical parameters and numerical
  run settings.
- `empirical.py`: tabulated latitude-dependent empirical fields, equatorial
  mirroring, interpolants, and IVP grid sampling.
- `physics.py`: local scalar/vectorized physical formulas. Keep these free of
  solver-specific logic.
- `initial_conditions.py`: scalar, default observational, and custom initial
  temperature profile builders.
- `ivp.py`: deterministic method-of-lines IVP solver using `scipy.integrate.solve_ivp`.
- `bvp.py`: steady-state boundary value solver using `scipy.integrate.solve_bvp`.
- `sde.py`: stochastic fixed-step IMEX temperature solver with spatially
  correlated additive noise.
- `run.py`: high-level workflows, bifurcation sweeps, and `xarray.Dataset`
  serialization.
- `diagnostics.py`: diagnostics reconstructed from model states and saved
  dataset metadata.
- `plotting.py`: matplotlib plotting helpers.
- `paths.py`: repository and data-directory helpers.
- `time.py`: physical time-unit constants.

## Development Commands

Use Python 3.12. The checked-in `.python-version` is `3.12`.

Install or update the environment with:

```bash
uv sync
```

Run the full test suite with either:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

or, inside the uv environment:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests
```

Run one test module with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_ivp
```

Run scripts with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python3 scripts/run_warm_cold_state.py --help
```

## Numerical and Domain Conventions

- Normalized latitude is `x in [-1, 1]`, where the poles are near `-1` and `1`.
- The IVP grid intentionally uses near-pole shifted points controlled by
  `RunSettings.delta`; do not replace it with a naive uniform endpoint grid
  without checking pole behavior.
- Empirical data grids are not solver grids. Build interpolants first, then
  sample onto the fixed IVP grid when needed.
- The BVP uses continuous empirical coefficient functions and derivatives on
  an adaptive mesh. Do not reuse pre-sampled IVP arrays in the BVP ODE.
- The IVP transport term is in divergence form with zero boundary face flux.
  Preserve this no-flux pole condition.
- `physics.py` functions accept scalars or NumPy arrays and often return a
  Python scalar for scalar input. Preserve this behavior unless tests are
  updated deliberately.
- The stochastic solver freezes diffusion at the previous state, treats
  reaction explicitly, and solves the implicit diffusion step with a banded
  tridiagonal matrix. Keep deterministic and stochastic operator definitions
  consistent.
- Be careful with units. Internal time values are seconds. Some CLI options
  accept years or days and convert them before building settings; inspect the
  script before changing option defaults or help text.

## Data and Output Rules

- Generated NetCDF files belong in `data/` and are ignored by git.
- Generated figures belong in `figures/` and are ignored by git.
- Do not commit generated research outputs unless the user explicitly asks.
- Dataset writers use `xarray` with `engine="scipy"` and store parameters and
  run settings as attributes. If you add new settings or parameters, update
  serialization, reconstruction diagnostics, and tests together.

## Testing Expectations

- Add or update focused tests for behavior changes in `tests/`.
- For physics formula changes, update `tests/test_physics.py`.
- For empirical interpolation or grid behavior, update
  `tests/test_empirical.py` and `tests/test_interpolation.py`.
- For IVP/BVP/SDE solver behavior, use short final times or small grids when
  possible so tests remain practical.
- For workflow or CLI changes, update `tests/test_run.py`.
- For saved-dataset diagnostics, update `tests/test_diagnostics.py`.

## Coding Style

- Follow the existing dataclass-heavy, typed Python style.
- Keep solver assembly separate from local physical formulas.
- Prefer explicit NumPy array conversion at module boundaries over implicit
  list handling.
- Keep public functions documented with concise docstrings.
- Use `pathlib.Path` for filesystem paths.
- Avoid broad refactors while changing numerical behavior; isolate formula,
  interpolation, and workflow changes so they can be reviewed independently.

## Safety Notes for Agents

- The git worktree may contain user changes. Inspect status before editing and
  do not revert unrelated changes.
- Treat `matlab/` as archival context, not as the authority for new Python
  design. Do not force Python changes to mimic MATLAB-specific solver,
  plotting, or driver structure unless the user explicitly asks.
- Treat `analysis/` and `prototyping/` as exploratory unless the user asks for
  production changes there.
- Do not delete files in `data/` or `figures/` unless the user explicitly asks.
- Before changing defaults in `parameters.py`, check scripts, datasets,
  diagnostics, and tests that depend on serialized setting names and values.
