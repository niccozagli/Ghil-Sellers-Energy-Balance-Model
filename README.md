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
- basic tests for package import, default parameter values, and simple
  configuration validation

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

## Layout

- `matlab/`: reference implementation
- `src/gsebm/`: Python package for the new implementation
- `tests/`: Python test suite

## Development Approach

The port proceeds in small steps. The immediate goal is to translate the
model into clear Python modules while keeping the implementation easy to
compare against the reference code.

The expected sequence is:

1. parameters and default settings
2. empirical latitude-dependent data
3. shared physical relationships
4. IVP and BVP model components
5. diagnostics, plotting, and solver wiring

## Running Tests

At the current stage, the lightweight test suite can be run with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
