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
- a method-of-lines IVP solver in `src/gsebm/ivp.py`
- a steady-state BVP solver in `src/gsebm/bvp.py`
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

## Layout

- `matlab/`: reference implementation
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
