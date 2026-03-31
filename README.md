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
- basic tests for package import, default parameter values, and simple
  configuration validation

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
