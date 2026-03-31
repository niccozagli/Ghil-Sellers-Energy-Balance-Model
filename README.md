# Ghil-Sellers Energy Balance Model

This repository is being migrated from the original MATLAB implementation
of the Ghil-Sellers energy balance model into a Python package managed with
`uv`.

## Layout

- `matlab/`: original MATLAB reference implementation kept intact for
  verification during the port
- `src/gsebm/`: Python package for the new implementation
- `tests/`: Python test suite for the port

## Migration Approach

The port will proceed in small steps. The initial structure separates the
reference MATLAB code from the Python package so that each translated
component can be checked against the original implementation.
