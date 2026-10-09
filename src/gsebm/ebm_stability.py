"""Linear stability of EBM equilibria: the ground truth for mode-finding methods.

Equilibria solve the semi-discrete IVP tendency `IVPOperator.rhs = 0` on the
solver grid; the Jacobian is a one-sided finite difference of that tendency.
Rates are in yr⁻¹.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import root

from gsebm.ivp import IVPOperator, build_ivp_operator
from gsebm.parameters import default_model_parameters
from gsebm.time import YEAR


@dataclass(frozen=True)
class StabilityResult:
    """Equilibrium, Jacobian modes (slowest first) and forced response at one μ."""

    mu: float
    x: np.ndarray
    temperature: np.ndarray
    rates: np.ndarray
    modes: np.ndarray
    forced_response: np.ndarray


def operator_at(mu: float) -> IVPOperator:
    """IVP operator with default parameters and the given μ."""
    return build_ivp_operator(params=replace(default_model_parameters(), mu=mu))


def equilibrium(operator: IVPOperator, guess: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Solve rhs(T) = 0 from `guess`; raise if the residual exceeds `tolerance` K yr⁻¹."""
    tendency = lambda temperature: operator.rhs(0.0, temperature) * YEAR
    solution = root(tendency, np.asarray(guess, dtype=float), method="hybr", tol=1e-12)
    residual = float(np.max(np.abs(tendency(solution.x))))
    if residual > tolerance:
        raise ValueError(f"No equilibrium found (residual {residual:.2e} K/yr).")
    return solution.x


def tendency_jacobian(operator: IVPOperator, temperature: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Finite-difference Jacobian of the tendency at `temperature`, in yr⁻¹."""
    base = operator.rhs(0.0, temperature)
    jacobian = np.empty((temperature.size, temperature.size))
    for column in range(temperature.size):
        shifted = temperature.copy()
        shifted[column] += step
        jacobian[:, column] = (operator.rhs(0.0, shifted) - base) / step
    return jacobian * YEAR


def stability(mu: float, guess: np.ndarray, dmu: float = 2e-4) -> StabilityResult:
    """Equilibrium, Jacobian modes and dT*/dμ (centred difference, step `dmu`) at μ."""
    operator = operator_at(mu)
    temperature = equilibrium(operator, guess)
    upper = equilibrium(operator_at(mu + dmu), temperature)
    lower = equilibrium(operator_at(mu - dmu), temperature)
    eigenvalues, vectors = np.linalg.eig(tendency_jacobian(operator, temperature))
    order = np.argsort(-eigenvalues.real)
    return StabilityResult(
        mu=mu,
        x=operator.x,
        temperature=temperature,
        rates=eigenvalues[order],
        modes=vectors[:, order].T,
        forced_response=(upper - lower) / (2 * dmu),
    )
