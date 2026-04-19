"""Differentiable optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize


ObjectiveType = Literal[
    "minimize_residual",
    "maximize_performance",
    "maximize_margin",
    "minimize_power",
    "minimize_thermal_load",
]


@dataclass(slots=True)
class OptimizationProblem:
    """Continuous optimization problem inside admissible design region."""

    objective_type: ObjectiveType
    variable_names: list[str]
    initial_guess: np.ndarray
    bounds: list[tuple[float, float]]
    objective_fn: Callable[[np.ndarray], float]
    admissibility_fn: Callable[[dict[str, float]], bool]
    gradient_fn: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(slots=True)
class OptimizationResult:
    """Optimization result record."""

    success: bool
    x: dict[str, float]
    objective_value: float
    message: str


def optimize_admissible(problem: OptimizationProblem) -> OptimizationResult:
    """Run constrained optimization; reject evaluations outside admissible region."""

    maximize = problem.objective_type in {"maximize_performance", "maximize_margin"}

    def wrapped_objective(x: np.ndarray) -> float:
        values = {n: float(v) for n, v in zip(problem.variable_names, x, strict=True)}
        if not problem.admissibility_fn(values):
            return 1e9

        value = float(problem.objective_fn(x))
        return -value if maximize else value

    def wrapped_gradient(x: np.ndarray) -> np.ndarray:
        if problem.gradient_fn is None:
            raise RuntimeError("gradient_fn requested but missing")

        values = {n: float(v) for n, v in zip(problem.variable_names, x, strict=True)}
        if not problem.admissibility_fn(values):
            return np.zeros_like(x)

        grad = np.asarray(problem.gradient_fn(x), dtype=float)
        return -grad if maximize else grad

    result = minimize(
        wrapped_objective,
        x0=problem.initial_guess,
        jac=wrapped_gradient if problem.gradient_fn is not None else None,
        method="L-BFGS-B",
        bounds=problem.bounds,
    )

    x_map = {n: float(v) for n, v in zip(problem.variable_names, result.x, strict=True)}
    objective_value = float(-result.fun if maximize else result.fun)

    return OptimizationResult(
        success=bool(result.success),
        x=x_map,
        objective_value=objective_value,
        message=str(result.message),
    )
