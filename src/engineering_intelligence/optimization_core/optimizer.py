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


@dataclass(slots=True)
class OptimizationResult:
    """Optimization result record."""

    success: bool
    x: dict[str, float]
    objective_value: float
    message: str


def optimize_admissible(problem: OptimizationProblem) -> OptimizationResult:
    """Run constrained optimization; reject evaluations outside admissible region."""

    def wrapped_objective(x: np.ndarray) -> float:
        values = {n: float(v) for n, v in zip(problem.variable_names, x, strict=True)}
        if not problem.admissibility_fn(values):
            return 1e9

        value = problem.objective_fn(x)
        if problem.objective_type in {"maximize_performance", "maximize_margin"}:
            return -value
        return value

    result = minimize(
        wrapped_objective,
        x0=problem.initial_guess,
        method="L-BFGS-B",
        bounds=problem.bounds,
    )

    x_map = {n: float(v) for n, v in zip(problem.variable_names, result.x, strict=True)}
    objective_value = float(-result.fun if problem.objective_type in {"maximize_performance", "maximize_margin"} else result.fun)

    return OptimizationResult(
        success=bool(result.success),
        x=x_map,
        objective_value=objective_value,
        message=str(result.message),
    )
