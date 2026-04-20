from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scipy.optimize import minimize


@dataclass(slots=True)
class OptimizationResult:
    x: list[float]
    objective: float
    success: bool
    message: str


class Objectives:
    @staticmethod
    def minimize_residual(residual: Callable[[list[float]], float]) -> Callable[[list[float]], float]:
        return lambda x: float(residual(x))

    @staticmethod
    def maximize_metric(metric: Callable[[list[float]], float]) -> Callable[[list[float]], float]:
        return lambda x: -float(metric(x))

    @staticmethod
    def maximize_margin(margin: Callable[[list[float]], float]) -> Callable[[list[float]], float]:
        return lambda x: -float(margin(x))

    @staticmethod
    def minimize_load(load: Callable[[list[float]], float]) -> Callable[[list[float]], float]:
        return lambda x: float(load(x))


class AdmissibleOptimizer:
    """Continuous optimizer that evaluates objective only in admissible region."""

    def optimize(
        self,
        seed: list[float],
        objective: Callable[[list[float]], float],
        in_admissible_region: Callable[[list[float]], bool],
    ) -> OptimizationResult:
        if not in_admissible_region(seed):
            return OptimizationResult(seed, float("inf"), False, "seed outside admissible region")

        def wrapped(x):
            xv = [float(v) for v in x]
            if not in_admissible_region(xv):
                return 1e9
            return objective(xv)

        res = minimize(wrapped, seed, method="L-BFGS-B")
        return OptimizationResult([float(v) for v in res.x], float(res.fun), bool(res.success), str(res.message))
