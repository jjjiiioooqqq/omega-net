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


class AdmissibleOptimizer:
    """Continuous optimizer that accepts only symbolically admissible seeds."""

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
