"""Concrete simplified physics solver implementations."""

from __future__ import annotations

import numpy as np

from .interfaces import PINNReadyMixin, PhysicsCase, PhysicsResult, PhysicsSolver


class Heat1DResidualSolver(PhysicsSolver, PINNReadyMixin):
    """Finite-difference residual evaluator for 1D steady thermal diffusion.

    Governing equation:
        k * d2T/dx2 + q = 0
    Uses central difference on an evenly spaced grid.
    """

    def solve(self, case: PhysicsCase) -> PhysicsResult:
        if case.domain != "thermal":
            raise ValueError("Heat1DResidualSolver only handles thermal domain")

        k = case.parameters["k"]
        q = case.parameters["q"]
        length = case.parameters["length"]
        t_left = case.parameters["t_left"]
        t_right = case.parameters["t_right"]
        n_points = int(case.parameters.get("n_points", 21))

        x = np.linspace(0.0, length, n_points)
        t_linear = t_left + (t_right - t_left) * (x / length)
        dx = x[1] - x[0]

        second_derivative = (t_linear[2:] - 2 * t_linear[1:-1] + t_linear[:-2]) / (dx**2)
        residual_field = k * second_derivative + q
        l2_residual = float(np.sqrt(np.mean(residual_field**2)))

        return PhysicsResult(
            case_id=case.case_id,
            model_type="governing_equation",
            residual=l2_residual,
            metrics={"grid_spacing": float(dx), "max_abs_residual": float(np.max(np.abs(residual_field)))},
            verified=True,
            notes="Finite-difference PDE residual on linear trial field.",
        )
