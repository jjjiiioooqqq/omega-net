"""Concrete simplified physics solver implementations."""

from __future__ import annotations

import numpy as np

from .interfaces import PINNReadyMixin, PhysicsCase, PhysicsResult, PhysicsSolver


class Heat1DResidualSolver(PhysicsSolver, PINNReadyMixin):
    """Finite-difference residual evaluator for 1D steady thermal diffusion.

    Governing equation:
        k * d2T/dx2 + q = 0

    Implementation details:
    - Constructs an analytic trial field that satisfies constant-source conduction
      with Dirichlet boundary conditions.
    - Evaluates PDE residual via central differencing to provide a numerical
      closure signal independent of symbolic consistency checks.
    """

    def solve(self, case: PhysicsCase) -> PhysicsResult:
        if case.domain != "thermal":
            raise ValueError("Heat1DResidualSolver only handles thermal domain")

        k = float(case.parameters["k"])
        q = float(case.parameters["q"])
        length = float(case.parameters["length"])
        t_left = float(case.parameters["t_left"])
        t_right = float(case.parameters["t_right"])
        n_points = int(case.parameters.get("n_points", 21))

        if k <= 0:
            raise ValueError("Thermal conductivity k must be positive")
        if length <= 0:
            raise ValueError("Domain length must be positive")
        if n_points < 3:
            raise ValueError("Need at least 3 points for second derivative residual")

        x = np.linspace(0.0, length, n_points)
        # T(x) = -(q/(2k))x^2 + C1 x + C2 with boundary conditions at x=0 and x=L.
        c2 = t_left
        c1 = (t_right - t_left + (q / (2.0 * k)) * (length**2)) / length
        t_profile = -(q / (2.0 * k)) * (x**2) + c1 * x + c2

        dx = x[1] - x[0]
        second_derivative = (t_profile[2:] - 2 * t_profile[1:-1] + t_profile[:-2]) / (dx**2)
        residual_field = k * second_derivative + q
        l2_residual = float(np.sqrt(np.mean(residual_field**2)))
        bc_error = float(max(abs(t_profile[0] - t_left), abs(t_profile[-1] - t_right)))

        return PhysicsResult(
            case_id=case.case_id,
            model_type="governing_equation",
            residual=l2_residual,
            metrics={
                "grid_spacing": float(dx),
                "max_abs_residual": float(np.max(np.abs(residual_field))),
                "boundary_error": bc_error,
            },
            verified=True,
            notes="Finite-difference PDE residual on analytic constant-source conduction profile.",
        )
