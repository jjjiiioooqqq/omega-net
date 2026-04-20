from __future__ import annotations

import numpy as np

from .interfaces import EvidenceTier, GoverningEquation, PhysicsResult


class ThermalDiffusionResidualSolver:
    """MVP residual evaluator for 1D thermal diffusion: dT/dt - alpha*d2T/dx2 = 0."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def evaluate(self, state: np.ndarray, dx: float, dt: float) -> PhysicsResult:
        if state.ndim != 2:
            raise ValueError("state must be shape [time, space]")
        if state.shape[0] < 2 or state.shape[1] < 3:
            raise ValueError("state too small for finite differencing")

        dt_term = (state[1:, 1:-1] - state[:-1, 1:-1]) / dt
        laplacian = (state[:-1, 2:] - 2 * state[:-1, 1:-1] + state[:-1, :-2]) / (dx**2)
        residual_grid = dt_term - self.alpha * laplacian
        residual = float(np.sqrt(np.mean(np.square(residual_grid))))

        equation = GoverningEquation(
            name="thermal_diffusion_1d",
            equation_type="heat_transport",
            evidence_tier=EvidenceTier.GOVERNING,
            notes="Finite-difference residual only; not a full validated PDE solve.",
        )
        return PhysicsResult(
            solver_name="ThermalDiffusionResidualSolver",
            governing_equations=[equation],
            residual=residual,
            metrics={"rms_residual": residual},
            surrogate_metrics={"mean_temperature": float(np.mean(state))},
            learned_approximation_metrics={"pinn_loss_proxy": float(np.var(residual_grid))},
            verified=True,
        )


class PlaceholderPhysicsSolver:
    """Explicit placeholder for unsupported domains (Maxwell/Navier/Stuctural wrappers)."""

    def __init__(self, name: str, equation_type: str) -> None:
        self.name = name
        self.equation_type = equation_type

    def evaluate(self, state: np.ndarray, dx: float, dt: float) -> PhysicsResult:
        equation = GoverningEquation(
            name=self.name,
            equation_type=self.equation_type,
            evidence_tier=EvidenceTier.SURROGATE,
            notes="Placeholder adapter; connect external validated solver.",
        )
        return PhysicsResult(
            solver_name=f"{self.name}Placeholder",
            governing_equations=[equation],
            residual=float("nan"),
            metrics={},
            surrogate_metrics={"state_mean": float(np.mean(state)) if state.size else 0.0},
            learned_approximation_metrics={},
            verified=False,
        )
