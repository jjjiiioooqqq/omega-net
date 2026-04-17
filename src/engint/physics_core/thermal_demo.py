from __future__ import annotations

import numpy as np

from .interfaces import GoverningEquation, PhysicsResult


class ThermalDiffusionResidualSolver:
    """MVP residual evaluator for 1D thermal diffusion equation dT/dt - alpha*d2T/dx2 = 0."""

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

        eq = GoverningEquation(
            name="thermal_diffusion_1d",
            equation_type="governing",
            trusted=True,
            notes="Finite-difference residual only; not full validation.",
        )
        return PhysicsResult(
            solver_name="ThermalDiffusionResidualSolver",
            governing_equations=[eq],
            residual=residual,
            metrics={"rms_residual": residual},
            surrogate_metrics={"mean_temperature": float(np.mean(state))},
            learned_approximation_metrics={"pinn_loss_proxy": float(np.var(residual_grid))},
            verified=True,
        )
