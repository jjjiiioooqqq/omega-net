from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class GoverningEquation:
    name: str
    equation_type: str
    trusted: bool
    notes: str


@dataclass(slots=True)
class PhysicsResult:
    solver_name: str
    governing_equations: list[GoverningEquation]
    residual: float
    metrics: dict[str, float]
    surrogate_metrics: dict[str, float] = field(default_factory=dict)
    learned_approximation_metrics: dict[str, float] = field(default_factory=dict)
    verified: bool = False


class PhysicsSolver(Protocol):
    def evaluate(self, state: np.ndarray, dx: float, dt: float) -> PhysicsResult:
        ...
