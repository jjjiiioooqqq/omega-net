from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class EvidenceTier(str, Enum):
    GOVERNING = "governing"
    SURROGATE = "surrogate"
    LEARNED_UNVERIFIED = "learned_unverified"


@dataclass(slots=True)
class GoverningEquation:
    name: str
    equation_type: str
    evidence_tier: EvidenceTier
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
    """Base interface for solver wrappers (classical or differentiable)."""

    def evaluate(self, state: np.ndarray, dx: float, dt: float) -> PhysicsResult:
        ...
