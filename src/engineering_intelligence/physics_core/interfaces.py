"""Interfaces for physics evaluations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


ModelType = Literal["governing_equation", "surrogate", "learned_unverified"]


@dataclass(slots=True)
class PhysicsCase:
    """Problem specification for a solver."""

    case_id: str
    domain: Literal["maxwell", "flow", "thermal", "structural"]
    parameters: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhysicsResult:
    """Outcome of a physics solve/evaluation."""

    case_id: str
    model_type: ModelType
    residual: float
    metrics: dict[str, float]
    verified: bool
    notes: str


class PhysicsSolver(ABC):
    """Abstract base for extensible physics backends."""

    @abstractmethod
    def solve(self, case: PhysicsCase) -> PhysicsResult:
        """Solve/evaluate case and return quantitative evidence."""


class PINNReadyMixin:
    """Marker + hook for future PINN integrations.

    PINN outputs must remain model_type='learned_unverified' until validated.
    """

    def train_surrogate(self, _case: PhysicsCase) -> None:
        """Placeholder for future learned-model integration."""
        return None
