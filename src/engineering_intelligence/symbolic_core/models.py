"""Data models for symbolic architecture pruning."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateArchitecture:
    """A candidate design with continuous or discrete variables."""

    architecture_id: str
    variables: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConstraintSpec:
    """Constraint specification used in pruning.

    expression: symbolic expression string using variable names (e.g. "power <= 100").
    kill_criterion: whether violation invalidates the entire architecture.
    """

    name: str
    expression: str
    kill_criterion: bool = True


@dataclass(slots=True)
class PruneResult:
    """Result of architecture pruning across candidate set."""

    admissible: list[CandidateArchitecture]
    rejected: dict[str, list[str]]
