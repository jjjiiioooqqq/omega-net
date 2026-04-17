from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateArchitecture:
    arch_id: str
    variables: dict[str, float]
    assumptions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConstraintSpec:
    name: str
    expression: str
    kind: str = "hard"


@dataclass(slots=True)
class PruneResult:
    survivors: list[CandidateArchitecture]
    rejected: dict[str, list[str]]
    canonical_forms: dict[str, str]
    satisfiable: bool
    metadata: dict[str, Any] = field(default_factory=dict)
