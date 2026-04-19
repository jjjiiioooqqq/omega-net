"""Machine-readable audit report with strict verification labeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

REQUIRED_SECTIONS = [
    "ASSUMED",
    "DERIVED",
    "VERIFIED",
    "UNVERIFIED",
    "FAILURE_MODES",
    "NEXT_TEST",
]


@dataclass(slots=True)
class AuditSection:
    """Named section in audit output."""

    entries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AuditReport:
    """Audit record for a single design evaluation."""

    design_id: str
    sections: dict[str, AuditSection]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "sections": {name: asdict(section) for name, section in self.sections.items()},
            "metadata": self.metadata,
        }


def _normalize_entry(text: str, *, unknown_fallback: str = "UNVERIFIED") -> str:
    stripped = text.strip()
    if not stripped:
        return unknown_fallback
    return stripped


def build_audit_report(
    design_id: str,
    *,
    assumed: list[str],
    derived: list[str],
    verified: list[str],
    unverified: list[str],
    failure_modes: list[str],
    next_test: list[str],
    metadata: dict[str, Any] | None = None,
) -> AuditReport:
    """Construct audit report while preserving explicit unknown labels."""
    sections = {
        "ASSUMED": AuditSection([_normalize_entry(x, unknown_fallback="UNKNOWN") for x in assumed] or ["UNKNOWN"]),
        "DERIVED": AuditSection([_normalize_entry(x) for x in derived] or ["UNVERIFIED"]),
        "VERIFIED": AuditSection([_normalize_entry(x) for x in verified] or ["UNVERIFIED"]),
        "UNVERIFIED": AuditSection([_normalize_entry(x) for x in unverified] or ["UNVERIFIED"]),
        "FAILURE_MODES": AuditSection([_normalize_entry(x) for x in failure_modes] or ["UNVERIFIED"]),
        "NEXT_TEST": AuditSection([_normalize_entry(x) for x in next_test] or ["UNVERIFIED"]),
    }

    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    if missing:
        raise ValueError(f"Missing required sections: {missing}")

    return AuditReport(design_id=design_id, sections=sections, metadata=metadata or {})
