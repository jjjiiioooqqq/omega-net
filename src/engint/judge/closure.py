from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ClosureInputs:
    residual_ok: bool
    hard_constraints_ok: bool
    margins_positive: bool
    manufacturable: bool
    controllable: bool
    power_thermal_closed: bool
    provenance_complete: bool
    unknowns_explicit: bool


@dataclass(slots=True)
class ClosureReport:
    classification: str
    checks: dict[str, bool]
    blockers: list[str] = field(default_factory=list)


class ClosureJudge:
    """Explicit policy for legendary/provisional/unverified/folkloric classes."""

    def evaluate(self, inputs: ClosureInputs) -> ClosureReport:
        checks = {
            "residual_ok": inputs.residual_ok,
            "hard_constraints_ok": inputs.hard_constraints_ok,
            "margins_positive": inputs.margins_positive,
            "manufacturable": inputs.manufacturable,
            "controllable": inputs.controllable,
            "power_thermal_closed": inputs.power_thermal_closed,
            "provenance_complete": inputs.provenance_complete,
            "unknowns_explicit": inputs.unknowns_explicit,
        }
        blockers = [k for k, v in checks.items() if not v]

        if all(checks.values()):
            classification = "legendary"
        elif inputs.hard_constraints_ok and inputs.residual_ok and inputs.provenance_complete:
            classification = "provisional"
        elif inputs.hard_constraints_ok:
            classification = "unverified"
        else:
            classification = "folkloric"
        return ClosureReport(classification=classification, checks=checks, blockers=blockers)
