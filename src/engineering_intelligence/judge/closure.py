"""Formal design closure and classification logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Classification = Literal["legendary", "provisional", "unverified", "folkloric"]


@dataclass(slots=True)
class ClosureInputs:
    """Inputs required for design closure decision."""

    residual: float
    residual_threshold: float
    hard_constraints_satisfied: bool
    margin: float
    manufacturable: bool
    controllable: bool
    power_budget_closed: bool
    thermal_budget_closed: bool
    provenance_complete: bool
    unknowns: list[str]


@dataclass(slots=True)
class ClosureResult:
    """Judge output containing explicit gating rationale."""

    classification: Classification
    reasons: list[str]


class ClosureJudge:
    """Classifies designs using explicit closure gates."""

    def evaluate(self, inputs: ClosureInputs) -> ClosureResult:
        reasons: list[str] = []

        if inputs.residual > inputs.residual_threshold:
            reasons.append("Residual above threshold")
        if not inputs.hard_constraints_satisfied:
            reasons.append("Hard constraints violated")
        if inputs.margin <= 0.0:
            reasons.append("Margin not positive")
        if not inputs.manufacturable:
            reasons.append("Manufacturability violated")
        if not inputs.controllable:
            reasons.append("Control feasibility violated")
        if not inputs.power_budget_closed:
            reasons.append("Power budget not closed")
        if not inputs.thermal_budget_closed:
            reasons.append("Thermal budget not closed")
        if not inputs.provenance_complete:
            reasons.append("Critical claims missing provenance")

        unknown_exists = len(inputs.unknowns) > 0

        if not reasons and not unknown_exists:
            return ClosureResult("legendary", ["All closure gates satisfied with no unknown carry."])

        if reasons and any(
            critical in reasons
            for critical in (
                "Hard constraints violated",
                "Manufacturability violated",
                "Control feasibility violated",
            )
        ):
            return ClosureResult("folkloric", reasons + (["UNKNOWN carry present"] if unknown_exists else []))

        if reasons:
            return ClosureResult("provisional", reasons + (["UNKNOWN carry present"] if unknown_exists else []))

        return ClosureResult("unverified", ["UNKNOWN carry present"])
