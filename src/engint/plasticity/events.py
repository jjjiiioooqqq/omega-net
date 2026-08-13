from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .cognition import AdaptiveComprehension, CognitionPlan, ProblemProfile

# Always-running means a sentinel, not continuous model thought:
# watch -> detect change -> calculate materiality -> wake appropriate agents.
TRIGGER_KINDS = (
    "sec_filing",
    "earnings_release",
    "research_paper",
    "competitor_update",
    "prospect_change",
    "prediction_resolved",
    "customer_response",
    "benchmark_failure",
    "scheduled_reevaluation",
)


@dataclass(frozen=True, slots=True)
class SourceEvent:
    kind: str
    subject: str
    observed_at: datetime
    materiality: float  # [0, 1], computed by the watcher, not by the model
    profile: ProblemProfile


@dataclass(slots=True)
class RoutedInvestigation:
    event: SourceEvent
    plan: CognitionPlan


@dataclass(slots=True)
class EventRouter:
    """Wakes cognition only when a change is material.

    Immaterial events update state silently; nothing is surfaced and no
    inference is spent. Material events are routed through the adaptive
    comprehension controller so consequential changes get deep treatment.
    """

    comprehension: AdaptiveComprehension = field(default_factory=AdaptiveComprehension)
    materiality_threshold: float = 0.2
    ignored: list[SourceEvent] = field(default_factory=list)
    routed: list[RoutedInvestigation] = field(default_factory=list)

    def route(self, event: SourceEvent, scar_hit: bool = False) -> RoutedInvestigation | None:
        if event.kind not in TRIGGER_KINDS:
            raise ValueError(f"unknown trigger kind '{event.kind}'; expected one of {TRIGGER_KINDS}")
        if event.materiality < self.materiality_threshold:
            self.ignored.append(event)
            return None
        plan = self.comprehension.plan(event.profile, scar_hit=scar_hit)
        investigation = RoutedInvestigation(event=event, plan=plan)
        self.routed.append(investigation)
        return investigation
