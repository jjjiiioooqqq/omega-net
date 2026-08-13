from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# Fundamentally different states that must never collapse into one number.
WEALTH_LADDER = (
    "idea",
    "research_supported_hypothesis",
    "customer_payment",
    "repeated_payment",
    "recurring_revenue",
    "owner_independent_operation",
    "transferable_equity",
    "realizable_owner_wealth",
)

# The only event kinds that can move the market-validation state. Research
# artifacts (papers, TAM estimates, funding news, LLM agreement, job postings)
# are structurally unable to change it.
MARKET_EVENTS = ("payment", "unrelated_second_payment", "recurring_commitment")


class FirewallViolation(RuntimeError):
    """Raised when research evidence is offered where a market event is required."""


@dataclass(slots=True)
class MarketValidationFirewall:
    """MARKET_VALIDATED changes only on predefined market events.

    Research can raise or lower whether testing is justified; it cannot
    fabricate demand.
    """

    events: list[str] = field(default_factory=list)

    @property
    def validated(self) -> bool:
        return all(kind in self.events for kind in MARKET_EVENTS)

    def observe(self, event_kind: str) -> None:
        if event_kind not in MARKET_EVENTS:
            raise FirewallViolation(
                f"'{event_kind}' is not a market event; only {MARKET_EVENTS} can move validation"
            )
        self.events.append(event_kind)

    def ladder_state(self) -> str:
        if "recurring_commitment" in self.events:
            return "recurring_revenue"
        if "unrelated_second_payment" in self.events:
            return "repeated_payment"
        if "payment" in self.events:
            return "customer_payment"
        return "research_supported_hypothesis"


class ExperimentFrozen(RuntimeError):
    """Raised on any post-hoc change to a precommitted experiment."""


@dataclass(slots=True)
class Experiment:
    """H -> test -> precommitted metric -> result -> update.

    The denominator is stored before the experiment begins. No changing the
    sample because the result looks bad; no inventing a new KPI afterward;
    no calling weak signals validation.
    """

    experiment_id: str
    hypothesis: str
    alternative: str
    metric: str
    denominator: int
    interpretation_rules: tuple[str, ...] = ()
    started: bool = False
    result: str | None = None

    def start(self) -> None:
        self.started = True

    def amend_denominator(self, value: int) -> None:
        if self.started:
            raise ExperimentFrozen("denominator is frozen once the experiment starts")
        self.denominator = value

    def record_result(self, result: str) -> None:
        if not self.started:
            raise ExperimentFrozen("cannot record a result before the experiment starts")
        if self.result is not None:
            raise ExperimentFrozen("result already recorded; experiments are not rescored")
        self.result = result


@dataclass(slots=True)
class Strategy:
    """A competing wealth strategy; never the religion of the system."""

    strategy_id: str
    name: str
    mechanism: str
    ownership_adjusted_potential: float
    time_to_decisive_evidence: float  # e.g. weeks
    capital_to_decisive_evidence: float
    kill_conditions: tuple[str, ...] = ()
    status: str = "candidate"  # candidate | active | rejected | paused

    @property
    def asymmetry(self) -> float:
        """ownership-adjusted potential / (time + capital to decisive evidence)."""
        cost = max(self.time_to_decisive_evidence + self.capital_to_decisive_evidence, 1e-6)
        return self.ownership_adjusted_potential / cost


class StrategyTournament:
    """Strategies compete for scarce founder time and capital."""

    def __init__(self) -> None:
        self.strategies: dict[str, Strategy] = {}

    def register(self, strategy: Strategy | None = None, **kwargs) -> Strategy:
        if strategy is None:
            strategy = Strategy(strategy_id=f"STR-{uuid.uuid4().hex[:6]}", **kwargs)
        self.strategies[strategy.strategy_id] = strategy
        return strategy

    def ranked(self) -> list[Strategy]:
        live = [s for s in self.strategies.values() if s.status in ("candidate", "active")]
        return sorted(live, key=lambda s: s.asymmetry, reverse=True)

    def kill(self, strategy_id: str, reason: str) -> None:
        strategy = self.strategies[strategy_id]
        strategy.status = "rejected"
        strategy.kill_conditions = strategy.kill_conditions + (f"killed: {reason}",)
