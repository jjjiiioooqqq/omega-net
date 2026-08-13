from __future__ import annotations

from dataclasses import dataclass

# Comprehension levels: C0 deterministic computation ... C6 competing
# architectures evolved against the problem.
COMPREHENSION_LEVELS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")


@dataclass(frozen=True, slots=True)
class ProblemProfile:
    """Inputs to the cognitive-budget formula, each in [0, 1]."""

    complexity: float  # D
    importance: float  # S: stakes / consequence of being wrong
    novelty: float  # N
    failure_rate: float  # F: historical failure rate on similar problems
    uncertainty: float  # U: unresolved uncertainty
    irreversibility: float = 0.0  # I: how hard the action is to undo


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """B = B0 * (1 + aD + bS + cN + dF + eU + zI).

    Model confidence is deliberately not an input: the controller changes the
    environment around the model, not the model's self-assessment.
    """

    base_budget: float = 1.0
    alpha: float = 1.0  # complexity
    beta: float = 2.0  # stakes dominate: cheap mistakes stay cheap
    gamma: float = 0.5  # novelty
    delta: float = 1.5  # prior failure -> more care around intellectual scars
    epsilon: float = 1.0  # uncertainty
    zeta: float = 2.0  # irreversibility weighs like stakes: undoable is cheap

    def budget(self, profile: ProblemProfile) -> float:
        return self.base_budget * (
            1.0
            + self.alpha * profile.complexity
            + self.beta * profile.importance
            + self.gamma * profile.novelty
            + self.delta * profile.failure_rate
            + self.epsilon * profile.uncertainty
            + self.zeta * profile.irreversibility
        )


@dataclass(frozen=True, slots=True)
class CognitionPlan:
    level: str
    budget: float
    escalated_by_scar: bool


class AdaptiveComprehension:
    """The system does not spend equal intelligence on every problem.

    A $20 arithmetic question never receives C6; a decision committing six
    months and $100,000 probably should. Known failure patterns escalate
    the level regardless of the raw budget.
    """

    def __init__(self, policy: BudgetPolicy | None = None) -> None:
        self.policy = policy or BudgetPolicy()
        # Budget thresholds for C1..C6; below the first is C0.
        self.thresholds = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)

    def plan(self, profile: ProblemProfile, scar_hit: bool = False) -> CognitionPlan:
        budget = self.policy.budget(profile)
        level_index = sum(1 for t in self.thresholds if budget >= t)
        if scar_hit and level_index < len(COMPREHENSION_LEVELS) - 1:
            level_index += 1
        return CognitionPlan(
            level=COMPREHENSION_LEVELS[level_index],
            budget=budget,
            escalated_by_scar=scar_hit,
        )


def cognition_roi(expected_decision_improvement: float, compute_cost: float, latency: float) -> float:
    """ROI of thinking harder; below ~1.0, stop spending cognition."""
    denominator = max(compute_cost + latency, 1e-6)
    return expected_decision_improvement / denominator
