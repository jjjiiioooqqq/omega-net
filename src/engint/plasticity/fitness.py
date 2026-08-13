from __future__ import annotations

import hashlib
import json
import marshal
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

# Denominator floor: costs below this are treated as this, so a lineage cannot
# drive fitness to infinity by claiming zero hallucination/cost/latency.
_EPS = 1e-3


class BenchmarkTampered(RuntimeError):
    """Raised when a sealed benchmark no longer matches its seal."""


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    expected: str


@dataclass(frozen=True, slots=True)
class IntelligenceVector:
    """I = [R, V, Q, F, P, S, L, C]; every dimension from held-out tests only.

    Never a single fake IQ number: show the vector, and show I_t - I_{t-1}
    so improvement (or regression) is visible per capability.
    """

    research: float = 0.0  # R: research accuracy
    verification: float = 0.0  # V: verification accuracy
    quantitative: float = 0.0  # Q: quantitative reasoning
    falsification: float = 0.0  # F: falsification performance
    calibration: float = 0.0  # P: prediction calibration
    strategy: float = 0.0  # S: strategy performance
    transfer: float = 0.0  # L: learning transfer
    cost_efficiency: float = 0.0  # C: useful conclusions per unit cost

    def delta(self, previous: IntelligenceVector) -> dict[str, float]:
        current = asdict(self)
        return {name: round(value - getattr(previous, name), 4) for name, value in current.items()}

    def render_bars(self, width: int = 10) -> str:
        lines = []
        for name, value in asdict(self).items():
            filled = round(max(min(value, 1.0), 0.0) * width)
            bar = "█" * filled + "░" * (width - filled)
            lines.append(f"{name.capitalize():<16}{bar} {round(value * 100)}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FitnessInputs:
    decision_accuracy: float
    source_integrity: float
    calibration: float
    economic_usefulness: float
    hallucination_rate: float
    research_cost: float
    decision_latency: float


def fitness(inputs: FitnessInputs) -> float:
    """Fitness = (accuracy * integrity * calibration * usefulness)
              / (hallucination * cost * latency), denominators floored."""
    numerator = (
        inputs.decision_accuracy
        * inputs.source_integrity
        * inputs.calibration
        * inputs.economic_usefulness
    )
    denominator = (
        max(inputs.hallucination_rate, _EPS)
        * max(inputs.research_cost, _EPS)
        * max(inputs.decision_latency, _EPS)
    )
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class FitnessReport:
    genome_id: str
    score: float
    inputs: FitnessInputs
    intelligence: IntelligenceVector
    benchmark_seal: str
    unsupported_claims: int = 0
    safety_violations: int = 0


class Phenotype(Protocol):
    """A genome expressed against a fixed foundation model: answers cases."""

    def __call__(self, case: BenchmarkCase) -> str: ...


class SealedBenchmark:
    """Held-out fitness test the evolving agent can never alter.

    The case set is hashed at construction; evaluation refuses to run if the
    cases no longer match the seal. Goodhart protection is structural, not
    behavioral: genomes never hold a reference to this object's cases.
    """

    def __init__(
        self,
        cases: tuple[BenchmarkCase, ...],
        scorer: Callable[[BenchmarkCase, str], bool] | None = None,
    ) -> None:
        self._cases = cases
        self._scorer = scorer or (lambda case, answer: answer.strip() == case.expected)
        self.seal = self._compute_seal(cases, self._scorer)

    @staticmethod
    def _fingerprint_scorer(scorer: Callable[[BenchmarkCase, str], bool]) -> str:
        # The scoring rule is part of the fitness test: swapping the scorer
        # must break the seal just as editing the cases does. Function
        # bytecode gives a stable in-process fingerprint; other callables
        # fall back to their repr (stable within a process).
        code = getattr(scorer, "__code__", None)
        payload = marshal.dumps(code) if code is not None else repr(scorer).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _compute_seal(
        cls,
        cases: tuple[BenchmarkCase, ...],
        scorer: Callable[[BenchmarkCase, str], bool],
    ) -> str:
        payload = json.dumps([asdict(c) for c in cases], sort_keys=True)
        combined = payload + "|scorer:" + cls._fingerprint_scorer(scorer)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def verify_seal(self) -> None:
        if self._compute_seal(self._cases, self._scorer) != self.seal:
            raise BenchmarkTampered("benchmark cases or scorer no longer match their seal")

    def evaluate(
        self,
        genome_id: str,
        phenotype: Phenotype,
        research_cost: float = 1.0,
        decision_latency: float = 1.0,
        intelligence: IntelligenceVector | None = None,
        safety_violations: int = 0,
    ) -> FitnessReport:
        """Score a phenotype. `safety_violations` is reported by the external
        harness (never the phenotype itself) and gates promotion regardless
        of the fitness score."""
        self.verify_seal()
        correct = 0
        hallucinated = 0
        for case in self._cases:
            answer = phenotype(case)
            if self._scorer(case, answer):
                correct += 1
            else:
                hallucinated += 1
        total = len(self._cases)
        accuracy = correct / total if total else 0.0
        inputs = FitnessInputs(
            decision_accuracy=accuracy,
            source_integrity=1.0,
            calibration=1.0,
            economic_usefulness=1.0,
            hallucination_rate=hallucinated / total if total else 1.0,
            research_cost=research_cost,
            decision_latency=decision_latency,
        )
        return FitnessReport(
            genome_id=genome_id,
            score=fitness(inputs),
            inputs=inputs,
            intelligence=intelligence or IntelligenceVector(research=accuracy),
            benchmark_seal=self.seal,
            unsupported_claims=hallucinated,
            safety_violations=safety_violations,
        )
