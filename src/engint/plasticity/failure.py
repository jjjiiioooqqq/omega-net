from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


@dataclass(slots=True)
class FailureRecord:
    """A diagnosed failure: the mechanism, not just the outcome.

    'I got NVIDIA wrong' is useless; 'the process treated management
    guidance as expected demand without modeling supply constraints'
    is a reusable pattern.
    """

    failure_id: str
    task: str
    prediction: str
    actual: str
    root_cause: str
    failed_procedure: str
    corrective_hypothesis: str


@dataclass(slots=True)
class Lesson:
    """A candidate lesson is not trusted until it survives transfer testing."""

    lesson_id: str
    failure_id: str
    new_procedure: str
    accepted: bool = False
    old_score: float | None = None
    new_score: float | None = None


@dataclass(slots=True)
class FailureMemory:
    """Scar tissue: indexed failure mechanisms plus transfer-gated lessons.

    Learning is demonstrated only when experience with X improves unseen X'.
    The defining metric: P(repeat failure | prior exposure) must fall below
    P(repeat failure | no prior exposure).
    """

    failures: dict[str, FailureRecord] = field(default_factory=dict)
    lessons: dict[str, Lesson] = field(default_factory=dict)
    repeat_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        task: str,
        prediction: str,
        actual: str,
        root_cause: str,
        failed_procedure: str,
        corrective_hypothesis: str,
    ) -> FailureRecord:
        record = FailureRecord(
            failure_id=f"F-{uuid.uuid4().hex[:8]}",
            task=task,
            prediction=prediction,
            actual=actual,
            root_cause=root_cause,
            failed_procedure=failed_procedure,
            corrective_hypothesis=corrective_hypothesis,
        )
        self.failures[record.failure_id] = record
        return record

    def similar(self, argument: str, threshold: float = 0.2) -> list[FailureRecord]:
        """Similarity search over failure mechanisms (scar-tissue lookup).

        current argument -> similarity search -> known failure pattern
        -> escalate cognition.
        """
        query = _tokens(argument)
        if not query:
            return []
        hits: list[tuple[float, FailureRecord]] = []
        for record in self.failures.values():
            pattern = _tokens(record.root_cause) | _tokens(record.failed_procedure)
            if not pattern:
                continue
            overlap = len(query & pattern) / len(query | pattern)
            if overlap >= threshold:
                hits.append((overlap, record))
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in hits]

    def propose_lesson(self, failure_id: str, new_procedure: str) -> Lesson:
        lesson = Lesson(
            lesson_id=f"L-{uuid.uuid4().hex[:8]}",
            failure_id=failure_id,
            new_procedure=new_procedure,
        )
        self.lessons[lesson.lesson_id] = lesson
        return lesson

    def transfer_test(self, lesson_id: str, old_score: float, new_score: float) -> bool:
        """Failure does not equal learning: accept only if the new procedure
        beats the old one on unseen tests."""
        lesson = self.lessons[lesson_id]
        lesson.old_score = old_score
        lesson.new_score = new_score
        lesson.accepted = new_score > old_score
        return lesson.accepted

    def note_repeat(self, failure_id: str) -> None:
        self.repeat_counts[failure_id] = self.repeat_counts.get(failure_id, 0) + 1

    def repeat_rate(self) -> float:
        """Fraction of recorded failures that recurred after being diagnosed."""
        if not self.failures:
            return 0.0
        repeated = sum(1 for fid in self.failures if self.repeat_counts.get(fid, 0) > 0)
        return repeated / len(self.failures)
