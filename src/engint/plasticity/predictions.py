from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class PredictionFrozen(RuntimeError):
    """Raised on any attempt to alter a prediction after it was committed."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """Frozen before outcomes occur; the system cannot rewrite history."""

    prediction_id: str
    statement: str
    probability: float
    committed_at: datetime
    evidence: tuple[str, ...]
    resolution_condition: str
    seal: str


@dataclass(frozen=True, slots=True)
class Resolution:
    """Immutable: an outcome, once recorded, cannot be edited in place."""

    prediction_id: str
    outcome: bool
    resolved_at: datetime


@dataclass(slots=True)
class PredictionLedger:
    """Precommitted predictions plus Brier-scored calibration.

    BS = (1/N) * sum((p_i - o_i)^2); lower is better, 0.25 is the score of
    an uninformed coin-flipper on balanced outcomes.
    """

    predictions: dict[str, Prediction] = field(default_factory=dict)
    resolutions: dict[str, Resolution] = field(default_factory=dict)

    @staticmethod
    def _seal(statement: str, probability: float, resolution_condition: str, committed_at: datetime) -> str:
        payload = json.dumps(
            {
                "statement": statement,
                "probability": probability,
                "resolution_condition": resolution_condition,
                "committed_at": committed_at.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def commit(
        self,
        statement: str,
        probability: float,
        resolution_condition: str,
        evidence: tuple[str, ...] = (),
        now: datetime | None = None,
        prediction_id: str | None = None,
    ) -> Prediction:
        """`prediction_id` lets reconstructed corpus predictions keep their
        original identifiers (e.g. PRD-003) so experiment records that
        reference them stay resolvable."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if prediction_id is not None and prediction_id in self.predictions:
            raise PredictionFrozen(f"prediction id '{prediction_id}' already exists")
        committed_at = now or datetime.now(timezone.utc)
        prediction = Prediction(
            prediction_id=prediction_id or f"P-{uuid.uuid4().hex[:8]}",
            statement=statement,
            probability=probability,
            committed_at=committed_at,
            evidence=evidence,
            resolution_condition=resolution_condition,
            seal=self._seal(statement, probability, resolution_condition, committed_at),
        )
        self.predictions[prediction.prediction_id] = prediction
        return prediction

    def verify_seal(self, prediction_id: str) -> None:
        p = self.predictions[prediction_id]
        expected = self._seal(p.statement, p.probability, p.resolution_condition, p.committed_at)
        if expected != p.seal:
            raise PredictionFrozen(f"prediction {prediction_id} no longer matches its seal")

    def resolve(self, prediction_id: str, outcome: bool, now: datetime | None = None) -> Resolution:
        if prediction_id in self.resolutions:
            raise PredictionFrozen(f"prediction {prediction_id} is already resolved")
        self.verify_seal(prediction_id)
        resolution = Resolution(
            prediction_id=prediction_id,
            outcome=outcome,
            resolved_at=now or datetime.now(timezone.utc),
        )
        self.resolutions[prediction_id] = resolution
        return resolution

    def brier_score(self) -> float | None:
        if not self.resolutions:
            return None
        total = 0.0
        for pid, resolution in self.resolutions.items():
            probability = self.predictions[pid].probability
            total += (probability - (1.0 if resolution.outcome else 0.0)) ** 2
        return total / len(self.resolutions)

    def calibration_dimension(self) -> float:
        """Map Brier score onto a [0, 1] intelligence dimension (1 = perfect)."""
        score = self.brier_score()
        if score is None:
            return 0.0
        return max(0.0, 1.0 - 2.0 * score)
