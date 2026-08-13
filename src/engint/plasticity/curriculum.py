from __future__ import annotations

from dataclasses import dataclass

from .fitness import IntelligenceVector


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    index: int
    name: str
    description: str
    dimensions: tuple[str, ...]  # IntelligenceVector fields this stage exercises
    pass_threshold: float


# T0-T10: train != test at every stage; promotion is scored on unseen holdouts.
CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(0, "factual_extraction", "Find a fact; must cite the primary source.", ("research",), 0.8),
    CurriculumStage(1, "source_verification", "Assess provenance and evidence quality.", ("verification",), 0.7),
    CurriculumStage(2, "contradiction_resolution", "Find two contradictory claims and resolve them.", ("research", "verification"), 0.7),
    CurriculumStage(3, "multi_document_synthesis", "Analyze many documents without unsupported claims.", ("research", "verification"), 0.7),
    CurriculumStage(4, "financial_modeling", "Build models that reconcile with reported figures.", ("quantitative",), 0.6),
    CurriculumStage(5, "forecasts", "Make precommitted predictions and score them later.", ("calibration",), 0.6),
    CurriculumStage(6, "falsification", "Attack an investment/business hypothesis.", ("falsification",), 0.6),
    CurriculumStage(7, "business_analysis", "Determine whether a market supports a viable business.", ("strategy", "quantitative"), 0.6),
    CurriculumStage(8, "competitive_substitution", "Argue our product unnecessary; survive the argument.", ("falsification", "strategy"), 0.6),
    CurriculumStage(9, "strategy_comparison", "Choose between wealth strategies under constraints.", ("strategy",), 0.6),
    CurriculumStage(10, "capital_time_allocation", "Allocate scarce capital and founder time.", ("strategy", "cost_efficiency", "transfer"), 0.6),
)


class Curriculum:
    """Progressively harder environments; promotion requires held-out scores."""

    def __init__(self, stages: tuple[CurriculumStage, ...] = CURRICULUM) -> None:
        self.stages = stages
        self.stage_index = 0
        self._passed_last = False

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def complete(self) -> bool:
        return self.stage_index >= len(self.stages) - 1 and self._passed_last

    def evaluate_promotion(self, intelligence: IntelligenceVector) -> bool:
        """Promote iff every exercised dimension meets the stage threshold."""
        stage = self.current
        passed = all(
            getattr(intelligence, dim) >= stage.pass_threshold for dim in stage.dimensions
        )
        if passed:
            if self.stage_index < len(self.stages) - 1:
                self.stage_index += 1
            else:
                self._passed_last = True
        return passed
