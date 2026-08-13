from __future__ import annotations

from dataclasses import dataclass, field

from .fitness import FitnessReport, Phenotype, SealedBenchmark

# Claude is the first primary engine, not an irreversible dependency: the
# complete agent architecture is the evolving organism, and any engine that
# demonstrably performs better on the same sealed benchmarks can take a task.


@dataclass(slots=True)
class EngineRecord:
    name: str
    phenotype: Phenotype
    reports: dict[str, FitnessReport] = field(default_factory=dict)  # domain -> report


class EngineRegistry:
    """Model-agnostic task interface: engines compete on identical holdouts.

    Routing decisions come from recorded benchmark reports, never from an
    engine's opinion of itself or of a rival.
    """

    def __init__(self) -> None:
        self.engines: dict[str, EngineRecord] = {}

    def register(self, name: str, phenotype: Phenotype) -> EngineRecord:
        record = EngineRecord(name=name, phenotype=phenotype)
        self.engines[name] = record
        return record

    def compare(self, domain: str, benchmark: SealedBenchmark) -> list[FitnessReport]:
        """Run every engine against the same sealed benchmark for a domain."""
        reports: list[FitnessReport] = []
        for record in self.engines.values():
            report = benchmark.evaluate(f"engine:{record.name}", record.phenotype)
            record.reports[domain] = report
            reports.append(report)
        reports.sort(key=lambda r: r.score, reverse=True)
        return reports

    def route(self, domain: str) -> str | None:
        """Best engine for a domain by measured score; None if nothing measured."""
        best_name: str | None = None
        best_score = float("-inf")
        for record in self.engines.values():
            report = record.reports.get(domain)
            if report is not None and report.score > best_score:
                best_score = report.score
                best_name = record.name
        return best_name
