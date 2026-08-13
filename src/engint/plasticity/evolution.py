from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .fitness import FitnessReport, Phenotype, SealedBenchmark
from .genome import Constitution, Genome, Mutator

# Archive statuses
CHAMPION = "champion"
ALIVE = "alive"
EXTINCT = "extinct"
UNTESTED = "untested"


@dataclass(slots=True)
class LineageRecord:
    genome: Genome
    status: str = UNTESTED
    report: FitnessReport | None = None
    extinction_reason: str | None = None

    @property
    def score(self) -> float | None:
        return self.report.score if self.report else None


class Archive:
    """Darwin-Gödel-style archive: variants compete, branches persist.

    Descendants are kept rather than overwriting a single agent, so different
    improvement branches can compete and cross over later.
    """

    def __init__(self) -> None:
        self.records: dict[str, LineageRecord] = {}
        self.champion_id: str | None = None

    def add(self, genome: Genome) -> LineageRecord:
        record = LineageRecord(genome=genome)
        self.records[genome.genome_id] = record
        return record

    def children_of(self, genome_id: str) -> list[LineageRecord]:
        return [r for r in self.records.values() if r.genome.parent_id == genome_id]

    def roots(self) -> list[LineageRecord]:
        return [r for r in self.records.values() if r.genome.parent_id is None]

    def render(self) -> str:
        lines: list[str] = []
        for root in self.roots():
            self._render_node(root, prefix="", lines=lines)
        return "\n".join(lines)

    def _render_node(self, record: LineageRecord, prefix: str, lines: list[str]) -> None:
        genome = record.genome
        marker = ""
        if record.status == EXTINCT:
            marker = f"   × extinct ({record.extinction_reason})"
        elif genome.genome_id == self.champion_id:
            marker = "   <- CURRENT CHAMPION"
        score = f" [F={record.score:.3f}]" if record.score is not None else ""
        lines.append(f"{prefix}{genome.genome_id} (gen {genome.generation}){score}{marker}")
        children = self.children_of(genome.genome_id)
        for index, child in enumerate(children):
            last = index == len(children) - 1
            branch = "└── " if last else "├── "
            extension = "    " if last else "│   "
            child_lines: list[str] = []
            self._render_node(child, prefix="", lines=child_lines)
            lines.append(f"{prefix}{branch}{child_lines[0]}")
            lines.extend(f"{prefix}{extension}{line}" for line in child_lines[1:])


class EvolutionEngine:
    """Artificial selection loop: mutate, express, test on holdout, select.

    The engine owns the sealed benchmark and the constitution; genomes never
    do. A descendant survives only if F(child) > F(parent) on unseen tests.
    """

    def __init__(
        self,
        benchmark: SealedBenchmark,
        constitution: Constitution | None = None,
        mutation_seed: int | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.constitution = constitution or Constitution(benchmark_seal=benchmark.seal)
        if self.constitution.benchmark_seal != benchmark.seal:
            raise ValueError("constitution seal does not match benchmark seal")
        self.mutator = Mutator(self.constitution, seed=mutation_seed)
        self.archive = Archive()

    def seed(self, genome: Genome, phenotype_factory: Callable[[Genome], Phenotype]) -> LineageRecord:
        record = self.archive.add(genome)
        record.report = self.benchmark.evaluate(genome.genome_id, phenotype_factory(genome))
        record.status = CHAMPION
        self.archive.champion_id = genome.genome_id
        return record

    def trial(
        self,
        child: Genome,
        phenotype_factory: Callable[[Genome], Phenotype],
    ) -> LineageRecord:
        """Evaluate a descendant against its parent; keep it only if fitter."""
        parent = self.archive.records.get(child.parent_id or "")
        if parent is None or parent.report is None:
            raise ValueError("descendant has no evaluated parent in the archive")
        record = self.archive.add(child)
        record.report = self.benchmark.evaluate(child.genome_id, phenotype_factory(child))
        if record.report.score > parent.report.score:
            record.status = ALIVE
            champion = self.archive.records[self.archive.champion_id]
            if champion.report is None or record.report.score > champion.report.score:
                champion.status = ALIVE
                record.status = CHAMPION
                self.archive.champion_id = child.genome_id
        else:
            record.status = EXTINCT
            record.extinction_reason = (
                f"fitness {record.report.score:.3f} <= parent {parent.report.score:.3f}"
            )
        return record
