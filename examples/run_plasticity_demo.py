"""Artificial Darwinian Plasticity demonstrator.

1. Loads Generation Zero from the frozen corpus in corpus/g0/.
2. Prints the reconstructed cognitive state (the real 'brain', not decoration).
3. Runs a small deterministic evolution loop against a sealed benchmark and
   prints the lineage tree and intelligence bars.
"""
from __future__ import annotations

from pathlib import Path

from engint.plasticity import (
    BenchmarkCase,
    EvolutionEngine,
    Genome,
    IntelligenceVector,
    SealedBenchmark,
    load_generation_zero,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    g0 = load_generation_zero(ROOT / "corpus" / "g0")
    print(g0.brain_summary())
    print()
    print("Frozen predictions:")
    for prediction in g0.memory.predictive.predictions.values():
        print(f"  p={prediction.probability:.2f}  {prediction.statement[:90]}...")
    print()
    print("Scar tissue (failure mechanisms on record):")
    for failure in g0.memory.failure.failures.values():
        print(f"  [{failure.failure_id}] {failure.root_cause[:90]}")
    print()

    # Toy sealed benchmark: genomes whose blueprint encodes higher accuracy
    # answer more holdout cases correctly. The benchmark is hashed; descendants
    # cannot touch it.
    cases = tuple(BenchmarkCase(f"c{i}", f"q{i}", f"a{i}") for i in range(10))
    benchmark = SealedBenchmark(cases)
    engine = EvolutionEngine(benchmark)

    def phenotype_factory(genome: Genome):
        accuracy = float(genome.blueprint.get("accuracy", "0.4"))

        def phenotype(case: BenchmarkCase) -> str:
            return case.expected if int(case.case_id[1:]) < accuracy * 10 else "hallucinated"

        return phenotype

    seed = Genome.seed(blueprint={"accuracy": "0.4"}, species="wealth_research")
    engine.seed(seed, phenotype_factory)
    for accuracy in ("0.6", "0.3", "0.8"):
        child = engine.mutator.mutate_blueprint(engine.archive.records[engine.archive.champion_id].genome, "accuracy", accuracy)
        engine.trial(child, phenotype_factory)

    print("Evolutionary lineage:")
    print(engine.archive.render())
    print()
    champion = engine.archive.records[engine.archive.champion_id]
    print("Champion intelligence (held-out, toy benchmark):")
    print(IntelligenceVector(research=champion.report.inputs.decision_accuracy).render_bars())


if __name__ == "__main__":
    main()
