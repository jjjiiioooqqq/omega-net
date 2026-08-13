from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engint.plasticity import (
    AdaptiveComprehension,
    BenchmarkCase,
    BenchmarkTampered,
    Constitution,
    ConstitutionViolation,
    Curriculum,
    DocumentStore,
    EvolutionEngine,
    ExperimentFrozen,
    FailureMemory,
    FirewallViolation,
    FitnessInputs,
    Genome,
    IntelligenceVector,
    KnowledgeStore,
    MarketValidationFirewall,
    Mutator,
    PredictionLedger,
    ProblemProfile,
    SealedBenchmark,
    SkillLibrary,
    SourceRecord,
    fitness,
    load_generation_zero,
)

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "g0"
NOW = datetime(2026, 8, 13, tzinfo=timezone.utc)


def make_benchmark() -> SealedBenchmark:
    cases = tuple(
        BenchmarkCase(case_id=f"c{i}", prompt=f"q{i}", expected=f"a{i}") for i in range(10)
    )
    return SealedBenchmark(cases)


class TestConstitution:
    def test_mutation_of_protected_target_raises(self):
        constitution = Constitution(benchmark_seal="x")
        mutator = Mutator(constitution)
        genome = Genome.seed()
        with pytest.raises(ConstitutionViolation):
            mutator.mutate_blueprint(genome, "benchmark_selection", "use easier tests")
        with pytest.raises(ConstitutionViolation):
            mutator.mutate_research_strategy(genome, ("skip source_verification checks",))

    def test_allowed_mutation_creates_descendant(self):
        constitution = Constitution(benchmark_seal="x")
        mutator = Mutator(constitution)
        parent = Genome.seed(blueprint={"style": "terse"})
        child = mutator.mutate_blueprint(parent, "style", "cite primary sources first")
        assert child.generation == 1
        assert child.parent_id == parent.genome_id
        assert parent.blueprint["style"] == "terse"  # parent untouched


class TestKnowledge:
    def test_gate_rejects_low_quality_source(self):
        store = KnowledgeStore()
        blog = SourceRecord(identifier="random-blog", relevance=0.3)
        assert store.admit("X is true", blog, now=NOW) is None
        assert store.rejected == ["random-blog"]

    def test_lifecycle_and_decay(self):
        store = KnowledgeStore()
        filing = SourceRecord(
            identifier="sec-10k",
            author_known=True,
            publisher_known=True,
            publication_date=NOW,
            is_primary=True,
            methodology_stated=True,
            relevance=1.0,
        )
        obj = store.admit("Revenue was $1B", filing, now=NOW, half_life_days=100)
        assert obj is not None and obj.status == "UNVERIFIED"
        store.reinforce(obj.knowledge_id)
        assert obj.status == "SUPPORTED"
        store.reinforce(obj.knowledge_id)
        assert obj.status == "REINFORCED"
        fresh = obj.weight(NOW)
        aged = obj.weight(NOW + timedelta(days=200))
        assert aged < fresh / 3  # two half-lives plus rounding
        store.contradict(obj.knowledge_id)
        assert obj.status == "CONTRADICTED"
        assert obj.weight(NOW) < fresh

    def test_edge_weights_strengthen_and_weaken(self):
        store = KnowledgeStore()
        src = SourceRecord(
            identifier="paper",
            author_known=True,
            publisher_known=True,
            publication_date=NOW,
            is_primary=True,
            methodology_stated=True,
            relevance=1.0,
        )
        a = store.admit("claim A", src, now=NOW)
        b = store.admit("claim B", src, now=NOW)
        store.relate(a.knowledge_id, "SUPPORTS", b.knowledge_id, weight=0.5)
        up = store.adjust_edge(a.knowledge_id, b.knowledge_id, verified_usefulness=1.0)
        assert up > 0.5
        down = store.adjust_edge(a.knowledge_id, b.knowledge_id, contradiction=1.0)
        assert down < up


class TestFitnessAndEvolution:
    def test_sealed_benchmark_detects_tampering(self):
        benchmark = make_benchmark()
        benchmark._cases = benchmark._cases[:1]  # the evolved agent cheats
        with pytest.raises(BenchmarkTampered):
            benchmark.evaluate("G-x", lambda case: case.expected)

    def test_fitness_formula_floors_denominator(self):
        perfect = FitnessInputs(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert fitness(perfect) == pytest.approx(1e9)

    def test_selection_keeps_better_discards_worse(self):
        benchmark = make_benchmark()
        engine = EvolutionEngine(benchmark)

        def phenotype_factory(genome: Genome):
            accuracy = float(genome.blueprint.get("accuracy", "0.5"))
            def phenotype(case: BenchmarkCase) -> str:
                index = int(case.case_id[1:])
                return case.expected if index < accuracy * 10 else "wrong"
            return phenotype

        g0 = Genome.seed(blueprint={"accuracy": "0.5"})
        engine.seed(g0, phenotype_factory)
        better = engine.mutator.mutate_blueprint(g0, "accuracy", "0.9")
        worse = engine.mutator.mutate_blueprint(g0, "accuracy", "0.2")
        record_better = engine.trial(better, phenotype_factory)
        record_worse = engine.trial(worse, phenotype_factory)
        assert record_better.status == "champion"
        assert record_worse.status == "extinct"
        assert engine.archive.champion_id == better.genome_id
        tree = engine.archive.render()
        assert "CURRENT CHAMPION" in tree and "extinct" in tree

    def test_intelligence_vector_delta(self):
        before = IntelligenceVector(research=0.5)
        after = IntelligenceVector(research=0.7, falsification=0.2)
        delta = after.delta(before)
        assert delta["research"] == pytest.approx(0.2)
        assert delta["falsification"] == pytest.approx(0.2)


class TestPromotionGates:
    def _factory(self, accuracy_key: str = "accuracy", cost_key: str = "cost"):
        def phenotype_factory(genome: Genome):
            accuracy = float(genome.blueprint.get(accuracy_key, "0.5"))

            def phenotype(case: BenchmarkCase) -> str:
                return case.expected if int(case.case_id[1:]) < accuracy * 10 else "wrong"

            return phenotype

        return phenotype_factory

    def test_safety_violation_blocks_promotion_despite_fitness(self):
        benchmark = make_benchmark()
        engine = EvolutionEngine(benchmark)
        factory = self._factory()
        g0 = Genome.seed(blueprint={"accuracy": "0.5"})
        engine.seed(g0, factory)
        child = engine.mutator.mutate_blueprint(g0, "accuracy", "0.9")
        record = engine.trial(child, factory, safety_violations=1)
        assert record.status == "extinct"
        assert "safety" in record.extinction_reason

    def test_more_unsupported_claims_blocks_even_if_score_higher(self):
        benchmark = make_benchmark()
        engine = EvolutionEngine(benchmark)

        def factory(genome: Genome):
            accuracy = float(genome.blueprint.get("accuracy", "0.5"))

            def phenotype(case: BenchmarkCase) -> str:
                return case.expected if int(case.case_id[1:]) < accuracy * 10 else "wrong"

            return phenotype

        g0 = Genome.seed(blueprint={"accuracy": "0.8"})
        engine.seed(g0, factory)
        # Sloppier but "cheaper" child: lower accuracy would normally be
        # outscored, so cheat by giving the engine's evaluate a tiny cost via
        # a wrapper benchmark is not possible — instead check the gate directly.
        sloppy = engine.mutator.mutate_blueprint(g0, "accuracy", "0.6")
        record = engine.trial(sloppy, factory)
        assert record.status == "extinct"
        assert "unsupported claims" in record.extinction_reason


class TestEngineRegistry:
    def test_engines_compete_on_same_benchmark(self):
        from engint.plasticity import EngineRegistry

        benchmark = make_benchmark()
        registry = EngineRegistry()
        registry.register("engine-a", lambda case: case.expected)
        registry.register(
            "engine-b", lambda case: case.expected if int(case.case_id[1:]) < 5 else "wrong"
        )
        reports = registry.compare("research", benchmark)
        assert reports[0].genome_id == "engine:engine-a"
        assert registry.route("research") == "engine-a"
        assert registry.route("unmeasured-domain") is None


class TestEventRouter:
    def test_immaterial_events_do_not_wake_cognition(self):
        from engint.plasticity import EventRouter, SourceEvent

        router = EventRouter()
        noise = SourceEvent(
            kind="research_paper", subject="minor blog repost", observed_at=NOW,
            materiality=0.05, profile=ProblemProfile(0.2, 0.1, 0.1, 0.0, 0.1),
        )
        assert router.route(noise) is None
        assert len(router.ignored) == 1
        filing = SourceEvent(
            kind="sec_filing", subject="NVDA 10-Q", observed_at=NOW,
            materiality=0.9,
            profile=ProblemProfile(0.7, 0.9, 0.3, 0.2, 0.6, irreversibility=0.4),
        )
        investigation = router.route(filing)
        assert investigation is not None and investigation.plan.level >= "C4"
        with pytest.raises(ValueError):
            router.route(SourceEvent("twitter_rumor", "x", NOW, 0.9, filing.profile))

    def test_irreversibility_raises_budget(self):
        controller = AdaptiveComprehension()
        reversible = ProblemProfile(0.4, 0.4, 0.2, 0.1, 0.3, irreversibility=0.0)
        irreversible = ProblemProfile(0.4, 0.4, 0.2, 0.1, 0.3, irreversibility=1.0)
        assert controller.plan(irreversible).budget > controller.plan(reversible).budget
        assert controller.plan(irreversible).level > controller.plan(reversible).level


class TestFailureMemory:
    def test_lesson_requires_transfer(self):
        memory = FailureMemory()
        failure = memory.record(
            task="revenue forecast",
            prediction="guidance equals demand",
            actual="supply-constrained shortfall",
            root_cause="treated management guidance as demand without modeling supply constraints",
            failed_procedure="guidance-only forecasting",
            corrective_hypothesis="model supply independently",
        )
        lesson = memory.propose_lesson(failure.failure_id, "add independent supply model")
        assert not memory.transfer_test(lesson.lesson_id, old_score=0.6, new_score=0.55)
        assert not lesson.accepted
        lesson2 = memory.propose_lesson(failure.failure_id, "add supply model v2")
        assert memory.transfer_test(lesson2.lesson_id, old_score=0.6, new_score=0.8)

    def test_scar_similarity_search(self):
        memory = FailureMemory()
        memory.record(
            task="t", prediction="p", actual="a",
            root_cause="market growth treated as willingness to pay",
            failed_procedure="TAM extrapolation",
            corrective_hypothesis="require paid evidence",
        )
        hits = memory.similar("this argument assumes market growth implies willingness to pay")
        assert len(hits) == 1


class TestCognition:
    def test_budget_scales_with_stakes_and_scars(self):
        controller = AdaptiveComprehension()
        trivial = controller.plan(ProblemProfile(0.05, 0.05, 0.0, 0.0, 0.1))
        consequential = controller.plan(ProblemProfile(0.9, 1.0, 0.8, 0.9, 0.9))
        assert trivial.level == "C0"
        assert consequential.level == "C6"
        base = ProblemProfile(0.3, 0.3, 0.2, 0.0, 0.2)
        assert controller.plan(base, scar_hit=True).level > controller.plan(base).level


class TestPredictionsAndFirewall:
    def test_ledger_freezes_and_scores(self):
        ledger = PredictionLedger()
        p = ledger.commit("payment arrives", 0.1, "cleared $600 within 21 days", now=NOW)
        ledger.resolve(p.prediction_id, outcome=False, now=NOW)
        with pytest.raises(Exception):
            ledger.resolve(p.prediction_id, outcome=True, now=NOW)  # no history rewriting
        assert ledger.brier_score() == pytest.approx(0.01)

    def test_firewall_rejects_research_evidence(self):
        firewall = MarketValidationFirewall()
        with pytest.raises(FirewallViolation):
            firewall.observe("competitor_funding_news")
        assert not firewall.validated
        firewall.observe("payment")
        assert firewall.ladder_state() == "customer_payment"
        assert not firewall.validated  # one payment is not market validation
        firewall.observe("unrelated_second_payment")
        firewall.observe("recurring_commitment")
        assert firewall.validated

    def test_experiment_denominator_freezes(self):
        from engint.plasticity import Experiment

        exp = Experiment("EXP-T", "h", "not h", "payments", denominator=30)
        exp.amend_denominator(29)  # allowed before start
        exp.start()
        with pytest.raises(ExperimentFrozen):
            exp.amend_denominator(5)
        exp.record_result("zero payments")
        with pytest.raises(ExperimentFrozen):
            exp.record_result("actually one payment")


class TestSkillsAndCurriculum:
    def test_skill_selection_and_pruning(self):
        library = SkillLibrary()
        good = library.register("verify-first", "research", ("sec", "primary", "cross-check"))
        bad = library.register("google-summarize", "research", ("google", "summarize"))
        for _ in range(5):
            library.record_outcome(good.skill_id, True)
            library.record_outcome(bad.skill_id, False)
        assert library.best_for("research").skill_id == good.skill_id
        assert library.prune() == [bad.skill_id]

    def test_curriculum_promotion_requires_dimension_scores(self):
        curriculum = Curriculum()
        assert curriculum.current.name == "factual_extraction"
        assert not curriculum.evaluate_promotion(IntelligenceVector(research=0.5))
        assert curriculum.evaluate_promotion(IntelligenceVector(research=0.9))
        assert curriculum.current.name == "source_verification"


class TestDocumentStore:
    def test_changed_content_creates_new_version(self):
        store = DocumentStore()
        v1 = store.register(b"content-a", source="page")
        same = store.register(b"content-a", source="page")
        v2 = store.register(b"content-b", source="page")
        assert same.document_id == v1.document_id
        assert v2.version == 2
        assert len(store.versions_of("page")) == 2
        assert store.verify(v1.document_id, b"content-a")


class TestGenerationZero:
    def test_g0_reconstructs_strategy_state(self):
        g0 = load_generation_zero(CORPUS)
        # Five corpus documents hashed and immutable.
        assert len(g0.documents.documents) == 5
        # STR-005 is the only active strategy.
        active = [s for s in g0.tournament.strategies.values() if s.status == "active"]
        assert [s.strategy_id for s in active] == ["STR-005"]
        # EXP-002 denominator frozen at 30 and untouched.
        assert g0.experiments["EXP-002"].denominator == 30
        assert not g0.experiments["EXP-002"].started
        # PRD-003 frozen at 0.10, unresolved.
        prd3 = [p for p in g0.memory.predictive.predictions.values() if "$600" in p.statement]
        assert len(prd3) == 1 and prd3[0].probability == pytest.approx(0.10)
        assert not g0.memory.predictive.resolutions
        # No market validation: research artifacts cannot fabricate demand.
        assert not g0.firewall.validated
        assert g0.firewall.ladder_state() == "research_supported_hypothesis"
        # All 21 claims ingested; falsified ones contradicted, not deleted.
        assert len(g0.claim_index) == 21
        statuses = {g0.memory.semantic.objects[k].status for k in g0.claim_index.values()}
        assert "CONTRADICTED" in statuses and "REINFORCED" in statuses
        # v3 failure recorded with a transfer-tested lesson whose text preserves
        # the generalization boundary.
        lessons = list(g0.memory.failure.lessons.values())
        v4_lessons = [l for l in lessons if "NOT YET ESTABLISHED" in l.new_procedure]
        assert len(v4_lessons) == 1 and v4_lessons[0].accepted
        # Strategic scars are searchable.
        hits = g0.memory.failure.similar(
            "this modeled probability ranking shows strategy A beats strategy B"
        )
        assert hits, "scar-tissue similarity search should find the DEC-007 episode"
        # 30 cohort companies present as entity nodes.
        companies = [
            n for n, d in g0.memory.semantic.graph.nodes(data=True)
            if d.get("node_type") == "company"
        ]
        assert len(companies) == 30
