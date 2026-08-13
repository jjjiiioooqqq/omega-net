from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .evidence import DocumentStore
from .genome import Genome
from .knowledge import SourceRecord
from .memory import MemorySystem
from .validation import Experiment, MarketValidationFirewall, Strategy, StrategyTournament

# Claim-status mapping from the strategy-state schema onto the knowledge
# lifecycle. Falsified claims are kept, not deleted: a contradicted claim
# with its audit trail is more informative than an absence.
_STATUS_REINFORCEMENTS = {
    "unverified": 0,
    "source-supported": 1,
    "independently-reproduced": 2,
    "externally-observed": 2,
}


@dataclass(slots=True)
class GenerationZero:
    """ADP-G0: the system's first cognitive environment, built from the real
    wealth-project records rather than arbitrary benchmark questions."""

    memory: MemorySystem
    firewall: MarketValidationFirewall
    tournament: StrategyTournament
    experiments: dict[str, Experiment]
    genome: Genome
    documents: DocumentStore = field(default_factory=DocumentStore)
    claim_index: dict[str, str] = field(default_factory=dict)  # CLM-id -> knowledge id

    def brain_summary(self) -> str:
        store = self.memory.semantic
        statuses: dict[str, int] = {}
        for obj in store.objects.values():
            statuses[obj.status] = statuses.get(obj.status, 0) + 1
        lines = [
            "GENERATION ZERO — external cognition summary",
            f"  knowledge objects: {len(store.objects)} "
            f"({', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))})",
            f"  graph nodes/edges: {store.graph.number_of_nodes()}/{store.graph.number_of_edges()}",
            f"  episodes: {len(self.memory.episodic.episodes)}",
            f"  skills: {len(self.memory.procedural.skills)}",
            f"  frozen predictions: {len(self.memory.predictive.predictions)} "
            f"(resolved: {len(self.memory.predictive.resolutions)})",
            f"  failures on record: {len(self.memory.failure.failures)} "
            f"(lessons: {len(self.memory.failure.lessons)})",
            f"  strategies: {len(self.tournament.strategies)} "
            f"(active: {sum(1 for s in self.tournament.strategies.values() if s.status == 'active')})",
            f"  market validated: {self.firewall.validated}",
            f"  wealth ladder state: {self.firewall.ladder_state()}",
        ]
        return "\n".join(lines)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_generation_zero(corpus_dir: Path) -> GenerationZero:
    """Decompose the G0 corpus into the five memory types.

    Requirement: never know less tomorrow than can be defensibly
    reconstructed from these records today, while remaining capable of
    deleting or downgrading beliefs that future evidence falsifies.
    """
    state = json.loads((corpus_dir / "strategy-state.json").read_text(encoding="utf-8"))
    oracle = json.loads((corpus_dir / "ORACLE_COMPARISON.json").read_text(encoding="utf-8"))
    cohort = json.loads(
        (corpus_dir / "final_30_prospect_cohort.redacted.json").read_text(encoding="utf-8")
    )

    # --- immutable evidence store: hash the corpus files unchanged ---
    documents = DocumentStore()
    for name, doc_type in (
        ("strategy-state.json", "strategy_state"),
        ("ORACLE_COMPARISON.json", "technical_evidence"),
        ("PILOT_OFFER.md", "commercial_procedure"),
        ("SCOPE_AND_DATA_TERMS_DRAFT.md", "commercial_terms"),
        ("final_30_prospect_cohort.redacted.json", "entity_cohort"),
    ):
        documents.register_file(corpus_dir / name, doc_type=doc_type)

    memory = MemorySystem()
    updated_at = _parse_ts(state["updated_at"])
    coverage = {src["id"]: src for src in state["coverage"]}

    # --- semantic memory: claims with provenance and lifecycle status ---
    claim_index: dict[str, str] = {}
    for claim in state["claims"]:
        src_id = claim["source_ids"][0] if claim["source_ids"] else "unknown"
        src = coverage.get(src_id, {})
        source = SourceRecord(
            identifier=f"{src_id}: {src.get('title', 'unknown source')}",
            author_known=True,
            publisher_known=src.get("source_type") in ("primary", "secondary"),
            publication_date=updated_at,
            is_primary=src.get("source_type") == "primary",
            methodology_stated=bool(claim.get("independent_check")),
            # Corpus claims are all central to the domain; the claim's own
            # confidence is uncertainty about the claim, not source relevance,
            # and is already expressed through the UNVERIFIED/... status.
            relevance=1.0,
        )
        half_life = 180.0 if claim["claim_type"] in ("market", "legal", "forecast") else 365.0
        obj = memory.semantic.admit(
            claim=claim["text"],
            source=source,
            now=updated_at,
            applicability=(claim["claim_type"],),
            half_life_days=half_life,
        )
        if obj is None:
            continue
        claim_index[claim["id"]] = obj.knowledge_id
        status = claim["status"]
        if status == "falsified":
            memory.semantic.contradict(obj.knowledge_id)
        else:
            for _ in range(_STATUS_REINFORCEMENTS.get(status, 0)):
                memory.semantic.reinforce(obj.knowledge_id)

    # --- entity graph: frozen cohort companies (redacted to company level) ---
    anchor = claim_index.get("CLM-021")
    for row in cohort.get("recipients", []):
        node_id = f"company:{row['company']}"
        memory.semantic.graph.add_node(
            node_id, node_type="company", source=row["source_id"], sequence=row["sequence"]
        )
        if anchor:
            memory.semantic.graph.add_edge(anchor, node_id, relation="DEPENDS_ON", weight=0.5)

    # --- episodic memory: every decision and run, in order ---
    for decision in state["decisions"]:
        memory.episodic.record(
            timestamp=_parse_ts(decision["timestamp"]),
            kind="decision",
            description=f"{decision['decision']} — {decision['reason']}",
            evidence=tuple(decision.get("evidence_ids", ())),
            supersedes=decision.get("supersedes"),
            episode_id=decision["id"],
        )
    for run in state.get("run_log", []):
        memory.episodic.record(
            timestamp=_parse_ts(run["timestamp"]),
            kind="run",
            description=run["action"],
            evidence=tuple(run.get("validators", ())),
            episode_id=run["id"],
        )

    # --- predictive memory: frozen, uncalibrated, unresolved ---
    for prd in state["predictions"]:
        memory.predictive.commit(
            statement=prd["text"],
            probability=float(prd["probability"]),
            resolution_condition="per pre-registered interpretation rules",
            evidence=tuple(prd.get("evidence_ids", ())),
            now=_parse_ts(prd["recorded_at"]),
        )

    # --- failure memory: diagnosed mechanisms, not outcomes ---
    fail = memory.failure
    v3 = fail.record(
        task="foreign-schema blind rehearsal v3 (trace evaluation)",
        prediction="pre-oracle diagnostic finds all seeded contract defects",
        actual="2 TP, 0 FP, 2 FN; precision 1.0, recall 0.5",
        root_cause="diagnostic process missed two seeded defect classes; completeness "
        "of detection was assumed rather than established before the oracle opened",
        failed_procedure="v3 rehearsal fulfillment process",
        corrective_hypothesis="a clean-sheet process with contract and baseline frozen "
        "before candidate access will detect all seeded classes",
    )
    lesson = fail.propose_lesson(
        v3.failure_id,
        "v4 clean-sheet blind rehearsal: freeze contract and baseline before candidate "
        "access; issue and reproduce pre-oracle verdict; open sealed oracle last. "
        "Generalization to unseeded/production defect classes NOT YET ESTABLISHED "
        f"(oracle comparison: {oracle['unique_defect_confusion_matrix']['true_positive_count']} TP, "
        f"{oracle['unique_defect_confusion_matrix']['false_positive_count']} FP, "
        f"{oracle['unique_defect_confusion_matrix']['false_negative_count']} FN, synthetic only)",
    )
    fail.transfer_test(
        lesson.lesson_id,
        old_score=0.5,  # v3 recall on seeded defects
        new_score=float(oracle["unique_defect_confusion_matrix"]["recall"]),  # v4, same class of test
    )
    # Strategic scar tissue: each rule stays linked to the episode that produced it.
    fail.record(
        task="Intuit exploratory outreach gate (DEC-003)",
        prediction="frozen 50-account universe holds >= 50 compliant research channels",
        actual="row-level audit cleared 9 of 50",
        root_cause="a public route was treated as evidence of reachable demand; "
        "public route does not imply demand or permission",
        failed_procedure="structural role matching without row-level channel audit",
        corrective_hypothesis="audit contact channels row-by-row before any denominator freeze",
    )
    fail.record(
        task="$149 target-scale Intuit wealth case (DEC-002)",
        prediction="target-scale case has an owner-independent economic path",
        actual="support alone requires 67.33 founder hours/month vs an 8-hour cap at 404 accounts",
        root_cause="market-size arithmetic ignored per-account support physics; "
        "market growth does not imply serviceable willingness to pay",
        failed_procedure="wealth-scale design without a costed owner-replacement model",
        corrective_hypothesis="reject candidates whose target-scale support exceeds founder caps "
        "before build",
    )
    fail.record(
        task="Atlassian existing-app earn-in screen (DEC-014)",
        prediction="public listings identify at least one qualified earn-in target",
        actual="zero strictly qualified targets in 20 public listings",
        root_cause="market activity was treated as accessible deal supply; visible market "
        "activity does not imply our product or deal has a moat or a counterparty",
        failed_procedure="public-listing screen as a proxy for seller intent and authority",
        corrective_hypothesis="require verified seller consent, receipts, and code rights before "
        "treating a route as a strategy",
    )
    fail.record(
        task="distribution-first numerical winner claim (DEC-007)",
        prediction="modeled priors establish distribution-first as more likely to reach the target",
        actual="midpoint gap 0.0656pp; rank reverses under a sub-one-point assumption change",
        root_cause="a modeled probability was treated as an empirical probability; "
        "the ranking had no matched cohort behind it",
        failed_procedure="probability-ranked strategy selection from judgment priors",
        corrective_hypothesis="use constraint dominance and falsifiable paid gates, never "
        "fabricated numerical rankings, for strategy selection",
    )

    # --- procedural memory: workflows that have already run, with real tallies ---
    skills = memory.procedural
    rehearsal = skills.register(
        name="blind frozen-oracle rehearsal",
        domain="trace_evaluation",
        workflow=(
            "freeze contract and baseline before candidate access",
            "run pre-oracle structural diagnostic",
            "freeze verdict artifacts and hashes",
            "independently reproduce the verdict",
            "open sealed oracle only after freeze",
            "score confusion matrix against seeded causes",
        ),
    )
    skills.record_outcome(rehearsal.skill_id, success=False)  # v3: recall 0.5
    skills.record_outcome(rehearsal.skill_id, success=True)  # v4: 7 TP, 0 FP, 0 FN
    cohort_screen = skills.register(
        name="strict frozen-cohort screen",
        domain="prospect_qualification",
        workflow=tuple(cohort["cohort_rules"]["required_gates"]),
    )
    skills.record_outcome(cohort_screen.skill_id, success=True)  # 30/30 frozen
    hardening = skills.register(
        name="hostile-review evaluator hardening",
        domain="software_verification",
        workflow=(
            "independent false-PASS probes",
            "malformed-shape fuzzing",
            "fix and regression-test each found bypass",
            "freeze suite and hashes",
        ),
    )
    skills.record_outcome(hardening.skill_id, success=True)  # 48/48 after hostile audit

    # --- strategies and experiments ---
    tournament = StrategyTournament()
    for strat in state["strategies"]:
        tournament.register(
            Strategy(
                strategy_id=strat["id"],
                name=strat["name"],
                mechanism=strat["mechanism"],
                # DEC-007 removed numerical strategy ranking as a selection
                # basis; no asymmetry numbers are fabricated here.
                ownership_adjusted_potential=0.0,
                time_to_decisive_evidence=1.0,
                capital_to_decisive_evidence=0.0,
                kill_conditions=(strat["rejection_reason"],) if strat.get("rejection_reason") else (),
                status=strat["status"] if strat["status"] in ("active", "rejected", "paused") else "candidate",
            )
        )

    experiments: dict[str, Experiment] = {}
    for exp in state["experiments"]:
        experiment = Experiment(
            experiment_id=exp["id"],
            hypothesis=exp["hypothesis"],
            alternative=exp["alternative"],
            metric="cleared prepayment or funded escrow within pre-registered ceilings",
            denominator=30 if exp["id"] == "EXP-002" else 12,
            interpretation_rules=tuple(exp.get("interpretation_rules", ())),
        )
        experiments[experiment.experiment_id] = experiment

    # --- market validation: firewalled; research cannot move it ---
    firewall = MarketValidationFirewall()  # zero market events have occurred

    genome = Genome.seed(
        blueprint={
            "objective": state["objective"]["desired_outcome"],
            "epistemics": "separate FACT / INFERENCE / FORECAST; cite primary sources; "
            "never treat research proxies as market validation",
        },
        skills=(rehearsal.skill_id, cohort_screen.skill_id, hardening.skill_id),
        research_strategy=(
            "official primary sources first",
            "row-level audit before any denominator freeze",
            "independent hostile review of load-bearing claims",
            "precommit predictions before outcomes",
        ),
        agent_organization=("researcher", "falsifier", "evidence_auditor"),
        species="wealth_research",
    )

    return GenerationZero(
        memory=memory,
        firewall=firewall,
        tournament=tournament,
        experiments=experiments,
        genome=genome,
        documents=documents,
        claim_index=claim_index,
    )
