from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx

RELATIONS = (
    "SUPPORTS",
    "CONTRADICTS",
    "DEPENDS_ON",
    "CITES",
    "CAUSES",
    "COMPETES_WITH",
    "INVALIDATES",
    "PREDICTS",
    "LEARNED_FROM",
    "FAILED_BECAUSE",
)

# Claim lifecycle: UNVERIFIED -> SUPPORTED -> REINFORCED, or any state
# -> CONTRADICTED -> SUPERSEDED once a replacement claim invalidates it.
CLAIM_STATES = ("UNVERIFIED", "SUPPORTED", "REINFORCED", "CONTRADICTED", "SUPERSEDED")


@dataclass(slots=True)
class SourceRecord:
    """Verification-gate inputs extracted from a document before admission."""

    identifier: str
    author_known: bool = False
    publisher_known: bool = False
    publication_date: datetime | None = None
    is_primary: bool = False
    methodology_stated: bool = False
    citation_count: int = 0
    relevance: float = 0.0  # [0, 1]


@dataclass(slots=True)
class KnowledgeObject:
    """K = (C, S, Q, T, A): a claim under selection pressure."""

    knowledge_id: str
    claim: str  # C
    source: SourceRecord  # S
    quality: float  # Q: source/evidence quality in [0, 1]
    acquired_at: datetime  # T: temporal validity anchor
    half_life_days: float  # T: decay rate
    applicability: tuple[str, ...] = ()  # A
    status: str = "UNVERIFIED"
    reinforcements: int = 0
    contradictions: int = 0
    superseded_by: str | None = None

    def weight(self, now: datetime) -> float:
        """Current evidential weight: quality decayed by age, shifted by use.

        Old information weakens; reinforced information persists; contradicted
        information collapses. This is the knowledge metabolism:
        acquire -> verify -> use -> reinforce/contradict -> decay/archive.
        """
        age_days = max((now - self.acquired_at).total_seconds() / 86400.0, 0.0)
        decay = math.pow(0.5, age_days / self.half_life_days)
        support = 1.0 + 0.1 * self.reinforcements
        doubt = 1.0 + 0.5 * self.contradictions
        return self.quality * decay * support / doubt


class VerificationGate:
    """More documents != more intelligence; only verified material enters."""

    def __init__(self, admission_threshold: float = 0.5) -> None:
        self.admission_threshold = admission_threshold

    def assess(self, source: SourceRecord) -> float:
        score = 0.0
        score += 0.15 if source.author_known else 0.0
        score += 0.15 if source.publisher_known else 0.0
        score += 0.10 if source.publication_date is not None else 0.0
        score += 0.20 if source.is_primary else 0.0
        score += 0.15 if source.methodology_stated else 0.0
        score += 0.10 * min(source.citation_count / 10.0, 1.0)
        score += 0.15 * max(min(source.relevance, 1.0), 0.0)
        return round(score, 4)

    def admits(self, source: SourceRecord) -> bool:
        return self.assess(source) >= self.admission_threshold


class KnowledgeStore:
    """Typed knowledge graph with admission gating and temporal decay.

    Nodes are knowledge objects; edges carry one of RELATIONS. The graph is
    the agent's accumulated external cognition — every edge traces back to
    the source document that generated it.
    """

    def __init__(
        self,
        gate: VerificationGate | None = None,
        default_half_life_days: float = 365.0,
        archive_threshold: float = 0.05,
    ) -> None:
        self.gate = gate or VerificationGate()
        self.default_half_life_days = default_half_life_days
        self.archive_threshold = archive_threshold
        self.graph = nx.DiGraph()
        self.objects: dict[str, KnowledgeObject] = {}
        self.rejected: list[str] = []
        self.archived: dict[str, KnowledgeObject] = {}

    def admit(
        self,
        claim: str,
        source: SourceRecord,
        now: datetime | None = None,
        applicability: tuple[str, ...] = (),
        half_life_days: float | None = None,
    ) -> KnowledgeObject | None:
        quality = self.gate.assess(source)
        if quality < self.gate.admission_threshold:
            self.rejected.append(source.identifier)
            return None
        obj = KnowledgeObject(
            knowledge_id=f"K-{uuid.uuid4().hex[:8]}",
            claim=claim,
            source=source,
            quality=quality,
            acquired_at=now or datetime.now(timezone.utc),
            half_life_days=half_life_days or self.default_half_life_days,
            applicability=applicability,
        )
        self.objects[obj.knowledge_id] = obj
        self.graph.add_node(obj.knowledge_id, claim=claim, source=source.identifier)
        return obj

    def relate(self, source_id: str, relation: str, target_id: str, weight: float = 0.5) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation '{relation}'; expected one of {RELATIONS}")
        self.graph.add_edge(source_id, target_id, relation=relation, weight=weight)
        if relation in ("CONTRADICTS", "INVALIDATES"):
            self._register_contradiction(target_id)
        elif relation == "SUPPORTS":
            self.reinforce(target_id)

    def reinforce(self, knowledge_id: str) -> None:
        obj = self.objects[knowledge_id]
        obj.reinforcements += 1
        if obj.status == "UNVERIFIED":
            obj.status = "SUPPORTED"
        elif obj.status == "SUPPORTED":
            obj.status = "REINFORCED"

    def contradict(self, knowledge_id: str) -> None:
        """Register a contradiction observed outside the graph (e.g. an audit)."""
        self._register_contradiction(knowledge_id)

    def _register_contradiction(self, knowledge_id: str) -> None:
        obj = self.objects[knowledge_id]
        obj.contradictions += 1
        if obj.status != "SUPERSEDED":
            obj.status = "CONTRADICTED"

    def supersede(self, old_id: str, new_id: str) -> None:
        """A newer verified claim replaces an outdated one (INVALIDATES edge)."""
        self.relate(new_id, "INVALIDATES", old_id)
        old = self.objects[old_id]
        old.status = "SUPERSEDED"
        old.superseded_by = new_id

    def adjust_edge(
        self,
        source_id: str,
        target_id: str,
        verified_usefulness: float = 0.0,
        contradiction: float = 0.0,
        eta: float = 0.1,
        lam: float = 0.2,
    ) -> float:
        """w(t+1) = w(t) + eta*V - lambda*C, clamped to [0, 1].

        Heavily verified connections visibly strengthen; repeatedly
        falsified relationships weaken toward removal.
        """
        data = self.graph.edges[source_id, target_id]
        weight = data.get("weight", 0.5) + eta * verified_usefulness - lam * contradiction
        data["weight"] = max(min(weight, 1.0), 0.0)
        return data["weight"]

    def contradictions(self) -> list[tuple[str, str]]:
        return [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if data.get("relation") in ("CONTRADICTS", "INVALIDATES")
        ]

    def metabolize(self, now: datetime) -> list[str]:
        """Archive knowledge whose weight has decayed below the threshold."""
        expired = [
            kid for kid, obj in self.objects.items() if obj.weight(now) < self.archive_threshold
        ]
        for kid in expired:
            self.archived[kid] = self.objects.pop(kid)
            self.graph.remove_node(kid)
        return expired
