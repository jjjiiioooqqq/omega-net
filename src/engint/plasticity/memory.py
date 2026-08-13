from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .failure import FailureMemory
from .knowledge import KnowledgeStore
from .predictions import PredictionLedger
from .skills import SkillLibrary


@dataclass(slots=True)
class Episode:
    """Episodic memory entry: what happened, when, and with what outcome."""

    episode_id: str
    timestamp: datetime
    kind: str  # decision | run | experiment | outcome
    description: str
    evidence: tuple[str, ...] = ()
    supersedes: str | None = None


@dataclass(slots=True)
class EpisodeLog:
    episodes: dict[str, Episode] = field(default_factory=dict)

    def record(
        self,
        timestamp: datetime,
        kind: str,
        description: str,
        evidence: tuple[str, ...] = (),
        supersedes: str | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        episode = Episode(
            episode_id=episode_id or f"E-{uuid.uuid4().hex[:8]}",
            timestamp=timestamp,
            kind=kind,
            description=description,
            evidence=evidence,
            supersedes=supersedes,
        )
        self.episodes[episode.episode_id] = episode
        return episode

    def timeline(self) -> list[Episode]:
        return sorted(self.episodes.values(), key=lambda e: e.timestamp)


@dataclass(slots=True)
class MemorySystem:
    """M = M_semantic + M_episodic + M_procedural + M_predictive + M_failure.

    The separation is load-bearing: learning a fact about one company must
    not rewrite how a valuation is performed, and a diagnosed failure must
    not silently edit the prediction record.
    """

    semantic: KnowledgeStore = field(default_factory=KnowledgeStore)  # what is known
    episodic: EpisodeLog = field(default_factory=EpisodeLog)  # what happened
    procedural: SkillLibrary = field(default_factory=SkillLibrary)  # how to do things
    predictive: PredictionLedger = field(default_factory=PredictionLedger)  # frozen forecasts
    failure: FailureMemory = field(default_factory=FailureMemory)  # scar tissue
