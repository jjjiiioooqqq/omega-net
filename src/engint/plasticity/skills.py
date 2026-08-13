from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass(slots=True)
class Skill:
    """A reusable workflow distilled from repeated task outcomes.

    Skills accumulate like Voyager's library: an ordered workflow plus the
    empirical record of how well it performed when applied.
    """

    skill_id: str
    name: str
    domain: str
    workflow: tuple[str, ...]
    trials: int = 0
    successes: int = 0
    total_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.trials if self.trials else 0.0


class SkillLibrary:
    """Skill-level selection: workflows that keep failing are retired."""

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}
        self.retired: dict[str, Skill] = {}

    def register(self, name: str, domain: str, workflow: tuple[str, ...]) -> Skill:
        skill = Skill(
            skill_id=f"S-{uuid.uuid4().hex[:8]}",
            name=name,
            domain=domain,
            workflow=workflow,
        )
        self.skills[skill.skill_id] = skill
        return skill

    def record_outcome(self, skill_id: str, success: bool, cost: float = 0.0) -> None:
        skill = self.skills[skill_id]
        skill.trials += 1
        skill.total_cost += cost
        if success:
            skill.successes += 1

    def best_for(self, domain: str) -> Skill | None:
        candidates = [s for s in self.skills.values() if s.domain == domain and s.trials > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.success_rate, -s.mean_cost))

    def prune(self, min_trials: int = 5, min_success_rate: float = 0.3) -> list[str]:
        doomed = [
            sid
            for sid, s in self.skills.items()
            if s.trials >= min_trials and s.success_rate < min_success_rate
        ]
        for sid in doomed:
            self.retired[sid] = self.skills.pop(sid)
        return doomed
