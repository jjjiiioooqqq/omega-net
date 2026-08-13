from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field, replace


class ConstitutionViolation(RuntimeError):
    """Raised when a mutation attempts to alter a protected invariant."""


@dataclass(frozen=True, slots=True)
class Constitution:
    """Invariants the evolving agent can never modify.

    These are the physics of the agent's environment: holdout benchmarks,
    evidence requirements, loss constraints, and approval boundaries live
    outside the genome and are never subject to mutation or crossover.
    """

    benchmark_seal: str
    max_ruin_probability: float = 0.05
    evidence_required: bool = True
    protected_targets: frozenset[str] = frozenset(
        {
            "benchmark",
            "fitness",
            "holdout",
            "permissions",
            "source_verification",
            "human_approval",
        }
    )

    def check_target(self, target: str) -> None:
        lowered = target.lower()
        for protected in self.protected_targets:
            if protected in lowered:
                raise ConstitutionViolation(
                    f"mutation target '{target}' touches protected invariant '{protected}'"
                )


@dataclass(slots=True)
class Genome:
    """G = (P, S, M, E, T, R): the evolvable architecture around a fixed model.

    The foundation model is only the inference engine underneath this genome;
    descendants change the architecture, never the fitness test.
    """

    genome_id: str
    generation: int
    parent_id: str | None
    blueprint: dict[str, str] = field(default_factory=dict)  # P: prompts/system blueprint
    skills: tuple[str, ...] = ()  # S: skill ids from the library
    memory_policy: dict[str, float] = field(default_factory=dict)  # M
    evaluators: tuple[str, ...] = ()  # E: decision procedures
    tool_policies: dict[str, str] = field(default_factory=dict)  # T
    research_strategy: tuple[str, ...] = ()  # R: ordered retrieval steps
    agent_organization: tuple[str, ...] = ()  # A: species/subagent roster
    model_routing: dict[str, str] = field(default_factory=dict)  # domain -> engine name
    species: str = "generalist"

    @staticmethod
    def seed(**components) -> Genome:
        return Genome(
            genome_id=f"G-{uuid.uuid4().hex[:8]}",
            generation=0,
            parent_id=None,
            **components,
        )

    def descend(self) -> Genome:
        return replace(
            self,
            genome_id=f"G-{uuid.uuid4().hex[:8]}",
            generation=self.generation + 1,
            parent_id=self.genome_id,
            blueprint=dict(self.blueprint),
            memory_policy=dict(self.memory_policy),
            tool_policies=dict(self.tool_policies),
            model_routing=dict(self.model_routing),
        )


class Mutator:
    """Generates architectural descendants; refuses protected targets."""

    def __init__(self, constitution: Constitution, seed: int | None = None) -> None:
        self.constitution = constitution
        self.rng = random.Random(seed)

    def mutate_blueprint(self, genome: Genome, key: str, value: str) -> Genome:
        # Both key and value are screened: a protected directive must not
        # enter the genome under an innocuous key.
        self.constitution.check_target(key)
        self.constitution.check_target(value)
        child = genome.descend()
        child.blueprint[key] = value
        return child

    def mutate_research_strategy(self, genome: Genome, steps: tuple[str, ...]) -> Genome:
        for step in steps:
            self.constitution.check_target(step)
        child = genome.descend()
        child.research_strategy = steps
        return child

    def mutate_memory_policy(self, genome: Genome, key: str, value: float) -> Genome:
        self.constitution.check_target(key)
        child = genome.descend()
        child.memory_policy[key] = value
        return child

    def mutate_model_routing(self, genome: Genome, domain: str, engine: str) -> Genome:
        """Route a task domain to a different engine; the routing choice is
        itself selected on benchmarks, never on an engine's self-report."""
        self.constitution.check_target(domain)
        child = genome.descend()
        child.model_routing[domain] = engine
        return child

    def add_skill(self, genome: Genome, skill_id: str) -> Genome:
        child = genome.descend()
        if skill_id not in child.skills:
            child.skills = child.skills + (skill_id,)
        return child

    def crossover(self, primary: Genome, donor: Genome) -> Genome:
        """Species crossover: strong components from two lineages combine."""
        child = primary.descend()
        child.skills = tuple(dict.fromkeys(primary.skills + donor.skills))
        child.research_strategy = donor.research_strategy or primary.research_strategy
        merged = dict(donor.blueprint)
        merged.update(primary.blueprint)
        child.blueprint = merged
        return child
