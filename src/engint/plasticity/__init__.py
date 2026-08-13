"""Artificial Darwinian Plasticity: an evolving cognitive architecture
around a fixed foundation model.

Genome -> Phenotype -> Environment -> Fitness -> Selection -> Mutation
-> next generation. The foundation model is only the inference engine;
the genome (blueprint, skills, memory, evaluators, tool policies,
research strategy, agent organization) is what evolves, and the fitness
test is sealed outside the reach of the evolving agent.
"""

from .bootstrap import GenerationZero, load_generation_zero
from .cognition import (
    COMPREHENSION_LEVELS,
    AdaptiveComprehension,
    BudgetPolicy,
    CognitionPlan,
    ProblemProfile,
    cognition_roi,
)
from .curriculum import CURRICULUM, Curriculum, CurriculumStage
from .events import TRIGGER_KINDS, EventRouter, RoutedInvestigation, SourceEvent
from .evidence import Document, DocumentStore
from .evolution import ALIVE, CHAMPION, EXTINCT, UNTESTED, Archive, EvolutionEngine, LineageRecord
from .failure import FailureMemory, FailureRecord, Lesson
from .fitness import (
    BenchmarkCase,
    BenchmarkTampered,
    FitnessInputs,
    FitnessReport,
    IntelligenceVector,
    SealedBenchmark,
    fitness,
)
from .genome import Constitution, ConstitutionViolation, Genome, Mutator
from .knowledge import (
    CLAIM_STATES,
    RELATIONS,
    KnowledgeObject,
    KnowledgeStore,
    SourceRecord,
    VerificationGate,
)
from .memory import Episode, EpisodeLog, MemorySystem
from .predictions import Prediction, PredictionFrozen, PredictionLedger, Resolution
from .runtime import EngineRecord, EngineRegistry
from .skills import Skill, SkillLibrary
from .validation import (
    MARKET_EVENTS,
    WEALTH_LADDER,
    Experiment,
    ExperimentFrozen,
    FirewallViolation,
    MarketValidationFirewall,
    Strategy,
    StrategyTournament,
)

__all__ = [
    "ALIVE",
    "CHAMPION",
    "CLAIM_STATES",
    "COMPREHENSION_LEVELS",
    "CURRICULUM",
    "EXTINCT",
    "MARKET_EVENTS",
    "RELATIONS",
    "TRIGGER_KINDS",
    "UNTESTED",
    "WEALTH_LADDER",
    "AdaptiveComprehension",
    "Archive",
    "BenchmarkCase",
    "BenchmarkTampered",
    "BudgetPolicy",
    "CognitionPlan",
    "Constitution",
    "ConstitutionViolation",
    "Curriculum",
    "CurriculumStage",
    "Document",
    "DocumentStore",
    "EngineRecord",
    "EngineRegistry",
    "Episode",
    "EpisodeLog",
    "EventRouter",
    "EvolutionEngine",
    "Experiment",
    "ExperimentFrozen",
    "FailureMemory",
    "FailureRecord",
    "FirewallViolation",
    "FitnessInputs",
    "FitnessReport",
    "GenerationZero",
    "Genome",
    "IntelligenceVector",
    "KnowledgeObject",
    "KnowledgeStore",
    "Lesson",
    "LineageRecord",
    "MarketValidationFirewall",
    "MemorySystem",
    "Mutator",
    "Prediction",
    "PredictionFrozen",
    "PredictionLedger",
    "ProblemProfile",
    "Resolution",
    "RoutedInvestigation",
    "SealedBenchmark",
    "SourceEvent",
    "Skill",
    "SkillLibrary",
    "SourceRecord",
    "Strategy",
    "StrategyTournament",
    "VerificationGate",
    "cognition_roi",
    "fitness",
    "load_generation_zero",
]
