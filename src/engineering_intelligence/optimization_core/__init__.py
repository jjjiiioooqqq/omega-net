"""Optimization core constrained by symbolic admissibility."""

from .optimizer import ObjectiveType, OptimizationProblem, OptimizationResult, optimize_admissible

__all__ = ["ObjectiveType", "OptimizationProblem", "OptimizationResult", "optimize_admissible"]
