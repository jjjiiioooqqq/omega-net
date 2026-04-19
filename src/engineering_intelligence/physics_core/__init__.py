"""Physics solver interfaces and demonstrators."""

from .interfaces import PhysicsCase, PhysicsResult, PhysicsSolver
from .solvers import Heat1DResidualSolver

__all__ = ["PhysicsCase", "PhysicsResult", "PhysicsSolver", "Heat1DResidualSolver"]
