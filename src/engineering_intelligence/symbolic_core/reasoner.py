"""SymPy and Z3 based symbolic reasoning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sympy as sp
from z3 import Real, RealVal, Solver, sat

from .models import CandidateArchitecture, ConstraintSpec, PruneResult


@dataclass(slots=True)
class SymbolicReasoner:
    """Applies symbolic canonicalization and hard satisfiability pruning."""

    def canonicalize(self, expression: str) -> str:
        """Canonicalize symbolic expression using SymPy simplification."""
        return str(sp.simplify(sp.sympify(expression)))

    def dimensionally_consistent(self, lhs: str, rhs: str) -> bool:
        """Best-effort dimension sanity check by symbol matching.

        Notes:
            This is not a full units engine. It catches trivial mismatches where
            no symbolic dimensions overlap.
        """
        lhs_symbols = sp.sympify(lhs).free_symbols
        rhs_symbols = sp.sympify(rhs).free_symbols
        return bool(lhs_symbols & rhs_symbols) or lhs_symbols == rhs_symbols

    def prune_architectures(
        self,
        candidates: Iterable[CandidateArchitecture],
        constraints: Iterable[ConstraintSpec],
    ) -> PruneResult:
        """Prune candidates using hard constraints with kill-criteria semantics."""
        admissible: list[CandidateArchitecture] = []
        rejected: dict[str, list[str]] = {}

        constraints = list(constraints)
        for candidate in candidates:
            failures = self._evaluate_candidate(candidate, constraints)
            if failures:
                rejected[candidate.architecture_id] = failures
            else:
                admissible.append(candidate)

        return PruneResult(admissible=admissible, rejected=rejected)

    def _evaluate_candidate(
        self,
        candidate: CandidateArchitecture,
        constraints: list[ConstraintSpec],
    ) -> list[str]:
        solver = Solver()
        z3_vars = {name: Real(name) for name in candidate.variables}
        for name, value in candidate.variables.items():
            solver.add(z3_vars[name] == RealVal(str(value)))

        failures: list[str] = []
        env = {**z3_vars}

        for constraint in constraints:
            try:
                expr = eval(constraint.expression, {"__builtins__": {}}, env)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{constraint.name}: parse_error={exc}")
                continue

            solver.push()
            solver.add(expr)
            if solver.check() != sat:
                failures.append(constraint.name)
                solver.pop()
                if constraint.kill_criterion:
                    break
            else:
                solver.pop()

        return failures
