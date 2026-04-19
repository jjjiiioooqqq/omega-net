"""SymPy and Z3 based symbolic reasoning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sympy as sp
from z3 import ArithRef, BoolRef, Real, RealVal, Solver, sat

from .models import CandidateArchitecture, ConstraintSpec, PruneResult


@dataclass(slots=True)
class SymbolicReasoner:
    """Applies symbolic canonicalization and hard satisfiability pruning."""

    def canonicalize(self, expression: str) -> str:
        """Canonicalize symbolic expression using SymPy simplification."""
        return str(sp.simplify(sp.sympify(expression)))

    def dimensionally_consistent(self, lhs: str, rhs: str) -> bool:
        """Best-effort dimension sanity check by comparing symbol sets.

        This is intentionally conservative and should be replaced by a full units
        implementation in future integrations.
        """
        lhs_symbols = sp.sympify(lhs).free_symbols
        rhs_symbols = sp.sympify(rhs).free_symbols
        return lhs_symbols == rhs_symbols

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

        for constraint in constraints:
            try:
                sym_expr = sp.sympify(constraint.expression)
                z3_expr = self._sympy_to_z3(sym_expr, z3_vars)
            except (sp.SympifyError, ValueError, KeyError) as exc:
                failures.append(f"{constraint.name}: parse_error={exc}")
                if constraint.kill_criterion:
                    break
                continue

            solver.push()
            solver.add(z3_expr)
            if solver.check() != sat:
                failures.append(constraint.name)
                solver.pop()
                if constraint.kill_criterion:
                    break
            else:
                solver.pop()

        return failures

    def _sympy_to_z3(self, expr: sp.Expr, z3_vars: dict[str, ArithRef]) -> BoolRef | ArithRef:
        if isinstance(expr, sp.And):
            return sp_to_bool([self._sympy_to_z3(arg, z3_vars) for arg in expr.args], op="and")
        if isinstance(expr, sp.Or):
            return sp_to_bool([self._sympy_to_z3(arg, z3_vars) for arg in expr.args], op="or")
        if isinstance(expr, sp.Equality):
            return self._sympy_to_z3(expr.lhs, z3_vars) == self._sympy_to_z3(expr.rhs, z3_vars)
        if isinstance(expr, sp.StrictLessThan):
            return self._sympy_to_z3(expr.lhs, z3_vars) < self._sympy_to_z3(expr.rhs, z3_vars)
        if isinstance(expr, sp.LessThan):
            return self._sympy_to_z3(expr.lhs, z3_vars) <= self._sympy_to_z3(expr.rhs, z3_vars)
        if isinstance(expr, sp.StrictGreaterThan):
            return self._sympy_to_z3(expr.lhs, z3_vars) > self._sympy_to_z3(expr.rhs, z3_vars)
        if isinstance(expr, sp.GreaterThan):
            return self._sympy_to_z3(expr.lhs, z3_vars) >= self._sympy_to_z3(expr.rhs, z3_vars)
        if isinstance(expr, sp.Symbol):
            key = str(expr)
            if key not in z3_vars:
                raise KeyError(f"unknown variable '{key}'")
            return z3_vars[key]
        if isinstance(expr, sp.Number):
            return RealVal(str(float(expr)))
        if isinstance(expr, sp.Add):
            args = [self._sympy_to_z3(arg, z3_vars) for arg in expr.args]
            return sum(args[1:], args[0])
        if isinstance(expr, sp.Mul):
            args = [self._sympy_to_z3(arg, z3_vars) for arg in expr.args]
            out = args[0]
            for item in args[1:]:
                out = out * item
            return out
        if isinstance(expr, sp.Pow):
            base = self._sympy_to_z3(expr.base, z3_vars)
            exponent = self._sympy_to_z3(expr.exp, z3_vars)
            return base**exponent
        raise ValueError(f"Unsupported sympy expression: {type(expr)}")


def sp_to_bool(items: list[BoolRef | ArithRef], op: str) -> BoolRef:
    """Convert composite boolean structures to z3 operators."""
    from z3 import And, Or

    if op == "and":
        return And(*items)
    if op == "or":
        return Or(*items)
    raise ValueError(f"Unsupported boolean op: {op}")
