from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sympy as sp
import z3

from .models import CandidateArchitecture, ConstraintSpec, PruneResult


@dataclass(slots=True)
class SymbolicEngine:
    """SymPy canonicalization + Z3 kill-criteria pruning for candidate architectures."""

    def canonicalize(self, expression: str) -> str:
        """Return canonical SymPy representation for a constraint expression."""
        return str(sp.simplify(sp.sympify(expression)))

    def prune(
        self,
        candidates: Iterable[CandidateArchitecture],
        constraints: list[ConstraintSpec],
    ) -> PruneResult:
        """Reject candidates that violate any hard constraint (kill criteria)."""
        survivors: list[CandidateArchitecture] = []
        rejected: dict[str, list[str]] = {}
        canonical_forms = {c.name: self.canonicalize(c.expression) for c in constraints}

        for candidate in candidates:
            failures: list[str] = []
            solver = z3.Solver()
            z3_vars = {name: z3.Real(name) for name in candidate.variables}

            for name, value in candidate.variables.items():
                solver.add(z3_vars[name] == z3.RealVal(value))

            for constraint in constraints:
                expr = sp.sympify(constraint.expression)
                if not self._is_relational(expr):
                    failures.append(f"kill:{constraint.name}:non_relational")
                    continue
                z3_expr = self._sympy_to_z3(expr, z3_vars)
                if constraint.kind == "hard":
                    solver.push()
                    solver.add(z3.Not(z3_expr))
                    if solver.check() == z3.sat:
                        failures.append(f"kill:{constraint.name}")
                    solver.pop()

            if failures:
                rejected[candidate.arch_id] = failures
            else:
                survivors.append(candidate)

        return PruneResult(
            survivors=survivors,
            rejected=rejected,
            canonical_forms=canonical_forms,
            satisfiable=len(survivors) > 0,
            metadata={"kill_criteria": [c.name for c in constraints if c.kind == "hard"]},
        )

    def _is_relational(self, expr: sp.Expr) -> bool:
        return isinstance(expr, sp.core.relational.Relational)

    def _sympy_to_z3(self, expr: sp.Expr, z3_vars: dict[str, z3.ArithRef]) -> z3.ExprRef:
        if isinstance(expr, sp.And):
            return z3.And(*[self._sympy_to_z3(arg, z3_vars) for arg in expr.args])
        if isinstance(expr, sp.Or):
            return z3.Or(*[self._sympy_to_z3(arg, z3_vars) for arg in expr.args])
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
            symbol = str(expr)
            if symbol not in z3_vars:
                raise ValueError(f"variable '{symbol}' missing from candidate")
            return z3_vars[symbol]
        if isinstance(expr, sp.Number):
            return z3.RealVal(float(expr))
        if isinstance(expr, sp.Add):
            args = [self._sympy_to_z3(arg, z3_vars) for arg in expr.args]
            out = args[0]
            for item in args[1:]:
                out = out + item
            return out
        if isinstance(expr, sp.Mul):
            args = [self._sympy_to_z3(arg, z3_vars) for arg in expr.args]
            out = args[0]
            for item in args[1:]:
                out = out * item
            return out
        if isinstance(expr, sp.Pow):
            base = self._sympy_to_z3(expr.base, z3_vars)
            exp = self._sympy_to_z3(expr.exp, z3_vars)
            return base**exp
        raise ValueError(f"Unsupported sympy expression: {type(expr)}")
