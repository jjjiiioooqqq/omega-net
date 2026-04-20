from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sympy")
pytest.importorskip("z3")

from engint.audit import AuditWriter
from engint.judge import ClosureInputs, ClosureJudge
from engint.physics_core import ThermalDiffusionResidualSolver
from engint.symbolic_core import CandidateArchitecture, ConstraintSpec, SymbolicEngine


def test_pruning_and_judgement(tmp_path: Path):
    engine = SymbolicEngine()
    candidates = [
        CandidateArchitecture("a", {"power": 50.0, "margin": 0.1}),
        CandidateArchitecture("b", {"power": 150.0, "margin": -0.1}),
    ]
    constraints = [ConstraintSpec("power", "power <= 100"), ConstraintSpec("margin", "margin > 0")]
    result = engine.prune(candidates, constraints)
    assert [c.arch_id for c in result.survivors] == ["a"]

    state = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    phys = ThermalDiffusionResidualSolver(alpha=1.0).evaluate(state, dx=1.0, dt=1.0)

    report = ClosureJudge().evaluate(
        ClosureInputs(
            residual_ok=phys.residual < 1e-6,
            hard_constraints_ok=result.satisfiable,
            margins_positive=True,
            manufacturable=True,
            controllable=True,
            power_thermal_closed=True,
            provenance_complete=True,
            unknowns_explicit=True,
        )
    )
    assert report.classification == "legendary"

    bundle = AuditWriter().build_bundle(ASSUMED=["UNKNOWN: x"], VERIFIED=["v"])
    out = tmp_path / "audit.json"
    AuditWriter().write_json(bundle, out)
    assert out.exists()
