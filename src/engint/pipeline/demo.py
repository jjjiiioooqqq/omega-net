from __future__ import annotations

from pathlib import Path

import numpy as np

from engint.audit import AuditWriter
from engint.judge import ClosureInputs, ClosureJudge
from engint.knowledge_graph import DigitalThreadGraph, NodeRecord
from engint.physics_core import ThermalDiffusionResidualSolver
from engint.symbolic_core import CandidateArchitecture, ConstraintSpec, SymbolicEngine


def run_demo(output_path: Path) -> dict[str, object]:
    symbolic = SymbolicEngine()
    candidates = [
        CandidateArchitecture("arch_good", {"power": 80.0, "mass": 1.2, "margin": 0.15}),
        CandidateArchitecture("arch_bad", {"power": 130.0, "mass": 0.4, "margin": -0.2}),
    ]
    constraints = [
        ConstraintSpec("power_budget", "power <= 100"),
        ConstraintSpec("mass_min", "mass >= 1.0"),
        ConstraintSpec("positive_margin", "margin > 0"),
    ]
    prune = symbolic.prune(candidates, constraints)

    state = np.array(
        [
            [300.0, 305.0, 307.0, 309.0, 310.0],
            [300.2, 304.8, 306.9, 308.7, 309.4],
            [300.3, 304.6, 306.7, 308.5, 309.0],
        ]
    )
    physics = ThermalDiffusionResidualSolver(alpha=1.0e-5).evaluate(state=state, dx=0.1, dt=0.25)

    graph = DigitalThreadGraph()
    graph.add_node(NodeRecord("assumption_1", "assumption", {"text": "steady material alpha"}, ["demo_input"]))
    graph.add_node(NodeRecord("equation_1", "equation", {"name": "thermal_diffusion_1d"}, ["textbook_ref"]))
    graph.add_node(NodeRecord("claim_1", "claim", {"text": "architecture survives hard constraints"}, ["z3_result"]))
    graph.relate("assumption_1", "supports", "equation_1")
    graph.relate("equation_1", "supports", "claim_1")

    judge = ClosureJudge()
    closure = judge.evaluate(
        ClosureInputs(
            residual_ok=physics.residual < 100.0,
            hard_constraints_ok=prune.satisfiable,
            margins_positive=True,
            manufacturable=True,
            controllable=True,
            power_thermal_closed=True,
            provenance_complete=(len(graph.claims_missing_provenance()) == 0),
            unknowns_explicit=True,
        )
    )

    audit = AuditWriter().build_bundle(
        assumed=["alpha is treated constant and isotropic", "UNKNOWN: manufacturing tolerance model"],
        derived=[f"survivor_count={len(prune.survivors)}", f"thermal_rms_residual={physics.residual:.6f}"],
        verified=["Z3 hard constraints evaluated", "Finite-difference residual computed"],
        unverified=["UNVERIFIED: external CFD/FEA correlation", "UNVERIFIED: closed-loop controller robustness"],
        failure_modes=["thermal hotspot under boundary condition drift", "sensor dropout can invalidate controllability"],
        next_test=["Run mesh-refined thermal solve", "Perform hardware-in-the-loop control test"],
    )
    AuditWriter().write_json(audit, output_path)

    return {
        "survivors": [c.arch_id for c in prune.survivors],
        "rejected": prune.rejected,
        "physics_residual": physics.residual,
        "classification": closure.classification,
        "audit_file": str(output_path),
    }


if __name__ == "__main__":
    out = run_demo(Path("output_data/demo_audit.json"))
    print(out)
