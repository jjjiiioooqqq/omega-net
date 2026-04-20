from __future__ import annotations

from pathlib import Path

import numpy as np

from engint.audit import AuditWriter
from engint.judge import ClosureInputs, ClosureJudge
from engint.knowledge_graph import DigitalThreadGraph, NodeRecord, NodeType
from engint.physics_core import PlaceholderPhysicsSolver, ThermalDiffusionResidualSolver
from engint.symbolic_core import CandidateArchitecture, ConstraintSpec, SymbolicEngine


def run_demo(output_path: Path) -> dict[str, object]:
    """End-to-end contradiction elimination and closure classification demo."""
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

    # Explicit placeholders for unsupported domains to avoid pretending closure.
    maxwell = PlaceholderPhysicsSolver("maxwell_stub", "maxwell").evaluate(np.zeros((2, 3)), dx=1.0, dt=1.0)
    flow = PlaceholderPhysicsSolver("navier_stokes_stub", "flowfield").evaluate(np.zeros((2, 3)), dx=1.0, dt=1.0)

    graph = DigitalThreadGraph()
    graph.add_node(NodeRecord("assumption_1", NodeType.ASSUMPTION, {"text": "constant alpha"}, ["demo_input"]))
    graph.add_node(NodeRecord("equation_1", NodeType.EQUATION, {"name": "thermal_diffusion_1d"}, ["textbook_ref"]))
    graph.add_node(
        NodeRecord(
            "claim_1",
            NodeType.CLAIM,
            {"text": "architecture survives hard constraints"},
            ["z3_result"],
            critical=True,
        )
    )
    graph.relate("assumption_1", "supports", "equation_1")
    graph.relate("equation_1", "supports", "claim_1")

    provenance_complete = len(graph.claims_missing_provenance()) == 0 and len(graph.critical_nodes_missing_provenance()) == 0

    closure = ClosureJudge().evaluate(
        ClosureInputs(
            residual_ok=physics.residual < 100.0,
            hard_constraints_ok=prune.satisfiable,
            margins_positive=True,
            manufacturable=True,
            controllable=True,
            power_thermal_closed=True,
            provenance_complete=provenance_complete,
            unknowns_explicit=True,
        )
    )

    audit = AuditWriter().build_bundle(
        ASSUMED=["alpha treated constant/isotropic", "UNKNOWN: manufacturing tolerance model"],
        DERIVED=[f"survivor_count={len(prune.survivors)}", f"thermal_rms_residual={physics.residual:.6f}"],
        VERIFIED=["Z3 hard constraints evaluated", "Finite-difference thermal residual computed"],
        UNVERIFIED=[
            "UNVERIFIED: Maxwell closure uses placeholder adapter",
            "UNVERIFIED: Navier-Stokes closure uses placeholder adapter",
            f"UNVERIFIED: {maxwell.solver_name} residual unavailable",
            f"UNVERIFIED: {flow.solver_name} residual unavailable",
        ],
        FAILURE_MODES=["thermal hotspot with boundary drift", "sensor dropout can invalidate controllability"],
        NEXT_TEST=["integrate external CFD/EM solvers", "perform hardware-in-the-loop control test"],
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
