"""End-to-end demonstrator for contradiction elimination and closure."""

from __future__ import annotations

from dataclasses import asdict

from engineering_intelligence.audit import build_audit_report
from engineering_intelligence.judge import ClosureInputs, ClosureJudge
from engineering_intelligence.knowledge_graph import KnowledgeGraph, NodeRecord
from engineering_intelligence.physics_core import Heat1DResidualSolver, PhysicsCase
from engineering_intelligence.symbolic_core import CandidateArchitecture, ConstraintSpec, SymbolicReasoner


def run_demo() -> dict:
    """Run demonstrator pipeline and return machine-readable result."""
    reasoner = SymbolicReasoner()
    candidates = [
        CandidateArchitecture("design_A", {"power": 80.0, "mass": 18.0, "margin": 0.2, "k": 10.0}),
        CandidateArchitecture("design_B", {"power": 140.0, "mass": 22.0, "margin": -0.1, "k": 8.0}),
    ]
    constraints = [
        ConstraintSpec("power_cap", "power <= 100", kill_criterion=True),
        ConstraintSpec("mass_cap", "mass <= 20", kill_criterion=True),
        ConstraintSpec("positive_margin", "margin > 0", kill_criterion=True),
        ConstraintSpec("positive_k", "k > 0", kill_criterion=True),
    ]

    prune_result = reasoner.prune_architectures(candidates, constraints)
    if not prune_result.admissible:
        raise RuntimeError("No admissible architectures found in demo")

    selected = prune_result.admissible[0]

    solver = Heat1DResidualSolver()
    physics_case = PhysicsCase(
        case_id=f"thermal_{selected.architecture_id}",
        domain="thermal",
        parameters={
            "k": selected.variables["k"],
            "q": 100.0,
            "length": 1.0,
            "t_left": 350.0,
            "t_right": 300.0,
            "n_points": 21,
        },
    )
    physics_result = solver.solve(physics_case)

    kg = KnowledgeGraph()
    kg.add_node(NodeRecord("assump_1", "assumption", {"text": "Steady-state 1D conduction"}))
    kg.add_node(NodeRecord("eq_1", "equation", {"text": "k*d2T/dx2 + q = 0"}))
    kg.add_node(NodeRecord("solver_1", "solver_output", {"residual": physics_result.residual}))
    kg.add_node(NodeRecord("cite_1", "citation", {"source": "Incropera textbook ref placeholder"}))
    kg.add_node(NodeRecord("claim_1", "claim", {"text": "Residual below threshold"}))
    kg.add_edge("assump_1", "claim_1", "supports")
    kg.add_edge("eq_1", "claim_1", "supports")
    kg.add_edge("solver_1", "claim_1", "supports")
    kg.add_edge("cite_1", "claim_1", "supports")

    judge = ClosureJudge()
    closure = judge.evaluate(
        ClosureInputs(
            residual=physics_result.residual,
            residual_threshold=120.0,
            hard_constraints_satisfied=selected.architecture_id not in prune_result.rejected,
            margin=selected.variables["margin"],
            manufacturable=True,
            controllable=True,
            power_budget_closed=selected.variables["power"] <= 100.0,
            thermal_budget_closed=physics_result.residual <= 120.0,
            provenance_complete=kg.has_provenance("claim_1"),
            unknowns=["Material aging coefficient UNKNOWN"],
        )
    )

    audit = build_audit_report(
        design_id=selected.architecture_id,
        assumed=["Steady-state conduction", "Constant thermal conductivity"],
        derived=[f"PDE L2 residual = {physics_result.residual:.3f}"],
        verified=["Hard constraints satisfied by Z3", "Residual computed from governing equation discretization"],
        unverified=["Material aging coefficient UNKNOWN", "No experimental calibration yet"],
        failure_modes=["Thermal source term q mismatch", "Boundary condition drift"],
        next_test=["Run transient thermal test with measured heat flux"],
        metadata={"classification": closure.classification, "reasons": closure.reasons},
    )

    return {
        "selected_design": asdict(selected),
        "prune_result": {
            "admissible_ids": [c.architecture_id for c in prune_result.admissible],
            "rejected": prune_result.rejected,
        },
        "physics_result": asdict(physics_result),
        "closure": asdict(closure),
        "knowledge_graph": kg.export_json(),
        "audit": audit.to_json(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_demo(), indent=2))
