from engineering_intelligence.symbolic_core import CandidateArchitecture, ConstraintSpec, SymbolicReasoner


def test_prune_architectures_kill_criteria() -> None:
    reasoner = SymbolicReasoner()
    candidates = [
        CandidateArchitecture("ok", {"power": 90.0, "mass": 15.0}),
        CandidateArchitecture("bad", {"power": 120.0, "mass": 15.0}),
    ]
    constraints = [
        ConstraintSpec("power_cap", "power <= 100", kill_criterion=True),
        ConstraintSpec("mass_cap", "mass <= 20", kill_criterion=True),
    ]

    result = reasoner.prune_architectures(candidates, constraints)

    assert [c.architecture_id for c in result.admissible] == ["ok"]
    assert "bad" in result.rejected


def test_prune_reports_unknown_variable_parse_error() -> None:
    reasoner = SymbolicReasoner()
    candidates = [CandidateArchitecture("ok", {"power": 90.0})]
    constraints = [ConstraintSpec("bad_var", "mass <= 20", kill_criterion=False)]

    result = reasoner.prune_architectures(candidates, constraints)

    assert result.admissible == []
    assert "unknown variable" in result.rejected["ok"][0]
