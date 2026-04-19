from engineering_intelligence.audit import build_audit_report
from engineering_intelligence.judge import ClosureInputs, ClosureJudge


def test_legendary_requires_no_unknowns() -> None:
    judge = ClosureJudge()
    result = judge.evaluate(
        ClosureInputs(
            residual=1.0,
            residual_threshold=10.0,
            hard_constraints_satisfied=True,
            margin=0.2,
            manufacturable=True,
            controllable=True,
            power_budget_closed=True,
            thermal_budget_closed=True,
            provenance_complete=True,
            unknowns=[],
        )
    )
    assert result.classification == "legendary"


def test_audit_sections_present() -> None:
    report = build_audit_report(
        design_id="x",
        assumed=["A"],
        derived=["B"],
        verified=["C"],
        unverified=[],
        failure_modes=["D"],
        next_test=["E"],
    )
    data = report.to_json()
    assert set(data["sections"].keys()) == {
        "ASSUMED",
        "DERIVED",
        "VERIFIED",
        "UNVERIFIED",
        "FAILURE_MODES",
        "NEXT_TEST",
    }
