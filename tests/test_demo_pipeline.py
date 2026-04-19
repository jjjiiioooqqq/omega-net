from engineering_intelligence.demo.pipeline import run_demo


def test_demo_pipeline_runs() -> None:
    output = run_demo()
    assert output["selected_design"]["architecture_id"] == "design_A"
    assert output["closure"]["classification"] in {"unverified", "provisional", "legendary", "folkloric"}
    assert "audit" in output
