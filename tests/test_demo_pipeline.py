from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sympy")
pytest.importorskip("z3")

from engint.pipeline.demo import run_demo


def test_demo_pipeline(tmp_path: Path):
    result = run_demo(tmp_path / "demo_audit.json")
    assert result["classification"] in {"legendary", "provisional", "unverified", "folkloric"}
    assert len(result["survivors"]) >= 1
