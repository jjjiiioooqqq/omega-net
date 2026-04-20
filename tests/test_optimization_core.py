import numpy as np

from engineering_intelligence.optimization_core import OptimizationProblem, optimize_admissible


def test_admissible_optimization_with_gradient() -> None:
    problem = OptimizationProblem(
        objective_type="minimize_power",
        variable_names=["x"],
        initial_guess=np.array([0.9]),
        bounds=[(0.0, 1.0)],
        objective_fn=lambda arr: float((arr[0] - 0.2) ** 2),
        gradient_fn=lambda arr: np.array([2 * (arr[0] - 0.2)]),
        admissibility_fn=lambda vals: vals["x"] >= 0.1,
    )

    result = optimize_admissible(problem)

    assert result.success
    assert result.x["x"] >= 0.1
    assert result.objective_value < 1e-4


def test_inadmissible_optimizer_solution_is_rejected() -> None:
    problem = OptimizationProblem(
        objective_type="minimize_power",
        variable_names=["x"],
        initial_guess=np.array([0.0]),
        bounds=[(0.0, 1.0)],
        objective_fn=lambda arr: float(2e9 + arr[0]),
        admissibility_fn=lambda vals: vals["x"] >= 0.8,
    )

    result = optimize_admissible(problem)

    assert not result.success
    assert result.x["x"] < 0.8
    assert np.isinf(result.objective_value)
    assert "Rejected inadmissible optimizer output." in result.message
