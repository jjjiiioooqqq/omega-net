# Engineering Intelligence Foundation

A Python foundation for contradiction elimination and engineering design closure.

## Philosophy

This system is intentionally **evidence-bound**:

- It rejects impossible architectures via hard constraints.
- It distinguishes governing-equation evidence from surrogate or unverified learned outputs.
- It records digital-thread provenance for claims.
- It classifies closure explicitly (`legendary`, `provisional`, `unverified`, `folkloric`).
- It emits audit artifacts that separate VERIFIED and UNVERIFIED content.

## Architecture

```text
engineering_intelligence/
  symbolic_core/      # SymPy + Z3 canonicalization and pruning
  physics_core/       # Physics solver interfaces + thermal residual demonstrator
  optimization_core/  # Admissibility-gated differentiable optimization
  knowledge_graph/    # Graph-backed provenance store (digital thread)
  judge/              # Formal closure evaluator and classification
  audit/              # Machine-readable audit report generation
  demo/               # End-to-end pipeline example
```

## Module details

### symbolic_core
- `SymbolicReasoner.canonicalize(...)` uses SymPy simplification.
- `SymbolicReasoner.prune_architectures(...)` uses Z3 to enforce hard constraints.
- Kill criteria are explicit per `ConstraintSpec`.

### physics_core
- `PhysicsSolver` interface supports domain-tagged cases (`maxwell`, `flow`, `thermal`, `structural`).
- `Heat1DResidualSolver` is a concrete governing-equation residual evaluator for 1D steady conduction.
- `PINNReadyMixin` provides a placeholder integration point while keeping learned models explicitly unverified.

### optimization_core
- `optimize_admissible(...)` performs continuous optimization with admissibility gating.
- Objective modes include residual, performance, margin, power, and thermal load targets.

### knowledge_graph
- Local directed graph (`networkx`) with explicit node-type schema.
- Supports provenance checks from claims back to assumptions/equations/solver evidence/citations.

### judge
- `ClosureJudge.evaluate(...)` encodes explicit closure gates:
  residual threshold, hard constraints, margin, manufacturability, control, budget closure, provenance, unknown carry.

### audit
- `build_audit_report(...)` produces machine-readable records with required sections:
  `ASSUMED`, `DERIVED`, `VERIFIED`, `UNVERIFIED`, `FAILURE_MODES`, `NEXT_TEST`.

## End-to-end demonstrator

The demo pipeline:
1. Defines two candidate architectures.
2. Prunes them with Z3 hard constraints.
3. Runs a thermal PDE residual evaluator for the surviving design.
4. Builds a knowledge graph and checks claim provenance.
5. Evaluates closure classification.
6. Emits machine-readable audit output.

Run:

```bash
python -m engineering_intelligence.demo.pipeline
```


## PyCharm setup (recommended)

1. Open the repository root in PyCharm.
2. Create a Python 3.10+ interpreter/venv for the project.
3. Install dependencies from files (works well with PyCharm package UI or terminal):

```bash
pip install -r requirements-dev.txt
```

4. Run tests directly from PyCharm or terminal:

```bash
pytest
```

5. Run the demo (no PYTHONPATH editing required):

```bash
python examples/run_demo.py
```

Optional: if you install the package in editable mode (`pip install -e .`), you can
also run:

```bash
python -m engineering_intelligence --output examples/demo_audit_output.json
```

The repo includes `tests/conftest.py` and a path-bootstrap in `examples/run_demo.py`
so execution works in typical PyCharm run/test configurations without manual
`PYTHONPATH` editing.

## Development

Install dependencies (example with pip):

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

## Current limitations

- Dimensional consistency is best-effort and not a full units system.
- Thermal demonstrator is simplified (linear trial field with FD residual) and not a validated production simulation.
- No external CFD/FEA/EM solver adapters yet.
- No experimental data ingestion pipeline yet.
- Optimization layer is generic and does not yet include automatic differentiation backends.

## Future integrations

- OpenFOAM / SU2 / FEniCS wrappers for higher-fidelity physics.
- EM and structural solver adapters with standardized result schema.
- Formal units engine for rigorous dimensional analysis.
- Experiment ingestion + validation workflows.
- Probabilistic uncertainty propagation and robust design criteria.
- Control synthesis hooks for dynamic feasibility closure.
