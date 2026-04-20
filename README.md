# Engineering Intelligence Foundation (`engint`)

`engint` is a Python foundation for **contradiction elimination and design closure**, not a conversational plausibility engine.
It prioritizes:
- invariant consistency,
- hard contradiction detection,
- physical closure checks,
- explicit provenance,
- explicit unknown labeling,
- auditability.

## Architecture

- `symbolic_core/`
  - SymPy canonicalization and expression handling.
  - Z3 hard-constraint pruning with kill criteria.
  - Candidate architectures are rejected on any violated hard constraint.

- `physics_core/`
  - Extensible solver interfaces.
  - Concrete thermal diffusion residual demonstrator (`dT/dt - alpha*d2T/dx2`).
  - Explicit placeholders for Maxwell / Navier–Stokes / structural-style wrappers.
  - Distinguishes governing vs surrogate vs learned-unverified evidence tiers.

- `optimization_core/`
  - Admissible-region constrained optimizer.
  - Objective helpers for minimizing residual/load and maximizing margin/metric.
  - Optimization executes only when seed is admissible.

- `knowledge_graph/`
  - Local graph-backed digital thread via `networkx`.
  - Explicit schema for assumptions, equations, constraints, geometry, solver outputs, experiments, materials, claims.
  - Provenance checks for claims and critical nodes.

- `judge/`
  - Explicit closure policy classifying designs as:
    - `legendary`,
    - `provisional`,
    - `unverified`,
    - `folkloric`.

- `audit/`
  - Machine-readable audit output with required sections:
    - `ASSUMED`, `DERIVED`, `VERIFIED`, `UNVERIFIED`, `FAILURE_MODES`, `NEXT_TEST`.
  - Unknowns are expected to be explicitly labeled `UNKNOWN`/`UNVERIFIED` in content.

- `pipeline/demo.py`
  - End-to-end demonstrator:
    1) define candidate architectures,
    2) prune contradictions,
    3) run simplified physics residual evaluation,
    4) evaluate closure class,
    5) emit audit report JSON.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

## Run demonstrator

```bash
python examples/run_demo.py
cat output_data/demo_audit.json
```

## Run tests

```bash
pytest
```

## Explicit limitations (truthful scope)

1. Thermal module is residual evaluation, not a full verified PDE solve/validation workflow.
2. Maxwell/Navier/structural modules are explicit placeholders awaiting external solver wrappers.
3. Judge thresholds are policy gates; they do not constitute empirical validation.
4. Optimization is local gradient-based and not guaranteed globally optimal.
5. Dimensional checks in symbolic constraints are minimal sanity checks (relational form), not full unit algebra.

## Planned integrations

1. External EM/CFD/FEA solver adapters with structured provenance capture.
2. Unit-aware symbolic constraints with stronger dimensional analysis.
3. Multi-fidelity optimization loops with robust uncertainty quantification.
4. Experiment ingestion and calibration loop to transition from unverified to verified claims.
