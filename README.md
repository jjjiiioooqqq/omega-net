# Engineering Intelligence Foundation (engint)

A truthful foundation for contradiction-elimination and design-closure, plus a mocap-less scene extraction bridge for Blender.

## Philosophy
The system prioritizes invariant consistency, contradiction detection, physical closure, provenance, and testability over plausibility.

## Package structure
- `symbolic_core/`: SymPy canonicalization + Z3 hard-constraint pruning with kill criteria.
- `physics_core/`: extensible solver interface and 1D thermal diffusion residual demonstrator.
- `optimization_core/`: admissible-region constrained continuous optimization.
- `knowledge_graph/`: digital thread graph schema and provenance checks.
- `judge/`: explicit closure classifier (`legendary`, `provisional`, `unverified`, `folkloric`).
- `audit/`: machine-readable audit bundle writer with ASSUMED/DERIVED/VERIFIED/UNVERIFIED/FAILURE MODES/NEXT TEST.
- `mocapless/`: scene processor for 2D video -> 3D manifest.
- `plasticity/`: Artificial Darwinian Plasticity — an evolving cognitive architecture around a fixed foundation model (see below).

## Folder layout
```
/workspace/omega-net
  /input_video
  /output_data
  /blender_scripts
  /examples
  /src/engint
  /tests
```

## Phase 1: setup commands
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
pip install opencv-python mediapipe torch torchvision numpy
pip install git+https://github.com/facebookresearch/segment-anything.git
# Optional for DepthAnything v2 integration entrypoint:
# pip install timm einops
```

## Run the engineering demonstrator
```bash
python examples/run_demo.py
cat output_data/demo_audit.json
```

## Run mocap-less scene processor
```bash
python -c "from pathlib import Path; from engint.mocapless.scene_processor import SceneProcessor; SceneProcessor().process_video(Path('input_video/sample.mp4'), Path('output_data/scene_manifest.json'), (200,100,450,700))"
```

## Blender bridge
Open Blender Scripting tab and run:
```python
from blender_scripts.import_manifest import main
main('/absolute/path/to/output_data/scene_manifest.json')
```

## Artificial Darwinian Plasticity (`engint.plasticity`)

The system does not pretend to rewire foundation-model neurons; what evolves is
the versioned cognitive architecture around a fixed model. The genome
`G = (P, S, M, E, T, R, A)` (blueprint, skills, memory policy, evaluators,
tool policies, research strategy, agent organization) mutates and competes;
the fitness test is sealed outside its reach.

Components:
- `genome`: mutable genome + immutable `Constitution`; mutations touching
  benchmarks, fitness, permissions, or source-verification rules raise
  `ConstitutionViolation`.
- `fitness`: hash-sealed holdout benchmarks (`SealedBenchmark`, tamper-detecting),
  the fitness ratio, and the 8-dimension `IntelligenceVector` `[R,V,Q,F,P,S,L,C]`
  with per-dimension deltas — never a single fake IQ.
- `evolution`: Darwin-Gödel-style archive; a descendant survives only if it
  clears every promotion gate on unseen tests — higher fitness alone never
  promotes: unsupported claims must not increase, safety violations must be
  zero, and the benchmark seal must be intact. Lineage tree rendering keeps
  extinct branches as scar tissue.
- `runtime`: model-agnostic `EngineRegistry` — Claude is the first primary
  engine, not an irreversible dependency; engines compete on identical sealed
  benchmarks and routing follows measured scores, never self-report.
- `events`: sentinel-based operation — watch → detect change → materiality →
  wake cognition. Immaterial events never spend inference; material ones route
  through the C0–C6 controller (budget now includes irreversibility, ζI).
- `knowledge`: verification-gated knowledge objects `K = (C,S,Q,T,A)` with the
  UNVERIFIED → SUPPORTED → REINFORCED / CONTRADICTED → SUPERSEDED lifecycle,
  typed relations, weighted edges (`w += ηV − λC`), temporal decay, and a
  metabolize/archive cycle. More documents ≠ more intelligence.
- `evidence`: content-addressed immutable `DocumentStore`; changed upstream
  content becomes a new version, never a silent edit.
- `memory`: `M = semantic + episodic + procedural + predictive + failure`,
  kept separate so learning a fact never rewrites a procedure.
- `failure`: diagnosed failure mechanisms, scar-tissue similarity search, and
  transfer-gated lessons — failure does not equal learning until a new
  procedure beats the old one on unseen tests.
- `predictions`: hash-frozen prediction ledger with Brier-scored calibration;
  history cannot be rewritten after outcomes.
- `validation`: the wealth ladder, the market-validation firewall (only
  payment / unrelated second payment / recurring commitment can move
  `MARKET_VALIDATED`), frozen-denominator experiments, and the strategy
  tournament.
- `cognition`: adaptive comprehension `B = B0(1+αD+βI+γN+δF+εU)` routing
  problems to levels C0–C6, with escalation on known failure patterns.
- `curriculum`: T0–T10 training stages promoted only on held-out scores.
- `bootstrap`: Generation Zero (`corpus/g0/`) — the real wealth-project records
  (strategy state, oracle comparison, pilot offer, terms, redacted cohort)
  decomposed into the five memory types instead of starting from an empty
  database. The cohort file is committed company-level only; person names and
  contact routes were removed before committing.

Run the demonstrator:
```bash
PYTHONPATH=src python3 examples/run_plasticity_demo.py
python3 -m pytest tests/test_plasticity.py
```

## Known limitations (explicit)
- SAM integration currently uses bbox-mask fallback unless SAM checkpoint runtime wiring is added.
- Depth estimator is currently a deterministic placeholder (grayscale proxy), not validated DepthAnything v2 inference.
- Physics demonstrator is residual evaluation only and not a full PDE solve/validation loop.
- Blender bridge currently animates root transform only.

## Future integrations
- Real SAM checkpoint loading and temporal mask propagation.
- DepthAnything v2 torch wrapper with calibrated camera model.
- Structured manufacturability/control constraints in symbolic core.
- CFD/FEA external solver adapters with provenance capture.
