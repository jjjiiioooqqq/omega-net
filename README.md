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
- `spatial_audio/`: SoF-inity speaker-array rendering — acoustic mapping (time-alignment + level compensation from measured geometry), spatiotemporal multiplexing (sample-interpolated VBAP for moving sources), physical propagation (exact Doppler via emission-time delay, ISO 9613-1 air absorption, shoebox image-source reflections), and Gerzon energy-vector localization metrics.

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

## Run the SoF-inity spatial audio demo
```bash
python examples/run_sofinity_demo.py
# writes output_data/sofinity_demo_8ch.wav (8-channel speaker feeds)
# and output_data/sofinity_demo_metrics.json (localization + level metrics)
```

## Blender bridge
Open Blender Scripting tab and run:
```python
from blender_scripts.import_manifest import main
main('/absolute/path/to/output_data/scene_manifest.json')
```

## Known limitations (explicit)
- SAM integration currently uses bbox-mask fallback unless SAM checkpoint runtime wiring is added.
- Depth estimator is currently a deterministic placeholder (grayscale proxy), not validated DepthAnything v2 inference.
- Physics demonstrator is residual evaluation only and not a full PDE solve/validation loop.
- Blender bridge currently animates root transform only.
- SoF-inity pans in the horizontal plane only (speaker and path elevation collapse to azimuth). Reflections are specular shoebox image sources with uniform, frequency-independent wall absorption; air absorption is block-wise zero-phase (no dispersion); supersonic radial closing speeds are rejected rather than modeled.

## Future integrations
- Real SAM checkpoint loading and temporal mask propagation.
- DepthAnything v2 torch wrapper with calibrated camera model.
- Structured manufacturability/control constraints in symbolic core.
- CFD/FEA external solver adapters with provenance capture.
