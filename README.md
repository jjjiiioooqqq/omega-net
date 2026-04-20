# Mocap-less 2D Video → 3D Scene Pipeline

This project builds a high-fidelity Python application that extracts motion and spatial data from 2D video and exports a JSON manifest usable in Blender (and adaptable to Maya).

## Folder Structure

```text
omega-net/
├── input_video/
│   └── your_shot.mp4
├── output_data/
│   └── your_shot_manifest.json
├── blender_scripts/
│   └── import_manifest.py
├── scene_processor.py
└── requirements.txt
```

## Phase 1: Environment + Dependencies

### Required Libraries
- OpenCV (`opencv-python`) for video I/O and tracking.
- MediaPipe (`mediapipe`) for 33-point pose landmark inference.
- Segment Anything (`segment-anything`) for target identity masking.
- Torch (`torch`, `torchvision`) for model execution.
- DepthAnything v2 (`transformers` model `depth-anything/Depth-Anything-V2-Small-hf`) for monocular depth.
- NumPy (`numpy`) for coordinate and matrix math.
- SciPy (`scipy`) for robust transform math.
- tqdm (`tqdm`) for processing progress.

### Clean Virtual Environment Setup (Exact Commands)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### SAM Checkpoint
Download one SAM checkpoint and pass it to `scene_processor.py`:
- `sam_vit_h_4b8939.pth` (recommended highest quality)
- `sam_vit_l_0b3195.pth`
- `sam_vit_b_01ec64.pth`

## Phase 2: Core Python Logic (Scene Processor)

### What `scene_processor.py` Does
1. **Identity Scanner**: On frame zero, you draw the target ROI. SAM turns it into a binary mask.
2. **Identity Tracking**: OpenCV CSRT tracks the target, SAM refines mask each frame.
3. **3D Keypoint Extraction**: MediaPipe detects 33 landmarks from masked target.
4. **2D→3D Conversion**: Depth map + camera FOV convert pixel points to meter-space world points.
5. **Spatial Memory Mapping**: RANSAC fits ground and wall planes from depth point cloud.
6. **Feet Ground Snapping**: Foot/ankle keypoints are projected onto ground plane.
7. **Universal Exporter**: A JSON manifest stores per-frame transforms and environment planes.
8. **Lost Target Handling**: If target goes off-screen/occluded, frame is marked `target_lost`. Long consecutive loss stops early (configurable).

### Run the Scene Processor

```bash
python scene_processor.py \
  --video input_video/your_shot.mp4 \
  --output-dir output_data \
  --sam-checkpoint /absolute/path/to/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h
```

Optional flags:
- `--depth-model depth-anything/Depth-Anything-V2-Small-hf`
- `--fov-deg 60`
- `--max-lost-frames 45`
- `--cpu`

## Phase 3: Blender Bridge

`blender_scripts/import_manifest.py` reads the JSON manifest, creates a metarig-like armature, keys frame animation, and adds ground/wall environment planes.

It includes `Global_Transform` to adjust:
- **Scale** (character height)
- **Rotation** (heading)
- **Location** (world position)

while preserving captured motion dynamics.

### Run in Blender

```bash
blender --python blender_scripts/import_manifest.py -- \
  --manifest output_data/your_shot_manifest.json \
  --location 0 0 0 \
  --rotation 0 0 0 \
  --scale 1.0
```

## Phase 4: Maya Integration (Data Contract)

The generated manifest is DCC-agnostic JSON containing:
- Frame metadata
- `target_lost` state
- `bbox_xywh`
- `transform` (`translation`, `rotation_euler_xyz`, `scale`)
- `landmarks_world_m` (33 entries)
- `environment` planes (`ground_plane`, `left_wall`, `right_wall`)

You can consume the same JSON in Maya (Python) to build joints, apply keyframes, and create proxy planes.

