# Ω-Net Mocap-less Video-to-3D Pipeline

This repository provides a production-oriented Python pipeline that:

1. Scans a target identity from frame 0 using Segment Anything (SAM).
2. Tracks the target through the video with mask re-segmentation.
3. Extracts 33 MediaPipe skeletal keypoints.
4. Lifts keypoints into normalized 3D metric space with monocular depth.
5. Fits a ground plane and projects feet keypoints to remove sliding/floating.
6. Exports a full frame-by-frame JSON manifest.
7. Imports the manifest into Blender with a generated metarig and global transforms.

## Folder Structure

```text
omega-net/
├── input_video/
│   └── your_shot.mp4
├── output_data/
│   └── scene_manifest.json
├── blender_scripts/
│   └── blender_bridge.py
├── src/
│   └── scene_processor.py
└── README.md
```

## Phase 1 — Environment + Dependencies

### 1) Create a clean virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Install all required libraries

```bash
pip install numpy opencv-python mediapipe
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers timm safetensors huggingface_hub
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
```

> CPU-only Torch alternative:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3) Download SAM checkpoint

Download one checkpoint file and place it in `models/`, for example:

- `models/sam_vit_h_4b8939.pth`

### 4) Optional DepthAnything V2 checkpoint

If you use native DepthAnything V2 checkpoints, set:

```bash
export DEPTHANYTHINGV2_CKPT=/absolute/path/to/depth_anything_v2_vitb.pth
```

If this variable is not set, the script falls back to Hugging Face depth-anything model.

## Phase 2 — Core Scene Processor

Run the processor:

```bash
python src/scene_processor.py \
  --video input_video/your_shot.mp4 \
  --sam-checkpoint models/sam_vit_h_4b8939.pth \
  --output output_data/scene_manifest.json \
  --target-id actor_A \
  --device cuda \
  --sam-model-type vit_h
```

At launch, an ROI selection window appears on frame 0. Draw the target rectangle and press Enter.

### Lost Target Handling

The processor marks a frame as `"lost_target": true` when:

- Optical flow points drop below threshold.
- Re-detected features inside the mask are insufficient.
- Pose detection fails for that frame.

These frames are still exported, preserving timeline integrity.

## Phase 3 — Blender Bridge

Import the manifest into Blender:

```bash
blender --python blender_scripts/blender_bridge.py -- \
  --manifest output_data/scene_manifest.json \
  --scale 1.0 \
  --rot-deg 0 \
  --x 0 --y 0 --z 0
```

What this does:

- Builds environment wall meshes from manifest.
- Creates an armature metarig from first valid keypoint frame.
- Animates root transform frame-by-frame.
- Applies `Global_Transform` adjustments while preserving original motion timing and physics.

## Output: JSON Manifest

Each frame contains:

- `translation` (root motion)
- `rotation_euler_xyz`
- `scale_xyz`
- `confidence`
- `lost_target`
- `keypoints_3d_m` (33 named landmarks)

Environment includes:

- Ground plane equation coefficients
- Wall meshes (vertices + faces)
