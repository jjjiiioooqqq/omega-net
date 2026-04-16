# Universal Mocap-less 2D Video → 3D Scene Pipeline

This tool scans one person from regular 2D video and exports a structured 3D motion/scene manifest for Blender and Maya.

## Folder Layout

```text
omega-net/
  input_video/
    shot.mp4
  checkpoints/
    sam_vit_h_4b8939.pth
  output_data/
    manifest.json
    masks/
      frame_000000.png
      frame_000001.png
      ...
  blender_scripts/
    blender_bridge.py
    maya_bridge.py
  scene_processor.py
  requirements.txt
```

## Phase 1 — Environment + Dependencies

### Create clean environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Install all required libraries

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy scipy tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
```

CPU-only PyTorch alternative:

```bash
pip install torch torchvision torchaudio
```

### Download SAM checkpoint

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Phase 2 — Core Scene Processor

### Universal run (headless/any video orientation)

```bash
python scene_processor.py \
  --input_video input_video/shot.mp4 \
  --output_manifest output_data/manifest.json \
  --output_masks output_data/masks \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --depth_model_id depth-anything/Depth-Anything-V2-Small-hf \
  --auto_target \
  --no_gui
```

### Optional manual memory-bank target bbox

```bash
python scene_processor.py \
  --input_video input_video/shot.mp4 \
  --output_manifest output_data/manifest.json \
  --output_masks output_data/masks \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --init_bbox 640,180,300,620
```

### Optional interactive target scanning

```bash
python scene_processor.py \
  --input_video input_video/shot.mp4 \
  --output_manifest output_data/manifest.json \
  --output_masks output_data/masks \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth
```

This produces one JSON manifest containing:

- frame-by-frame target mask path
- target bbox + tracking confidence
- 3D landmarks in meters
- root rotation/translation/scale
- environment plane and mesh data
- lost-target state when subject is occluded/off-screen

## Phase 3 — Blender Bridge

1. Open Blender → **Scripting**.
2. Open `blender_scripts/blender_bridge.py`.
3. Set `MANIFEST_PATH`.
4. Run script.

Behavior:

- Builds a Rigify human metarig if available (automatic).
- Falls back to a generated custom armature if Rigify is unavailable.
- Applies animation from the manifest.
- Creates a `Global_Transform` control object for scale/heading/location.
- Imports environment mesh from manifest.

## Maya Bridge (optional)

1. Open Maya Script Editor (Python).
2. Load and run `blender_scripts/maya_bridge.py`.
3. Set `MANIFEST_PATH` if needed.

## Lost Target Handling

When target is off-screen or fully occluded:

- frame is flagged as `lost_target: true`
- SAM reacquisition is attempted around last known target position
- if reacquisition fails, previous stable world pose/root transform are propagated to avoid motion spikes
