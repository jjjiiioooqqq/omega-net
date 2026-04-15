# Mocap-less Scene Decomposer

This project converts a monocular 2D video into:

1. Per-frame tracked identity masks.
2. 3D skeletal landmarks (MediaPipe 33 points) in meter-scale world coordinates.
3. Environment geometry estimates (ground plane + wall planes) from monocular depth (Depth Anything v2 via PyTorch).
4. A JSON manifest that can be imported into Blender for automatic rig creation and animation.

## Folder Structure

```text
omega-net/
  input_video/
    shot.mp4
  output_data/
    manifest.json
    masks/
      frame_000000.png
      ...
  blender_scripts/
    blender_bridge.py
  scene_processor.py
```

## Phase 1 — Environment Setup

Create and activate a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scipy tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> If you are CPU-only, replace the PyTorch install line with:
>
> ```bash
> pip install torch torchvision torchaudio
> ```

Download a SAM checkpoint (example):

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Phase 2 — Run the Scene Processor

```bash
python scene_processor.py \
  --input_video input_video/shot.mp4 \
  --output_manifest output_data/manifest.json \
  --output_masks output_data/masks \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --depth_model_id depth-anything/Depth-Anything-V2-Small-hf
```

How target scanning works:

- The first frame opens in a window.
- Draw a rectangle around the target with the mouse.
- Press `ENTER` (or `SPACE`) to confirm.
- Press `c` to cancel and redraw.

## Phase 3 — Blender Bridge

In Blender:

1. Open **Scripting** workspace.
2. Open `blender_scripts/blender_bridge.py`.
3. Set `MANIFEST_PATH` in the script (or pass your own path in Blender text editor and run).
4. Run script.

The script creates:

- An armature from MediaPipe topology.
- Keyframed pose animation.
- A parent `Mocap_Global_CTRL` empty that applies global scale/rotation/location without breaking source motion.

## Lost Target Handling

If target tracking confidence collapses or the target exits/occludes:

- The frame is marked `lost_target = true` in JSON.
- The processor attempts automatic reacquisition using SAM in a search window around the last known position.
- If reacquisition fails, transforms are smoothly propagated from previous valid frame to avoid discontinuities.

## Maya

The JSON manifest is DCC-neutral and includes per-frame transform and skeleton data. You can import it with a Python script in Maya using `cmds.joint`, then keyframe transforms from the same manifest channels.
