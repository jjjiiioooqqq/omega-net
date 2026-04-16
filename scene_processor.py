#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from segment_anything import SamPredictor, sam_model_registry


MP_POSE = mp.solutions.pose
LANDMARK_NAMES = [l.name for l in MP_POSE.PoseLandmark]
POSE_CONNECTIONS = sorted({tuple(sorted((a.value, b.value))) for a, b in MP_POSE.POSE_CONNECTIONS})


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class TrackState:
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float
    lost_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal mocap-less scene processor")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_masks", required=True)

    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--depth_model_id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--fov_degrees", type=float, default=60.0)
    parser.add_argument("--target_height_m", type=float, default=1.72)
    parser.add_argument("--max_lost_frames", type=int, default=45)
    parser.add_argument("--sam_refresh_stride", type=int, default=3, help="Refresh SAM mask every N frames for stability")

    parser.add_argument("--auto_target", action="store_true", help="Auto-detect person in frame 0 if no ROI provided")
    parser.add_argument("--init_bbox", type=str, default="", help="Manual init bbox as x,y,w,h")
    parser.add_argument("--no_gui", action="store_true", help="Disable interactive ROI selection")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_camera_intrinsics(width: int, height: int, fov_degrees: float) -> CameraIntrinsics:
    fov = math.radians(fov_degrees)
    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def pixel_to_camera_xyz(u: float, v: float, z: float, K: CameraIntrinsics) -> np.ndarray:
    return np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], dtype=np.float32)


def parse_bbox_text(text: str, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--init_bbox must be x,y,w,h")
    x, y, w, h = [int(p) for p in parts]
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    w = int(np.clip(w, 1, width - x))
    h = int(np.clip(h, 1, height - y))
    return x, y, w, h


def interactive_select_bbox(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    title = "Select Target (ENTER/SPACE confirm, C cancel)"
    bbox = cv2.selectROI(title, frame_bgr.copy(), fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        raise ValueError("Interactive target selection failed (empty ROI).")
    return x, y, w, h


def auto_detect_person_bbox(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with MP_POSE.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)
    if not res.pose_landmarks:
        raise RuntimeError("Auto target failed: no person detected in frame 0. Use --init_bbox or interactive selection.")

    h, w = frame_bgr.shape[:2]
    xs = []
    ys = []
    for lm in res.pose_landmarks.landmark:
        if lm.visibility < 0.3:
            continue
        xs.append(np.clip(lm.x * w, 0, w - 1))
        ys.append(np.clip(lm.y * h, 0, h - 1))

    if len(xs) < 6:
        raise RuntimeError("Auto target failed: insufficient visible landmarks.")

    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    pad_x = int(max(15, (x1 - x0) * 0.25))
    pad_y = int(max(15, (y1 - y0) * 0.20))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5 or len(ys) < 5:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def sam_mask_from_bbox(predictor: SamPredictor, frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
    x, y, w, h = bbox_xywh
    box = np.array([x, y, x + w, y + h], dtype=np.float32)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    idx = int(np.argmax(scores))
    return (masks[idx].astype(np.uint8) * 255), float(scores[idx])


def init_sam_predictor(model_type: str, checkpoint: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def make_tracker(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]):
    tracker = None
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        tracker = cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        tracker = cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        tracker = cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        tracker = cv2.TrackerKCF_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        tracker = cv2.legacy.TrackerMOSSE_create()
    else:
        raise RuntimeError("No OpenCV tracker available. Install opencv-contrib-python for better compatibility.")

    tracker.init(frame_bgr, tuple(map(float, bbox)))
    return tracker


def track_bbox(tracker, frame_bgr: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
    ok, b = tracker.update(frame_bgr)
    if not ok:
        return False, (0, 0, 0, 0)
    x, y, w, h = [int(v) for v in b]
    if w <= 0 or h <= 0:
        return False, (0, 0, 0, 0)
    return True, (x, y, w, h)


def reacquire_with_sam(predictor: SamPredictor, frame_bgr: np.ndarray, prev_bbox: Tuple[int, int, int, int], expand=2.0):
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = prev_bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    nw, nh = int(bw * expand), int(bh * expand)
    nx = max(0, int(cx - nw / 2.0))
    ny = max(0, int(cy - nh / 2.0))
    nw = min(w - nx, nw)
    nh = min(h - ny, nh)
    if nw < 5 or nh < 5:
        return None

    mask, score = sam_mask_from_bbox(predictor, frame_bgr, (nx, ny, nw, nh))
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    return mask, bbox, score


def load_depth_model(model_id: str, device: str):
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return proc, model


def estimate_depth_map(frame_bgr: np.ndarray, processor, model, device: str) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        d = model(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(d.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth = d.detach().cpu().numpy().astype(np.float32)
    depth = np.maximum(depth, 1e-6)
    depth = depth / (np.percentile(depth, 95) + 1e-6)
    depth *= 4.0
    return depth


def fit_plane_ransac(points: np.ndarray, iterations=250, threshold=0.03) -> Tuple[np.ndarray, float]:
    if len(points) < 3:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32), 0.0

    best_n = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    best_d = 0.0
    best_count = 0

    for _ in range(iterations):
        idx = np.random.choice(points.shape[0], 3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-7:
            continue
        n = n / nn
        d = -np.dot(n, p1)
        dist = np.abs(points @ n + d)
        count = int((dist < threshold).sum())
        if count > best_count:
            best_count = count
            best_n = n.astype(np.float32)
            best_d = float(d)

    if best_n[1] > 0:
        best_n *= -1
        best_d *= -1

    return best_n, best_d


def project_point_to_plane(pt: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return pt - (np.dot(n, pt) + d) * n


def camera_to_world_basis_from_ground(ground_n: np.ndarray) -> np.ndarray:
    y_axis = -ground_n / (np.linalg.norm(ground_n) + 1e-7)
    z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(y_axis, z_cam)
    if np.linalg.norm(x_axis) < 1e-5:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-7)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-7)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def build_environment(depth: np.ndarray, K: CameraIntrinsics) -> Dict:
    h, w = depth.shape
    pts = []
    for y in range(int(h * 0.55), h, 5):
        for x in range(0, w, 5):
            z = float(depth[y, x])
            if z <= 0:
                continue
            pts.append(pixel_to_camera_xyz(x, y, z, K))

    if len(pts) < 30:
        ground_n, ground_d = np.array([0.0, -1.0, 0.0], dtype=np.float32), 0.0
    else:
        arr = np.stack(pts, axis=0)
        ground_n, ground_d = fit_plane_ransac(arr)

    cam_to_world = camera_to_world_basis_from_ground(ground_n)

    left_depth = float(np.median(depth[int(h * 0.35):int(h * 0.7), :int(w * 0.15)]))
    right_depth = float(np.median(depth[int(h * 0.35):int(h * 0.7), int(w * 0.85):]))
    back_depth = float(np.median(depth[:int(h * 0.25), int(w * 0.25):int(w * 0.75)]))

    extent_x = 4.0
    extent_z = max(4.0, back_depth * 1.8)
    wall_h = 2.8
    vertices = [
        [-extent_x, 0.0, 0.0], [extent_x, 0.0, 0.0], [extent_x, 0.0, extent_z], [-extent_x, 0.0, extent_z],
        [-extent_x, wall_h, extent_z], [extent_x, wall_h, extent_z],
        [-extent_x, wall_h, 0.0], [extent_x, wall_h, 0.0],
    ]
    faces = [[0, 1, 2, 3], [3, 2, 5, 4], [0, 3, 4, 6], [1, 2, 5, 7]]

    return {
        "ground_plane": {"normal_camera": ground_n.tolist(), "d_camera": float(ground_d)},
        "camera_to_world_basis": cam_to_world.tolist(),
        "walls": {
            "left": {"depth_hint": left_depth},
            "right": {"depth_hint": right_depth},
            "back": {"depth_hint": back_depth},
        },
        "mesh": {"vertices": vertices, "faces": faces},
    }


def run_pose(frame_bgr: np.ndarray, mask: np.ndarray, pose_model) -> Optional[List[Dict]]:
    masked = frame_bgr.copy()
    masked[mask == 0] = 0
    rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    res = pose_model.process(rgb)
    if not res.pose_landmarks:
        return None

    out = []
    for i, lm in enumerate(res.pose_landmarks.landmark):
        out.append({
            "id": i,
            "name": LANDMARK_NAMES[i],
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })
    return out


def lift_landmarks_to_world(
    landmarks_2d: List[Dict], depth: np.ndarray, K: CameraIntrinsics, env: Dict, target_height_m: float
) -> List[Dict]:
    h, w = depth.shape
    cam_pts = []

    for lm in landmarks_2d:
        u = np.clip(lm["x"] * w, 0, w - 1)
        v = np.clip(lm["y"] * h, 0, h - 1)
        z = float(depth[int(v), int(u)])
        cam_pts.append(pixel_to_camera_xyz(u, v, z, K))

    cam_pts = np.stack(cam_pts, axis=0)

    nose = cam_pts[MP_POSE.PoseLandmark.NOSE.value]
    lank = cam_pts[MP_POSE.PoseLandmark.LEFT_ANKLE.value]
    rank = cam_pts[MP_POSE.PoseLandmark.RIGHT_ANKLE.value]
    est_h = np.linalg.norm(nose - (lank + rank) * 0.5)
    scale = target_height_m / max(est_h, 1e-4)
    cam_pts *= scale

    basis = np.array(env["camera_to_world_basis"], dtype=np.float32)
    world_pts = (basis.T @ cam_pts.T).T

    out = []
    for i, p in enumerate(world_pts):
        out.append({
            "id": int(i),
            "name": LANDMARK_NAMES[i],
            "xyz": p.tolist(),
            "visibility": landmarks_2d[i]["visibility"],
        })
    return out


def snap_feet_to_ground(world_landmarks: List[Dict], env: Dict) -> List[Dict]:
    n_cam = np.array(env["ground_plane"]["normal_camera"], dtype=np.float32)
    d_cam = float(env["ground_plane"]["d_camera"])
    basis = np.array(env["camera_to_world_basis"], dtype=np.float32)

    # Transform plane into world coordinates
    n_world = basis.T @ n_cam
    n_world = n_world / (np.linalg.norm(n_world) + 1e-7)
    if n_world[1] > 0:
        n_world *= -1
        d_cam *= -1

    idx_map = {lm["id"]: i for i, lm in enumerate(world_landmarks)}
    la = np.array(world_landmarks[idx_map[MP_POSE.PoseLandmark.LEFT_ANKLE.value]]["xyz"], dtype=np.float32)
    ra = np.array(world_landmarks[idx_map[MP_POSE.PoseLandmark.RIGHT_ANKLE.value]]["xyz"], dtype=np.float32)
    foot_center = (la + ra) * 0.5

    # approximate world plane d by mapping via inverse basis relationship
    d_world = d_cam
    snapped_center = project_point_to_plane(foot_center, n_world, d_world)
    offset = snapped_center - foot_center

    snapped = []
    for lm in world_landmarks:
        p = np.array(lm["xyz"], dtype=np.float32)
        q = p + offset
        x = dict(lm)
        x["xyz"] = q.tolist()
        snapped.append(x)
    return snapped


def compute_root_transform(world_landmarks: List[Dict], target_height_m: float) -> Dict:
    m = {lm["id"]: np.array(lm["xyz"], dtype=np.float32) for lm in world_landmarks}
    lhip = m[MP_POSE.PoseLandmark.LEFT_HIP.value]
    rhip = m[MP_POSE.PoseLandmark.RIGHT_HIP.value]
    lsho = m[MP_POSE.PoseLandmark.LEFT_SHOULDER.value]
    rsho = m[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]

    pelvis = 0.5 * (lhip + rhip)
    shoulder_mid = 0.5 * (lsho + rsho)

    right = rhip - lhip
    up = shoulder_mid - pelvis
    right = right / (np.linalg.norm(right) + 1e-7)
    up = up / (np.linalg.norm(up) + 1e-7)
    forward = np.cross(right, up)
    forward = forward / (np.linalg.norm(forward) + 1e-7)

    rotm = np.stack([right, up, forward], axis=1)
    quat_xyzw = R.from_matrix(rotm).as_quat().tolist()

    h = np.linalg.norm(m[MP_POSE.PoseLandmark.NOSE.value] - 0.5 * (
        m[MP_POSE.PoseLandmark.LEFT_ANKLE.value] + m[MP_POSE.PoseLandmark.RIGHT_ANKLE.value]
    ))
    s = h / max(target_height_m, 1e-4)

    return {
        "translation": pelvis.tolist(),
        "rotation_quat_xyzw": quat_xyzw,
        "scale": [s, s, s],
    }


def init_target(frame0: np.ndarray, args: argparse.Namespace, predictor: SamPredictor) -> TrackState:
    h, w = frame0.shape[:2]
    bbox = parse_bbox_text(args.init_bbox, w, h)
    if bbox is None:
        if args.auto_target or args.no_gui:
            bbox = auto_detect_person_bbox(frame0)
        else:
            bbox = interactive_select_bbox(frame0)

    mask, score = sam_mask_from_bbox(predictor, frame0, bbox)
    mb = bbox_from_mask(mask)
    if mb is not None:
        bbox = mb
    return TrackState(bbox=bbox, mask=mask, confidence=score, lost_frames=0)


def main() -> None:
    args = parse_args()

    ensure_parent(args.output_manifest)
    os.makedirs(args.output_masks, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read frame 0")

    intr = get_camera_intrinsics(width, height, args.fov_degrees)
    predictor = init_sam_predictor(args.sam_model_type, args.sam_checkpoint, args.device)
    depth_proc, depth_model = load_depth_model(args.depth_model_id, args.device)

    track = init_target(frame0, args, predictor)
    tracker = make_tracker(frame0, track.bbox)

    pose_model = MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    manifest = {
        "schema_version": "2.0",
        "pipeline": "omega-net-mocapless",
        "source_video": args.input_video,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "units": "meters",
        "landmark_topology": {"names": LANDMARK_NAMES, "connections": POSE_CONNECTIONS},
        "frames": [],
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_world = None
    prev_root = None

    for i in tqdm(range(total), desc="Processing"):
        ok, frame = cap.read()
        if not ok:
            break

        if i == 0:
            bbox = track.bbox
            mask = track.mask
            confidence = track.confidence
            ok_track = True
        else:
            ok_track, bbox = track_bbox(tracker, frame)
            if ok_track and (i % max(1, args.sam_refresh_stride) == 0):
                try:
                    mask, confidence = sam_mask_from_bbox(predictor, frame, bbox)
                    bb = bbox_from_mask(mask)
                    if bb is not None:
                        bbox = bb
                except Exception:
                    ok_track = False
                    mask = np.zeros((height, width), dtype=np.uint8)
                    confidence = 0.0
            elif ok_track:
                x, y, w, h = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                confidence = 0.35
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                confidence = 0.0

        lost_target = False
        if not ok_track or mask.sum() < 500:
            reacq = reacquire_with_sam(predictor, frame, track.bbox, expand=2.0)
            if reacq is not None:
                mask, bbox, confidence = reacq
                track.lost_frames = 0
                tracker = make_tracker(frame, bbox)
            else:
                lost_target = True
                track.lost_frames += 1
                bbox = track.bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                confidence = 0.0

        if track.lost_frames > args.max_lost_frames:
            lost_target = True

        track.bbox = bbox
        track.mask = mask
        track.confidence = confidence

        mask_path = os.path.join(args.output_masks, f"frame_{i:06d}.png")
        cv2.imwrite(mask_path, mask)

        depth = estimate_depth_map(frame, depth_proc, depth_model, args.device)
        env = build_environment(depth, intr)

        record = {
            "frame_index": i,
            "time_sec": i / fps,
            "target_bbox_xywh": [int(v) for v in bbox],
            "target_mask_path": mask_path,
            "tracking_confidence": float(confidence),
            "lost_target": bool(lost_target),
            "environment": env,
        }

        landmarks_2d = None
        if not lost_target:
            landmarks_2d = run_pose(frame, mask, pose_model)
            if landmarks_2d is None:
                lost_target = True
                record["lost_target"] = True

        if lost_target:
            if prev_world is not None and prev_root is not None:
                record["landmarks_world"] = prev_world
                record["root_transform"] = prev_root
            else:
                record["landmarks_world"] = []
                record["root_transform"] = {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            manifest["frames"].append(record)
            continue

        world = lift_landmarks_to_world(landmarks_2d, depth, intr, env, args.target_height_m)
        world = snap_feet_to_ground(world, env)
        root = compute_root_transform(world, args.target_height_m)

        record["landmarks_world"] = world
        record["root_transform"] = root
        manifest["frames"].append(record)

        prev_world = world
        prev_root = root

    cap.release()
    pose_model.close()

    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {args.output_manifest}")


if __name__ == "__main__":
    main()
