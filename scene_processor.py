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

MP_CONNECTIONS = sorted({tuple(sorted((a.value, b.value))) for a, b in MP_POSE.POSE_CONNECTIONS})


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
    lost_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mocap-less video to 3D scene processor")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_masks", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--depth_model_id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fov_degrees", type=float, default=60.0)
    parser.add_argument("--target_height_m", type=float, default=1.72)
    parser.add_argument("--max_lost_frames", type=int, default=30)
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def select_target_bbox(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    clone = frame_bgr.copy()
    bbox = cv2.selectROI("Select Target (ENTER/SPACE=confirm, C=cancel)", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target (ENTER/SPACE=confirm, C=cancel)")
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        raise ValueError("No target selected. Please rerun and draw a valid ROI.")
    return x, y, w, h


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def get_camera_intrinsics(width: int, height: int, fov_degrees: float) -> CameraIntrinsics:
    fov = math.radians(fov_degrees)
    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def pixel_to_camera_xyz(u: float, v: float, z: float, K: CameraIntrinsics) -> np.ndarray:
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return np.array([x, y, z], dtype=np.float32)


def fit_plane_ransac(points: np.ndarray, iterations: int = 200, threshold: float = 0.03) -> Tuple[np.ndarray, float]:
    if points.shape[0] < 3:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32), 0.0

    best_inliers = 0
    best_plane = (np.array([0.0, -1.0, 0.0], dtype=np.float32), 0.0)

    for _ in range(iterations):
        idx = np.random.choice(points.shape[0], 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        distances = np.abs(points @ normal + d)
        inliers = int((distances < threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal.astype(np.float32), float(d))
    return best_plane


def project_point_to_plane(point: np.ndarray, plane_n: np.ndarray, plane_d: float) -> np.ndarray:
    dist = np.dot(plane_n, point) + plane_d
    return point - dist * plane_n


def build_environment_mesh(depth: np.ndarray, K: CameraIntrinsics) -> Dict:
    h, w = depth.shape
    samples = []
    for y in range(int(h * 0.55), h, 6):
        for x in range(0, w, 6):
            z = float(depth[y, x])
            if z <= 0:
                continue
            samples.append(pixel_to_camera_xyz(x, y, z, K))

    if len(samples) < 20:
        ground_n, ground_d = np.array([0.0, -1.0, 0.0], dtype=np.float32), 0.0
    else:
        pts = np.stack(samples, axis=0)
        ground_n, ground_d = fit_plane_ransac(pts)
        if ground_n[1] > 0:
            ground_n = -ground_n
            ground_d = -ground_d

    mid_y = int(h * 0.5)
    left_depth = np.median(depth[mid_y - 10:mid_y + 10, : int(w * 0.2)])
    right_depth = np.median(depth[mid_y - 10:mid_y + 10, int(w * 0.8):])
    back_depth = np.median(depth[: int(h * 0.35), int(w * 0.35): int(w * 0.65)])

    extent = 5.0
    verts = [
        [-extent, 0.0, 0.0],
        [extent, 0.0, 0.0],
        [extent, 0.0, 2 * extent],
        [-extent, 0.0, 2 * extent],
        [-extent, 2.5, 2 * extent],
        [extent, 2.5, 2 * extent],
    ]
    faces = [[0, 1, 2, 3], [3, 2, 5, 4], [0, 3, 4], [1, 2, 5]]

    return {
        "ground_plane": {"normal": ground_n.tolist(), "d": float(ground_d)},
        "walls": {
            "left": {"depth_hint": float(left_depth)},
            "right": {"depth_hint": float(right_depth)},
            "back": {"depth_hint": float(back_depth)},
        },
        "mesh": {"vertices": verts, "faces": faces},
    }


def load_depth_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def estimate_depth_map(frame_bgr: np.ndarray, processor, model, device: str) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        depth = model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.cpu().numpy().astype(np.float32)
    depth_np = np.maximum(depth_np, 1e-5)
    depth_np = depth_np / np.percentile(depth_np, 95)
    depth_np *= 4.0
    return depth_np


def run_pose(frame_bgr: np.ndarray, mask: np.ndarray, pose_model) -> Optional[List[Dict]]:
    masked = frame_bgr.copy()
    masked[mask == 0] = 0
    frame_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    result = pose_model.process(frame_rgb)
    if not result.pose_landmarks:
        return None
    landmarks = []
    for i, lm in enumerate(result.pose_landmarks.landmark):
        landmarks.append(
            {
                "id": i,
                "name": LANDMARK_NAMES[i],
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
        )
    return landmarks


def landmarks_to_world(
    landmarks: List[Dict],
    depth_map: np.ndarray,
    K: CameraIntrinsics,
    scale_hint_m: float,
) -> List[Dict]:
    h, w = depth_map.shape
    world = []
    pts = []

    for lm in landmarks:
        u = np.clip(lm["x"] * w, 0, w - 1)
        v = np.clip(lm["y"] * h, 0, h - 1)
        z = float(depth_map[int(v), int(u)])
        p = pixel_to_camera_xyz(u, v, z, K)
        pts.append(p)
        world.append({"id": lm["id"], "name": lm["name"], "xyz": p.tolist(), "visibility": lm["visibility"]})

    pts = np.stack(pts, axis=0)
    top_idx = MP_POSE.PoseLandmark.NOSE.value
    left_ankle_idx = MP_POSE.PoseLandmark.LEFT_ANKLE.value
    right_ankle_idx = MP_POSE.PoseLandmark.RIGHT_ANKLE.value
    ankle_y = 0.5 * (pts[left_ankle_idx, 1] + pts[right_ankle_idx, 1])
    est_height = abs(pts[top_idx, 1] - ankle_y)
    if est_height > 1e-4:
        scale = scale_hint_m / est_height
    else:
        scale = 1.0

    for i, item in enumerate(world):
        scaled = (pts[i] * scale).astype(np.float32)
        item["xyz"] = scaled.tolist()

    return world


def compute_root_transform(world_landmarks: List[Dict]) -> Dict:
    id_to_p = {x["id"]: np.array(x["xyz"], dtype=np.float32) for x in world_landmarks}
    lhip = id_to_p[MP_POSE.PoseLandmark.LEFT_HIP.value]
    rhip = id_to_p[MP_POSE.PoseLandmark.RIGHT_HIP.value]
    lsho = id_to_p[MP_POSE.PoseLandmark.LEFT_SHOULDER.value]
    rsho = id_to_p[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]

    pelvis = 0.5 * (lhip + rhip)
    shoulder_mid = 0.5 * (lsho + rsho)

    right_vec = (rhip - lhip)
    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)

    up_vec = (shoulder_mid - pelvis)
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)

    forward = np.cross(right_vec, up_vec)
    forward = forward / (np.linalg.norm(forward) + 1e-6)

    rot_m = np.stack([right_vec, up_vec, forward], axis=1)
    rot = R.from_matrix(rot_m)
    quat = rot.as_quat()

    body_height = np.linalg.norm(
        id_to_p[MP_POSE.PoseLandmark.NOSE.value] -
        0.5 * (id_to_p[MP_POSE.PoseLandmark.LEFT_ANKLE.value] + id_to_p[MP_POSE.PoseLandmark.RIGHT_ANKLE.value])
    )

    return {
        "translation": pelvis.tolist(),
        "rotation_quat_xyzw": quat.tolist(),
        "scale": [1.0, body_height / 1.72, 1.0],
    }


def snap_feet_to_ground(world_landmarks: List[Dict], ground_n: np.ndarray, ground_d: float) -> List[Dict]:
    ids = [MP_POSE.PoseLandmark.LEFT_ANKLE.value, MP_POSE.PoseLandmark.RIGHT_ANKLE.value]
    snapped = [dict(x) for x in world_landmarks]
    id_to_idx = {x["id"]: i for i, x in enumerate(snapped)}

    feet_points = []
    for fid in ids:
        p = np.array(snapped[id_to_idx[fid]]["xyz"], dtype=np.float32)
        feet_points.append(p)

    avg_foot = np.mean(feet_points, axis=0)
    snapped_avg = project_point_to_plane(avg_foot, ground_n, ground_d)
    offset = snapped_avg - avg_foot

    for i in range(len(snapped)):
        p = np.array(snapped[i]["xyz"], dtype=np.float32)
        snapped[i]["xyz"] = (p + offset).tolist()
    return snapped


def init_sam_predictor(model_type: str, checkpoint: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def sam_mask_from_bbox(predictor: SamPredictor, frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    box = np.array([x, y, x + w, y + h], dtype=np.float32)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    return best


def track_with_csrt(tracker, frame_bgr: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
    ok, bbox = tracker.update(frame_bgr)
    if not ok:
        return False, (0, 0, 0, 0)
    x, y, w, h = [int(v) for v in bbox]
    return True, (x, y, w, h)


def make_tracker(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_bgr, tuple(map(float, bbox)))
    return tracker


def reacquire_with_sam(
    predictor: SamPredictor,
    frame_bgr: np.ndarray,
    prev_bbox: Tuple[int, int, int, int],
    expand_ratio: float = 1.6,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = prev_bbox
    cx, cy = x + bw / 2, y + bh / 2
    nw, nh = int(bw * expand_ratio), int(bh * expand_ratio)
    nx = max(0, int(cx - nw / 2))
    ny = max(0, int(cy - nh / 2))
    nw = min(w - nx, nw)
    nh = min(h - ny, nh)
    if nw <= 2 or nh <= 2:
        return None

    mask = sam_mask_from_bbox(predictor, frame_bgr, (nx, ny, nw, nh))
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    return mask, bbox


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_manifest)
    os.makedirs(args.output_masks, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Unable to read first frame from video.")

    K = get_camera_intrinsics(width, height, args.fov_degrees)
    predictor = init_sam_predictor(args.sam_model_type, args.sam_checkpoint, args.device)
    depth_processor, depth_model = load_depth_model(args.depth_model_id, args.device)

    init_bbox = select_target_bbox(first_frame)
    init_mask = sam_mask_from_bbox(predictor, first_frame, init_bbox)
    track_state = TrackState(bbox=init_bbox, mask=init_mask)
    tracker = make_tracker(first_frame, init_bbox)

    pose_model = MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    manifest = {
        "schema_version": "1.0",
        "source_video": args.input_video,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "landmark_topology": {
            "names": LANDMARK_NAMES,
            "connections": MP_CONNECTIONS,
        },
        "frames": [],
    }

    prev_good_world = None
    prev_good_transform = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx == 0:
            mask = track_state.mask
            bbox = track_state.bbox
            tracking_ok = True
        else:
            tracking_ok, bbox = track_with_csrt(tracker, frame)
            if tracking_ok:
                try:
                    mask = sam_mask_from_bbox(predictor, frame, bbox)
                    mb = bbox_from_mask(mask)
                    if mb is not None:
                        bbox = mb
                except Exception:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    tracking_ok = False
            else:
                mask = np.zeros((height, width), dtype=np.uint8)

        lost_target = False
        if not tracking_ok or mask.sum() < 500:
            reacq = reacquire_with_sam(predictor, frame, track_state.bbox)
            if reacq is None:
                track_state.lost_frames += 1
                lost_target = True
                mask = np.zeros((height, width), dtype=np.uint8)
                bbox = track_state.bbox
            else:
                mask, bbox = reacq
                track_state.lost_frames = 0
                tracker = make_tracker(frame, bbox)

        if track_state.lost_frames > args.max_lost_frames:
            lost_target = True

        track_state.bbox = bbox
        track_state.mask = mask

        mask_path = os.path.join(args.output_masks, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(mask_path, mask)

        depth_map = estimate_depth_map(frame, depth_processor, depth_model, args.device)
        env = build_environment_mesh(depth_map, K)
        ground_n = np.array(env["ground_plane"]["normal"], dtype=np.float32)
        ground_d = float(env["ground_plane"]["d"])

        frame_record = {
            "frame_index": frame_idx,
            "time_sec": frame_idx / fps,
            "target_bbox_xywh": list(map(int, bbox)),
            "target_mask_path": mask_path,
            "lost_target": bool(lost_target),
            "environment": env,
        }

        if not lost_target:
            landmarks_2d = run_pose(frame, mask, pose_model)
            if landmarks_2d is None:
                lost_target = True
                frame_record["lost_target"] = True

        if lost_target:
            if prev_good_world is not None and prev_good_transform is not None:
                frame_record["landmarks_world"] = prev_good_world
                frame_record["root_transform"] = prev_good_transform
            else:
                frame_record["landmarks_world"] = []
                frame_record["root_transform"] = {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            manifest["frames"].append(frame_record)
            continue

        world = landmarks_to_world(landmarks_2d, depth_map, K, args.target_height_m)
        world = snap_feet_to_ground(world, ground_n, ground_d)
        root_transform = compute_root_transform(world)

        frame_record["landmarks_world"] = world
        frame_record["root_transform"] = root_transform

        prev_good_world = world
        prev_good_transform = root_transform
        manifest["frames"].append(frame_record)

    cap.release()
    pose_model.close()

    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
