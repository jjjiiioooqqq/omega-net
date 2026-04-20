#!/usr/bin/env python3
"""Scene processor for mocap-less extraction.

Pipeline:
1) User picks target on frame 0.
2) SAM generates binary identity mask.
3) OpenCV tracker + SAM refinement tracks identity.
4) MediaPipe pose extracts 33 landmarks.
5) DepthAnything v2 estimates scene depth.
6) Ground/wall planes are estimated and feet are snapped to ground.
7) All per-frame data exported to JSON manifest for DCC ingestion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:
    raise ImportError(
        "segment-anything is required. Install via `pip install segment-anything`."
    ) from exc


# MediaPipe pose landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_FOOT = 31
RIGHT_FOOT = 32
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
FOOT_INDICES = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT]


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_sec: float
    target_lost: bool
    bbox_xywh: Optional[List[float]]
    transform: Dict[str, List[float]]
    landmarks_world_m: List[Optional[List[float]]]


class IdentityScanner:
    """Tracks identity using OpenCV tracker and SAM binary mask refinement."""

    def __init__(self, sam_checkpoint: str, sam_model_type: str, device: str = "cpu") -> None:
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")

        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam_model.to(device=device)
        self.predictor = SamPredictor(sam_model)
        self.device = device
        self.tracker = cv2.TrackerCSRT_create()
        self.last_bbox = None

    @staticmethod
    def _sanitize_box(box_xywh: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
        x, y, w, h = box_xywh
        x = int(max(0, min(width - 2, x)))
        y = int(max(0, min(height - 2, y)))
        w = int(max(2, min(width - x, w)))
        h = int(max(2, min(height - y, h)))
        return x, y, w, h

    def initialize_target(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """User selects initial ROI. SAM generates initial binary mask."""
        init_box = cv2.selectROI("Select Target", frame_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Target")

        if init_box == (0, 0, 0, 0):
            raise ValueError("No ROI selected. Please re-run and select a valid target.")

        h, w = frame_bgr.shape[:2]
        init_box = self._sanitize_box(init_box, w, h)
        self.tracker.init(frame_bgr, init_box)
        self.last_bbox = init_box

        mask = self._run_sam_mask(frame_bgr, init_box)
        return mask, init_box

    def _run_sam_mask(self, frame_bgr: np.ndarray, box_xywh: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box_xywh
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)
        masks, scores, _ = self.predictor.predict(
            box=box_xyxy[None, :],
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)].astype(np.uint8)
        return best_mask

    def track(self, frame_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        ok, tracked_box = self.tracker.update(frame_bgr)
        if not ok:
            return False, None, None

        h, w = frame_bgr.shape[:2]
        tracked_box = self._sanitize_box(tuple(map(float, tracked_box)), w, h)
        mask = self._run_sam_mask(frame_bgr, tracked_box)

        area = int(mask.sum())
        if area < 150:
            return False, None, None

        self.last_bbox = tracked_box
        return True, mask, tracked_box


class DepthEstimator:
    """DepthAnything v2 wrapper via transformers."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cpu") -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-6)
        # Relative depth to pseudo-meters for DCC-friendly scale.
        depth_m = 0.5 + (1.0 - depth) * 4.5
        return depth_m


class SceneProcessor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        self.identity = IdentityScanner(args.sam_checkpoint, args.sam_model_type, self.device)
        self.depth = DepthEstimator(args.depth_model, self.device)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def depth_to_world(depth_m: np.ndarray, fov_deg: float = 60.0) -> np.ndarray:
        h, w = depth_m.shape
        f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx, cy = w / 2.0, h / 2.0

        ys, xs = np.indices((h, w))
        z = depth_m
        x = (xs - cx) * z / f
        y = (ys - cy) * z / f
        points = np.stack([x, y, z], axis=-1)
        return points

    @staticmethod
    def fit_plane_ransac(points: np.ndarray, iterations: int = 120, threshold: float = 0.02) -> Tuple[np.ndarray, float]:
        if len(points) < 3:
            raise ValueError("Not enough points for plane fit.")

        best_inliers = -1
        best_model = None
        rng = np.random.default_rng(42)

        for _ in range(iterations):
            sample = points[rng.choice(points.shape[0], size=3, replace=False)]
            v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            d = -np.dot(normal, sample[0])
            dist = np.abs(np.dot(points, normal) + d)
            inliers = int((dist < threshold).sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_model = (normal, d)

        if best_model is None:
            raise RuntimeError("Plane fitting failed.")
        return best_model[0], float(best_model[1])

    @staticmethod
    def project_point_to_plane(point: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
        signed = np.dot(normal, point) + d
        return point - signed * normal

    def _extract_landmarks(self, frame_bgr: np.ndarray, mask: np.ndarray, world_grid: np.ndarray) -> List[Optional[np.ndarray]]:
        masked = frame_bgr.copy()
        masked[mask == 0] = 0
        pose_result = self.pose.process(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))

        if not pose_result.pose_landmarks:
            return [None] * 33

        h, w = frame_bgr.shape[:2]
        landmarks_world: List[Optional[np.ndarray]] = []

        for lm in pose_result.pose_landmarks.landmark:
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            p3 = world_grid[py, px].copy()
            p3[1] *= -1.0
            landmarks_world.append(p3)

        return landmarks_world

    @staticmethod
    def _compute_transform(landmarks_world: List[Optional[np.ndarray]]) -> Dict[str, List[float]]:
        def fetch(i: int) -> Optional[np.ndarray]:
            return landmarks_world[i] if i < len(landmarks_world) else None

        lhip, rhip = fetch(LEFT_HIP), fetch(RIGHT_HIP)
        lsho, rsho = fetch(LEFT_SHOULDER), fetch(RIGHT_SHOULDER)

        if any(v is None for v in [lhip, rhip, lsho, rsho]):
            return {
                "translation": [0.0, 0.0, 0.0],
                "rotation_euler_xyz": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }

        pelvis = 0.5 * (lhip + rhip)
        shoulder_vec = rsho - lsho
        heading = math.atan2(float(shoulder_vec[0]), float(shoulder_vec[2] + 1e-6))

        shoulder_width = float(np.linalg.norm(shoulder_vec))
        scale = max(0.2, shoulder_width / 0.36)

        quat = R.from_euler("y", heading).as_euler("xyz", degrees=False)
        return {
            "translation": pelvis.tolist(),
            "rotation_euler_xyz": [float(quat[0]), float(quat[1]), float(quat[2])],
            "scale": [scale, scale, scale],
        }

    def run(self) -> Path:
        video_path = Path(self.args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Unable to read first frame.")

        init_mask, init_box = self.identity.initialize_target(frame0)
        h, w = frame0.shape[:2]

        records: List[FrameRecord] = []
        lost_counter = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for idx in tqdm(range(total_frames if total_frames > 0 else 10**9), desc="Processing"):
            ok, frame = cap.read()
            if not ok:
                break

            depth_m = self.depth.predict_depth(frame)
            world_grid = self.depth_to_world(depth_m, self.args.fov_deg)

            if idx == 0:
                mask = init_mask
                bbox = init_box
                target_ok = True
            else:
                target_ok, mask, bbox = self.identity.track(frame)

            if not target_ok or mask is None:
                lost_counter += 1
                records.append(
                    FrameRecord(
                        frame_index=idx,
                        timestamp_sec=idx / fps,
                        target_lost=True,
                        bbox_xywh=None,
                        transform={
                            "translation": [0.0, 0.0, 0.0],
                            "rotation_euler_xyz": [0.0, 0.0, 0.0],
                            "scale": [1.0, 1.0, 1.0],
                        },
                        landmarks_world_m=[None] * 33,
                    )
                )

                if lost_counter > self.args.max_lost_frames:
                    print(
                        f"[WARN] Lost target for {lost_counter} consecutive frames. "
                        "Stopping early to avoid invalid output."
                    )
                    break
                continue

            lost_counter = 0
            landmarks_world = self._extract_landmarks(frame, mask, world_grid)

            ys, xs = np.where(mask > 0)
            sample_points = world_grid[ys, xs]
            lower_points = sample_points[sample_points[:, 1] > np.percentile(sample_points[:, 1], 80)]
            if len(lower_points) < 100:
                lower_points = sample_points

            ground_normal, ground_d = self.fit_plane_ransac(lower_points)

            left_strip = world_grid[:, : max(5, w // 12), :].reshape(-1, 3)
            right_strip = world_grid[:, -max(5, w // 12) :, :].reshape(-1, 3)
            left_n, left_d = self.fit_plane_ransac(left_strip)
            right_n, right_d = self.fit_plane_ransac(right_strip)

            for foot_idx in FOOT_INDICES:
                if landmarks_world[foot_idx] is not None:
                    landmarks_world[foot_idx] = self.project_point_to_plane(
                        landmarks_world[foot_idx], ground_normal, ground_d
                    )

            transform = self._compute_transform(landmarks_world)

            records.append(
                FrameRecord(
                    frame_index=idx,
                    timestamp_sec=idx / fps,
                    target_lost=False,
                    bbox_xywh=list(map(float, bbox)),
                    transform=transform,
                    landmarks_world_m=[p.tolist() if p is not None else None for p in landmarks_world],
                )
            )

            env = {
                "ground_plane": {"normal": ground_normal.tolist(), "d": float(ground_d)},
                "left_wall": {"normal": left_n.tolist(), "d": float(left_d)},
                "right_wall": {"normal": right_n.tolist(), "d": float(right_d)},
            }

        cap.release()

        manifest = {
            "version": "1.0",
            "source_video": str(video_path),
            "fps": fps,
            "frame_width": w,
            "frame_height": h,
            "coordinate_system": {
                "units": "meters",
                "up_axis": "Y",
                "forward_axis": "-Z",
            },
            "environment": env,
            "frames": [r.__dict__ for r in records],
        }

        manifest_path = out_dir / f"{video_path.stem}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[OK] Manifest saved: {manifest_path}")
        return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mocap-less scene processor")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="output_data", help="Directory for output manifest")
    parser.add_argument("--sam-checkpoint", required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM encoder model type",
    )
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="DepthAnything v2 model id",
    )
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Estimated camera horizontal FOV")
    parser.add_argument("--max-lost-frames", type=int, default=45, help="Stop after this many consecutive lost frames")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = SceneProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()
