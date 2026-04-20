#!/usr/bin/env python3
"""Ω-Net Mocap-less Scene Processor.

Extracts a target identity from 2D video, estimates 3D pose and environment geometry,
and exports an animation manifest consumable by Blender.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise RuntimeError("mediapipe is required. Install dependencies first.") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required. Install dependencies first.") from exc

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "segment-anything is required. Install from git+https://github.com/facebookresearch/segment-anything.git"
    ) from exc


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class FrameTransform:
    frame_idx: int
    timestamp_sec: float
    lost_target: bool
    translation: List[float]
    rotation_euler_xyz: List[float]
    scale_xyz: List[float]
    confidence: float
    keypoints_3d_m: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Manifest:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    target_id: str
    model_info: Dict[str, str]
    environment: Dict[str, object]
    frames: List[FrameTransform]

    def to_json(self) -> Dict[str, object]:
        return {
            "video_path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "resolution": {"width": self.width, "height": self.height},
            "target_id": self.target_id,
            "model_info": self.model_info,
            "environment": self.environment,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "timestamp_sec": f.timestamp_sec,
                    "lost_target": f.lost_target,
                    "translation": f.translation,
                    "rotation_euler_xyz": f.rotation_euler_xyz,
                    "scale_xyz": f.scale_xyz,
                    "confidence": f.confidence,
                    "keypoints_3d_m": f.keypoints_3d_m,
                }
                for f in self.frames
            ],
        }


class DepthEstimator:
    """DepthAnything V2 wrapper with a robust fallback path."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model = None
        self.backend = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

            encoder = "vitb"
            self.model = DepthAnythingV2(encoder=encoder, features=128, out_channels=[96, 192, 384, 768])
            ckpt_env = os.environ.get("DEPTHANYTHINGV2_CKPT", "")
            if ckpt_env and Path(ckpt_env).exists():
                state = torch.load(ckpt_env, map_location=self.device)
                self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.backend = "depth_anything_v2"
            return
        except Exception:
            pass

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

            model_id = "LiheYoung/depth-anything-small-hf"
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.backend = "transformers"
            return
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize depth model. Install depth-anything-v2 or transformers fallback."
            ) from exc

    @torch.no_grad()
    def infer_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "depth_anything_v2":
            depth = self.model.infer_image(rgb)  # type: ignore[attr-defined]
            depth = np.asarray(depth, dtype=np.float32)
            return depth

        inputs = self.image_processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        return depth


class GroundPlaneEstimator:
    def __init__(self) -> None:
        self.plane: Optional[np.ndarray] = None  # ax + by + cz + d = 0

    def fit_plane_ransac(self, depth_map: np.ndarray, max_points: int = 5000, iterations: int = 200) -> np.ndarray:
        h, w = depth_map.shape
        ys, xs = np.where(depth_map > np.percentile(depth_map, 10))
        if len(xs) < 100:
            raise RuntimeError("Insufficient depth points for plane fit.")

        idx = np.random.choice(len(xs), size=min(max_points, len(xs)), replace=False)
        xs = xs[idx]
        ys = ys[idx]
        zs = depth_map[ys, xs]

        points = np.stack([xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32)], axis=1)
        best_plane = None
        best_inliers = -1
        thresh = np.std(zs) * 0.08 + 1e-3

        for _ in range(iterations):
            sample_idx = np.random.choice(points.shape[0], size=3, replace=False)
            p1, p2, p3 = points[sample_idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-8:
                continue
            normal = normal / n_norm
            d = -np.dot(normal, p1)
            dist = np.abs(points @ normal + d)
            inliers = int((dist < thresh).sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.concatenate([normal, np.array([d], dtype=np.float32)])

        if best_plane is None:
            raise RuntimeError("RANSAC failed to find a ground plane.")

        self.plane = best_plane.astype(np.float32)
        return self.plane

    def project_to_plane(self, p: np.ndarray) -> np.ndarray:
        if self.plane is None:
            return p
        n = self.plane[:3]
        d = self.plane[3]
        t = (np.dot(n, p) + d) / (np.dot(n, n) + 1e-8)
        return p - t * n


class IdentityScanner:
    def __init__(self, sam_checkpoint: Path, model_type: str = "vit_h", device: str = "cpu") -> None:
        if not sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.initialized = False
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None

    def init_from_first_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        cv2.namedWindow("Select Target", cv2.WINDOW_NORMAL)
        x, y, w, h = cv2.selectROI("Select Target", frame_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target")
        if w <= 0 or h <= 0:
            raise RuntimeError("No valid ROI selected for target identity.")

        self.predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        box = np.array([x, y, x + w, y + h])
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=4, mask=mask)
        if pts is None or len(pts) < 8:
            raise RuntimeError("Unable to detect enough target features for tracking.")

        self.prev_gray = gray
        self.prev_pts = pts
        self.prev_bbox = (x, y, w, h)
        self.initialized = True
        return mask

    def track(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool, float, Tuple[int, int, int, int]]:
        if not self.initialized:
            raise RuntimeError("IdentityScanner not initialized. Call init_from_first_frame first.")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        assert self.prev_gray is not None and self.prev_pts is not None and self.prev_bbox is not None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        good_old = self.prev_pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1] if next_pts is not None else np.empty((0, 1, 2), dtype=np.float32)

        lost_target = False
        confidence = 0.0

        if len(good_new) < 8:
            lost_target = True
            x, y, w, h = self.prev_bbox
        else:
            flow = (good_new - good_old).reshape(-1, 2)
            dx, dy = flow.mean(axis=0)
            x, y, w, h = self.prev_bbox
            x = int(np.clip(x + dx, 0, frame_bgr.shape[1] - 1))
            y = int(np.clip(y + dy, 0, frame_bgr.shape[0] - 1))
            x2 = int(np.clip(x + w, 1, frame_bgr.shape[1]))
            y2 = int(np.clip(y + h, 1, frame_bgr.shape[0]))
            w = max(1, x2 - x)
            h = max(1, y2 - y)
            confidence = float(min(1.0, len(good_new) / 80.0))

        self.predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        box = np.array([x, y, x + w, y + h])
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)
        confidence = max(confidence, float(scores[best_idx]))

        new_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=4, mask=mask)
        if new_pts is None or len(new_pts) < 6:
            lost_target = True
            new_pts = self.prev_pts

        self.prev_gray = gray
        self.prev_pts = new_pts
        self.prev_bbox = (x, y, w, h)
        return mask, lost_target, confidence, self.prev_bbox


class SceneProcessor:
    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]

    def __init__(
        self,
        video_path: Path,
        output_manifest: Path,
        sam_ckpt: Path,
        target_id: str,
        device: str = "cpu",
        sam_model_type: str = "vit_h",
    ) -> None:
        self.video_path = video_path
        self.output_manifest = output_manifest
        self.target_id = target_id
        self.device = device
        self.identity_scanner = IdentityScanner(sam_ckpt, model_type=sam_model_type, device=device)
        self.depth_estimator = DepthEstimator(device=device)
        self.ground_estimator = GroundPlaneEstimator()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _masked_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = cv2.bitwise_and(frame, frame, mask=(mask * 255).astype(np.uint8))
        return masked

    @staticmethod
    def _landmarks_to_world(
        landmarks, width: int, height: int, depth_map: np.ndarray, meter_scale: float = 2.0
    ) -> Dict[str, List[float]]:
        pts3d: Dict[str, List[float]] = {}
        for idx, lm in enumerate(landmarks):
            x_px = int(np.clip(lm.x * width, 0, width - 1))
            y_px = int(np.clip(lm.y * height, 0, height - 1))
            z_d = float(depth_map[y_px, x_px])
            x_n = (lm.x - 0.5) * meter_scale
            y_n = -(lm.y - 0.5) * meter_scale
            z_n = -z_d / (np.percentile(depth_map, 95) + 1e-6) * meter_scale
            pts3d[SceneProcessor.LANDMARK_NAMES[idx]] = [float(x_n), float(y_n), float(z_n)]
        return pts3d

    @staticmethod
    def _compute_root_transform(kps: Dict[str, List[float]]) -> Tuple[List[float], List[float], List[float]]:
        l_hip = np.array(kps.get("left_hip", [0.0, 0.0, 0.0]), dtype=np.float32)
        r_hip = np.array(kps.get("right_hip", [0.0, 0.0, 0.0]), dtype=np.float32)
        l_sh = np.array(kps.get("left_shoulder", [0.0, 0.0, 0.0]), dtype=np.float32)
        r_sh = np.array(kps.get("right_shoulder", [0.0, 0.0, 0.0]), dtype=np.float32)

        root = (l_hip + r_hip) * 0.5
        shoulder_center = (l_sh + r_sh) * 0.5
        forward = shoulder_center - root
        heading = math.atan2(forward[0], max(1e-6, forward[2]))
        rot = [0.0, float(heading), 0.0]

        shoulder_span = np.linalg.norm(l_sh - r_sh)
        hip_span = np.linalg.norm(l_hip - r_hip)
        s = max(0.2, (shoulder_span + hip_span) * 0.5)
        scale = [float(s), float(s), float(s)]
        return root.tolist(), rot, scale

    def process(self) -> Manifest:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            raise RuntimeError("Could not read first frame.")

        mask0 = self.identity_scanner.init_from_first_frame(frame0)
        depth0 = self.depth_estimator.infer_depth(frame0)
        plane = self.ground_estimator.fit_plane_ransac(depth0)

        wall_meshes = self._estimate_walls(depth0)
        env = {
            "ground_plane": {"a": float(plane[0]), "b": float(plane[1]), "c": float(plane[2]), "d": float(plane[3])},
            "walls": wall_meshes,
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_out: List[FrameTransform] = []

        for frame_idx in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx == 0:
                mask = mask0
                lost = False
                conf = 1.0
            else:
                mask, lost, conf, _ = self.identity_scanner.track(frame)

            masked = self._masked_frame(frame, mask)
            rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            if not result.pose_landmarks:
                frames_out.append(
                    FrameTransform(
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / fps,
                        lost_target=True,
                        translation=[0.0, 0.0, 0.0],
                        rotation_euler_xyz=[0.0, 0.0, 0.0],
                        scale_xyz=[1.0, 1.0, 1.0],
                        confidence=0.0,
                        keypoints_3d_m={},
                    )
                )
                continue

            depth = self.depth_estimator.infer_depth(frame)
            landmarks = result.pose_landmarks.landmark
            kps_3d = self._landmarks_to_world(landmarks, width, height, depth)

            for foot_key in ["left_heel", "right_heel", "left_foot_index", "right_foot_index", "left_ankle", "right_ankle"]:
                if foot_key in kps_3d:
                    p = np.array(kps_3d[foot_key], dtype=np.float32)
                    p_projected = self.ground_estimator.project_to_plane(p)
                    kps_3d[foot_key] = [float(v) for v in p_projected]

            translation, rotation, scale = self._compute_root_transform(kps_3d)

            frames_out.append(
                FrameTransform(
                    frame_idx=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    lost_target=bool(lost),
                    translation=translation,
                    rotation_euler_xyz=rotation,
                    scale_xyz=scale,
                    confidence=float(conf),
                    keypoints_3d_m=kps_3d,
                )
            )

        cap.release()

        return Manifest(
            video_path=str(self.video_path),
            fps=fps,
            frame_count=len(frames_out),
            width=width,
            height=height,
            target_id=self.target_id,
            model_info={
                "sam": "segment-anything",
                "pose": "mediapipe_pose_33",
                "depth": self.depth_estimator.backend or "unknown",
            },
            environment=env,
            frames=frames_out,
        )

    def _estimate_walls(self, depth_map: np.ndarray) -> List[Dict[str, object]]:
        h, w = depth_map.shape
        patches = {
            "left": depth_map[:, : max(8, w // 8)],
            "right": depth_map[:, max(0, w - w // 8):],
            "back": depth_map[: max(8, h // 8), :],
        }
        meshes = []
        for name, patch in patches.items():
            z = float(np.median(patch))
            if name == "left":
                verts = [[-1.5, -1.0, -z], [-1.5, 1.5, -z], [-1.5, 1.5, -z - 2.0], [-1.5, -1.0, -z - 2.0]]
            elif name == "right":
                verts = [[1.5, -1.0, -z], [1.5, 1.5, -z], [1.5, 1.5, -z - 2.0], [1.5, -1.0, -z - 2.0]]
            else:
                verts = [[-1.5, 1.5, -z], [1.5, 1.5, -z], [1.5, 1.5, -z - 2.0], [-1.5, 1.5, -z - 2.0]]
            meshes.append(
                {
                    "name": f"wall_{name}",
                    "vertices": verts,
                    "faces": [[0, 1, 2], [0, 2, 3]],
                }
            )
        return meshes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process 2D video into mocap-less 3D manifest.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video.")
    parser.add_argument("--sam-checkpoint", required=True, type=Path, help="Path to SAM checkpoint (.pth).")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest path.")
    parser.add_argument("--target-id", default="target_001", help="Target identity label.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    _safe_mkdir(args.output.parent)

    processor = SceneProcessor(
        video_path=args.video,
        output_manifest=args.output,
        sam_ckpt=args.sam_checkpoint,
        target_id=args.target_id,
        device=args.device,
        sam_model_type=args.sam_model_type,
    )

    manifest = processor.process()
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)

    print(f"Manifest exported: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
