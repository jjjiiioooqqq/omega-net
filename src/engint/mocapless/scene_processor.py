from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .manifest import FramePose, SceneManifest


class LostTargetError(RuntimeError):
    """Raised when target cannot be tracked for too many consecutive frames."""


class SceneProcessor:
    """Mocap-less pipeline using SAM mask hook + MediaPipe + monocular-depth placeholder."""

    def __init__(self, max_lost_frames: int = 15) -> None:
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.max_lost_frames = max_lost_frames

    def _sam_mask_for_target(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        """Identity Scanner hook.

        NOTE: Integrates with SAM when model weights are available. This MVP uses bbox mask fallback.
        """
        x1, y1, x2, y2 = bbox_xyxy
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 1
        return mask

    def _estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """DepthAnything-v2 integration point; returns normalized pseudo depth for now."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth = cv2.GaussianBlur(gray, (9, 9), 0)
        depth = depth.astype(np.float32) / 255.0
        return depth

    def _fit_ground_plane(self, depth: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = depth.shape
        ys, xs = np.mgrid[int(0.6 * h) : h, 0:w]
        zs = depth[int(0.6 * h) : h, :]
        pts = np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[-1]
        d = -float(np.dot(normal, centroid))
        return normal, d

    def _landmarks_to_world(self, landmarks: list[Any], depth: np.ndarray, ground_plane: tuple[np.ndarray, float]) -> list[list[float]]:
        h, w = depth.shape
        normal, d = ground_plane
        out: list[list[float]] = []
        for lm in landmarks:
            x_pix = int(np.clip(lm.x * w, 0, w - 1))
            y_pix = int(np.clip(lm.y * h, 0, h - 1))
            z_m = float(depth[y_pix, x_pix]) * 3.0
            out.append([x_pix / w * 4.0, y_pix / h * 2.5, z_m])

        for foot_idx in (29, 30, 31, 32):
            if foot_idx < len(out):
                p = np.array(out[foot_idx])
                dist = (np.dot(normal, p) + d) / (np.linalg.norm(normal) + 1e-8)
                p_proj = p - dist * normal
                out[foot_idx] = [float(v) for v in p_proj]
        return out

    def process_video(self, input_video: Path, output_manifest: Path, target_bbox_xyxy: tuple[int, int, int, int]) -> SceneManifest:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        manifest = SceneManifest(source_video=str(input_video), fps=float(fps))
        lost_count = 0

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = self._sam_mask_for_target(frame, target_bbox_xyxy)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            depth = self._estimate_depth_map(masked)
            plane = self._fit_ground_plane(depth)

            result = self.pose.process(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks is None:
                lost_count += 1
                manifest.frames.append(
                    FramePose(
                        frame_index=idx,
                        keypoints_3d_m=[],
                        rotation_xyz=[0.0, 0.0, 0.0],
                        translation_xyz_m=[0.0, 0.0, 0.0],
                        scale_xyz=[1.0, 1.0, 1.0],
                        lost_target=True,
                    )
                )
                if lost_count > self.max_lost_frames:
                    raise LostTargetError("Target lost for too many consecutive frames")
                idx += 1
                continue

            lost_count = 0
            landmarks = result.pose_landmarks.landmark
            keypoints_3d = self._landmarks_to_world(landmarks, depth, plane)
            manifest.frames.append(
                FramePose(
                    frame_index=idx,
                    keypoints_3d_m=keypoints_3d,
                    rotation_xyz=[0.0, 0.0, 0.0],
                    translation_xyz_m=keypoints_3d[0] if keypoints_3d else [0.0, 0.0, 0.0],
                    scale_xyz=[1.0, 1.0, 1.0],
                    lost_target=False,
                )
            )
            idx += 1

        cap.release()
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest.open("w", encoding="utf-8") as fh:
            json.dump({
                "source_video": manifest.source_video,
                "fps": manifest.fps,
                "frames": [frame.__dict__ for frame in manifest.frames],
                "environment_mesh": {
                    "ground_plane": "derived_from_monocular_depth",
                    "walls": "placeholder_vertical_planes",
                },
            }, fh, indent=2)
        return manifest
