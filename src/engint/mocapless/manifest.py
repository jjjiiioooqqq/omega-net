from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FramePose:
    frame_index: int
    keypoints_3d_m: list[list[float]]
    rotation_xyz: list[float]
    translation_xyz_m: list[float]
    scale_xyz: list[float]
    lost_target: bool = False


@dataclass(slots=True)
class SceneManifest:
    source_video: str
    fps: float
    frames: list[FramePose] = field(default_factory=list)
    environment_mesh: dict[str, object] = field(default_factory=dict)
