"""Blender bridge for Ω-Net mocap-less manifest import.

Usage (inside Blender):
blender --python blender_scripts/blender_bridge.py -- --manifest output_data/scene_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
from mathutils import Euler, Vector


@dataclass
class Global_Transform:
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_euler_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply_location(self, v: Vector) -> Vector:
        return v + Vector(self.location)

    def apply_rotation(self, v: Vector) -> Vector:
        r = Euler(self.rotation_euler_xyz, "XYZ")
        return r.to_matrix() @ v

    def apply_scale(self, v: Vector) -> Vector:
        return Vector((v.x * self.scale[0], v.y * self.scale[1], v.z * self.scale[2]))

    def apply(self, v: Vector) -> Vector:
        return self.apply_location(self.apply_rotation(self.apply_scale(v)))


class ManifestImporter:
    def __init__(self, manifest_path: Path, global_transform: Global_Transform) -> None:
        self.manifest_path = manifest_path
        self.global_transform = global_transform
        self.data: Dict[str, object] = {}
        self.armature_obj: bpy.types.Object | None = None
        self.bone_map: Dict[str, str] = {}

    def load_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def build_environment(self) -> None:
        env = self.data.get("environment", {})
        walls = env.get("walls", []) if isinstance(env, dict) else []
        for wall in walls:
            name = wall["name"]
            verts = wall["vertices"]
            faces = wall["faces"]
            mesh = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            mesh.from_pydata(verts, [], faces)
            mesh.update()

    def create_metarig(self) -> None:
        if "frames" not in self.data or not self.data["frames"]:
            raise RuntimeError("Manifest has no frame data.")

        first_frame = self.data["frames"][0]
        keypoints = first_frame.get("keypoints_3d_m", {})
        if not keypoints:
            raise RuntimeError("First frame missing keypoints_3d_m.")

        arm_data = bpy.data.armatures.new("OmegaMetarig")
        arm_obj = bpy.data.objects.new("OmegaMetarig", arm_data)
        bpy.context.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        def add_bone(name: str, head: Vector, tail: Vector, parent: str | None = None) -> None:
            eb = arm_data.edit_bones.new(name)
            eb.head = head
            eb.tail = tail if (tail - head).length > 1e-4 else head + Vector((0, 0.1, 0))
            if parent:
                eb.parent = arm_data.edit_bones[parent]
            self.bone_map[name] = name

        kp = {k: Vector(v) for k, v in keypoints.items()}

        pelvis = (kp["left_hip"] + kp["right_hip"]) * 0.5
        chest = (kp["left_shoulder"] + kp["right_shoulder"]) * 0.5
        add_bone("root", pelvis, chest)
        add_bone("spine", pelvis, chest, parent="root")
        add_bone("neck", chest, kp["nose"], parent="spine")

        add_bone("thigh.L", kp["left_hip"], kp["left_knee"], parent="root")
        add_bone("shin.L", kp["left_knee"], kp["left_ankle"], parent="thigh.L")
        add_bone("foot.L", kp["left_ankle"], kp["left_foot_index"], parent="shin.L")

        add_bone("thigh.R", kp["right_hip"], kp["right_knee"], parent="root")
        add_bone("shin.R", kp["right_knee"], kp["right_ankle"], parent="thigh.R")
        add_bone("foot.R", kp["right_ankle"], kp["right_foot_index"], parent="shin.R")

        add_bone("upper_arm.L", kp["left_shoulder"], kp["left_elbow"], parent="spine")
        add_bone("forearm.L", kp["left_elbow"], kp["left_wrist"], parent="upper_arm.L")

        add_bone("upper_arm.R", kp["right_shoulder"], kp["right_elbow"], parent="spine")
        add_bone("forearm.R", kp["right_elbow"], kp["right_wrist"], parent="upper_arm.R")

        bpy.ops.object.mode_set(mode="OBJECT")
        self.armature_obj = arm_obj

    def animate(self) -> None:
        if self.armature_obj is None:
            raise RuntimeError("Metarig has not been created.")

        frames = self.data.get("frames", [])
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode="POSE")

        for frame_data in frames:
            frame_idx = int(frame_data["frame_idx"])
            bpy.context.scene.frame_set(frame_idx + 1)

            lost = frame_data.get("lost_target", False)
            if lost:
                continue

            root_translation = Vector(frame_data["translation"])
            root_rotation = Euler(frame_data["rotation_euler_xyz"], "XYZ")
            root_scale = frame_data.get("scale_xyz", [1.0, 1.0, 1.0])

            root_translation = self.global_transform.apply(root_translation)
            root_rotation = Euler(
                (
                    root_rotation.x + self.global_transform.rotation_euler_xyz[0],
                    root_rotation.y + self.global_transform.rotation_euler_xyz[1],
                    root_rotation.z + self.global_transform.rotation_euler_xyz[2],
                ),
                "XYZ",
            )

            pb = self.armature_obj.pose.bones
            if "root" in pb:
                pb["root"].location = root_translation
                pb["root"].rotation_mode = "XYZ"
                pb["root"].rotation_euler = root_rotation
                pb["root"].scale = (
                    root_scale[0] * self.global_transform.scale[0],
                    root_scale[1] * self.global_transform.scale[1],
                    root_scale[2] * self.global_transform.scale[2],
                )
                pb["root"].keyframe_insert(data_path="location")
                pb["root"].keyframe_insert(data_path="rotation_euler")
                pb["root"].keyframe_insert(data_path="scale")

        bpy.ops.object.mode_set(mode="OBJECT")


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--rot-deg", default=0.0, type=float, help="Global Yaw in degrees")
    parser.add_argument("--x", default=0.0, type=float)
    parser.add_argument("--y", default=0.0, type=float)
    parser.add_argument("--z", default=0.0, type=float)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    gtf = Global_Transform(
        scale=(args.scale, args.scale, args.scale),
        rotation_euler_xyz=(0.0, math.radians(args.rot_deg), 0.0),
        location=(args.x, args.y, args.z),
    )

    importer = ManifestImporter(args.manifest, gtf)
    importer.load_manifest()
    importer.build_environment()
    importer.create_metarig()
    importer.animate()
    print(f"Imported manifest: {args.manifest}")


if __name__ == "__main__":
    main()
