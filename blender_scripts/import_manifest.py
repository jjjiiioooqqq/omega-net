"""Blender bridge for mocap-less manifest import.

Usage in Blender:
blender --python blender_scripts/import_manifest.py -- --manifest /path/to/output_manifest.json
"""

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Vector, Euler

LANDMARK_NAMES = [f"LM_{i:02d}" for i in range(33)]
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


class Global_Transform:
    """Applies global post-transform while preserving per-frame human motion."""

    def __init__(self, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=1.0):
        self.location = Vector(location)
        self.rotation = Euler(rotation, "XYZ")
        self.scale = float(scale)

    def matrix(self) -> Matrix:
        t = Matrix.Translation(self.location)
        r = self.rotation.to_matrix().to_4x4()
        s = Matrix.Scale(self.scale, 4)
        return t @ r @ s


class ManifestImporter:
    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

    def build_metarig(self):
        bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
        arm_obj = bpy.context.object
        arm_obj.name = "MocapLess_Metarig"
        arm = arm_obj.data
        arm.name = "MocapLess_Metarig_Data"

        edit_bones = arm.edit_bones
        root = edit_bones[0]
        root.name = "ROOT"
        root.head = (0.0, 0.0, 0.0)
        root.tail = (0.0, 0.2, 0.0)

        created = {}
        for a, b in POSE_CONNECTIONS:
            bone_name = f"B_{a}_{b}"
            bone = edit_bones.new(bone_name)
            bone.parent = root
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.08, 0.0)
            created[(a, b)] = bone.name

        bpy.ops.object.mode_set(mode="POSE")
        return arm_obj, created

    def animate(self, arm_obj, created_bones, global_transform: Global_Transform):
        frames = self.data.get("frames", [])
        if not frames:
            raise RuntimeError("Manifest has no frame data.")

        for rec in frames:
            frame_idx = int(rec["frame_index"]) + 1
            bpy.context.scene.frame_set(frame_idx)

            if rec.get("target_lost", False):
                continue

            transform = rec["transform"]
            loc = Vector(transform["translation"])
            rot = Euler(transform["rotation_euler_xyz"], "XYZ")
            sc = transform["scale"][0]

            m_local = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Scale(sc, 4)
            m_global = global_transform.matrix() @ m_local

            arm_obj.matrix_world = m_global
            arm_obj.keyframe_insert(data_path="location", frame=frame_idx)
            arm_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            arm_obj.keyframe_insert(data_path="scale", frame=frame_idx)

            lms = rec["landmarks_world_m"]
            for (a, b), bone_name in created_bones.items():
                pa = lms[a]
                pb = lms[b]
                if pa is None or pb is None:
                    continue

                pbone = arm_obj.pose.bones[bone_name]
                head = Vector(pa)
                tail = Vector(pb)
                delta = tail - head
                if delta.length < 1e-5:
                    continue

                pbone.location = head
                pbone.scale = (1.0, delta.length, 1.0)
                pbone.keyframe_insert(data_path="location", frame=frame_idx)
                pbone.keyframe_insert(data_path="scale", frame=frame_idx)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = len(frames)

    def add_environment(self):
        env = self.data.get("environment", {})

        def create_plane(name, normal, d, size=8.0):
            bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
            obj = bpy.context.object
            obj.name = name
            n = Vector(normal).normalized()
            up = Vector((0, 0, 1))
            q = up.rotation_difference(n)
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = q
            obj.location = n * (-d)

        if "ground_plane" in env:
            gp = env["ground_plane"]
            create_plane("GroundPlane", gp["normal"], gp["d"], size=12.0)

        if "left_wall" in env:
            wp = env["left_wall"]
            create_plane("LeftWall", wp["normal"], wp["d"], size=12.0)

        if "right_wall" in env:
            wp = env["right_wall"]
            create_plane("RightWall", wp["normal"], wp["d"], size=12.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--location", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--rotation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    importer = ManifestImporter(args.manifest)
    arm_obj, bones = importer.build_metarig()
    gtx = Global_Transform(tuple(args.location), tuple(args.rotation), args.scale)
    importer.animate(arm_obj, bones, gtx)
    importer.add_environment()
    print(f"[OK] Imported manifest: {args.manifest}")


if __name__ == "__main__":
    main()
