import json
import math
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

MANIFEST_PATH = "//output_data/manifest.json"


class Global_Transform:
    def __init__(self, driven_obj: bpy.types.Object, name: str = "Mocap_Global_CTRL"):
        self.ctrl = bpy.data.objects.new(name, None)
        self.ctrl.empty_display_type = "PLAIN_AXES"
        self.ctrl.empty_display_size = 0.6
        bpy.context.collection.objects.link(self.ctrl)
        driven_obj.parent = self.ctrl

    def set_scale(self, uniform_scale: float):
        self.ctrl.scale = Vector((uniform_scale, uniform_scale, uniform_scale))

    def set_heading_deg(self, yaw_deg: float):
        self.ctrl.rotation_mode = "XYZ"
        self.ctrl.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))

    def set_location(self, xyz):
        self.ctrl.location = Vector(xyz)


def _load_manifest(path: str):
    p = Path(bpy.path.abspath(path))
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clear_previous(prefixes=("Mocap_", "metarig")):
    for obj in list(bpy.data.objects):
        if any(obj.name.startswith(p) for p in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)


def _first_valid_frame(frames):
    for fr in frames:
        if not fr.get("lost_target", False) and fr.get("landmarks_world"):
            return fr
    raise RuntimeError("No valid frame with landmarks_world found.")


def _create_metarig_or_armature(name="Mocap_Armature"):
    # Try Rigify metarig for true metarig generation.
    try:
        if not bpy.context.preferences.addons.get("rigify"):
            bpy.ops.preferences.addon_enable(module="rigify")
        bpy.ops.object.armature_human_metarig_add()
        rig = bpy.context.active_object
        rig.name = "Mocap_Metarig"
        return rig
    except Exception:
        # Fallback for environments without Rigify.
        arm_data = bpy.data.armatures.new(name)
        arm_obj = bpy.data.objects.new(name, arm_data)
        bpy.context.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        return arm_obj


def _build_custom_skeleton_if_needed(arm_obj, frame_data, connections):
    if arm_obj.name.startswith("Mocap_Metarig"):
        bpy.ops.object.mode_set(mode="OBJECT")
        return

    id_to_vec = {lm["id"]: Vector(lm["xyz"]) for lm in frame_data["landmarks_world"]}
    ebones = arm_obj.data.edit_bones

    for a, b in connections:
        if a not in id_to_vec or b not in id_to_vec:
            continue
        head = id_to_vec[a]
        tail = id_to_vec[b]
        if (tail - head).length < 1e-4:
            tail = head + Vector((0.0, 0.015, 0.0))
        bn = f"Mocap_{a}_{b}"
        eb = ebones.new(bn)
        eb.head = head
        eb.tail = tail

    bpy.ops.object.mode_set(mode="POSE")


def _animate_custom_skeleton(arm_obj, manifest, connections):
    if arm_obj.name.startswith("Mocap_Metarig"):
        return

    fps = int(round(float(manifest.get("fps", 30.0))))
    bpy.context.scene.render.fps = fps

    pose_bones = arm_obj.pose.bones
    name_map = {(a, b): f"Mocap_{a}_{b}" for a, b in connections}

    for fr in manifest["frames"]:
        fi = int(fr["frame_index"]) + 1
        bpy.context.scene.frame_set(fi)

        if fr.get("lost_target", False) or not fr.get("landmarks_world"):
            continue

        id_to_vec = {lm["id"]: Vector(lm["xyz"]) for lm in fr["landmarks_world"]}
        root = fr.get("root_transform", {})

        arm_obj.location = Vector(root.get("translation", [0.0, 0.0, 0.0]))
        q = root.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        arm_obj.rotation_mode = "QUATERNION"
        arm_obj.rotation_quaternion = Quaternion((q[3], q[0], q[1], q[2]))
        arm_obj.keyframe_insert("location", frame=fi)
        arm_obj.keyframe_insert("rotation_quaternion", frame=fi)

        for edge in connections:
            bn = name_map.get(edge)
            if bn not in pose_bones:
                continue
            a, b = edge
            if a not in id_to_vec or b not in id_to_vec:
                continue

            pb = pose_bones[bn]
            target = id_to_vec[b] - id_to_vec[a]
            if target.length < 1e-7:
                continue

            rest = pb.bone.tail_local - pb.bone.head_local
            if rest.length < 1e-7:
                continue

            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rest.normalized().rotation_difference(target.normalized())
            pb.keyframe_insert("rotation_quaternion", frame=fi)


def _fit_metarig_pose_keys(metarig_obj, manifest):
    if not metarig_obj.name.startswith("Mocap_Metarig"):
        return

    # For universal compatibility, keyframe global transform channels on metarig object.
    fps = int(round(float(manifest.get("fps", 30.0))))
    bpy.context.scene.render.fps = fps

    for fr in manifest["frames"]:
        fi = int(fr["frame_index"]) + 1
        rt = fr.get("root_transform", {})
        metarig_obj.location = Vector(rt.get("translation", [0.0, 0.0, 0.0]))
        q = rt.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        metarig_obj.rotation_mode = "QUATERNION"
        metarig_obj.rotation_quaternion = Quaternion((q[3], q[0], q[1], q[2]))
        metarig_obj.keyframe_insert("location", frame=fi)
        metarig_obj.keyframe_insert("rotation_quaternion", frame=fi)


def _create_environment_mesh(manifest):
    if not manifest.get("frames"):
        return
    env = manifest["frames"][0].get("environment", {})
    mesh = env.get("mesh", {})
    verts = mesh.get("vertices", [])
    faces = mesh.get("faces", [])
    if not verts or not faces:
        return

    m = bpy.data.meshes.new("Mocap_Environment")
    obj = bpy.data.objects.new("Mocap_Environment", m)
    bpy.context.collection.objects.link(obj)
    m.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    m.update()


def build_scene_from_manifest(manifest_path=MANIFEST_PATH):
    manifest = _load_manifest(manifest_path)
    frames = manifest.get("frames", [])
    if not frames:
        raise RuntimeError("Manifest has no frames.")

    connections = manifest.get("landmark_topology", {}).get("connections", [])
    first_valid = _first_valid_frame(frames)

    _clear_previous()
    arm_obj = _create_metarig_or_armature()

    bpy.context.view_layer.objects.active = arm_obj
    if arm_obj.mode != "EDIT":
        bpy.ops.object.mode_set(mode="EDIT")

    _build_custom_skeleton_if_needed(arm_obj, first_valid, connections)
    _animate_custom_skeleton(arm_obj, manifest, connections)
    _fit_metarig_pose_keys(arm_obj, manifest)
    _create_environment_mesh(manifest)

    gt = Global_Transform(arm_obj)
    gt.set_scale(1.0)
    gt.set_heading_deg(0.0)
    gt.set_location((0.0, 0.0, 0.0))
    return arm_obj, gt.ctrl


if __name__ == "__main__":
    build_scene_from_manifest(MANIFEST_PATH)
    print("Mocap import complete.")
