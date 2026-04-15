import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

# Update this path before running.
MANIFEST_PATH = "//output_data/manifest.json"

# MediaPipe edges (same as exported topology if needed fallback)
MEDIAPIPE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


class Global_Transform:
    """
    Global transform wrapper to modify scale/heading/location while preserving
    underlying frame animation channels.
    """

    def __init__(self, armature_obj: bpy.types.Object, name: str = "Mocap_Global_CTRL"):
        self.ctrl = bpy.data.objects.new(name, None)
        self.ctrl.empty_display_type = 'PLAIN_AXES'
        self.ctrl.empty_display_size = 0.5
        bpy.context.collection.objects.link(self.ctrl)

        self.armature = armature_obj
        self.armature.parent = self.ctrl

    def set_scale(self, uniform_height_scale: float):
        self.ctrl.scale = Vector((uniform_height_scale, uniform_height_scale, uniform_height_scale))

    def set_heading_deg(self, yaw_degrees: float):
        self.ctrl.rotation_mode = 'XYZ'
        self.ctrl.rotation_euler[2] = math.radians(yaw_degrees)

    def set_location(self, xyz):
        self.ctrl.location = Vector(xyz)


def _clear_scene_objects(prefixes=("Mocap_",)):
    for obj in list(bpy.data.objects):
        if any(obj.name.startswith(p) for p in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)


def _load_manifest(path: str):
    p = Path(bpy.path.abspath(path))
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _create_armature(name="Mocap_Armature"):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    return arm_obj


def _build_bones_from_first_valid_frame(arm_obj, frame_data, connections):
    landmarks = frame_data["landmarks_world"]
    if not landmarks:
        raise RuntimeError("First valid frame has no landmarks_world.")

    id_to_vec = {lm["id"]: Vector(lm["xyz"]) for lm in landmarks}

    edit_bones = arm_obj.data.edit_bones
    created = {}

    for a, b in connections:
        if a not in id_to_vec or b not in id_to_vec:
            continue
        head = id_to_vec[a]
        tail = id_to_vec[b]
        if (tail - head).length < 1e-4:
            tail = head + Vector((0, 0.02, 0))

        bname = f"Mocap_{a}_{b}"
        eb = edit_bones.new(bname)
        eb.head = head
        eb.tail = tail
        created[(a, b)] = bname

    bpy.ops.object.mode_set(mode='POSE')
    return created


def _find_first_valid_frame(frames):
    for fr in frames:
        if not fr.get("lost_target", False) and fr.get("landmarks_world"):
            return fr
    raise RuntimeError("No valid tracked frame found in manifest.")


def _pose_bone_name_map(connections):
    mapping = {}
    for a, b in connections:
        mapping[(a, b)] = f"Mocap_{a}_{b}"
    return mapping


def _quat_from_segment(v_from: Vector, v_to: Vector):
    if v_from.length < 1e-8 or v_to.length < 1e-8:
        return Quaternion((1, 0, 0, 0))
    return v_from.normalized().rotation_difference(v_to.normalized())


def _animate_armature(arm_obj, manifest, connections):
    frames = manifest["frames"]
    fps = float(manifest.get("fps", 30.0))

    bpy.context.scene.render.fps = int(round(fps))
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    pose_bones = arm_obj.pose.bones
    bone_map = _pose_bone_name_map(connections)

    for fr in frames:
        fi = int(fr["frame_index"]) + 1
        bpy.context.scene.frame_set(fi)

        if fr.get("lost_target", False) or not fr.get("landmarks_world"):
            continue

        id_to_vec = {lm["id"]: Vector(lm["xyz"]) for lm in fr["landmarks_world"]}

        root = fr.get("root_transform", {})
        tr = Vector(root.get("translation", [0.0, 0.0, 0.0]))
        q = root.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        q_bl = Quaternion((q[3], q[0], q[1], q[2]))

        arm_obj.location = tr
        arm_obj.rotation_mode = 'QUATERNION'
        arm_obj.rotation_quaternion = q_bl
        arm_obj.keyframe_insert(data_path="location", frame=fi)
        arm_obj.keyframe_insert(data_path="rotation_quaternion", frame=fi)

        for edge in connections:
            a, b = edge
            bname = bone_map.get(edge)
            if bname not in pose_bones:
                continue
            if a not in id_to_vec or b not in id_to_vec:
                continue

            pb = pose_bones[bname]
            target_vec = id_to_vec[b] - id_to_vec[a]
            if target_vec.length < 1e-6:
                continue

            rest_head = pb.bone.head_local
            rest_tail = pb.bone.tail_local
            rest_vec = rest_tail - rest_head

            q_delta = _quat_from_segment(rest_vec, target_vec)
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = q_delta
            pb.keyframe_insert(data_path="rotation_quaternion", frame=fi)


def _create_environment_meshes(manifest):
    frames = manifest["frames"]
    if not frames:
        return

    env = frames[0].get("environment", {})
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

    connections = manifest.get("landmark_topology", {}).get("connections", MEDIAPIPE_CONNECTIONS)

    _clear_scene_objects()
    arm_obj = _create_armature()

    first_valid = _find_first_valid_frame(frames)
    _build_bones_from_first_valid_frame(arm_obj, first_valid, connections)
    _animate_armature(arm_obj, manifest, connections)
    _create_environment_meshes(manifest)

    gt = Global_Transform(arm_obj)
    gt.set_scale(1.0)
    gt.set_heading_deg(0.0)
    gt.set_location((0.0, 0.0, 0.0))

    return arm_obj, gt.ctrl


if __name__ == "__main__":
    build_scene_from_manifest(MANIFEST_PATH)
    print("Blender bridge import complete.")
