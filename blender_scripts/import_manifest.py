"""Blender bridge script: load mocap manifest and animate a metarig-like armature."""

import json
from mathutils import Vector, Euler
import bpy


class Global_Transform:
    def __init__(self, scale=1.0, heading_deg=0.0, location=(0.0, 0.0, 0.0)):
        self.scale = scale
        self.heading_deg = heading_deg
        self.location = Vector(location)

    def apply(self, obj):
        obj.scale = (self.scale, self.scale, self.scale)
        obj.rotation_euler = Euler((0.0, 0.0, self.heading_deg * 3.1415926535 / 180.0), "XYZ")
        obj.location = self.location


def build_armature(name="TargetRig"):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bone = arm_data.edit_bones.new("root")
    bone.head = (0, 0, 0)
    bone.tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def animate_from_manifest(manifest, global_xform=None):
    rig = build_armature()
    if global_xform:
        global_xform.apply(rig)

    for frame in manifest.get("frames", []):
        idx = frame["frame_index"]
        bpy.context.scene.frame_set(idx)
        if frame.get("lost_target"):
            continue
        t = frame.get("translation_xyz_m", [0.0, 0.0, 0.0])
        rig.location = Vector((t[0], t[1], t[2]))
        rig.keyframe_insert(data_path="location", frame=idx)


def main(manifest_path):
    manifest = load_manifest(manifest_path)
    animate_from_manifest(manifest, Global_Transform(scale=1.0, heading_deg=0.0, location=(0, 0, 0)))


# Usage in Blender scripting tab:
# main('/absolute/path/to/output_data/scene_manifest.json')
