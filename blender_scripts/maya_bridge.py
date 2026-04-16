"""
Run inside Autodesk Maya Python tab.
Builds a joint hierarchy from manifest and keys world transforms.
"""
import json

import maya.cmds as cmds

MANIFEST_PATH = "output_data/manifest.json"


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_previous(prefix="Mocap_"):
    objs = cmds.ls(prefix + "*", long=True) or []
    if objs:
        cmds.delete(objs)


def build_joints(first_frame, connections):
    landmarks = {lm["id"]: lm for lm in first_frame["landmarks_world"]}
    joint_map = {}

    for a, b in connections:
        if a not in landmarks or b not in landmarks:
            continue
        name = f"Mocap_{a}_{b}_JNT"
        pos = landmarks[a]["xyz"]
        j = cmds.joint(name=name, p=(pos[0], pos[1], pos[2]))
        joint_map[(a, b)] = j
        cmds.select(clear=True)

    return joint_map


def animate_joints(manifest, joint_map):
    fps = int(round(float(manifest.get("fps", 30.0))))
    cmds.currentUnit(time=f"{fps}fps")

    for fr in manifest["frames"]:
        frame_num = int(fr["frame_index"]) + 1
        if fr.get("lost_target", False) or not fr.get("landmarks_world"):
            continue

        lm_map = {lm["id"]: lm for lm in fr["landmarks_world"]}
        for edge, jnt in joint_map.items():
            a, _ = edge
            if a not in lm_map:
                continue
            p = lm_map[a]["xyz"]
            cmds.setKeyframe(jnt, attribute="translateX", t=frame_num, v=p[0])
            cmds.setKeyframe(jnt, attribute="translateY", t=frame_num, v=p[1])
            cmds.setKeyframe(jnt, attribute="translateZ", t=frame_num, v=p[2])


def main(manifest_path=MANIFEST_PATH):
    manifest = load_manifest(manifest_path)
    frames = manifest.get("frames", [])
    if not frames:
        raise RuntimeError("Manifest has no frames")

    first_valid = None
    for fr in frames:
        if not fr.get("lost_target", False) and fr.get("landmarks_world"):
            first_valid = fr
            break
    if first_valid is None:
        raise RuntimeError("No valid tracked frame")

    connections = manifest.get("landmark_topology", {}).get("connections", [])
    clear_previous()
    joint_map = build_joints(first_valid, connections)
    animate_joints(manifest, joint_map)
    print("Maya bridge import complete.")


if __name__ == "__main__":
    main()
