"""Retarget a MimicKit MJCF motion to the local Unitree G1 with GMR.

This intentionally uses only the local MimicKit source character and the
checked-out GMR utility.  It does not depend on a task-specific simulator.
"""

import argparse
import json
import pickle
import pathlib
import shutil
import sys
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _load_motion(path):
    with open(path, "rb") as stream:
        data = pickle.load(stream)
    frames = np.asarray(data["frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] < 7:
        raise ValueError("MimicKit motion must contain a 2-D frames array")
    return data, frames


def _qpos_from_motion(frame):
    root_quat_xyzw = Rotation.from_rotvec(frame[3:6]).as_quat()
    root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate((frame[:3], root_quat_wxyz, frame[6:])).astype(np.float64)


def _human_frames(source_xml, frames):
    model = mujoco.MjModel.from_xml_path(str(source_xml))
    data = mujoco.MjData(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(model.nbody)]
    body_ids = {name: i for i, name in enumerate(body_names)}
    mapping = {
        "pelvis": "pelvis",
        "spine3": "torso",
        "left_hip": "left_thigh",
        "right_hip": "right_thigh",
        "left_knee": "left_shin",
        "right_knee": "right_shin",
        "left_foot": "left_foot",
        "right_foot": "right_foot",
        "left_shoulder": "left_upper_arm",
        "right_shoulder": "right_upper_arm",
        "left_elbow": "left_lower_arm",
        "right_elbow": "right_lower_arm",
        "left_wrist": "left_hand",
        "right_wrist": "right_hand",
    }
    missing = sorted(set(mapping.values()) - set(body_ids))
    if missing:
        raise ValueError("source character lacks bodies: {}".format(missing))

    out = []
    for frame in frames:
        data.qpos[:] = _qpos_from_motion(frame)
        mujoco.mj_forward(model, data)
        out.append({
            target: [data.xpos[body_ids[source]].copy(),
                     data.xquat[body_ids[source]].copy()]
            for target, source in mapping.items()
        })
    return out


def _patch_g1_ik_config(path):
    config = json.load(open(path))
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config[table_name]
        for old, new in (("left_toe_link", "left_ankle_roll_link"),
                         ("right_toe_link", "right_ankle_roll_link")):
            if old in table:
                table[new] = table.pop(old)
    return config


def _upsample_qpos(qpos, source_fps, target_fps):
    """Interpolate retargeted poses when the requested output FPS is higher."""
    if target_fps == source_fps:
        return qpos
    if target_fps % source_fps != 0:
        raise ValueError("target fps must be an integer multiple of source fps")
    ratio = target_fps // source_fps
    old_t = np.arange(qpos.shape[0], dtype=np.float64)
    new_t = np.arange((qpos.shape[0] - 1) * ratio + 1,
                      dtype=np.float64) / ratio
    out = np.empty((new_t.shape[0], qpos.shape[1]), dtype=np.float32)
    out[:, :3] = np.column_stack([
        np.interp(new_t, old_t, qpos[:, axis]) for axis in range(3)])
    root_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    root_interp = Slerp(old_t, Rotation.from_quat(root_xyzw))(new_t)
    out[:, 3:7] = root_interp.as_quat()[:, [3, 0, 1, 2]]
    for axis in range(7, qpos.shape[1]):
        out[:, axis] = np.interp(new_t, old_t, qpos[:, axis])
    return out


def _minimum_geom_height(model, data):
    minimum = np.inf
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        position = data.geom_xpos[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            height = position[2] - np.sum(np.abs(rotation[2]) * size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            height = position[2] - size[0]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            height = position[2] - abs(rotation[2, 2]) * size[1] - size[0]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radial = np.sqrt(max(0.0, 1.0 - rotation[2, 2] ** 2))
            height = position[2] - abs(rotation[2, 2]) * size[1] - radial * size[0]
        else:
            height = position[2] - model.geom_rbound[geom_id]
        minimum = min(minimum, float(height))
    return minimum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-xml", type=pathlib.Path, required=True)
    parser.add_argument("--g1-xml", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-root", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ground-clearance", type=float, default=0.01)
    args = parser.parse_args()

    source_data, source_frames = _load_motion(args.input)
    source_fps = int(source_data["fps"])
    if source_fps >= args.fps:
        if source_fps % args.fps != 0:
            raise ValueError(
                "source fps {} is not divisible by target fps {}".format(
                    source_fps, args.fps))
        source_frames = source_frames[::source_fps // args.fps]
    elif args.fps % source_fps != 0:
        raise ValueError(
            "target fps {} must be an integer multiple of source fps {}".format(
                args.fps, source_fps))
    human_frames = _human_frames(args.source_xml, source_frames)

    sys.path.insert(0, str(args.gmr_root))
    import general_motion_retargeting as gmr

    # Use the local repository XML, while retaining the checked-in GMR IK
    # geometry and robot-independent solver implementation.
    gmr.params.ROBOT_XML_DICT["unitree_g1"] = args.g1_xml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as stream:
        json.dump(_patch_g1_ik_config(
            gmr.params.IK_CONFIG_DICT["smplx"]["unitree_g1"]), stream)
        ik_path = pathlib.Path(stream.name)
    gmr.params.IK_CONFIG_DICT["smplx"]["unitree_g1"] = ik_path

    solver = gmr.GeneralMotionRetargeting(
        src_human="smplx", tgt_robot="unitree_g1", verbose=False)
    qpos_frames = np.asarray([solver.retarget(frame) for frame in human_frames],
                             dtype=np.float32)
    qpos_frames = _upsample_qpos(qpos_frames, source_fps, args.fps)

    # Keep every frame above the local G1 collision geometry.  This is a
    # conservative post-process and does not alter joint ordering.
    robot_data = mujoco.MjData(solver.model)
    offsets = []
    for qpos in qpos_frames:
        robot_data.qpos[:] = qpos
        mujoco.mj_forward(solver.model, robot_data)
        offset = args.ground_clearance - _minimum_geom_height(solver.model, robot_data)
        qpos[2] += offset
        offsets.append(offset)

    root_rot_xyzw = qpos_frames[:, 3:7][:, [1, 2, 3, 0]]
    root_rot_exp = Rotation.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    output_frames = np.concatenate(
        (qpos_frames[:, :3], root_rot_exp, qpos_frames[:, 7:]), axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump({"loop_mode": source_data.get("loop_mode", 0),
                     "fps": args.fps,
                     "frames": output_frames.tolist()}, stream)
    print("input={} output={} frames={} width={} fps={} ground_offset=[{:.4f},{:.4f}]".format(
        args.input, args.output, len(output_frames), output_frames.shape[1],
        args.fps, min(offsets), max(offsets)))


if __name__ == "__main__":
    main()
