"""Retarget a MimicKit SMPL motion to a GMR-supported humanoid.

This bridge is useful when the original AMASS/SMPL-X archive is unavailable but
the repository already contains the equivalent MimicKit SMPL motion.  It
reconstructs global SMPL link transforms with MimicKit FK, feeds those
transforms to GMR, and writes a MimicKit motion directly.
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from anim.motion import LoopMode, Motion  # noqa: E402
from util import torch_util  # noqa: E402


SMPL_TO_SMPLX = {
    "pelvis": "Pelvis",
    "spine3": "Chest",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_foot": "L_Ankle",
    "right_foot": "R_Ankle",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}


def _load_source(path):
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or "frames" not in data or "fps" not in data:
        raise ValueError("Expected a single MimicKit motion dictionary")
    frames = np.asarray(data["frames"], dtype=np.float32)
    return data, frames


def _global_smpl_transforms(frames, char_file):
    model = MJCFCharModel(device="cpu")
    model.load(str(char_file))
    if frames.shape[1] != 6 + model.get_dof_size():
        raise ValueError(
            "Motion/SMPL character mismatch: frame width {} != {}".format(
                frames.shape[1], 6 + model.get_dof_size()))

    frame_tensor = torch.from_numpy(frames)
    root_pos = frame_tensor[:, :3]
    root_rot = torch_util.exp_map_to_quat(frame_tensor[:, 3:6])
    joint_rot = model.dof_to_rot(frame_tensor[:, 6:])
    body_pos, body_rot = model.forward_kinematics(
        root_pos, root_rot, joint_rot)
    return (
        model.get_body_names(),
        body_pos.numpy(),
        body_rot.numpy(),
    )


def _geom_min_height(model, data, mujoco):
    min_height = np.inf
    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        pos = data.geom_xpos[geom_id]
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            begin = model.mesh_vertadr[mesh_id]
            end = begin + model.mesh_vertnum[mesh_id]
            verts = model.mesh_vert[begin:end]
            height = np.min(verts @ rot[2] + pos[2])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            height = pos[2] - np.sum(np.abs(rot[2]) * size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            height = pos[2] - size[0]
        elif geom_type in (
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis_z = abs(rot[2, 2])
            radial_z = np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                height = pos[2] - axis_z * size[1] - size[0]
            else:
                height = pos[2] - axis_z * size[1] - radial_z * size[0]
        else:
            # A conservative fallback for uncommon geometry types.
            height = pos[2] - model.geom_rbound[geom_id]
        min_height = min(min_height, float(height))
    return min_height


def _ground_motion(model, qpos, clearance):
    import mujoco

    data = mujoco.MjData(model)
    grounded = qpos.copy()
    offsets = np.zeros(qpos.shape[0], dtype=np.float32)
    for frame_id, pose in enumerate(grounded):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        offsets[frame_id] = clearance - _geom_min_height(
            model, data, mujoco)
        grounded[frame_id, 2] += offsets[frame_id]
    return grounded, offsets


def retarget(source, source_char, output, gmr_root, robot, loop,
             ground_clearance, target_fps):
    source_data, frames = _load_source(source)
    source_fps = int(source_data["fps"])
    if target_fps is not None:
        if source_fps % target_fps != 0:
            raise ValueError("Source FPS must be divisible by target FPS")
        frames = frames[::source_fps // target_fps]
    body_names, body_pos, body_rot_xyzw = _global_smpl_transforms(
        frames, source_char)
    body_ids = {name: i for i, name in enumerate(body_names)}

    missing = sorted(set(SMPL_TO_SMPLX.values()) - set(body_ids))
    if missing:
        raise ValueError("Source character lacks required links: {}".format(missing))

    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting

    solver = GeneralMotionRetargeting(
        src_human="smplx", tgt_robot=robot, verbose=False)
    qpos_frames = []
    for frame_id in range(frames.shape[0]):
        human = {}
        for target_name, source_name in SMPL_TO_SMPLX.items():
            idx = body_ids[source_name]
            quat_xyzw = body_rot_xyzw[frame_id, idx]
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            human[target_name] = [body_pos[frame_id, idx], quat_wxyz]
        qpos_frames.append(solver.retarget(human))

    qpos = np.asarray(qpos_frames, dtype=np.float32)
    if ground_clearance is not None:
        qpos, ground_offsets = _ground_motion(
            solver.model, qpos, ground_clearance)
        print("ground_offset=[{:.4f}, {:.4f}]".format(
            float(np.min(ground_offsets)), float(np.max(ground_offsets))))
    root_pos = qpos[:, :3]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    root_rot_exp = torch_util.quat_to_exp_map(
        torch.from_numpy(root_rot_xyzw)).numpy()
    output_frames = np.concatenate(
        [root_pos, root_rot_exp, qpos[:, 7:]], axis=-1)

    loop_mode = LoopMode.WRAP if loop == "wrap" else LoopMode.CLAMP
    motion = Motion(
        loop_mode=loop_mode,
        fps=target_fps if target_fps is not None else source_fps,
        frames=output_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    motion.save(str(output))

    print("source={}".format(source))
    print("output={}".format(output))
    print("frames={} fps={} width={}".format(
        output_frames.shape[0], motion.fps, output_frames.shape[1]))
    print("robot={} dofs={}".format(robot, qpos.shape[1] - 7))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--source-char", type=pathlib.Path,
        default=REPO_ROOT / "data/assets/smpl/smpl.xml")
    parser.add_argument("--gmr-root", type=pathlib.Path, required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--loop", choices=("wrap", "clamp"), default="clamp")
    parser.add_argument(
        "--ground-clearance", type=float, default=0.01,
        help="Per-frame minimum robot geometry height; use a negative value "
             "only when deliberate ground penetration is required")
    parser.add_argument("--target-fps", type=int, default=30)
    args = parser.parse_args()
    retarget(
        args.input, args.source_char, args.output, args.gmr_root,
        args.robot, args.loop, args.ground_clearance, args.target_fps)


if __name__ == "__main__":
    main()
