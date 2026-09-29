"""Convert Hybrid Motion Imitation whole-body references to MimicKit format.

The Hybrid files use IsaacLab's ``joint_pos`` convention:
``[root_xyz, root_quat_wxyz, 29 joint positions]``.  MimicKit stores
``[root_xyz, root_expmap, 29 joint positions]``.  The converter deliberately
keeps the official joint order and writes object trajectories separately so a
task can decide whether to replay a fixed or moving object.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def convert(input_path: Path, output_path: Path, object_output: Path | None) -> None:
    data = np.load(input_path, allow_pickle=True)
    required = {"fps", "joint_pos", "joint_names"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"missing required NPZ fields: {sorted(missing)}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    names = [str(x) for x in data["joint_names"].tolist()]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"expected joint_pos [N,36], got {joint_pos.shape}")
    if len(names) != 29:
        raise ValueError(f"expected 29 joint names, got {len(names)}")
    if len(set(names)) != 29:
        raise ValueError("joint names are not unique")

    # The NPZ quaternion is IsaacLab's wxyz; scipy accepts xyzw.
    root_quat_xyzw = joint_pos[:, 3:7][:, [1, 2, 3, 0]]
    root_expmap = Rotation.from_quat(root_quat_xyzw).as_rotvec()
    frames = np.concatenate((joint_pos[:, :3], root_expmap, joint_pos[:, 7:]), axis=1)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    with output_path.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps, "frames": frames.astype(np.float32).tolist()}, stream)

    if object_output is not None:
        object_keys = [
            "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w",
        ]
        if not all(key in data.files for key in object_keys):
            raise ValueError("requested object output but NPZ has no complete object trajectory")
        object_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            object_output,
            fps=np.asarray([fps], dtype=np.int64),
            **{key: np.asarray(data[key]) for key in object_keys},
        )

    print(
        f"input={input_path} output={output_path} frames={len(frames)} "
        f"width={frames.shape[1]} fps={fps} joints={len(names)} "
        f"max_dof_step={np.max(np.abs(np.diff(frames[:, 6:], axis=0))):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-output", type=Path)
    args = parser.parse_args()
    convert(args.input, args.output, args.object_output)


if __name__ == "__main__":
    main()
