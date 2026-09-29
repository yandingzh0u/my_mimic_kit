"""Upsample a MimicKit motion without changing its joint ordering.

MimicKit stores root orientation as an exponential map followed by the robot
DOF positions.  Interpolating the exponential-map coordinates directly can
create an artificial angular spike near pi, so the root is interpolated on
SO(3) and the remaining coordinates linearly.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


def _smooth_motion(frames: np.ndarray, sigma: float) -> np.ndarray:
    """Remove isolated retargeting spikes while preserving pose topology."""
    if sigma <= 0:
        return frames
    out = frames.copy()
    out[:, :3] = gaussian_filter1d(frames[:, :3], sigma, axis=0, mode="nearest")
    quat = Rotation.from_rotvec(frames[:, 3:6]).as_quat()
    # Quaternion signs are arbitrary; make the sequence continuous before
    # filtering, otherwise a sign flip would look like a 2*pi excursion.
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    quat = gaussian_filter1d(quat, sigma, axis=0, mode="nearest")
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-8)
    out[:, 3:6] = Rotation.from_quat(quat).as_rotvec()
    out[:, 6:] = gaussian_filter1d(frames[:, 6:], sigma, axis=0, mode="nearest")
    return out


def upsample_motion(
    input_path: Path, output_path: Path, target_fps: int, smooth_sigma: float = 0.0
) -> None:
    with input_path.open("rb") as stream:
        data = pickle.load(stream)
    source_fps = int(data["fps"])
    frames = np.asarray(data["frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 7:
        raise ValueError(f"expected frames [N, 6 + dof], got {frames.shape}")
    if target_fps <= source_fps or target_fps % source_fps:
        raise ValueError(
            f"target fps must be an integer multiple greater than source fps; "
            f"got {source_fps} -> {target_fps}"
        )

    frames = _smooth_motion(frames, smooth_sigma)
    ratio = target_fps // source_fps
    old_t = np.arange(frames.shape[0], dtype=np.float64)
    new_t = np.arange((frames.shape[0] - 1) * ratio + 1, dtype=np.float64) / ratio
    out = np.empty((new_t.size, frames.shape[1]), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(new_t, old_t, frames[:, axis])

    # MimicKit's exp-map is converted to/from xyzw quaternions for SLERP.
    rotations = Rotation.from_rotvec(frames[:, 3:6])
    out[:, 3:6] = Slerp(old_t, rotations)(new_t).as_rotvec()
    for axis in range(6, frames.shape[1]):
        out[:, axis] = np.interp(new_t, old_t, frames[:, axis])

    result = dict(data)
    result["fps"] = int(target_fps)
    result["frames"] = out.astype(np.float32).tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(result, stream)

    old_delta = np.max(np.abs(np.diff(frames[:, 6:], axis=0)), axis=0)
    new_delta = np.max(np.abs(np.diff(out[:, 6:], axis=0)), axis=0)
    print(
        f"input={input_path} output={output_path} "
        f"frames={frames.shape[0]}->{out.shape[0]} fps={source_fps}->{target_fps} "
        f"smooth_sigma={smooth_sigma} "
        f"max_dof_step={old_delta.max():.6f}->{new_delta.max():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=int, required=True)
    parser.add_argument(
        "--smooth-sigma", type=float, default=0.0,
        help="Gaussian time-domain smoothing in source frames (0 disables)",
    )
    args = parser.parse_args()
    upsample_motion(args.input, args.output, args.target_fps, args.smooth_sigma)


if __name__ == "__main__":
    main()
