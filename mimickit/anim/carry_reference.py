"""A control-rate recording of a humanoid and a free box in the same world frame."""
from xml.etree import ElementTree

import numpy as np
import torch

from anim.motion import LoopMode, load_motion
from util.torch_util import exp_map_to_quat


class CarryReference:
    """MotionLib interface for one recording of complete simulator states."""

    def __init__(self, motion_file, device):
        with np.load(motion_file, allow_pickle=False) as data:
            self.fps = float(data["fps"])
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Carry reference requires a positive frame rate")
            frames = len(data["root"])
            if frames < 2:
                raise ValueError("Carry reference requires at least two frames")

            def load(name, shape, dtype=torch.float32):
                value = data[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Expected finite {name} with shape {shape}")
                return torch.as_tensor(value, device=device, dtype=dtype)

            self.root = load("root", (frames, 13))
            self.dof_pos = load("dof_pos", (frames, 28))
            self.dof_vel = load("dof_vel", (frames, 28))
            self.box = load("box", (frames, 13))
            self.body_vel = load("body_vel", (frames, 15, 3))
            self.body_ang_vel = load("body_ang_vel", (frames, 15, 3))
            self.hand_contact = load("hand_contact", (frames, 2), torch.bool)
            self.box_size = load("box_size", (3,))
            self.goal = load("goal", (3,))
            self.body_names = tuple(data["names"].tolist())
            self.dof_names = tuple(data["dof_names"].tolist())

        if not (self.box_size > 0).all():
            raise ValueError("Box dimensions must be positive")
        for state in (self.root, self.box):
            norms = state[:, 3:7].norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0):
                raise ValueError("Carry reference must use unit XYZW quaternions")

    def bind_character(self, character, char_file):
        if tuple(character.get_body_names()) != self.body_names:
            raise ValueError("Recorded body order does not match the carry character")
        world = ElementTree.parse(char_file).find("worldbody")
        dof_names = tuple(
            joint.get("name") for body in world.iter("body") for joint in body.findall("joint"))
        if dof_names != self.dof_names:
            raise ValueError("Recorded DOF order does not match the carry character")
        self._character = character
        self._joint_rot = character.dof_to_rot(self.dof_pos)
        body_pos, _ = character.forward_kinematics(self.root[:, :3], self.root[:, 3:7], self._joint_rot)
        self.hand_ids = [self.body_names.index(name) for name in ("left_hand", "right_hand")]
        self.hand_pos = body_pos[:, self.hand_ids]

    def validate_motion(self, motion_file):
        motion = load_motion(motion_file)
        if (motion.loop_mode != LoopMode.CLAMP or motion.fps != self.fps
                or motion.frames.shape != (len(self.root), 6 + self.dof_pos.shape[1])):
            raise ValueError("Carry motion and reference must have matching frames, FPS and nonlooping playback")
        frames = torch.as_tensor(motion.frames, device=self.root.device)
        rot = exp_map_to_quat(frames[:, 3:6])
        root_rot = self.root[:, 3:7]
        rot_delta = torch.minimum((rot - root_rot).norm(dim=-1), (rot + root_rot).norm(dim=-1))
        if (not torch.allclose(frames[:, :3], self.root[:, :3], atol=1e-5, rtol=0)
                or not torch.allclose(frames[:, 6:], self.dof_pos, atol=1e-5, rtol=0)
                or not (rot_delta <= 1e-5).all()):
            raise ValueError("Carry motion and synchronized reference contain different poses")

    def frame_at(self, times):
        return (times * self.fps).round().long().clamp(0, len(self.root) - 1)

    def time_at(self, frames):
        return frames.float() / self.fps

    def get_motion_duration(self, motion_id=0):
        return (len(self.root) - 1) / self.fps

    def get_motion_length(self, motion_ids):
        return self.time_at(torch.full_like(motion_ids, len(self.root) - 1))

    def get_motion_loop_mode(self, motion_ids):
        return torch.full_like(motion_ids, LoopMode.CLAMP.value)

    def sample_motions(self, count):
        return torch.zeros(count, dtype=torch.long, device=self.root.device)

    def sample_time(self, motion_ids):
        # The last frame represents an already completed motion.
        frames = torch.randint(len(self.root) - 1, motion_ids.shape, device=motion_ids.device)
        return self.time_at(frames)

    def calc_motion_phase(self, motion_ids, times):
        return self.frame_at(times).float() / (len(self.root) - 1)

    def calc_motion_frame(self, motion_ids, times):
        frame = self.frame_at(times)
        root = self.root[frame]
        return root[..., :3], root[..., 3:7], root[..., 7:10], root[..., 10:13], self._joint_rot[frame], self.dof_vel[frame]

    def joint_rot_to_dof(self, joint_rot):
        return self._character.rot_to_dof(joint_rot)
