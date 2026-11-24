"""Imitate a synchronized humanoid and freely simulated box."""
import math

import torch

from anim.carry_reference import CarryReference
from engines.engine import ObjType
from envs.add_env import ADDEnv, build_disc_error_groups
from envs.base_env import EnvMode
from util import torch_util
from util.carry_geometry import CarryGeometry


def rotation_obs(matrix):
    return torch.cat((matrix[..., :, 0], matrix[..., :, 2]), dim=-1)


class CarryEnv(ADDEnv):
    tracks_box = True

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        if not env_config["global_obs"] or (
            env_config.get("disc_observation", "add") != "amp" and env_config["num_disc_obs_steps"] != 1
        ):
            raise ValueError("Carry requires global observations and one ADD discriminator frame")
        reference_file = env_config.get("carry_reference_file")
        self._reference = CarryReference(reference_file or env_config["motion_file"], device)
        if reference_file is not None:
            self._reference.validate_motion(env_config["motion_file"])
        self._box_size = self._reference.box_size
        self._box_goal = self._reference.goal
        self._geometry = CarryGeometry(self._box_size).to(device)
        self.tracks_box_velocity = bool(env_config.get("box_velocity_error", False))
        self.grip_release_aware = bool(env_config.get("grip_release_aware", False))
        if self.grip_release_aware:
            frames = torch.arange(len(self._reference.box), device=device)[:, None]
            self._last_grip_frames = torch.where(self._reference.hand_contact, frames, -1).amax(dim=0)
        self._box_motion_preview = bool(env_config.get("box_motion_preview", False))
        if self._box_motion_preview:
            steps = env_config.get("tar_obs_steps", [1])
            if not env_config["enable_tar_obs"] or not steps or max(steps) < 1:
                raise ValueError("Box motion preview requires at least one future pose target")
            self._box_preview_steps = max(steps)
        self._reference_centered_action = bool(env_config.get("reference_centered_action", False))
        if self._reference_centered_action:
            if engine_config.get("control_mode") != "pos" or not env_config["enable_tar_obs"]:
                raise ValueError("Reference-centered carry actions require position control and pose preview")
            self._action_preview = max(env_config["tar_obs_steps"])
        self._configure_resets(env_config)
        super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)

    def _configure_resets(self, config):
        # Match the phase-zero evaluation distribution as well as training from
        # recorded intermediate contact states (hybrid reference initialization).
        self._start0_prob = float(config.get("start0_prob", 0.5))
        has_focus_frames = "focus_reset_frames" in config
        if has_focus_frames != ("focus_reset_prob" in config):
            raise ValueError("Set focus_reset_frames and focus_reset_prob together")
        if has_focus_frames and "pregrasp_reset_prob" in config:
            raise ValueError("Choose either focus resets or pregrasp resets")
        self._focus_reset_prob = float(config.get("focus_reset_prob", config.get("pregrasp_reset_prob", 0.0)))
        if not 0 <= self._start0_prob <= 1 or not 0 <= self._focus_reset_prob <= 1 - self._start0_prob:
            raise ValueError("Phase-zero and focused reset probabilities must sum to at most one")

        if has_focus_frames:
            frames = config["focus_reset_frames"]
            if (not isinstance(frames, (list, tuple)) or len(frames) != 2
                    or any(type(frame) is not int for frame in frames)
                    or not 0 <= frames[0] < frames[1] <= len(self._reference.box) - 1):
                raise ValueError("focus_reset_frames must be a nonempty [start, end) inside the motion before its last frame")
            self._focus_reset_frames = tuple(frames)
        elif self._focus_reset_prob:
            contact = self._reference.hand_contact.any(dim=-1).nonzero().flatten()
            if not len(contact) or int(contact[0]) < 2:
                raise ValueError("Pregrasp resets require a reference that approaches the box before contact")
            end = int(contact[0])
            self._focus_reset_frames = (max(1, end - round(self._reference.fps / 3)), end)

    def _load_motions(self, motion_file):
        if not math.isclose(self._reference.fps * self._engine.get_timestep(), 1.0):
            raise ValueError("Carry recording FPS must equal the simulator control rate")
        if not math.isclose(self._episode_length, self._reference.get_motion_duration(), abs_tol=1e-6):
            raise ValueError("Episode length must equal the carry recording duration")
        self._motion_lib = self._reference

    def _build_sim_tensors(self, config):
        self._reference.bind_character(self._kin_char_model, config["char_file"])
        super()._build_sim_tensors(config)

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        start = self._reference.box[0].cpu().numpy()
        self._box_id = self._engine.create_obj(
            env_id, ObjType.rigid, config["box_asset"], "carry_box",
            start_pos=start[:3], start_rot=start[3:7], fix_root=False)

    def _get_motion_times(self, env_ids=None):
        steps, offsets = self._timestep_buf, self._motion_time_offsets
        if env_ids is not None:
            steps, offsets = steps[env_ids], offsets[env_ids]
        return self._reference.time_at(self._reference.frame_at(offsets) + steps)

    def _sample_motion_times(self, n):
        ids, times = super()._sample_motion_times(n)
        if self._mode == EnvMode.TRAIN and self._rand_reset:
            coin = torch.rand_like(times)
            if self._focus_reset_prob:
                start, end = self._focus_reset_frames
                frames = torch.randint(start, end, times.shape, device=times.device)
                times = torch.where(coin >= 1 - self._focus_reset_prob, self._reference.time_at(frames), times)
            times = times.masked_fill(coin < self._start0_prob, 0)
        return ids, times

    def _ref_state_init(self, env_ids):
        char_id = self._get_char_id()
        frame = self._reference.frame_at(self._get_motion_times(env_ids))
        root = self._reference.root[frame]
        self._set_rigid_state(env_ids, char_id, root)
        self._engine.set_dof_pos(env_ids, char_id, self._reference.dof_pos[frame])
        self._engine.set_dof_vel(env_ids, char_id, self._reference.dof_vel[frame])
        self._engine.set_body_vel(env_ids, char_id, self._reference.body_vel[frame])
        self._engine.set_body_ang_vel(env_ids, char_id, self._reference.body_ang_vel[frame])
        self._set_rigid_state(env_ids, self._box_id, self._reference.box[frame])

    def _set_rigid_state(self, env_ids, obj, state):
        self._engine.set_root_pos(env_ids, obj, state[:, :3])
        self._engine.set_root_rot(env_ids, obj, state[:, 3:7])
        self._engine.set_root_vel(env_ids, obj, state[:, 7:10])
        self._engine.set_root_ang_vel(env_ids, obj, state[:, 10:13])

    def _apply_action(self, actions):
        if self._reference_centered_action:
            low, high = self._action_bound_low, self._action_bound_high
            residual = 2 * (actions - low) / (high - low) - 1
            frame = self._reference.frame_at(self._get_motion_times()) + self._action_preview
            guide = self._reference.dof_pos[frame.clamp_max(len(self._reference.dof_pos) - 1)].clamp(low, high)
            # Zero follows the observed reference pose; either endpoint still
            # reaches the original actuator limit. No recorded PD commands.
            actions = guide + residual * torch.where(residual >= 0, high - guide, guide - low)
        super()._apply_action(actions)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        heading = torch_util.calc_heading_quat_inv(self._engine.get_root_rot(char_id))
        hands = self._engine.get_body_pos(char_id)[:, self._reference.hand_ids]
        pos, rot = self._box_pose(env_ids)
        vel = self._engine.get_root_vel(self._box_id)
        ang_vel = self._engine.get_root_ang_vel(self._box_id)
        if env_ids is not None:
            root_pos, heading, hands = root_pos[env_ids], heading[env_ids], hands[env_ids]
            vel, ang_vel = vel[env_ids], ang_vel[env_ids]

        frame = self._reference.frame_at(self._get_motion_times(env_ids))
        target = self._reference.box[frame]
        ref_pos = target[:, :3]
        actual_rot, ref_rot, residual_rot = self._geometry.align_box_rotation(rot, target[:, 3:7])
        local_hands = (hands - pos[:, None]) @ actual_rot
        ref_hands = (self._reference.hand_pos[frame] - ref_pos[:, None]) @ ref_rot
        obs = torch.cat((
            obs, torch_util.quat_rotate(heading, pos - root_pos),
            rotation_obs(torch_util.quat_to_matrix(heading) @ actual_rot),
            torch_util.quat_rotate(heading, vel), torch_util.quat_rotate(heading, ang_vel),
            torch_util.quat_rotate(heading, ref_pos - pos), rotation_obs(residual_rot),
            local_hands.flatten(1), (ref_hands - local_hands).flatten(1),
        ), dim=-1)
        if self._box_motion_preview:
            preview_frame = (frame + self._box_preview_steps).clamp_max(len(self._reference.box) - 1)
            preview = self._reference.box[preview_frame]
            obs = torch.cat((obs,
                             torch_util.quat_rotate(heading, preview[:, :3] - pos),
                             torch_util.quat_rotate(heading, preview[:, 7:10] - vel)), dim=-1)
        return obs

    def _box_pose(self, env_ids):
        pos = self._engine.get_root_pos(self._box_id)
        rot = self._engine.get_root_rot(self._box_id)
        return (pos, rot) if env_ids is None else (pos[env_ids], rot[env_ids])

    def _compute_policy_disc_obs(self, env_ids):
        human = super()._compute_policy_disc_obs(env_ids)
        box = self._box_pose(env_ids)
        if self.tracks_box_velocity:
            velocity = self._engine.get_root_vel(self._box_id)
            box += (velocity if env_ids is None else velocity[env_ids],)
        if self.grip_release_aware:
            box += (self._active_grips(self._get_motion_times(env_ids)).to(human),)
        return torch.cat((human, *box), dim=-1)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        human = super()._compute_disc_obs_demo(motion_ids, motion_times0)
        state_dim = 10 if self.tracks_box_velocity else 7
        box = self._reference.box[self._reference.frame_at(motion_times0), :state_dim]
        if self.grip_release_aware:
            box = torch.cat((box, self._active_grips(motion_times0).to(box)), dim=-1)
        return torch.cat((human, box), dim=-1)

    def _active_grips(self, times):
        # Before approach and through the last contact, track the hand in the
        # box frame. On final withdrawal, the humanoid pose already tracks it.
        return self._reference.frame_at(times)[..., None] <= self._last_grip_frames

    def get_disc_error_groups(self):
        char_id = self._get_char_id()
        box = (("box_pos", (172, 173, 174)), ("box_rot", (175, 176, 177, 178)))
        if self.tracks_box_velocity:
            box += (("box_vel", (179, 180, 181)),)
        if self.grip_release_aware:
            first = box[-1][1][-1] + 1
            box += (("active_grips", (first, first + 1)),)
        human = build_disc_error_groups(
            self._num_disc_obs_steps, self._kin_char_model.get_num_joints(),
            self._engine.get_obj_num_bodies(char_id), self._engine.get_obj_num_dofs(char_id),
            total_dim=int(self.get_disc_obs_space().shape[0]) - sum(len(indices) for _, indices in box))
        return human + box
