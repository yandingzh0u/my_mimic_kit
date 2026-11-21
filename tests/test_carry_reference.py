"""Check synchronization and contact geometry without starting Isaac Lab."""

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
import torch

from anim.carry_reference import CarryReference
from anim.mjcf_char_model import MJCFCharModel
from anim.motion import LoopMode, Motion
from envs.add_env import ADDEnv
from envs.base_env import EnvMode
from envs.carry_env import CarryEnv
from util.carry_geometry import CarryGeometry


CHAR_FILE = Path(__file__).resolve().parents[1] / "data/assets/humanoid/humanoid.xml"


def test_carry_uses_recorded_frame_and_velocities_after_every_reset(tmp_path):
    fps, count = 30, 166
    character = MJCFCharModel("cpu")
    character.load(str(CHAR_FILE))
    bodies = ElementTree.parse(CHAR_FILE).find("worldbody").iter("body")
    dof_names = [joint.get("name") for body in bodies for joint in body.findall("joint")]
    rng = np.random.default_rng(0)
    root = np.zeros((count, 13), dtype=np.float32)
    box = root.copy()
    for state in (root, box):
        state[:, :3] = rng.normal(size=(count, 3))
        state[:, 6] = 1
        state[:, 7:] = rng.normal(size=(count, 6))
    dof_pos = rng.uniform(-0.3, 0.3, (count, 28)).astype(np.float32)
    dof_vel = rng.normal(size=(count, 28)).astype(np.float32)
    path = tmp_path / "state.npz"
    np.savez(path, fps=fps, root=root, box=box, dof_pos=dof_pos, dof_vel=dof_vel,
             body_vel=np.zeros((count, 15, 3)), body_ang_vel=np.zeros((count, 15, 3)),
             hand_contact=np.zeros((count, 2), dtype=bool), box_size=[0.4] * 3,
             goal=[0.0, 0.0, 0.2], names=character.get_body_names(), dof_names=dof_names)
    reference = CarryReference(path, "cpu")
    reference.bind_character(character, CHAR_FILE)

    # Cover every control step from every recorded reset frame until the end.
    start, frame = torch.ones(count, count, dtype=torch.bool).triu().nonzero().T
    env = SimpleNamespace(_reference=reference, _timestep_buf=frame - start,
                          _motion_time_offsets=start.float() / fps)
    times = CarryEnv._get_motion_times(env)
    ids = reference.sample_motions(len(times))
    assert torch.equal(reference.frame_at(times), frame)
    assert torch.equal(times >= reference.get_motion_length(ids), frame == count - 1)
    pos, rot, vel, ang_vel, joint_rot, joint_vel = reference.calc_motion_frame(ids, times)
    torch.testing.assert_close(torch.cat((pos, rot, vel, ang_vel), -1),
                               torch.from_numpy(root)[frame], rtol=0, atol=0)
    torch.testing.assert_close(joint_rot, character.dof_to_rot(torch.from_numpy(dof_pos))[frame],
                               rtol=0, atol=0)
    torch.testing.assert_close(joint_vel, torch.from_numpy(dof_vel)[frame], rtol=0, atol=0)
    torch.testing.assert_close(reference.box[reference.frame_at(times)], torch.from_numpy(box)[frame],
                               rtol=0, atol=0)

    frames = np.concatenate((root[:, :3], np.zeros((count, 3)), dof_pos), axis=1)
    native_path = tmp_path / "carry.pkl"
    Motion(LoopMode.CLAMP, fps, frames).save(native_path)
    reference.validate_motion(native_path)
    Motion(LoopMode.CLAMP, fps, np.roll(frames, 1, axis=0)).save(native_path)
    with pytest.raises(ValueError, match="different poses"):
        reference.validate_motion(native_path)


@pytest.mark.parametrize("config, window", [
    ({"pregrasp_reset_prob": 0.25}, (24, 34)),
    ({"focus_reset_prob": 0.25, "focus_reset_frames": [78, 140]}, (78, 140)),
])
def test_focused_resets_preserve_phase_zero_and_leave_test_unchanged(config, window):
    env = CarryEnv.__new__(CarryEnv)
    env._device = "cpu"
    env._rand_reset = True
    # A fixed full-motion sample makes it possible to see which resets change.
    env._motion_lib = env._reference = SimpleNamespace(
        box=torch.zeros(200, 13), fps=30, hand_contact=(torch.arange(200)[:, None] >= 34),
        sample_motions=lambda n: torch.zeros(n, dtype=torch.long),
        sample_time=lambda ids: torch.full(ids.shape, 150 / 30),
        time_at=lambda frames: frames / 30,
    )
    env._configure_resets(config)
    env._mode = EnvMode.TRAIN
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _, times = env._sample_motion_times(10_000)
    frames = (30 * times).round().long()
    at_start = frames == 0
    focused = (window[0] <= frames) & (frames < window[1])
    assert (at_start | focused | (frames == 150)).all()
    assert 0.48 < at_start.float().mean() < 0.52
    assert 0.23 < focused.float().mean() < 0.27
    env._mode = EnvMode.TEST
    assert torch.count_nonzero(env._sample_motion_times(64)[1]) == 0
    env._test_random_start = True
    assert (env._sample_motion_times(64)[1] == 150 / 30).all()


@pytest.mark.parametrize("override", [
    {"pregrasp_reset_prob": 0}, {"focus_reset_prob": 0.6},
    {"focus_reset_frames": [78, 78]}, {"focus_reset_frames": [78.5, 140]},
    {"focus_reset_frames": [-1, 140]}, {"focus_reset_frames": [78, 200]},
])
def test_invalid_focused_reset_configuration_is_rejected(override):
    env = CarryEnv.__new__(CarryEnv)
    env._reference = SimpleNamespace(box=torch.zeros(200, 13))
    with pytest.raises(ValueError):
        env._configure_resets({"focus_reset_prob": 0.25, "focus_reset_frames": [78, 140], **override})


def test_reference_centered_actions_use_pose_preview_and_keep_both_limits():
    env = CarryEnv.__new__(CarryEnv)
    env._reference_centered_action = True
    env._action_preview = 3
    env._action_bound_low = torch.tensor([-2., 1.])
    env._action_bound_high = torch.tensor([4., 5.])
    env._char_ids = [0]
    env._get_motion_times = lambda: torch.tensor([0, 0, 0, 3], dtype=torch.long)
    poses = torch.tensor([[0., 3.], [0., 3.], [0., 3.], [-1., 4.], [3., 2.]])
    commands = []
    env._reference = SimpleNamespace(frame_at=lambda frames: frames, dof_pos=poses)
    env._engine = SimpleNamespace(set_cmd=lambda _, cmd: commands.append(cmd))
    midpoint = (env._action_bound_low + env._action_bound_high) / 2
    env._apply_action(torch.stack((env._action_bound_low, midpoint, env._action_bound_high, midpoint)))
    torch.testing.assert_close(commands[0], torch.stack((env._action_bound_low, poses[3],
                                                         env._action_bound_high, poses[-1])))


def test_box_preview_aligns_with_future_frame_and_robot_heading(monkeypatch):
    env = CarryEnv.__new__(CarryEnv)
    env._char_ids, env._box_id = [0], 1
    identity = torch.tensor([0., 0., 0., 1.])
    heading = torch.tensor([0., 0., 2**-0.5, 2**-0.5])
    rotations = (heading.expand(2, -1), identity.expand(2, -1))
    positions = (torch.zeros(2, 3), torch.tensor([[1., 1., 0.], [2., 1., 0.]]))
    velocities = torch.tensor([[1., 1., 0.], [2., 1., 0.]])
    env._engine = SimpleNamespace(
        get_root_pos=lambda obj: positions[obj], get_root_rot=lambda obj: rotations[obj],
        get_root_vel=lambda _: velocities, get_root_ang_vel=lambda _: torch.zeros(2, 3),
        get_body_pos=lambda _: torch.zeros(2, 2, 3),
    )
    box = torch.zeros(5, 13)
    box[:, 3:7] = identity
    box[:, 0] = torch.arange(5.)
    box[:, 7:9] = torch.tensor([3., 4.])
    env._reference = SimpleNamespace(
        box=box, hand_ids=[0, 1], hand_pos=torch.zeros(5, 2, 3), frame_at=lambda frame: frame,
    )
    env._geometry = CarryGeometry([.4, .4, .4])
    frame = torch.tensor([0, 4])
    env._get_motion_times = lambda ids: frame if ids is None else frame[ids]
    monkeypatch.setattr(ADDEnv, "_compute_obs",
                        lambda _, ids: torch.zeros(2 if ids is None else len(ids), 5))
    env._box_motion_preview = False
    old = env._compute_obs()
    env._box_motion_preview, env._box_preview_steps = True, 3
    new = env._compute_obs()
    torch.testing.assert_close(new[:, :-6], old)
    torch.testing.assert_close(new[:, -6:], torch.tensor([
        [-1., -2., 0., 3., -2., 0.], [-1., -2., 0., 3., -1., 0.],
    ]))
    torch.testing.assert_close(env._compute_obs(torch.tensor([1])), new[1:])


def test_carry_geometry_distinguishes_lost_contact_and_respects_cube_symmetry():
    geometry = CarryGeometry([0.4, 0.4, 0.4])
    reference = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    hands = torch.tensor([[[0.0, 0.24, 0.0], [0.0, -0.24, 0.0]]]).expand(3, -1, -1)
    box = reference.clone()
    box[0, 0] = 0.03  # Slide parallel to the grasped faces.
    box[1, 1] = 0.03  # Push one hand off and into the other.
    box[2, 5:7] = 2 ** -0.5  # Relabel the cube axes by rotating 90 degrees around Z.

    error = geometry(box, reference, hands, hands)
    torch.testing.assert_close(error[:2, 0], torch.tensor([0.03, 0.03]))
    assert torch.all(error[1, 2:] > error[0, 2:])
    torch.testing.assert_close(error[2], torch.zeros(4), rtol=0, atol=1e-6)
