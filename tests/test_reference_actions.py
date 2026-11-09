"""Reference-centered PD commands preserve the existing action contract."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from anim.kin_char_model import JointType
from anim.mjcf_char_model import MJCFCharModel
from engines.engine import ControlMode
from envs.char_env import CharEnv
from envs.deepmimic_env import DeepMimicEnv, compose_reference_pd_targets
from util import torch_util


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def humanoid():
    model = MJCFCharModel("cpu")
    model.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
    assert model.get_dof_size() == 28
    return model


def _joint(model, kind):
    return next((i, model.get_joint(i))
                for i in range(1, model.get_num_joints())
                if model.get_joint(i).joint_type == kind)


def _assert_same_rotations(actual, expected):
    # Quaternion signs may differ while representing the same rotation.
    alignment = torch.abs(torch.sum(actual * expected, dim=-1))
    torch.testing.assert_close(alignment, torch.ones_like(alignment),
                               atol=2e-6, rtol=0)


def test_zero_residual_reproduces_reference_pose(humanoid):
    generator = torch.Generator().manual_seed(381)
    reference_dof = 0.5 * torch.randn(19, 28, generator=generator)
    reference_rot = humanoid.dof_to_rot(reference_dof)
    targets = compose_reference_pd_targets(
        humanoid, torch.zeros_like(reference_dof), reference_rot)
    _assert_same_rotations(humanoid.dof_to_rot(targets), reference_rot)
    torch.testing.assert_close(targets, reference_dof, atol=1e-6, rtol=1e-6)


def test_spherical_residual_uses_noncommuting_left_rotation(humanoid):
    joint_id, joint = _joint(humanoid, JointType.SPHERICAL)
    reference = torch.zeros(1, 28)
    residual = torch.zeros_like(reference)
    joint.set_joint_dof(torch.tensor([[0.9, 0.0, 0.0]]), reference)
    joint.set_joint_dof(torch.tensor([[0.0, 0.7, 0.0]]), residual)
    reference_rot = humanoid.dof_to_rot(reference)
    target = compose_reference_pd_targets(humanoid, residual, reference_rot)
    actual = humanoid.dof_to_rot(target)[:, joint_id - 1]
    q_reference = reference_rot[:, joint_id - 1]
    q_residual = joint.dof_to_rot(joint.get_joint_dof(residual))
    expected = torch_util.quat_mul(q_residual, q_reference)
    _assert_same_rotations(actual, expected)
    reversed_order = torch_util.quat_mul(q_reference, q_residual)
    coordinate_addition = joint.dof_to_rot(joint.get_joint_dof(reference + residual))
    assert torch.abs(torch.sum(actual * reversed_order)) < 0.98
    assert torch.abs(torch.sum(actual * coordinate_addition)) < 0.99


def test_hinge_angle_addition_is_not_wrapped_at_pi(humanoid):
    _, joint = _joint(humanoid, JointType.HINGE)
    reference = torch.zeros(2, 28)
    residual = torch.zeros_like(reference)
    joint.set_joint_dof(torch.tensor([[2.0], [-2.0]]), reference)
    joint.set_joint_dof(torch.tensor([[2.0], [-2.0]]), residual)
    target = compose_reference_pd_targets(
        humanoid, residual, humanoid.dof_to_rot(reference))
    torch.testing.assert_close(joint.get_joint_dof(target),
                               torch.tensor([[4.0], [-4.0]]))


def test_composition_handles_each_sample_and_leading_batch_axis(humanoid):
    generator = torch.Generator().manual_seed(64)
    reference = 0.3 * torch.randn(2, 3, 28, generator=generator)
    residual = 0.2 * torch.randn(2, 3, 28, generator=generator)
    reference_rot = humanoid.dof_to_rot(reference)
    reference_before = reference_rot.clone()
    residual_before = residual.clone()
    batch = compose_reference_pd_targets(humanoid, residual, reference_rot)
    per_sample = torch.stack([
        compose_reference_pd_targets(humanoid, r.unsqueeze(0), q.unsqueeze(0))[0]
        for r, q in zip(residual.reshape(-1, 28),
                        reference_rot.reshape(-1, reference_rot.shape[-2], 4))
    ]).reshape_as(batch)
    torch.testing.assert_close(batch, per_sample)
    assert torch.equal(reference_rot, reference_before)
    assert torch.equal(residual, residual_before)


def _env(humanoid, enabled=True, low=-4.0, high=4.0):
    env = object.__new__(DeepMimicEnv)
    env._reference_centered_actions = enabled
    env._kin_char_model = humanoid
    env._char_ids = [17]
    env._action_bound_low = torch.full((28,), low)
    env._action_bound_high = torch.full((28,), high)
    env._engine = SimpleNamespace(get_timestep=lambda: 1.0 / 30,
                                  set_cmd=mock.Mock())
    env._motion_ids = torch.tensor([3, 9])
    env._time_buf = torch.tensor([0.4, 0.8])
    env._motion_time_offsets = torch.tensor([1.1, 2.7])
    env._motion_lib = mock.Mock()
    # These caches can be stale in headless mode, including after reset.
    env._ref_dof_pos = torch.full((2, 28), float("nan"))
    env._ref_joint_rot = torch.full((2, humanoid.get_num_joints() - 1, 4),
                                    float("nan"))
    return env


def test_next_frame_query_includes_motion_offsets_and_reset(humanoid):
    env = _env(humanoid, low=-3, high=5)
    _, hinge = _joint(humanoid, JointType.HINGE)

    def frame(motion_ids, times):
        reference = torch.zeros(2, 28)
        hinge.set_joint_dof((times * 0.1).unsqueeze(-1), reference)
        return (None, None, None, None, humanoid.dof_to_rot(reference))

    env._motion_lib.calc_motion_frame.side_effect = frame
    midpoint = 0.5 * (env._action_bound_low + env._action_bound_high)
    for current, offset in (([0.4, 0.8], [1.1, 2.7]),
                            ([0.0, 0.0], [0.2, 0.6])):
        env._time_buf = torch.tensor(current)
        env._motion_time_offsets = torch.tensor(offset)
        env._apply_action(midpoint.expand(2, -1))
        queried_ids, queried_times = env._motion_lib.calc_motion_frame.call_args.args
        assert torch.equal(queried_ids, env._motion_ids)
        expected_times = torch.tensor(current) + torch.tensor(offset) + 1.0 / 30
        torch.testing.assert_close(queried_times, expected_times)
        char_id, command = env._engine.set_cmd.call_args.args
        assert char_id == 17
        assert torch.isfinite(command).all()
        torch.testing.assert_close(hinge.get_joint_dof(command).squeeze(-1),
                                   0.1 * expected_times)


def test_disabled_mode_uses_original_clipping_bitwise(humanoid):
    env = _env(humanoid, enabled=False, low=-0.7, high=1.2)
    actions = torch.linspace(-5, 5, 56).reshape(2, 28)
    before = actions.clone()
    env._apply_action(actions)
    command = env._engine.set_cmd.call_args.args[1].clone()
    env._engine.set_cmd.reset_mock()
    CharEnv._apply_action(env, actions)
    assert torch.equal(command, env._engine.set_cmd.call_args.args[1])
    assert torch.equal(actions, before)
    env._motion_lib.calc_motion_frame.assert_not_called()


@pytest.mark.parametrize("direction", [-1, 1])
def test_existing_input_saturation_limits_reference_residual(humanoid, direction):
    env = _env(humanoid, low=-0.4, high=0.4)
    reference = humanoid.dof_to_rot(torch.zeros(2, 28))
    env._motion_lib.calc_motion_frame.return_value = (
        None, None, None, None, reference)
    boundary = direction * 0.4 * torch.ones(2, 28)
    env._apply_action(boundary)
    expected = env._engine.set_cmd.call_args.args[1].clone()
    env._apply_action(direction * 1e3 * torch.ones_like(boundary))
    actual = env._engine.set_cmd.call_args.args[1]
    assert torch.equal(actual, expected)


def test_reference_target_keeps_original_absolute_target_limits(humanoid):
    env = _env(humanoid, low=-1, high=1)
    _, hinge = _joint(humanoid, JointType.HINGE)
    reference = torch.zeros(2, 28)
    residual = torch.zeros_like(reference)
    hinge.set_joint_dof(torch.tensor([[0.8], [-0.8]]), reference)
    hinge.set_joint_dof(torch.tensor([[0.8], [-0.8]]), residual)
    env._motion_lib.calc_motion_frame.return_value = (
        None, None, None, None, humanoid.dof_to_rot(reference))
    env._apply_action(residual)
    actual = env._engine.set_cmd.call_args.args[1]
    torch.testing.assert_close(hinge.get_joint_dof(actual),
                               torch.tensor([[1.0], [-1.0]]))
    assert torch.all(actual >= env._action_bound_low)
    assert torch.all(actual <= env._action_bound_high)


def test_reference_actions_are_opt_in():
    config = {"enable_early_termination": False, "ref_char_offset": [0, 0, 0]}
    with mock.patch.object(CharEnv, "__init__", return_value=None):
        env = DeepMimicEnv(config, {}, 1, "cpu", False)
    assert env._reference_centered_actions is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_ambiguous_reference_action_flag_is_rejected(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        DeepMimicEnv({"reference_centered_actions": value}, {}, 1, "cpu", False)


@pytest.mark.parametrize("mode", [ControlMode.none, ControlMode.vel, ControlMode.torque])
def test_reference_positions_cannot_be_used_as_velocity_or_torque_commands(mode):
    env = object.__new__(DeepMimicEnv)
    env._reference_centered_actions = True
    env._engine = SimpleNamespace(get_control_mode=lambda: mode)
    with mock.patch.object(CharEnv, "_build_sim_tensors", return_value=None):
        with pytest.raises(ValueError, match="requires position control"):
            env._build_sim_tensors({})
