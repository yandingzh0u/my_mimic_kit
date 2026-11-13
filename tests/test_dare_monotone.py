"""Production monotone scorer contracts, real humanoid FK and optimizer state."""

import copy
from pathlib import Path

import gymnasium.spaces as spaces
import pytest
import torch
import yaml

from anim.mjcf_char_model import MJCFCharModel
from envs.add_env import build_disc_error_groups, compute_disc_obs
from envs.deepmimic_env import compute_tracking_error
from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_monotone_agent import DAREMonotoneAgent
from learning.dare_monotone_model import (
    ADDPairErrorMetrics, DAREMonotoneModel, FEATURE_DIM, HIDDEN_WIDTHS,
    MixedELU, aggregate_error_metrics)
from learning.add_model import ADDModel
import util.torch_util as torch_util


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/agents/dare_monotone_humanoid_agent.yaml"


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Env:
    _global_obs = True

    def __init__(self):
        self._kin_char_model = MJCFCharModel("cpu")
        self._kin_char_model.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1., 1., shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        return build_disc_error_groups(1, 15, 15, 28, 172)

    def get_num_disc_obs_steps(self):
        return 1

    def get_num_envs(self):
        return 4


def config():
    return yaml.safe_load(CONFIG.read_text())


def physical_pair(env, count=128):
    """Different physical humanoid states; no synthetic invalid 6D rotations."""
    def sample():
        root = torch.randn(count, 3)
        rotation = torch_util.exp_map_to_quat(torch.randn(count, 3))
        joints = env._kin_char_model.dof_to_rot(torch.randn(count, 28))
        body_pos, body_rot = env._kin_char_model.forward_kinematics(root, rotation, joints)
        root_vel, root_ang_vel, dof_vel = torch.randn(count, 3), torch.randn(count, 3), torch.randn(count, 28)
        obs = compute_disc_obs(root[:, None], rotation[:, None], root_vel[:, None],
            root_ang_vel[:, None], joints[:, None], dof_vel[:, None], body_pos[:, None], True)
        state = (root, rotation, body_rot, body_pos, root_vel, root_ang_vel, dof_vel)
        return obs, state
    return sample(), sample()


def fill_buffers(agent, policy, reference):
    agent._exp_buffer.set_data_flat("disc_obs", policy)
    agent._exp_buffer.set_data_flat("disc_obs_demo", reference)
    agent._exp_buffer.set_data_flat("reward", policy.new_zeros(len(policy)))
    agent._disc_buffer.push({"disc_obs": policy[:32, None],
                             "disc_obs_demo": reference[:32, None]})


def test_metrics_match_real_fk_evaluator_including_body_world_rotation():
    env = Env()
    (policy, p), (reference, r) = physical_pair(env)
    features = ADDPairErrorMetrics(env)(policy, reference)
    actual = aggregate_error_metrics(features)
    expected = compute_tracking_error(
        root_pos=p[0], root_rot=p[1], body_rot=p[2], body_pos=p[3],
        tar_root_pos=r[0], tar_root_rot=r[1], tar_body_rot=r[2], tar_body_pos=r[3],
        root_vel=p[4], root_ang_vel=p[5], dof_vel=p[6],
        tar_root_vel=r[4], tar_root_ang_vel=r[5], tar_dof_vel=r[6])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert features.shape == (128, 64)
    assert (features >= 0).all()
    # The unfolded observations must retain each non-root body's own physical
    # error, rather than repeat the old group mean in a wider tensor.
    body_delta = ((r[3] - r[0][:, None]) - (p[3] - p[0][:, None]))
    torch.testing.assert_close(features[:, 2:16], body_delta[:, 1:].norm(dim=-1),
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(features[:, 16:30],
        torch_util.quat_diff_angle(p[2][:, 1:], r[2][:, 1:]).abs(),
        rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(features[:, 30:58], (r[6] - p[6]).abs())
    torch.testing.assert_close(features[:, 58:61], (r[4] - p[4]).abs())
    torch.testing.assert_close(features[:, 61:64], (r[5] - p[5]).abs())
    zero = ADDPairErrorMetrics(env)(policy, policy)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=1e-6)
    assert ADDPairErrorMetrics(env)(policy[:0], reference[:0]).shape == (0, 64)
    assert aggregate_error_metrics(features[:0]).shape == (0, 7)


def test_single_joint_velocity_identity_is_not_averaged_or_duplicated():
    env = Env()
    (base, _), _ = physical_pair(env, count=1)
    policy = base.expand(31, -1).clone()
    reference = policy.clone()
    # Each of the 28 DOFs and each of the three Root velocity axes has a
    # distinct input coordinate, even though the group evaluator averages it.
    reference[torch.arange(28), torch.arange(144, 172)] += 1.25
    reference[torch.arange(28, 31), torch.arange(138, 141)] += 1.25
    actual = ADDPairErrorMetrics(env)(policy, reference)
    expected = torch.zeros_like(actual)
    expected[torch.arange(31), torch.arange(30, 61)] = 1.25
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    aggregated = aggregate_error_metrics(actual)
    torch.testing.assert_close(aggregated[:28, 4], torch.full((28,), 1.25 / 28))
    torch.testing.assert_close(aggregated[28:, 5], torch.full((3,), 1.25 / 3))


@pytest.mark.parametrize("offset", [0., -1e-6, 1e-6, -.001, .001])
def test_pi_boundary_root_and_local_rotations_match_actual_fk(offset):
    env = Env()
    char = env._kin_char_model
    cases = [(0, axis) for axis in range(3)]
    cases += [(body, component) for body in range(1, 15)
              for component in range(char.get_joint_dof_dim(body))]
    count = len(cases)
    root_pos = torch.zeros(count, 3)
    root_exp_map = torch.zeros(count, 3)
    dof = torch.zeros(count, 28)
    # Use the character's actual DOF mapping, including each hinge's own axis.
    # Joint limits/dynamic reachability are outside this geometry-only test.
    for index, (body, component) in enumerate(cases):
        if body == 0:
            root_exp_map[index, component] = torch.pi + offset
        else:
            dof[index, char.get_joint_dof_idx(body) + component] = torch.pi + offset
    root_rot = torch_util.exp_map_to_quat(root_exp_map)
    joint_rot = char.dof_to_rot(dof)
    body_pos, body_rot = char.forward_kinematics(root_pos, root_rot, joint_rot)
    identity_root = torch.zeros_like(root_rot)
    identity_root[:, -1] = 1
    identity_joints = char.dof_to_rot(torch.zeros(count, 28))
    ref_pos, ref_rot = char.forward_kinematics(root_pos, identity_root, identity_joints)
    velocity = torch.zeros(count, 3)
    dof_velocity = torch.zeros(count, 28)
    policy = compute_disc_obs(root_pos[:, None], root_rot[:, None], velocity[:, None],
        velocity[:, None], joint_rot[:, None], dof_velocity[:, None], body_pos[:, None], True)
    reference = compute_disc_obs(root_pos[:, None], identity_root[:, None], velocity[:, None],
        velocity[:, None], identity_joints[:, None], dof_velocity[:, None], ref_pos[:, None], True)
    features = ADDPairErrorMetrics(env)(policy, reference)
    expected = compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
        root_pos, identity_root, ref_rot, ref_pos, velocity, velocity, dof_velocity,
        velocity, velocity, dof_velocity)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(aggregate_error_metrics(features), expected,
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(features[:, 16:30],
        torch_util.quat_diff_angle(body_rot[:, 1:], ref_rot[:, 1:]).abs(),
        rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(ADDPairErrorMetrics(env)(policy, policy),
                               torch.zeros_like(features), rtol=0, atol=1e-6)


@pytest.mark.parametrize("incompatible", ["global", "frames", "groups", "character"])
def test_unsupported_physical_layouts_fail_explicitly(incompatible):
    env = Env()
    if incompatible == "global":
        env._global_obs = False
    elif incompatible == "frames":
        env.get_num_disc_obs_steps = lambda: 2
    elif incompatible == "groups":
        env.get_disc_error_groups = lambda: ()
    else:
        env._kin_char_model.get_dof_size = lambda: 27
    with pytest.raises(ValueError, match="requires"):
        ADDPairErrorMetrics(env)


def test_same_actor_critic_and_no_new_loss_weights_or_reward_multipliers():
    env = Env()
    torch.manual_seed(3)
    baseline_config = config()["model"] | {"disc_net": "fc_2layers_1024units"}
    baseline = ADDModel(baseline_config, env)
    torch.manual_seed(3)
    monotone = DAREMonotoneModel(config()["model"], env)
    for a, b in zip(baseline.get_actor_params() + baseline.get_critic_params(),
                    monotone.get_actor_params() + monotone.get_critic_params()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    linears = [layer for layer in monotone._disc_energy if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (64, HIDDEN_WIDTHS[0]), (*HIDDEN_WIDTHS,), (HIDDEN_WIDTHS[1], 1)]
    assert all((layer.weight >= 0).all() for layer in linears)


def test_fixed_cmnn_activations_are_monotone_across_wide_input_range():
    values = torch.linspace(-100, 100, 2048)[:, None].expand(-1, 513).clone().requires_grad_(True)
    outputs = MixedELU()(values)
    gradients = torch.autograd.grad(outputs.sum(), values)[0]
    assert torch.isfinite(outputs).all() and torch.isfinite(gradients).all()
    assert (gradients >= 0).all()


def test_bce_updates_and_production_rewards_keep_every_metric_monotone():
    torch.manual_seed(47)
    env = Env()
    agent = DAREMonotoneAgent(config(), env, "cpu")
    (policy, _), (reference, _) = physical_pair(env)
    fill_buffers(agent, policy, reference)
    batch = {"disc_obs": policy[:16], "disc_obs_demo": reference[:16]}
    for _ in range(8):
        agent.train()
        losses = agent._compute_disc_loss(batch)
        assert all(torch.isfinite(value).all() for value in losses.values())
        agent._disc_optimizer.step(losses["disc_loss"])
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in agent._model.get_disc_params())
    agent.eval()
    before = torch.rand(128, FEATURE_DIM) * 20
    before = before[:, None].expand(-1, FEATURE_DIM, -1).reshape(-1, FEATURE_DIM).clone()
    after = before.clone()
    after[torch.arange(len(after)), torch.arange(FEATURE_DIM).repeat(128)] *= .9
    with torch.no_grad():
        scores = agent._model.eval_disc(torch.cat((before, after)))[:, 0]
    first, second = scores.chunk(2)
    assert (second >= first - 1e-6).all()
    r0, r1 = agent._calc_disc_rewards(before), agent._calc_disc_rewards(after)
    assert (r1 >= r0 - 1e-6).all()
    x = before[:128].requires_grad_(True)
    gradient = torch.autograd.grad(agent._model.eval_disc(x).sum(), x)[0]
    assert (gradient <= 1e-7).all()
    assert (gradient.norm(dim=-1) <= 1 + 5e-6).all()
    displacement = torch.linalg.vector_norm(before - after, dim=-1)
    assert ((second - first).abs() <= displacement + 5e-6).all()
    torch.testing.assert_close(agent._model.eval_disc(torch.zeros(1, FEATURE_DIM)).squeeze(),
                               agent._model._anchor_logit, rtol=0, atol=1e-7)
    for layer in agent._model._disc_energy:
        if isinstance(layer, torch.nn.Linear):
            assert (layer.weight >= 0).all()


@pytest.mark.parametrize("chunk", [0, 1, 9])
def test_paired_reward_api_uses_exact_original_add_clipping(chunk):
    env = Env()
    agent = DAREMonotoneAgent(config(), env, "cpu").eval()
    (policy, _), (reference, _) = physical_pair(env, count=33)
    agent._disc_eval_batch_size = chunk
    agent._disc_demo_batch_size = 7
    metrics = agent._model.pair_to_error_metrics(policy, reference)
    expected = calc_unscaled_disc_reward(agent._model.eval_disc(
        agent._disc_obs_norm.normalize(metrics)).squeeze(-1))
    actual = agent.calc_disc_reward_from_obs(policy, reference)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    assert not actual.requires_grad and actual.shape == (33,)
    assert agent.calc_disc_reward_from_obs(policy[:0], reference[:0]).shape == (0,)
    with pytest.raises(ValueError, match="matching"):
        agent.calc_disc_reward_from_obs(policy, reference[:1])
    for wrong_width in (7, 172, 344):
        with pytest.raises(ValueError, match="64"):
            agent._model.eval_disc(torch.zeros(8, wrong_width))


def test_normalizer_completion_preserves_zero_anchor_and_frozen_reward_function():
    env = Env()
    agent = DAREMonotoneAgent(config(), env, "cpu").eval()
    (policy, _), (reference, _) = physical_pair(env)
    fill_buffers(agent, policy, reference)
    agent._sample_count = agent._normalizer_samples - 1
    agent._compute_rewards()
    agent._disc_obs_norm.update()
    before = agent.calc_disc_reward_from_obs(policy, reference)
    model_state = copy.deepcopy(agent._model.state_dict())
    agent._sample_count = agent._normalizer_samples
    agent._compute_rewards()
    after = agent.calc_disc_reward_from_obs(policy, reference)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(agent._disc_obs_norm.normalize(torch.zeros(1, FEATURE_DIM)),
                               torch.zeros(1, FEATURE_DIM))
    assert (agent._disc_obs_norm.get_abs_mean() > 0).all()
    for name, value in model_state.items():
        torch.testing.assert_close(value, agent._model.state_dict()[name], rtol=0, atol=0)
    assert not hasattr(agent, "_disc_reward_scale")


def test_physical_normalizer_preserves_old_group_statistics_without_averaging_features():
    env = Env()
    agent = DAREMonotoneAgent(config(), env, "cpu").eval()
    (policy, _), (reference, _) = physical_pair(env, count=128)
    features = agent._model.pair_to_error_metrics(policy, reference)
    norm = agent._disc_obs_norm
    norm.record(features[:17])
    norm.record(features[17:])
    norm.update()
    expected_scales = aggregate_error_metrics(features).mean(0)
    torch.testing.assert_close(norm.get_abs_mean(), expected_scales, rtol=2e-6, atol=1e-6)
    normalized = norm.normalize(features)
    assert normalized.shape == features.shape
    for group, (start, stop) in enumerate(((0, 1), (1, 2), (2, 16), (16, 30),
                                         (30, 58), (58, 61), (61, 64))):
        group_factor = (stop - start) ** .5
        torch.testing.assert_close(normalized[:, start:stop],
                                   features[:, start:stop] /
                                   (expected_scales[group] * group_factor))
    torch.testing.assert_close(norm.unnormalize(normalized), features, rtol=2e-6, atol=1e-6)
    # A DOF's change uses the fixed group RMS factor, not a tunable reward
    # multiplier or a learned per-joint coefficient.
    correction = features.clone()
    correction[:, 30] += .25
    actual_delta = norm.normalize(correction) - normalized
    torch.testing.assert_close(actual_delta[:, 30],
                               torch.full((128,), .25 /
                                          (expected_scales[4] * 28 ** .5)),
                               rtol=2e-5, atol=1e-6)
    assert torch.count_nonzero(actual_delta[:, :30]) == 0
    assert torch.count_nonzero(actual_delta[:, 31:]) == 0


def test_strict_checkpoint_roundtrip_and_rejection_of_legacy_reward_semantics():
    env = Env()
    agent = DAREMonotoneAgent(config(), env, "cpu")
    (policy, _), (reference, _) = physical_pair(env)
    fill_buffers(agent, policy, reference)
    loss = agent._compute_disc_loss({"disc_obs": policy[:16], "disc_obs_demo": reference[:16]})
    agent._disc_optimizer.step(loss["disc_loss"])
    agent.eval()
    saved = copy.deepcopy(agent.state_dict())
    extra = copy.deepcopy(agent._get_checkpoint_extra_state())
    restored = DAREMonotoneAgent(config(), Env(), "cpu").eval()
    restored.load_state_dict(saved, strict=True)
    restored._load_checkpoint_extra_state(extra)
    restored._disc_optimizer.load_state_dict(copy.deepcopy(agent._disc_optimizer.state_dict()))
    torch.testing.assert_close(agent.calc_disc_reward_from_obs(policy, reference),
                               restored.calc_disc_reward_from_obs(policy, reference), rtol=0, atol=0)
    assert restored._disc_optimizer.get_steps() == 1
    bad = copy.deepcopy(saved)
    bad.pop("_model._monotone_schema_version")
    with pytest.raises(RuntimeError, match="monotone"):
        restored.load_state_dict(bad, strict=False)
    bad = copy.deepcopy(saved)
    bad["_model._monotone_schema_version"] = torch.tensor(2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="monotone"):
        restored.load_state_dict(bad, strict=False)
    bad = copy.deepcopy(saved)
    bad["_model._pair_metrics.parent_ids"][0] = 11
    with pytest.raises(RuntimeError, match="character schema"):
        restored.load_state_dict(bad, strict=False)
    bad = copy.deepcopy(saved)
    bad["_disc_obs_norm._feature_groups"][30] = 0
    with pytest.raises(RuntimeError, match="normalizer|feature|schema"):
        restored.load_state_dict(bad, strict=False)
    bad = copy.deepcopy(saved)
    bad.pop("_disc_obs_norm._feature_groups")
    with pytest.raises(RuntimeError, match="normalizer|feature|schema"):
        restored.load_state_dict(bad, strict=False)
    bad_extra = copy.deepcopy(extra)
    bad_extra["monotone_disc"]["reward"] = "scaled"
    with pytest.raises(ValueError, match="monotone"):
        restored._load_checkpoint_extra_state(bad_extra)
    bad_extra = copy.deepcopy(extra)
    bad_extra["dare_reward"]["version"] = 1
    with pytest.raises(ValueError, match="legacy scaled"):
        restored._load_checkpoint_extra_state(bad_extra)


def test_training_checkpoint_keeps_pending_group_scale_statistics(tmp_path):
    agent = DAREMonotoneAgent(config(), Env(), "cpu").eval()
    (policy, _), (reference, _) = physical_pair(agent._env)
    fill_buffers(agent, policy, reference)
    features = agent._model.pair_to_error_metrics(policy, reference)
    agent._disc_obs_norm.record(features[:17])
    checkpoint_path = tmp_path / "physical64_checkpoint.pt"
    agent.save_checkpoint(checkpoint_path, next_iter=8)
    payload = agent._torch_load(checkpoint_path)
    pending = payload["normalizer_training_states"]["_disc_obs_norm"]
    assert pending["new_count"] == 17
    torch.testing.assert_close(pending["new_sum_abs"],
                               aggregate_error_metrics(features[:17]).sum(0))
    restored = DAREMonotoneAgent(config(), Env(), "cpu").eval()
    restored.resume(checkpoint_path)
    assert restored._iter == 8
    for current in (agent, restored):
        current._disc_obs_norm.record(features[17:])
        current._disc_obs_norm.update()
    torch.testing.assert_close(agent._disc_obs_norm.get_abs_mean(),
                               aggregate_error_metrics(features).mean(0))
    torch.testing.assert_close(restored._disc_obs_norm.get_abs_mean(),
                               agent._disc_obs_norm.get_abs_mean(), rtol=0, atol=0)
    torch.testing.assert_close(restored.calc_disc_reward_from_obs(policy, reference),
                               agent.calc_disc_reward_from_obs(policy, reference), rtol=0, atol=0)


def test_structured_negatives_and_old_reward_scale_are_rejected():
    for name, value in (("disc_group_negatives", True), ("disc_reward_scale", 2)):
        candidate = config()
        candidate[name] = value
        with pytest.raises(ValueError):
            DAREMonotoneAgent(candidate, Env(), "cpu")
