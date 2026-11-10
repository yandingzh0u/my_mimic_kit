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
    ADDPairErrorMetrics, DAREMonotoneModel, HIDDEN_WIDTHS, MixedELU)
from learning.add_model import ADDModel
import util.torch_util as torch_util


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/agents/dare_monotone_humanoid_agent.yaml"


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
    actual = ADDPairErrorMetrics(env)(policy, reference)
    expected = compute_tracking_error(
        root_pos=p[0], root_rot=p[1], body_rot=p[2], body_pos=p[3],
        tar_root_pos=r[0], tar_root_rot=r[1], tar_body_rot=r[2], tar_body_pos=r[3],
        root_vel=p[4], root_ang_vel=p[5], dof_vel=p[6],
        tar_root_vel=r[4], tar_root_ang_vel=r[5], tar_dof_vel=r[6])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert (actual >= 0).all()
    zero = ADDPairErrorMetrics(env)(policy, policy)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=1e-6)
    assert ADDPairErrorMetrics(env)(policy[:0], reference[:0]).shape == (0, 7)


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
        (7, HIDDEN_WIDTHS[0]), (*HIDDEN_WIDTHS,), (HIDDEN_WIDTHS[1], 1)]
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
    before = torch.rand(2048, 7) * 20
    before = before[:, None].expand(-1, 7, -1).reshape(-1, 7).clone()
    after = before.clone()
    after[torch.arange(len(after)), torch.arange(7).repeat(2048)] *= .9
    with torch.no_grad():
        scores = agent._model.eval_disc(torch.cat((before, after)))[:, 0]
    first, second = scores.chunk(2)
    assert (second >= first - 1e-6).all()
    r0, r1 = agent._calc_disc_rewards(before), agent._calc_disc_rewards(after)
    assert (r1 >= r0 - 1e-6).all()
    x = before[:128].requires_grad_(True)
    gradient = torch.autograd.grad(agent._model.eval_disc(x).sum(), x)[0]
    assert (gradient <= 1e-7).all()
    torch.testing.assert_close(agent._model.eval_disc(torch.zeros(1, 7)).squeeze(),
                               agent._model._anchor_logit, rtol=0, atol=0)
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
    with pytest.raises(ValueError, match="seven"):
        agent._model.eval_disc(torch.zeros(8, 172))


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
    torch.testing.assert_close(agent._disc_obs_norm.normalize(torch.zeros(1, 7)), torch.zeros(1, 7))
    assert (agent._disc_obs_norm.get_abs_mean() > 0).all()
    for name, value in model_state.items():
        torch.testing.assert_close(value, agent._model.state_dict()[name], rtol=0, atol=0)
    assert not hasattr(agent, "_disc_reward_scale")


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
    bad["_model._pair_metrics.parent_ids"][0] = 11
    with pytest.raises(RuntimeError, match="character schema"):
        restored.load_state_dict(bad, strict=False)
    bad_extra = copy.deepcopy(extra)
    bad_extra["monotone_disc"]["reward"] = "scaled"
    with pytest.raises(ValueError, match="monotone"):
        restored._load_checkpoint_extra_state(bad_extra)
    bad_extra = copy.deepcopy(extra)
    bad_extra["dare_reward"]["version"] = 1
    with pytest.raises(ValueError, match="legacy scaled"):
        restored._load_checkpoint_extra_state(bad_extra)


def test_structured_negatives_and_old_reward_scale_are_rejected():
    for name, value in (("disc_group_negatives", True), ("disc_reward_scale", 2)):
        candidate = config()
        candidate[name] = value
        with pytest.raises(ValueError):
            DAREMonotoneAgent(candidate, Env(), "cpu")
