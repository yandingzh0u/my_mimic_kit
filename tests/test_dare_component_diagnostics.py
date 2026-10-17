"""Single-loss ablations must not silently change other normalizations."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from phasebalanced_component_probe import install_loss_hooks, component_objective
from test_ppo_phasebalanced import _agent
from learning.ppo_agent import PPOAgent


def test_no_actor_weight_matches_unit_weight_reference_and_leaves_buffer_alone():
    agent = _agent()
    batch = dict(agent._exp_buffer._flat_buffers)
    original = batch["phase_weight"].clone()
    unit = dict(batch, phase_weight=torch.ones_like(original))
    expected = PPOAgent._compute_actor_loss(agent, unit)["actor_loss"]
    install_loss_hooks(agent, "no_actor_weight")
    actual = agent._compute_actor_loss(batch)["actor_loss"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, agent._model.weight)[0],
                               torch.autograd.grad(expected, agent._model.weight)[0])
    torch.testing.assert_close(batch["phase_weight"], original)


def test_no_critic_weight_does_not_change_actor_loss_or_gradient():
    agent = _agent()
    batch = dict(agent._exp_buffer._flat_buffers)
    expected = agent._compute_actor_loss(batch)["actor_loss"]
    expected_grad = torch.autograd.grad(expected, agent._model.weight)[0]
    install_loss_hooks(agent, "no_critic_weight")
    actual = agent._compute_actor_loss(batch)["actor_loss"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, agent._model.weight)[0], expected_grad)


def test_weighted_advantage_normalization_is_one_global_affine_transform():
    agent = _agent()
    raw = agent._exp_buffer.get_data_flat("adv")
    weights = agent._exp_buffer.get_data_flat("phase_weight")
    mw, sw = agent._weighted_mean_std(raw, weights)
    mu, su = agent._weighted_mean_std(raw, torch.ones_like(weights))
    weighted = (raw - mw) / sw
    uniform = (raw - mu) / su
    # No independent per-phase advantage centers/scales are estimated.
    torch.testing.assert_close(uniform, (sw / su) * weighted + (mw - mu) / su)
    torch.testing.assert_close((weighted * weights).sum() / weights.sum(), torch.tensor(0.), atol=1e-6, rtol=0)


def test_component_objective_uses_ablation_specific_global_denominators():
    agent = _agent()
    agent._model.eval_critic = lambda obs: obs[:, :1] * 0.3
    agent._exp_buffer.set_data("tar_val", torch.ones(1, agent.get_num_envs()))
    batch = dict(agent._exp_buffer._flat_buffers)
    for branch in ("all", "no_actor_weight", "no_critic_weight", "no_adv_weight"):
        agent._component_branch = branch
        install_loss_hooks(agent, branch)
        actual = component_objective(agent)
        expected_actor = agent._compute_actor_loss(batch)
        expected_critic = agent._compute_critic_loss(batch)
        assert abs(actual["actor_total_loss"] - float(expected_actor["actor_loss"])) < 1e-6
        assert abs(actual["critic_mse"] - float(expected_critic["critic_loss"])) < 1e-6
