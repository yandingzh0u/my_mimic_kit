from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

from learning.action_pullback_add_agent import (
    build_private_generator,
    differentiable_add_reward,
    linearized_pullback_loss,
    response_target_from_closure,
    split_aligned_command,
    update_response_normalizers_from_train,
    zero_motion_skill,
)


def test_response_target_is_exact_closure_increment():
    self_dim = 3
    command_dim = 5
    self_obs = torch.randn(7, self_dim)
    next_self_obs = torch.randn(7, self_dim)
    error = torch.randn(7, command_dim)
    motion = torch.randn(7, command_dim)
    realized_delta = torch.randn(7, command_dim)
    next_error = error + motion - realized_delta
    obs = torch.cat([self_obs, error, motion], dim=-1)
    next_obs = torch.cat(
        [next_self_obs, next_error, torch.randn_like(motion)], dim=-1)

    target = response_target_from_closure(
        obs, next_obs, self_dim, command_dim)
    assert torch.allclose(target, realized_delta, atol=1e-6, rtol=1e-6)


def test_split_rejects_wrong_observation_size():
    with pytest.raises(ValueError):
        split_aligned_command(torch.zeros(2, 9), 2, 3)


def test_linearized_loss_moves_action_along_reward_gradient():
    action = torch.tensor([[0.2, -0.3]], requires_grad=True)
    reward_grad = torch.tensor([[1.5, -2.0]])
    reference = action.detach().clone()
    loss = linearized_pullback_loss(action, reward_grad, reference)
    grad = torch.autograd.grad(loss, action)[0]
    assert torch.allclose(grad, -reward_grad)
    updated = action.detach() - 0.01 * grad
    displacement = updated - action.detach()
    assert torch.sum(displacement * reward_grad) > 0


def test_linearized_loss_truncates_outside_local_box_support():
    action = torch.tensor([[0.2, -0.2]], requires_grad=True)
    reward_grad = torch.tensor([[1.0, -1.0]])
    reference = torch.zeros_like(action)
    loss = linearized_pullback_loss(
        action, reward_grad, reference, action_delta_clip=0.05)
    grad = torch.autograd.grad(loss, action)[0]
    assert torch.equal(grad, torch.zeros_like(grad))


def test_private_generator_does_not_advance_global_rng_and_samples_full_range():
    torch.manual_seed(1234)
    global_state = torch.get_rng_state().clone()
    generator = build_private_generator("cpu", 0xA0D17)
    idx = torch.randperm(128, generator=generator)[:16]
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.any(idx >= 16)


class _RecordingNormalizer:
    def __init__(self):
        self.recorded = None
        self.num_updates = 0

    def record(self, value):
        self.recorded = value.detach().clone()

    def update(self):
        self.num_updates += 1


def test_response_normalizers_only_observe_training_indices():
    self_obs = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    target_delta = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    train_idx = torch.tensor([0, 2, 5])
    self_norm = _RecordingNormalizer()
    delta_norm = _RecordingNormalizer()

    update_response_normalizers_from_train(
        self_obs, target_delta, train_idx, self_norm, delta_norm)

    assert torch.equal(self_norm.recorded, self_obs[train_idx])
    assert torch.equal(delta_norm.recorded, target_delta[train_idx])
    assert self_norm.num_updates == 1
    assert delta_norm.num_updates == 1


def test_response_skill_is_zero_motion_baseline_skill_not_r_squared():
    loss = torch.tensor(0.25)
    zero_loss = torch.tensor(1.0)
    assert zero_motion_skill(loss, zero_loss).item() == pytest.approx(0.75)


def test_differentiable_reward_matches_add_rollout_formula():
    logits = torch.linspace(-8.0, 12.0, 41, requires_grad=True)
    reward = differentiable_add_reward(logits, 2.0)
    prob = torch.sigmoid(logits)
    expected = -torch.log(torch.maximum(
        1.0 - prob, torch.tensor(1e-4))) * 2.0
    assert torch.allclose(reward, expected)
    grad = torch.autograd.grad(reward.sum(), logits)[0]
    assert torch.all(torch.isfinite(grad))


def test_config_and_builder_contract():
    config = yaml.safe_load((
        ROOT / "data/agents/action_pullback_add_humanoid_agent.yaml"
    ).read_text())
    assert config["agent_name"] == "ACTION_PULLBACK_ADD"
    assert config["response_epochs"] > 0
    assert config["response_validation_size"] > 0
    assert config["pullback_weight"] > 0
    assert config["pullback_normalize_direction"] is True
    assert config["pullback_min_grad_norm"] > 0
    assert config["pullback_action_delta_clip"] > 0
    assert config["pullback_audit_batch_size"] > 0
    assert config["model"]["response_hidden_units"] == [512, 512]

    builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'agent_name == "ACTION_PULLBACK_ADD"' in builder

    smoke_args = (
        ROOT / "args/action_pullback_add_humanoid_roll_smoke_args.txt"
    ).read_text()
    formal_args = (
        ROOT / "args/action_pullback_add_humanoid_roll_2k_8192_args.txt"
    ).read_text()
    pilot_args = (
        ROOT / "args/action_pullback_add_humanoid_roll_50_8192_args.txt"
    ).read_text()
    assert "--num_envs 64" in smoke_args
    assert "--num_envs 8192" in formal_args
    assert "--max_samples 524288000" in formal_args
    assert "--num_envs 8192" in pilot_args
    assert "--max_samples 13107200" in pilot_args
