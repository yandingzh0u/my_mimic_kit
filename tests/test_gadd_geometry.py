import pathlib
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from envs.add_env import build_disc_error_groups
from learning.add_agent import (
    calc_influence_allocation_loss,
    calc_unscaled_disc_reward,
)
from learning.add_model import ADDModel


class _Env:
    def __init__(self, dim=12):
        self._obs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self._action = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def get_obs_space(self):
        return self._obs

    def get_disc_obs_space(self):
        return self._obs

    def get_action_space(self):
        return self._action


def _config(geometry="add"):
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "disc_geometry": geometry,
    }


@pytest.mark.parametrize(
    "geometry,input_dim", [("add", 12), ("ref_concat", 24)])
def test_discriminator_geometry_shapes(geometry, input_dim):
    model = ADDModel(_config(geometry), _Env())
    diff = torch.randn(7, 12)
    context = torch.randn(7, 12)
    disc_input = model.build_disc_input(diff, context)
    assert disc_input.shape == (7, input_dim)
    assert model.eval_disc(diff, context).shape == (7, 1)


def test_discriminator_uses_original_unparametrized_add_layers():
    model = ADDModel(_config(), _Env())
    hidden = [
        layer for layer in model._disc_layers.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert len(hidden) == 2
    assert all(not hasattr(layer, "parametrizations") for layer in hidden)
    assert not hasattr(model._disc_logits, "parametrizations")


def test_disc_error_groups_cover_multiframe_input_exactly_once():
    num_steps = 2
    num_joints = 5
    num_bodies = 5
    num_dofs = 9
    frame_dim = (
        3 + 6 + 6 * (num_joints - 1) + 3 * num_bodies + 3 + 3
        + num_dofs)
    total_dim = num_steps * frame_dim
    groups = build_disc_error_groups(
        num_steps, num_joints, num_bodies, num_dofs, total_dim)

    assert [name for name, _ in groups] == [
        "root_pos", "root_rot", "body_pos", "body_rot",
        "root_vel", "root_ang_vel", "dof_vel"]
    all_indices = [index for _, indices in groups for index in indices]
    assert sorted(all_indices) == list(range(total_dim))
    assert len(all_indices) == len(set(all_indices))


def test_disc_error_groups_reject_layout_mismatch():
    with pytest.raises(ValueError, match="layout mismatch"):
        build_disc_error_groups(1, 5, 5, 9, total_dim=3)


def test_signed_allocation_loss_uses_margin_targets_and_factual_gain():
    gains = torch.tensor([-0.2, 0.8], requires_grad=True)
    margin = torch.tensor(2.0, requires_grad=True)
    target = torch.tensor([0.75, 0.25], requires_grad=True)
    loss, desired = calc_influence_allocation_loss(gains, margin, target)
    loss.backward()

    assert torch.allclose(desired, torch.tensor([1.5, 0.5]))
    assert gains.grad[0] < 0  # gradient descent raises the negative gain
    assert gains.grad[1] > 0  # and lowers an over-allocated gain
    assert margin.grad is None
    assert target.grad is None


def test_reward_space_allocation_rejects_saturated_logit_gain():
    factual_logit = torch.tensor(-35.0)
    counterfactual_logit = torch.full((7,), -29.0)
    positive_logit = torch.tensor(5.18)
    factual_reward = calc_unscaled_disc_reward(factual_logit)
    gains = calc_unscaled_disc_reward(counterfactual_logit) - factual_reward
    margin = calc_unscaled_disc_reward(positive_logit) - factual_reward
    target = torch.full((7,), 1.0 / 7.0)
    loss, desired = calc_influence_allocation_loss(gains, margin, target)

    assert torch.all(gains < 1e-10)
    assert torch.all(desired > 0.7)
    assert loss > 1.8


def test_gadd_config_keeps_original_gp_and_only_adds_allocation():
    path = ROOT / "data" / "agents" / "gadd_humanoid_agent.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_geometry"] == "add"
    assert "disc_spectral_norm" not in config
    assert config["disc_influence_allocation"] is True
    assert config["disc_grad_penalty"] == 2


def test_failed_metric_configs_are_removed():
    for filename in (
            "gadd_global_metric_humanoid_agent.yaml",
            "gadd_metric_raw_gp_humanoid_agent.yaml",
            "gadd_metric_z_gp_humanoid_agent.yaml"):
        assert not (ROOT / "data" / "agents" / filename).exists()
