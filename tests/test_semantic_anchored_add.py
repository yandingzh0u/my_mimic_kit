import pathlib
import sys

import gymnasium.spaces as spaces
import numpy as np
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from envs.add_env import build_disc_error_groups
from learning.add_model import ADDModel


class _Env:
    def __init__(self):
        self._obs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self._action = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def get_obs_space(self):
        return self._obs

    def get_action_space(self):
        return self._action

    def get_disc_obs_space(self):
        return self._obs

    def get_disc_error_groups(self):
        return (
            ("small", tuple(range(3))),
            ("large", tuple(range(3, 12))),
        )


def _config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def test_semantic_groups_cover_each_differential_coordinate_once():
    groups = build_disc_error_groups(
        num_steps=2, num_joints=4, num_bodies=5, num_dofs=8,
        total_dim=2 * (3 + 6 + 18 + 15 + 3 + 3 + 8))
    indices = [index for _, group in groups for index in group]
    assert len(groups) == 7
    assert sorted(indices) == list(range(len(indices)))


def test_zero_is_exact_anchor_and_global_logit_maximum():
    torch.manual_seed(7)
    model = ADDModel(_config(), _Env()).eval()
    zero = torch.zeros(4, 12)
    diff = torch.randn(32, 12)

    zero_embedding = model.eval_disc_embedding(zero)
    zero_logit = model.eval_disc(zero).squeeze(-1)
    diff_logit = model.eval_disc(diff).squeeze(-1)

    torch.testing.assert_close(zero_embedding, torch.zeros_like(zero_embedding))
    torch.testing.assert_close(
        zero_logit, model.get_disc_bias().expand_as(zero_logit))
    assert torch.all(diff_logit <= model.get_disc_bias() + 1e-7)


def test_centered_embedding_and_logit_are_empirically_one_lipschitz():
    torch.manual_seed(11)
    model = ADDModel(_config(), _Env()).eval()
    x = torch.randn(256, 12)
    y = torch.randn(256, 12)

    embedding_delta = torch.linalg.vector_norm(
        model.eval_disc_embedding(x) - model.eval_disc_embedding(y), dim=-1)
    input_delta = torch.linalg.vector_norm(x - y, dim=-1)
    logit_delta = torch.abs(
        model.eval_disc(x).squeeze(-1) - model.eval_disc(y).squeeze(-1))

    assert torch.all(embedding_delta <= input_delta * 1.01 + 1e-6)
    assert torch.all(logit_delta <= input_delta * 1.01 + 1e-6)


def test_all_embedding_linears_are_spectral_normalized_and_head_is_scalar():
    model = ADDModel(_config(), _Env())
    linears = [
        layer for layer in model._disc_layers.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert len(linears) == 3
    assert all(torch.nn.utils.parametrize.is_parametrized(layer, "weight")
               for layer in linears)
    assert not hasattr(model, "_disc_logits")
    assert tuple(model.get_disc_bias().shape) == (1,)


def test_discriminator_loss_updates_embedding_and_anchor_bias():
    torch.manual_seed(19)
    model = ADDModel(_config(), _Env())
    positive = model.eval_disc(torch.zeros(8, 12)).squeeze(-1)
    negative = model.eval_disc(torch.randn(8, 12)).squeeze(-1)
    loss = (torch.nn.functional.softplus(-positive).mean()
            + torch.nn.functional.softplus(negative).mean())
    loss.backward()

    encoder_grad = model._disc_layers.encoders[0][0].parametrizations.weight \
        .original.grad
    assert encoder_grad is not None
    assert torch.isfinite(encoder_grad).all()
    assert torch.linalg.vector_norm(encoder_grad) > 0
    assert model.get_disc_bias().grad is not None
    assert torch.isfinite(model.get_disc_bias().grad).all()


def test_sadd_config_removes_external_discriminator_regularizers():
    with (ROOT / "data/agents/sadd_humanoid_agent.yaml").open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_grad_penalty"] == 0
    assert config["disc_logit_reg"] == 0
    assert "disc_spectral_norm" not in config
    assert "disc_group_separable_frontend" not in config
