import inspect
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


def test_zero_is_exact_scalar_anchor():
    torch.manual_seed(7)
    model = ADDModel(_config(), _Env()).eval()
    with torch.no_grad():
        model.get_disc_anchor_bias().fill_(0.37)
    logits = model.eval_disc(torch.zeros(16, 12)).squeeze(-1)
    torch.testing.assert_close(logits, torch.full_like(logits, 0.37))


def test_sage_is_function_equivalent_to_unanchored_a30_score():
    torch.manual_seed(11)
    model = ADDModel(_config(), _Env()).eval()
    diff = torch.randn(64, 12)
    zero = torch.zeros(1, 12)

    # q and q(0) use the exact a30 groupwise-SN trunk and signed head.
    with torch.no_grad():
        features = model._disc_layers(torch.cat((diff, zero), dim=0))
        raw_scores = model._disc_relative_head(features).squeeze(-1)
        output_bias = torch.tensor(-0.23)
        a30_logits = output_bias + raw_scores[:-1]
        model.get_disc_anchor_bias().copy_(output_bias + raw_scores[-1])

    sage_logits = model.eval_disc(diff).squeeze(-1)
    torch.testing.assert_close(sage_logits, a30_logits, atol=1e-6, rtol=1e-6)


def test_positive_bce_gradient_is_isolated_to_anchor_bias():
    torch.manual_seed(19)
    model = ADDModel(_config(), _Env()).eval()
    positive = model.eval_disc(torch.zeros(8, 12)).squeeze(-1)
    torch.nn.functional.softplus(-positive).mean().backward()

    assert model.get_disc_anchor_bias().grad is not None
    assert torch.abs(model.get_disc_anchor_bias().grad) > 0
    for name, parameter in model.named_parameters():
        if name == "_disc_anchor_bias":
            continue
        if name.startswith("_disc_layers") or name.startswith(
                "_disc_relative_head"):
            if parameter.grad is not None:
                torch.testing.assert_close(
                    parameter.grad, torch.zeros_like(parameter.grad),
                    atol=1e-8, rtol=0)


def test_negative_bce_updates_signed_field_and_anchor():
    torch.manual_seed(23)
    model = ADDModel(_config(), _Env())
    negative = model.eval_disc(torch.randn(32, 12)).squeeze(-1)
    torch.nn.functional.softplus(negative).mean().backward()

    encoder_weight = model._disc_layers.encoders[0][0] \
        .parametrizations.weight.original
    head_weight = model._disc_relative_head.parametrizations.weight.original
    assert torch.linalg.vector_norm(encoder_weight.grad) > 0
    assert torch.linalg.vector_norm(head_weight.grad) > 0
    assert torch.abs(model.get_disc_anchor_bias().grad) > 0


def test_signed_score_is_empirically_one_lipschitz():
    torch.manual_seed(29)
    model = ADDModel(_config(), _Env()).eval()
    x = torch.randn(256, 12)
    y = torch.randn(256, 12)
    input_delta = torch.linalg.vector_norm(x - y, dim=-1)
    logit_delta = torch.abs(
        model.eval_disc(x).squeeze(-1) - model.eval_disc(y).squeeze(-1))
    assert torch.all(logit_delta <= input_delta * 1.01 + 1e-6)


def test_all_discriminator_linears_are_spectral_normalized():
    model = ADDModel(_config(), _Env())
    linears = [
        *[encoder[0] for encoder in model._disc_layers.encoders],
        *[layer for layer in model._disc_layers.trunk
          if isinstance(layer, torch.nn.Linear)],
        model._disc_relative_head,
    ]
    assert len(linears) == 4
    assert all(torch.nn.utils.parametrize.is_parametrized(layer, "weight")
               for layer in linears)
    assert model._disc_relative_head.bias is None
    assert tuple(model.get_disc_anchor_bias().shape) == (1,)


def test_no_radial_distance_path_remains():
    source = inspect.getsource(sys.modules[ADDModel.__module__])
    assert "vector_norm" not in source
    assert "eval_disc_distance" not in source


def test_sage_config_removes_external_discriminator_regularizers():
    with (ROOT / "data/agents/sage_add_humanoid_agent.yaml").open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_grad_penalty"] == 0
    assert config["disc_logit_reg"] == 0
    assert config["disc_reward_scale"] == 2
    assert "disc_spectral_norm" not in config
    assert "disc_group_separable_frontend" not in config
