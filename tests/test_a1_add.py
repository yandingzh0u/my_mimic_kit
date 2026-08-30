import inspect
import math
import pathlib
import sys

import gymnasium.spaces as spaces
import numpy as np
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from envs.add_env import build_disc_error_groups
import learning.add_agent as add_agent
import learning.add_model as add_model
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


def test_encoder_inputs_use_only_group_rms_scaling():
    model = ADDModel(_config(), _Env()).eval()
    captured = []
    handles = [encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()))
        for encoder in model._disc_layers.encoders]

    diff = torch.full((4, 12), 0.75)
    model.eval_disc(diff)
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(
        captured[0], diff[:, :3] / math.sqrt(3))
    torch.testing.assert_close(
        captured[1], diff[:, 3:] / math.sqrt(9))
    small_rms = torch.linalg.vector_norm(captured[0], dim=-1)
    large_rms = torch.linalg.vector_norm(captured[1], dim=-1)
    torch.testing.assert_close(small_rms, large_rms)


def test_signed_head_directly_scores_group_separable_features():
    torch.manual_seed(11)
    model = ADDModel(_config(), _Env()).eval()
    diff = torch.randn(64, 12)
    with torch.no_grad():
        expected = model._disc_logits(model._disc_layers(diff))
    torch.testing.assert_close(model.eval_disc(diff), expected)


def test_positive_zero_supervision_reaches_the_entire_a30_critic():
    torch.manual_seed(19)
    model = ADDModel(_config(), _Env()).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.fill_(0.1)

    positive = model.eval_disc(torch.zeros(8, 12)).squeeze(-1)
    torch.nn.functional.softplus(-positive).mean().backward()

    encoder_bias = model._disc_layers.encoders[0][0].bias
    trunk_weight = model._disc_layers.trunk[0] \
        .parametrizations.weight.original
    head_weight = model._disc_logits.parametrizations.weight.original
    assert torch.linalg.vector_norm(encoder_bias.grad) > 0
    assert torch.linalg.vector_norm(trunk_weight.grad) > 0
    assert torch.linalg.vector_norm(head_weight.grad) > 0
    assert torch.abs(model._disc_logits.bias.grad) > 0


def test_every_discriminator_linear_keeps_a30_spectral_normalization():
    model = ADDModel(_config(), _Env())
    linears = [
        *[encoder[0] for encoder in model._disc_layers.encoders],
        *[layer for layer in model._disc_layers.trunk
          if isinstance(layer, torch.nn.Linear)],
        model._disc_logits,
    ]
    assert len(linears) == 4
    assert all(torch.nn.utils.parametrize.is_parametrized(layer, "weight")
               for layer in linears)


def test_no_superseded_discriminator_mechanism_remains():
    source = inspect.getsource(add_agent) + inspect.getsource(add_model)
    forbidden = (
        "semantic_core",
        "cert_scale",
        "theoretical_floor",
        "build_semantic_contractions",
        "calc_pc_loss",
        "anchor",
        "distance_head",
        "ref_concat",
        "group_balanced",
    )
    assert all(token not in source.lower() for token in forbidden)


def test_a1_config_matches_a30_regularization_and_reward_protocol():
    with (ROOT / "data/agents/a1_add_humanoid_agent.yaml").open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_grad_penalty"] == 0
    assert config["disc_logit_reg"] == 0.01
    assert config["disc_reward_scale"] == 2
    assert config["task_reward_weight"] == 0
    assert config["disc_reward_weight"] == 1
    forbidden = (
        "disc_geometry", "disc_spectral_norm",
        "disc_group_separable_frontend", "disc_group_balanced_metric",
        "disc_group_balanced_gp", "disc_influence_allocation",
    )
    assert all(key not in config for key in forbidden)
