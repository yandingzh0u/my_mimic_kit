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


def _warm_spectral_norm(model, diff):
    model.train()
    with torch.no_grad():
        for _ in range(20):
            model.eval_disc(diff)
    return model.eval()


def test_semantic_groups_cover_each_differential_coordinate_once():
    groups = build_disc_error_groups(
        num_steps=2, num_joints=4, num_bodies=5, num_dofs=8,
        total_dim=2 * (3 + 6 + 18 + 15 + 3 + 3 + 8))
    indices = [index for _, group in groups for index in group]
    assert len(groups) == 7
    assert sorted(indices) == list(range(len(indices)))


def test_group_rms_coordinates_are_dimension_invariant():
    model = ADDModel(_config(), _Env()).eval()
    diff = torch.ones(5, 12) * 0.75
    with torch.no_grad():
        _, _, _, radii = model.eval_disc_components(diff)
    torch.testing.assert_close(radii[:, 0], radii[:, 1])
    torch.testing.assert_close(radii, torch.full_like(radii, 0.75))


def test_semantic_contraction_has_certified_positive_slope():
    torch.manual_seed(7)
    model = _warm_spectral_norm(ADDModel(_config(), _Env()),
                                torch.randn(256, 12))
    diff = torch.randn(128, 12)
    base_logit, _, _, radii = model.eval_disc_components(diff)
    base_logit = base_logit.squeeze(-1)
    tau = 0.37
    floor = 1.0 / (1.0 + 2.0 * math.sqrt(2.0))

    for group_id, (_, indices) in enumerate(_Env().get_disc_error_groups()):
        contracted = diff.clone()
        contracted[:, indices] *= tau
        contracted_logit = model.eval_disc(contracted).squeeze(-1)
        guaranteed_gain = floor * (1.0 - tau) * radii[:, group_id]
        assert torch.all(contracted_logit - base_logit
                         >= guaranteed_gain - 1e-5)


def test_zero_is_global_score_maximum_on_random_inputs():
    torch.manual_seed(13)
    model = _warm_spectral_norm(ADDModel(_config(), _Env()),
                                torch.randn(256, 12))
    diff = torch.randn(512, 12)
    zero_logit = model.eval_disc(torch.zeros_like(diff)).squeeze(-1)
    diff_logit = model.eval_disc(diff).squeeze(-1)
    assert torch.all(zero_logit > diff_logit)


def test_positive_zero_supervision_reaches_the_entire_residual_critic():
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


def test_full_score_is_empirically_one_lipschitz():
    torch.manual_seed(29)
    model = _warm_spectral_norm(ADDModel(_config(), _Env()),
                                torch.randn(256, 12))
    x = torch.randn(256, 12)
    y = torch.randn(256, 12)
    input_delta = torch.linalg.vector_norm(x - y, dim=-1)
    logit_delta = torch.abs(
        model.eval_disc(x).squeeze(-1) - model.eval_disc(y).squeeze(-1))
    assert torch.all(logit_delta <= input_delta + 1e-5)


def test_all_residual_linears_are_spectral_normalized():
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


def test_no_pc_ranking_path_remains():
    agent_source = inspect.getsource(add_agent)
    forbidden = (
        "build_semantic_contractions",
        "calc_pc_loss",
        "disc_pc_loss",
        "disc_pc_acc",
        "disc_pc_margin",
    )
    assert all(token not in agent_source for token in forbidden)


def test_pdr_config_has_no_external_discriminator_regularizer():
    with (ROOT / "data/agents/pdr_add_humanoid_agent.yaml").open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_grad_penalty"] == 0
    assert config["disc_logit_reg"] == 0
    assert config["disc_reward_scale"] == 2
    assert "disc_spectral_norm" not in config
    assert "disc_group_separable_frontend" not in config
