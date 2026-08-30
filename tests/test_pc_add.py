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
from learning.add_agent import build_semantic_contractions, calc_pc_loss
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


def test_each_counterfactual_contracts_exactly_one_group():
    diff = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    groups = _Env().get_disc_error_groups()
    tau = torch.tensor([[[0.25], [0.50]], [[0.75], [0.10]]])
    contractions = build_semantic_contractions(diff, groups, tau)

    for group_id, (_, indices) in enumerate(groups):
        complement = sorted(set(range(12)) - set(indices))
        torch.testing.assert_close(
            contractions[group_id, :, indices],
            diff[:, indices] * tau[group_id])
        torch.testing.assert_close(
            contractions[group_id, :, complement], diff[:, complement])


def test_ignored_group_has_log2_loss_and_nonzero_correction_gradient():
    pos = torch.zeros(1, requires_grad=True)
    neg = torch.zeros(5, requires_grad=True)
    contraction = torch.zeros(2, 5, requires_grad=True)
    loss, pos_loss, neg_loss, preference_loss = calc_pc_loss(
        pos, neg, contraction)

    expected = torch.tensor(np.log(2), dtype=torch.float32)
    torch.testing.assert_close(pos_loss, expected)
    torch.testing.assert_close(neg_loss, expected)
    torch.testing.assert_close(preference_loss, expected.expand(2))
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert torch.all(contraction.grad < 0)
    assert torch.all(neg.grad > 0)


def test_direct_signed_critic_has_no_zero_anchor_subtraction():
    torch.manual_seed(11)
    model = ADDModel(_config(), _Env()).eval()
    diff = torch.randn(64, 12)
    with torch.no_grad():
        expected = model._disc_logits(model._disc_layers(diff))
    torch.testing.assert_close(model.eval_disc(diff), expected)


def test_positive_zero_supervision_reaches_shared_critic():
    torch.manual_seed(19)
    model = ADDModel(_config(), _Env()).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.fill_(0.1)

    positive = model.eval_disc(torch.zeros(8, 12)).squeeze(-1)
    torch.nn.functional.softplus(-positive).mean().backward()

    head_weight = model._disc_logits.parametrizations.weight.original
    trunk_weight = model._disc_layers.trunk[0] \
        .parametrizations.weight.original
    assert torch.linalg.vector_norm(head_weight.grad) > 0
    assert torch.linalg.vector_norm(trunk_weight.grad) > 0
    assert torch.abs(model._disc_logits.bias.grad) > 0


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
        model._disc_logits,
    ]
    assert len(linears) == 4
    assert all(torch.nn.utils.parametrize.is_parametrized(layer, "weight")
               for layer in linears)
    assert model._disc_logits.bias is not None


def test_no_anchor_or_distance_path_remains():
    model_source = inspect.getsource(sys.modules[ADDModel.__module__])
    assert "anchor" not in model_source.lower()
    assert "relative_scores" not in model_source
    assert "vector_norm" not in model_source
    assert "eval_disc_distance" not in model_source


def test_pc_add_config_has_no_external_discriminator_regularizer():
    with (ROOT / "data/agents/pc_add_humanoid_agent.yaml").open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_grad_penalty"] == 0
    assert config["disc_logit_reg"] == 0
    assert config["disc_reward_scale"] == 2
    assert "disc_spectral_norm" not in config
    assert "disc_group_separable_frontend" not in config
