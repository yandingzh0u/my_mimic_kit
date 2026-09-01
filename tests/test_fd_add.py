import pathlib

import gymnasium.spaces as spaces
import torch

from learning.fd_add_model import FDADDModel
from learning.semantic_grouped_linear import SemanticGroupedLinear


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    dims = (3, 6, 45, 84, 3, 3, 28)

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        groups = []
        start = 0
        for group_id, dim in enumerate(self.dims):
            groups.append(("group{}".format(group_id),
                           tuple(range(start, start + dim))))
            start += dim
        return tuple(groups)


def _config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def test_grouped_linear_is_one_module_and_partitions_input():
    layer = SemanticGroupedLinear(
        172, _Env().get_disc_error_groups(), out_features=18).eval()
    assert layer.weight.shape == (18, 172)
    assert layer.bias.shape == (7, 18)
    assert not isinstance(layer, torch.nn.ModuleList)
    assert sum(1 for _ in layer.modules()) == 1


def test_every_group_block_has_unit_spectral_norm():
    layer = SemanticGroupedLinear(
        172, _Env().get_disc_error_groups(), out_features=18).eval()
    packed = layer.normalized_weight()
    for group_id, dim in enumerate(_Env.dims):
        block = packed[group_id, :, :dim]
        norm = torch.linalg.matrix_norm(block, ord=2)
        torch.testing.assert_close(norm, torch.ones_like(norm),
                                   rtol=2e-3, atol=2e-3)


def test_matches_explicit_group_linears_forward_and_backward():
    groups = _Env().get_disc_error_groups()
    layer = SemanticGroupedLinear(172, groups, out_features=18).eval()
    inputs = torch.randn(11, 172, requires_grad=True)
    actual = layer(inputs)

    packed = layer.normalized_weight()
    expected = []
    for group_id, (_, indices) in enumerate(groups):
        group_input = inputs[:, torch.tensor(indices)]
        expected.append(torch.nn.functional.linear(
            group_input, packed[group_id, :, :len(indices)],
            layer.bias[group_id]))
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.count_nonzero(layer.weight.grad) > 0


def test_fd_model_is_full_sn_and_has_a_single_grouped_frontend():
    model = FDADDModel(_config(), _Env()).train()
    semantic = model._disc_layers.semantic
    assert isinstance(semantic, SemanticGroupedLinear)
    assert not hasattr(model._disc_layers, "encoders")
    assert hasattr(model._disc_logits.parametrizations, "weight")
    trunk_linears = [module for module in model._disc_layers.trunk
                     if isinstance(module, torch.nn.Linear)]
    assert trunk_linears
    assert all(hasattr(module.parametrizations, "weight")
               for module in trunk_linears)
    logits = model.eval_disc(torch.randn(32, 172)).squeeze(-1)
    loss = (torch.nn.functional.softplus(logits).mean()
            + torch.nn.functional.softplus(
                -model.eval_disc(torch.zeros(1, 172))).mean())
    loss.backward()
    assert torch.isfinite(semantic.weight.grad).all()


def test_config_is_fixed_to_gp_zero_and_100_iter_logging():
    text = (ROOT / "data/agents/fd_add_humanoid_agent.yaml").read_text()
    assert 'agent_name: "FD_ADD"' in text
    assert "disc_grad_penalty: 0" in text
    assert "iters_per_output: 100" in text


def test_official_add_is_not_modified_by_fd_add():
    add_model = (ROOT / "mimickit/learning/add_model.py").read_text()
    add_agent = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "FDADD" not in add_model
    assert "SemanticGroupedLinear" not in add_model
    assert "FDADD" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent
