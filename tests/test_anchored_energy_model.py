from pathlib import Path
import sys

import gymnasium.spaces as spaces
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.add_model as add_model
from learning.anchored_energy_model import (
    AnchoredEnergyModel,
    ConditionalPositiveDefiniteEnergy,
    energy_actor_reward,
)


def _energy_module(residual_dim=5, rank=3, eigen_floor=0.1):
    torch.manual_seed(7)
    return ConditionalPositiveDefiniteEnergy(
        residual_dim=residual_dim,
        context_dim=2 * residual_dim,
        hidden_units=(13, 11),
        rank=rank,
        eigen_floor=eigen_floor,
    ).double()


def test_zero_is_context_independent_unique_anchor():
    module = _energy_module()
    context = torch.randn(9, 10, dtype=torch.float64)
    zero = torch.zeros(9, 5, dtype=torch.float64, requires_grad=True)

    energy = module.eval_energy(zero, context)
    gradient, = torch.autograd.grad(energy.sum(), zero)
    assert energy.shape == (9, 1)
    assert torch.equal(energy, torch.zeros_like(energy))
    assert torch.equal(gradient, torch.zeros_like(gradient))

    nonzero = torch.randn(9, 5, dtype=torch.float64)
    assert torch.all(module.eval_energy(nonzero, context) > 0)
    assert torch.all(energy_actor_reward(
        module.eval_energy(nonzero, context)) < 1)
    assert torch.equal(energy_actor_reward(energy), torch.ones_like(energy))


def test_trace_normalization_and_eigenvalue_floor():
    residual_dim = 6
    eigen_floor = 0.15
    module = _energy_module(residual_dim, rank=4, eigen_floor=eigen_floor)
    context = torch.randn(8, 2 * residual_dim, dtype=torch.float64)
    metric = module.eval_metric(context)
    eigenvalues = torch.linalg.eigvalsh(metric)

    assert torch.allclose(
        torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1),
        torch.full((8,), float(residual_dim), dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.all(eigenvalues >= eigen_floor - 1e-10)
    assert torch.allclose(metric, metric.transpose(-2, -1), atol=1e-12)


def test_efficient_energy_matches_explicit_quadratic_and_gradient():
    residual_dim = 5
    module = _energy_module(residual_dim)
    context = torch.randn(7, 2 * residual_dim, dtype=torch.float64)
    residual = torch.randn(
        7, residual_dim, dtype=torch.float64, requires_grad=True)

    metric = module.eval_metric(context)
    expected = 0.5 / residual_dim * torch.einsum(
        "bi,bij,bj->b", residual, metric, residual)
    actual = module.eval_energy(residual, context).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)

    gradient, = torch.autograd.grad(actual.sum(), residual)
    expected_gradient = torch.matmul(
        metric, residual.detach().unsqueeze(-1)).squeeze(-1) / residual_dim
    assert torch.allclose(
        gradient, expected_gradient, atol=1e-10, rtol=1e-10)


def test_energy_has_global_quadratic_bounds_and_radial_scaling():
    residual_dim = 5
    eigen_floor = 0.2
    module = _energy_module(residual_dim, eigen_floor=eigen_floor)
    context = torch.randn(12, 2 * residual_dim, dtype=torch.float64)
    residual = torch.randn(12, residual_dim, dtype=torch.float64)
    energy = module.eval_energy(residual, context).squeeze(-1)

    norm_squared = torch.sum(torch.square(residual), dim=-1)
    lower = eigen_floor * norm_squared / (2 * residual_dim)
    max_eigen_bound = eigen_floor + residual_dim * (1 - eigen_floor)
    upper = max_eigen_bound * norm_squared / (2 * residual_dim)
    assert torch.all(energy >= lower - 1e-12)
    assert torch.all(energy <= upper + 1e-12)

    scale = -2.75
    scaled = module.eval_energy(scale * residual, context).squeeze(-1)
    assert torch.allclose(scaled, scale ** 2 * energy,
                          atol=1e-10, rtol=1e-10)


def test_one_dimensional_metric_is_exactly_identity():
    module = _energy_module(residual_dim=1, rank=2, eigen_floor=0.37)
    context = torch.randn(6, 2, dtype=torch.float64)
    residual = torch.randn(6, 1, dtype=torch.float64)

    assert torch.allclose(
        module.eval_metric(context),
        torch.ones(6, 1, 1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        module.eval_energy(residual, context),
        0.5 * torch.square(residual),
        atol=1e-12,
        rtol=1e-12,
    )


def test_extreme_diagonal_logits_remain_finite():
    module = _energy_module()
    with torch.no_grad():
        module._metric_head.weight.fill_(1e20)
        module._metric_head.bias.fill_(1e20)
    context = torch.ones(4, 10, dtype=torch.float64)
    residual = torch.full((4, 5), 1e3, dtype=torch.float64)

    factors = module.metric_factors(context)
    energy = module.eval_energy(residual, context)
    assert all(torch.isfinite(value).all() for value in factors)
    assert torch.isfinite(energy).all()
    assert torch.all(energy > 0)


def test_low_rank_branch_receives_gradient_at_initialization():
    module = _energy_module()
    context = torch.randn(16, 10, dtype=torch.float64)
    residual = torch.randn(16, 5, dtype=torch.float64)
    module.eval_energy(residual, context).mean().backward()

    low_rank_size = module.residual_dim * module.rank
    head_gradient = module._metric_head.weight.grad[:low_rank_size]
    assert torch.isfinite(head_gradient).all()
    assert torch.linalg.vector_norm(head_gradient) > 0


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"residual_dim": 0, "context_dim": 0}, ValueError),
        ({"residual_dim": 3, "context_dim": 5}, ValueError),
        ({"residual_dim": 3, "context_dim": 6, "rank": 0}, ValueError),
        ({"residual_dim": 3, "context_dim": 6,
          "eigen_floor": 0.0}, ValueError),
        ({"residual_dim": 3, "context_dim": 6,
          "eigen_floor": 1.0}, ValueError),
    ],
)
def test_invalid_geometry_is_rejected(kwargs, exception):
    with pytest.raises(exception):
        ConditionalPositiveDefiniteEnergy(hidden_units=(), **kwargs)


def test_input_shape_contract_is_explicit():
    module = _energy_module()
    with pytest.raises(ValueError, match="residual last dimension"):
        module.eval_energy(torch.randn(2, 4, dtype=torch.float64),
                           torch.randn(2, 10, dtype=torch.float64))
    with pytest.raises(ValueError, match="context last dimension"):
        module.eval_energy(torch.randn(2, 5, dtype=torch.float64),
                           torch.randn(2, 9, dtype=torch.float64))
    with pytest.raises(ValueError, match="identical leading shapes"):
        module.eval_energy(torch.randn(2, 5, dtype=torch.float64),
                           torch.randn(3, 10, dtype=torch.float64))


class _FakeEnv:
    def __init__(self, residual_dim=5):
        self._obs_space = spaces.Box(-1.0, 1.0, (8,), dtype=float)
        self._action_space = spaces.Box(-1.0, 1.0, (3,), dtype=float)
        self._disc_obs_space = spaces.Box(
            -float("inf"), float("inf"), (residual_dim,), dtype=float)

    def get_obs_space(self):
        return self._obs_space

    def get_action_space(self):
        return self._action_space

    def get_disc_obs_space(self):
        return self._disc_obs_space


def _model_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "energy_hidden_units": [17, 13],
        "energy_rank": 3,
        "energy_eigen_floor": 0.1,
        "energy_beta_init": 0.4,
    }


def test_model_preserves_add_actor_critic_and_exposes_energy_interface():
    torch.manual_seed(11)
    model = AnchoredEnergyModel(_model_config(), _FakeEnv())
    assert isinstance(model, add_model.ADDModel)

    observation = torch.randn(4, 8)
    assert model.eval_actor(observation).mean.shape == (4, 3)
    assert model.eval_critic(observation).shape == (4, 1)

    residual = torch.randn(4, 5)
    context = torch.randn(4, 10)
    energy = model.eval_energy(residual, context)
    logits = model.eval_disc(residual, context)
    reward_before = model.eval_actor_reward(residual, context)
    assert energy.shape == logits.shape == reward_before.shape == (4, 1)
    assert torch.allclose(logits, model.disc_beta - energy)
    assert model.get_energy_bias() is model.disc_beta
    assert model.get_energy_epsilon() == pytest.approx(0.1)

    with torch.no_grad():
        model.disc_beta.add_(9.0)
    reward_after = model.eval_actor_reward(residual, context)
    assert torch.equal(reward_before, reward_after)


def test_discriminator_parameters_exclude_actor_and_critic():
    model = AnchoredEnergyModel(_model_config(), _FakeEnv())
    disc_params = model.get_disc_params()
    disc_ids = {id(parameter) for parameter in disc_params}
    actor_critic_ids = {
        id(parameter)
        for parameter in model.get_actor_params() + model.get_critic_params()
    }
    expected_ids = {
        id(parameter) for parameter in model._conditional_energy.parameters()
    } | {id(model.disc_beta)}

    assert disc_ids == expected_ids
    assert disc_ids.isdisjoint(actor_critic_ids)
    assert len(disc_ids) == len(disc_params)
