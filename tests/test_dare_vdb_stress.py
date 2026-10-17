import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_vdb_stress import (JointLayers, gaussian_kl, balanced_kl, dual_ascent,
                             stochastic_logits, vdb_objective)
from learning.dare_model import GroupSeparableDiscLayers


def small_layers():
    torch.manual_seed(113)
    grouped = GroupSeparableDiscLayers([("a", [0, 2]), ("b", [1, 3, 4])],
                                      8, [6, 6], nn.ReLU).eval()
    return grouped, JointLayers.from_grouped(grouped)


def test_full_vector_migration_preserves_values_gradients_and_can_learn_cross_terms():
    grouped, joint = small_layers()
    x = torch.randn(7, 5, requires_grad=True)
    old, new = grouped(x), joint(x)
    torch.testing.assert_close(old, new, rtol=2e-5, atol=1e-6)
    go = torch.autograd.grad(old.sum(), x, retain_graph=True)[0]
    gn = torch.autograd.grad(new.sum(), x)[0]
    torch.testing.assert_close(go, gn, rtol=2e-5, atol=1e-6)
    assert not any("group" in name for name, _ in joint.named_buffers())
    joint.joint.parametrizations.weight.original.grad = None
    joint(x).square().sum().backward()
    cross_grad = joint.joint.parametrizations.weight.original.grad[:4, [1, 3, 4]]
    assert cross_grad.abs().sum() > 0


def test_kl_sums_latent_coordinates_and_balances_one_anchor_with_many_negatives():
    mean = torch.ones(3, 7)
    torch.testing.assert_close(gaussian_kl(mean, torch.zeros_like(mean)), torch.full((3,), 3.5))
    assert balanced_kl(torch.tensor([2.]), torch.full((1000,), 6.)) == 4
    torch.testing.assert_close(gaussian_kl(torch.zeros_like(mean), torch.zeros_like(mean)),
                               torch.zeros(3))
    with pytest.raises(ValueError):
        gaussian_kl(torch.zeros(3), torch.zeros(3))


def test_dual_ascent_has_correct_sign_projection_and_no_backpropagation():
    beta = torch.tensor(0., requires_grad=True)
    kl = torch.tensor(1.5, requires_grad=True)
    out = dual_ascent(beta, kl)
    assert float(out) == pytest.approx(1e-5)
    assert not out.requires_grad
    assert dual_ascent(beta, torch.tensor(.1)) == 0
    assert float(dual_ascent(torch.tensor(.01), torch.tensor(.1))) < .01


def test_one_anchor_gets_independent_noise_and_mean_path_is_deterministic():
    head = nn.Linear(6, 1)
    mean = torch.zeros(1, 6)
    noise = torch.randn(128, 6)
    samples = stochastic_logits(head, mean, torch.zeros_like(mean), noise)
    assert samples.shape == (128,) and samples.std() > 0
    _, layers = small_layers()
    layers.eval()
    x = torch.randn(8, 5)
    torch.testing.assert_close(layers(x), layers(x), atol=0, rtol=0)
    mu, lv = layers.posterior(x)
    torch.testing.assert_close(mu, layers(x), atol=0, rtol=0)
    assert torch.count_nonzero(lv) == 0


def test_constraint_changes_gradients_but_noise_control_is_identical_at_beta_zero():
    _, layers = small_layers()
    head = nn.Linear(6, 1)
    inputs = torch.randn(24, 5)
    values = []
    for arm, beta in (("flat_noise", 0.), ("flat_vdb", 0.), ("flat_vdb", .3)):
        branch, classifier = copy.deepcopy(layers), copy.deepcopy(head)
        branch.eval()
        branch.beta.fill_(beta)
        rng = torch.Generator().manual_seed(71)
        loss, info = vdb_objective(branch, classifier, inputs, arm, rng)
        loss.backward()
        values.append((loss.detach(), branch.joint.parametrizations.weight.original.grad.clone(), info))
    torch.testing.assert_close(values[0][0], values[1][0], atol=0, rtol=0)
    torch.testing.assert_close(values[0][1], values[1][1], atol=0, rtol=0)
    assert not torch.allclose(values[1][1], values[2][1])
    assert values[0][2]["kl"] > 0


def test_joint_encoder_accepts_new_input_width_without_group_rules():
    layer = JointLayers(11, 8, nn.Sequential(nn.Linear(8, 6), nn.ReLU())).eval()
    mu, logvar = layer.posterior(torch.randn(13, 11))
    assert mu.shape == logvar.shape == (13, 6)
    assert gaussian_kl(mu, logvar).shape == (13,)


def test_archive_and_joint_snapshot_strict_loading_round_trip():
    from dare_vdb_learning import VDBAgent
    agent = VDBAgent.__new__(VDBAgent)
    nn.Module.__init__(agent)
    grouped, _ = small_layers()
    agent._device = "cpu"
    agent._model = nn.Module()
    agent._model._disc_layers = grouped
    agent._model._disc_logits = nn.Linear(6, 1)
    object.__setattr__(agent, "_grouped_constructor_template", copy.deepcopy(grouped))
    old = copy.deepcopy(agent.state_dict())
    agent._model._disc_layers = JointLayers.from_grouped(grouped)
    new = copy.deepcopy(agent.state_dict())
    agent.load_state_dict(old, strict=True)
    assert not isinstance(agent._model._disc_layers, JointLayers)
    agent.load_state_dict(new, strict=True)
    assert isinstance(agent._model._disc_layers, JointLayers)
    for key, value in new.items():
        torch.testing.assert_close(agent.state_dict()[key], value, atol=0, rtol=0)


def test_online_migration_preserves_rollout_randomness_and_original_ppo():
    from types import SimpleNamespace
    from dare_vdb_learning import configure
    grouped, _ = small_layers()
    agent = SimpleNamespace(_device="cpu", _model=SimpleNamespace(_disc_layers=grouped),
        _get_rng_state=torch.get_rng_state, _set_rng_state=torch.set_rng_state)
    torch.manual_seed(419)
    expected = torch.randperm(100)
    torch.manual_seed(419)
    _, skipped = configure(agent, {}, "flat_vdb")
    assert torch.equal(torch.randperm(100), expected)
    assert agent._actor_objective == "ppo"
    assert agent._vdb_arm == "flat_vdb"
    assert skipped == ("_disc_optimizer",)
    assert agent._bottleneck_rng.initial_seed() == 1419
