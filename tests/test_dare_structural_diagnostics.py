"""Safety and mathematical checks for standalone structural stress tests."""

from pathlib import Path
import sys

import torch
from torch.nn.utils import parametrize, parametrizations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from phasebalanced_actor_velocity_stress import VelocityActorInput
from phasebalanced_orthogonal_cpl_stress import orthogonalize_cpl
from dare_shared_reference import ConvexPotentialBlock


def test_velocity_branch_starts_with_exact_same_action_features():
    torch.manual_seed(716)
    old = torch.nn.Linear(4, 8)
    obs = torch.randn(17, 7)
    before = old(obs[:, :4].contiguous())
    wrapped = VelocityActorInput(old, 3)
    assert torch.equal(before, wrapped(obs))


def test_velocity_branch_has_gradient_and_can_be_disabled():
    torch.manual_seed(717)
    wrapped = VelocityActorInput(torch.nn.Linear(4, 8), 3)
    obs = torch.randn(17, 7)
    loss = (wrapped(obs) - torch.randn(17, 8)).square().mean()
    loss.backward()
    assert wrapped.velocity_weight.grad.norm() > 0
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=.1)
    optimizer.step()
    assert wrapped.velocity_weight.norm() > 0
    wrapped.ignore_velocity = True
    assert torch.equal(wrapped(obs), wrapped.old_linear(obs[:, :4].contiguous()))


def test_adding_velocity_parameter_preserves_existing_sgd_momentum():
    old = torch.nn.Linear(4, 8)
    optimizer = torch.optim.SGD(old.parameters(), lr=.1, momentum=.9)
    old(torch.randn(5, 4)).square().mean().backward()
    optimizer.step()
    momentum = {p: optimizer.state[p]["momentum_buffer"].clone()
                for p in old.parameters()}
    wrapped = VelocityActorInput(old, 3)
    optimizer.add_param_group({"params": [wrapped.velocity_weight]})
    for parameter, previous in momentum.items():
        assert torch.equal(previous, optimizer.state[parameter]["momentum_buffer"])
    assert wrapped.velocity_weight not in optimizer.state


def test_spectral_bound_does_not_prevent_cpl_direction_annihilation():
    block = ConvexPotentialBlock(2, torch.nn.ReLU).eval()
    parametrize.remove_parametrizations(block.linear, "weight", leave_parametrized=True)
    with torch.no_grad():
        block.linear.weight.copy_(torch.diag(torch.tensor([1., 2. ** -.5])))
    assert torch.linalg.matrix_norm(block.linear.weight, ord=2) == 1
    _, tangent = torch.autograd.functional.jvp(
        block, torch.ones(1, 2), torch.tensor([[0., 1.]]))
    assert tangent.norm() < 1e-6


def test_orthogonal_cpl_preserves_local_norm_after_training():
    torch.manual_seed(718)
    block = ConvexPotentialBlock(8, torch.nn.ReLU).eval()
    parametrize.remove_parametrizations(block.linear, "weight", leave_parametrized=True)
    parametrizations.orthogonal(block.linear, orthogonal_map="cayley")
    optimizer = torch.optim.SGD(block.parameters(), lr=.03, momentum=.9,
                                weight_decay=1e-4)
    inputs, target = torch.randn(16, 8), torch.randn(16, 8)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (block(inputs) - target).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in block.parameters()
                   if p.grad is not None)
        optimizer.step()
    weight = block.linear.weight.detach()
    torch.testing.assert_close(weight.T @ weight, torch.eye(8), atol=1e-5, rtol=0)
    direction = torch.randn_like(inputs)
    _, tangent = torch.autograd.functional.jvp(block, inputs, direction)
    torch.testing.assert_close(tangent.norm(dim=-1), direction.norm(dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_orthogonal_candidate_does_not_invalidate_original_parametrization():
    import copy

    model = torch.nn.Module()
    model._disc_layers = torch.nn.Module()
    model._disc_layers.trunk = torch.nn.ModuleList([
        ConvexPotentialBlock(8, torch.nn.ReLU)])
    model.eval()
    inputs = torch.randn(5, 8)
    original = model._disc_layers.trunk[0](inputs).detach().clone()
    candidate = orthogonalize_cpl(copy.deepcopy(model))
    matched = orthogonalize_cpl(copy.deepcopy(model), False)
    assert torch.equal(original, model._disc_layers.trunk[0](inputs))
    torch.testing.assert_close(candidate._disc_layers.trunk[0](inputs),
                               matched._disc_layers.trunk[0](inputs),
                               atol=1e-5, rtol=1e-5)


def test_norm_preserving_cpl_does_not_guarantee_scalar_score_sensitivity():
    block = ConvexPotentialBlock(2, torch.nn.ReLU).eval()
    parametrize.remove_parametrizations(block.linear, "weight", leave_parametrized=True)
    with torch.no_grad():
        block.linear.weight.copy_(torch.eye(2))
    inputs = torch.tensor([[1., 1.], [1., 2.]])
    hidden = block(inputs)
    # The hidden difference is fully preserved, but a valid unit-norm output
    # projection can ignore it completely.
    torch.testing.assert_close((hidden[1] - hidden[0]).norm(), torch.tensor(1.))
    score = hidden @ torch.tensor([1., 0.])
    assert score[0] == score[1]
