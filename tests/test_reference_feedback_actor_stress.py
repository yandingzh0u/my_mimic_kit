"""Tests of the isolated prototype, not changes to the production actor."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/diagnostics'))
from reference_feedback_actor_stress import ConditionalActor


def model_and_inputs():
    torch.set_num_threads(2)
    torch.manual_seed(72913)
    model = ConditionalActor(width=32)
    return model, torch.randn(64,172), torch.randn(64,464)


def test_anchor_exact_in_both_modes():
    model,e,r = model_and_inputs()
    for mode in (True,False):
        model.train(mode)
        assert torch.equal(model(torch.zeros_like(e),r),model.feedforward(r))


def test_secants_and_weight_bounds():
    model,e,r = model_and_inputs()
    for scale in (.001,.1,1.,10.):
        delta = scale*torch.randn_like(e)
        output = model(e+delta,r)-model(e,r)
        assert torch.all(output.norm(dim=-1) <= delta.norm(dim=-1)+1e-5)
    assert all(float(torch.linalg.svdvals(w.double())[0]) <= 1.
               for w in model.matrices())


def test_all_branches_receive_finite_gradients():
    model,e,r = model_and_inputs()
    model(e,r).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_projection_after_optimizer_update():
    model,e,r = model_and_inputs()
    optimizer = torch.optim.Adam(model.parameters(),lr=.1)
    (-model(e,r).square().mean()).backward()
    optimizer.step()
    model.project()
    assert all(float(torch.linalg.svdvals(w.double())[0]) <= 1.
               for w in model.matrices())
