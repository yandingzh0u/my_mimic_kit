"""Unit tests for the FlowADD potential-circulation flow discriminator.

These tests exercise the discriminator math without a simulator:
  - with S = 0 and A = 0 the model reduces exactly to ADD: z(x_prev, x_t) = f(x_t)
  - q_prog matches the potential identity E_S(x_prev) - E_S(x_t) with S = L L^T (PSD)
  - q_circ vanishes for pure scaling (x_t = c * x_prev) and is antisymmetric
    under time reversal
  - accumulated circulation over a path equals <A, Omega(path)>_F, and two paths
    with different signed-area matrices Omega are separable by some A even with
    identical endpoints
  - gradients on the joint input [x_prev, x_t] are finite, including at the
    ideal point (0, 0) (needed for the gradient penalty)
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import gymnasium.spaces as spaces

import learning.flow_add_model as flow_add_model

OBS_DIM = 15
ACTION_DIM = 8
DISC_OBS_DIM = 12

class FakeEnv:
    def get_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(low=-np.ones(ACTION_DIM, dtype=np.float32),
                          high=np.ones(ACTION_DIM, dtype=np.float32),
                          dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(DISC_OBS_DIM,), dtype=np.float32)

def build_model(disc_mode):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "disc_mode": disc_mode
    }
    model = flow_add_model.FlowADDModel(config, FakeEnv())
    return model

def randomize_flow_matrices(model, seed=0):
    # L is initialized near zero and B at zero, randomize them so the flow
    # terms are non-trivial
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if (model._has_potential()):
            L = model._disc_flow_potential
            L.copy_(0.3 * torch.randn(L.shape, generator=gen))
        if (model._has_circulation()):
            B = model._disc_flow_circulation
            B.copy_(0.3 * torch.randn(B.shape, generator=gen))
    return

def rand_inputs(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    disc_obs = torch.randn([n, DISC_OBS_DIM], generator=gen)
    disc_obs_prev = disc_obs + 0.3 * torch.randn([n, DISC_OBS_DIM], generator=gen)
    return disc_obs, disc_obs_prev

def signed_area(path):
    # Omega(path) = 0.5 * sum_t (x_t-1 x_t^T - x_t x_t-1^T)
    d = path.shape[-1]
    omega = torch.zeros([d, d])
    for t in range(1, path.shape[0]):
        x_prev = path[t - 1].unsqueeze(-1)
        x_curr = path[t].unsqueeze(-1)
        omega += 0.5 * (x_prev @ x_curr.t() - x_curr @ x_prev.t())
    return omega

def test_add_special_case():
    # with S = 0 and A = 0, FlowADD reduces exactly to ADD: z = f(x_t)
    model = build_model("flow")
    with torch.no_grad():
        model._disc_flow_potential.zero_()
        model._disc_flow_circulation.zero_()

    disc_obs, disc_obs_prev = rand_inputs(64)
    z = model.eval_disc(disc_obs, disc_obs_prev)
    f = model._disc_logits(model._disc_layers(disc_obs))
    assert torch.allclose(z, f, atol=1e-6)

@pytest.mark.parametrize("disc_mode", ["flow", "potential", "circulation"])
def test_static_transition_matches_static_scalarization(disc_mode):
    # x_prev = x_t gives q_prog = 0 and q_circ = 0, so z = f(x_t) even with
    # non-trivial S and A
    model = build_model(disc_mode)
    randomize_flow_matrices(model)

    disc_obs, _ = rand_inputs(64)
    z = model.eval_disc(disc_obs, disc_obs.clone())
    f = model._disc_logits(model._disc_layers(disc_obs))
    assert torch.allclose(z, f, atol=1e-4)

    # the ideal transition (0, 0) also reduces to f(0)
    zeros = torch.zeros([8, DISC_OBS_DIM])
    q_prog, q_circ = model.eval_flow_scores(zeros, zeros)
    assert torch.all(q_prog == 0)
    assert torch.all(q_circ == 0)

def test_potential_identity():
    # q_prog must equal E_S(x_prev) - E_S(x_t) with E_S(x) = 0.5 x^T (L L^T) x
    model = build_model("potential")
    randomize_flow_matrices(model)
    disc_obs, disc_obs_prev = rand_inputs(128)

    q_prog, q_circ = model.eval_flow_scores(disc_obs, disc_obs_prev)
    assert torch.all(q_circ == 0)

    L = model._disc_flow_potential.detach()
    S = L @ L.t()
    e_curr = 0.5 * torch.sum((disc_obs @ S) * disc_obs, dim=-1)
    e_prev = 0.5 * torch.sum((disc_obs_prev @ S) * disc_obs_prev, dim=-1)
    assert torch.allclose(q_prog, e_prev - e_curr, atol=1e-4)

    # S is PSD, so shrinking the error (x_t = 0.5 x_prev) gives positive progress
    q_shrink, _ = model.eval_flow_scores(0.5 * disc_obs_prev, disc_obs_prev)
    assert torch.all(q_shrink > 0)
    # and growing the error gives negative progress
    q_grow, _ = model.eval_flow_scores(2.0 * disc_obs_prev, disc_obs_prev)
    assert torch.all(q_grow < 0)

def test_circulation_ignores_pure_scaling():
    # x_t = c * x_prev has no rotation across objectives, so q_circ = 0
    model = build_model("circulation")
    randomize_flow_matrices(model)
    _, disc_obs_prev = rand_inputs(64)

    for c in [0.0, 0.5, 1.0, 2.0, -1.0]:
        q_prog, q_circ = model.eval_flow_scores(c * disc_obs_prev, disc_obs_prev)
        assert torch.all(q_prog == 0)
        assert torch.all(torch.abs(q_circ) < 1e-4)

def test_circulation_time_reversal_antisymmetry():
    # q_circ(x_prev, x_t) = -q_circ(x_t, x_prev): A -> B and B -> A transitions
    # get opposite circulation scores
    model = build_model("circulation")
    randomize_flow_matrices(model)
    disc_obs, disc_obs_prev = rand_inputs(128)

    _, q_fwd = model.eval_flow_scores(disc_obs, disc_obs_prev)
    _, q_bwd = model.eval_flow_scores(disc_obs_prev, disc_obs)
    assert torch.allclose(q_fwd, -q_bwd, atol=1e-4)
    assert torch.norm(q_fwd) > 1e-3

def test_circulation_equals_signed_area_inner_product():
    # accumulated circulation over a path Gamma satisfies
    # sum_t q_circ(x_t-1, x_t) = <A, Omega(Gamma)>_F
    model = build_model("circulation")
    randomize_flow_matrices(model)

    gen = torch.Generator().manual_seed(3)
    path = torch.randn([20, DISC_OBS_DIM], generator=gen)

    _, q_circ = model.eval_flow_scores(path[1:], path[:-1])
    total_circ = torch.sum(q_circ)

    B = model._disc_flow_circulation.detach()
    A = B - B.t()
    omega = signed_area(path)
    expected = torch.sum(A * omega)
    assert torch.allclose(total_circ, expected, atol=1e-3)

def test_paths_with_different_signed_area_are_separable():
    # theorem: Omega(G1) != Omega(G2) => exists A with C_A(G1) != C_A(G2),
    # even when both paths share the same start and end points;
    # take A = Omega(G1) - Omega(G2), then C_A(G1) - C_A(G2) = |A|_F^2 > 0
    gen = torch.Generator().manual_seed(4)
    start = torch.randn([DISC_OBS_DIM], generator=gen)
    end = torch.randn([DISC_OBS_DIM], generator=gen)
    mid1 = torch.randn([8, DISC_OBS_DIM], generator=gen)
    mid2 = torch.randn([8, DISC_OBS_DIM], generator=gen)
    path1 = torch.cat([start.unsqueeze(0), mid1, end.unsqueeze(0)], dim=0)
    path2 = torch.cat([start.unsqueeze(0), mid2, end.unsqueeze(0)], dim=0)

    omega1 = signed_area(path1)
    omega2 = signed_area(path2)
    a_gap = omega1 - omega2
    assert torch.norm(a_gap) > 1e-3

    # realize A = B - B^T with B = 0.5 * (Omega1 - Omega2), which is valid
    # since Omega1 - Omega2 is antisymmetric
    model = build_model("circulation")
    with torch.no_grad():
        model._disc_flow_circulation.copy_(0.5 * a_gap)

    _, q1 = model.eval_flow_scores(path1[1:], path1[:-1])
    _, q2 = model.eval_flow_scores(path2[1:], path2[:-1])
    gap = torch.sum(q1) - torch.sum(q2)
    assert torch.allclose(gap, torch.sum(torch.square(a_gap)), atol=1e-3)
    assert gap > 0

@pytest.mark.parametrize("disc_mode", ["flow", "potential", "circulation", "concat"])
def test_grad_penalty_finite(disc_mode):
    # gradient penalty is taken w.r.t. the joint input [x_prev, x_t], including
    # the ideal point (0, 0); first and second order gradients must be finite
    model = build_model(disc_mode)
    if (disc_mode != "concat"):
        randomize_flow_matrices(model)

    disc_obs, disc_obs_prev = rand_inputs(32)
    # include the ideal transition (0, 0) in the batch
    disc_obs[0] = 0
    disc_obs_prev[0] = 0

    disc_obs.requires_grad_(True)
    disc_obs_prev.requires_grad_(True)
    logit = model.eval_disc(disc_obs, disc_obs_prev)

    grads = torch.autograd.grad(logit, [disc_obs, disc_obs_prev],
                                grad_outputs=torch.ones_like(logit),
                                create_graph=True, retain_graph=True, only_inputs=True)
    for g in grads:
        assert torch.all(torch.isfinite(g))

    grad_sq = torch.sum(torch.square(grads[0]), dim=-1) + torch.sum(torch.square(grads[1]), dim=-1)
    penalty = torch.mean(grad_sq)
    penalty.backward()
    for p in model.get_disc_params():
        if (p.grad is not None):
            assert torch.all(torch.isfinite(p.grad))

def test_flow_changes_logit():
    # the discriminator must actually depend on the previous frame
    model = build_model("flow")
    randomize_flow_matrices(model)
    disc_obs, disc_obs_prev = rand_inputs(64)

    z0 = model.eval_disc(disc_obs, disc_obs.clone())
    z1 = model.eval_disc(disc_obs, disc_obs_prev)
    assert not torch.allclose(z0, z1)

def test_concat_mode():
    # concat mode consumes [x_t, v_t] as a single MLP input
    model = build_model("concat")
    first_layer = [m for m in model._disc_layers.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first_layer.in_features == 2 * DISC_OBS_DIM

    disc_obs, disc_obs_prev = rand_inputs(16)
    logit = model.eval_disc(disc_obs, disc_obs_prev)
    assert logit.shape == (16, 1)
    assert torch.all(torch.isfinite(logit))

    # the logit must depend on the previous frame through v
    logit_static = model.eval_disc(disc_obs, disc_obs.clone())
    assert not torch.allclose(logit, logit_static)

@pytest.mark.parametrize("disc_mode,num_matrices", [("flow", 2), ("potential", 1), ("circulation", 1), ("concat", 0)])
def test_disc_params_and_logit_weights(disc_mode, num_matrices):
    model = build_model(disc_mode)

    # trunk (2 linear layers => 4 tensors) + f head (weight + bias) + flow matrices
    params = model.get_disc_params()
    assert len(params) == 6 + num_matrices

    # logit reg covers the f-head weights and the flow matrices
    logit_weights = model.get_disc_logit_weights()
    trunk_out = 64  # fc_2layers_128units => [128, 64]
    expected = trunk_out + num_matrices * DISC_OBS_DIM * DISC_OBS_DIM
    assert logit_weights.numel() == expected

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
