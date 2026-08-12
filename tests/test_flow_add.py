"""Unit tests for the FlowADD differential-flow discriminator.

These tests exercise the discriminator math without a simulator:
  - q(delta, 0) = 0, so zero flow reduces to ADD's static scalarization f(delta)
  - the radial part of the preferred flow always points toward delta = 0
  - the tangential part is orthogonal to delta
  - q(delta, v) is maximized at v = v*
  - gradients are finite at the ideal point (delta, v) = (0, 0) (grad penalty)
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

def randomize_flow_heads(model, seed=0):
    # the flow heads are zero-initialized, randomize them so the flow terms are non-trivial
    gen = torch.Generator().manual_seed(seed)
    heads = [model._disc_flow_alpha, model._disc_flow_metric]
    if (model.get_disc_mode() == flow_add_model.DISC_MODE_FLOW):
        heads.append(model._disc_flow_tangent)
    for head in heads:
        with torch.no_grad():
            head.weight.copy_(0.1 * torch.randn(head.weight.shape, generator=gen))
            head.bias.copy_(0.1 * torch.randn(head.bias.shape, generator=gen))
    return

def rand_inputs(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    disc_obs = torch.randn([n, DISC_OBS_DIM], generator=gen)
    disc_flow = 0.1 * torch.randn([n, DISC_OBS_DIM], generator=gen)
    return disc_obs, disc_flow

@pytest.mark.parametrize("disc_mode", ["flow", "radial"])
def test_zero_flow_matches_static_scalarization(disc_mode):
    # q(delta, 0) = 0, so z(delta, 0) must equal the static head f(delta)
    model = build_model(disc_mode)
    randomize_flow_heads(model)
    disc_obs, _ = rand_inputs(64)

    z = model.eval_disc(disc_obs, torch.zeros_like(disc_obs))
    f = model._disc_logits(model._disc_layers(disc_obs))
    assert torch.allclose(z, f, atol=1e-5)

def test_radial_flow_points_inward():
    # delta^T v* = -alpha |delta|^2 < 0 for delta != 0
    model = build_model("flow")
    randomize_flow_heads(model)
    disc_obs, _ = rand_inputs(256)

    v_star, v_star_tan, G = model.eval_disc_flow(disc_obs)

    radial_dot = torch.sum(disc_obs * v_star, dim=-1)
    assert torch.all(radial_dot < 0)

    # tangential component is orthogonal to delta
    tan_dot = torch.sum(disc_obs * v_star_tan, dim=-1)
    scale = torch.norm(disc_obs, dim=-1) * torch.norm(v_star_tan, dim=-1) + 1e-8
    assert torch.all(torch.abs(tan_dot) / scale < 1e-5)

    # metric is strictly positive
    assert torch.all(G > 0)

def test_tangential_flow_nontrivial():
    # with randomized heads, the tangential component should be non-zero for
    # "flow" mode and exactly zero for "radial" mode
    flow_model = build_model("flow")
    randomize_flow_heads(flow_model)
    radial_model = build_model("radial")
    randomize_flow_heads(radial_model)
    disc_obs, _ = rand_inputs(64)

    _, flow_tan, _ = flow_model.eval_disc_flow(disc_obs)
    _, radial_tan, _ = radial_model.eval_disc_flow(disc_obs)

    assert torch.norm(flow_tan) > 1e-3
    assert torch.all(radial_tan == 0)

def test_preferred_flow_maximizes_q():
    # q(delta, v) is strictly concave in v with argmax v*, so
    # q(delta, v*) >= q(delta, v) for any v; q = z(delta, v) - z(delta, 0)
    model = build_model("flow")
    randomize_flow_heads(model)
    disc_obs, disc_flow = rand_inputs(128)

    v_star, _, _ = model.eval_disc_flow(disc_obs)

    z0 = model.eval_disc(disc_obs, torch.zeros_like(disc_obs))
    q_star = model.eval_disc(disc_obs, v_star) - z0
    q_rand = model.eval_disc(disc_obs, disc_flow) - z0
    assert torch.all(q_star >= q_rand - 1e-5)

    # q(delta, v*) = 0.5 v*^T G v* >= 0
    assert torch.all(q_star >= -1e-6)

@pytest.mark.parametrize("disc_mode", ["flow", "radial", "concat"])
def test_grad_penalty_finite(disc_mode):
    # gradient penalty is taken w.r.t. the joint input [delta, v], including the
    # ideal point (0, 0); gradients must be finite everywhere
    model = build_model(disc_mode)
    if (disc_mode != "concat"):
        randomize_flow_heads(model)

    disc_obs, disc_flow = rand_inputs(32)
    # include the ideal point (0, 0) in the batch
    disc_obs[0] = 0
    disc_flow[0] = 0

    disc_obs.requires_grad_(True)
    disc_flow.requires_grad_(True)
    logit = model.eval_disc(disc_obs, disc_flow)

    grads = torch.autograd.grad(logit, [disc_obs, disc_flow],
                                grad_outputs=torch.ones_like(logit),
                                create_graph=True, retain_graph=True, only_inputs=True)
    for g in grads:
        assert torch.all(torch.isfinite(g))

    # second order gradients (needed to optimize the penalty) must also be finite
    grad_sq = torch.sum(torch.square(grads[0]), dim=-1) + torch.sum(torch.square(grads[1]), dim=-1)
    penalty = torch.mean(grad_sq)
    penalty.backward()
    for p in model.get_disc_params():
        if (p.grad is not None):
            assert torch.all(torch.isfinite(p.grad))

def test_flow_changes_logit():
    # the discriminator must actually depend on v
    model = build_model("flow")
    randomize_flow_heads(model)
    disc_obs, disc_flow = rand_inputs(64)

    z0 = model.eval_disc(disc_obs, torch.zeros_like(disc_obs))
    z1 = model.eval_disc(disc_obs, disc_flow)
    assert not torch.allclose(z0, z1)

def test_concat_mode_input_dim():
    # concat mode consumes [delta, v] as a single input
    model = build_model("concat")
    first_layer = [m for m in model._disc_layers.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first_layer.in_features == 2 * DISC_OBS_DIM

    disc_obs, disc_flow = rand_inputs(16)
    logit = model.eval_disc(disc_obs, disc_flow)
    assert logit.shape == (16, 1)
    assert torch.all(torch.isfinite(logit))

@pytest.mark.parametrize("disc_mode,num_heads", [("flow", 4), ("radial", 3), ("concat", 1)])
def test_disc_params_and_logit_weights(disc_mode, num_heads):
    model = build_model(disc_mode)

    # trunk (2 linear layers => 4 tensors) + heads (weight + bias each)
    params = model.get_disc_params()
    assert len(params) == 4 + 2 * num_heads

    # logit reg covers the final-layer weights of every head that feeds the logit
    logit_weights = model.get_disc_logit_weights()
    trunk_out = 64  # fc_2layers_128units => [128, 64]
    expected = trunk_out  # f head
    if (disc_mode != "concat"):
        expected += trunk_out  # alpha head
        expected += trunk_out * DISC_OBS_DIM  # metric head
        if (disc_mode == "flow"):
            expected += trunk_out * DISC_OBS_DIM  # tangent head
    assert logit_weights.numel() == expected

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
