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

import learning.flow_add_agent as flow_add_agent
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

def build_model(disc_mode, **overrides):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "disc_mode": disc_mode
    }
    config.update(overrides)
    model = flow_add_model.FlowADDModel(config, FakeEnv())
    return model

def randomize_flow_matrices(model, seed=0):
    # L is initialized near zero and B at zero, randomize them so the flow
    # terms are non-trivial
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if (model.has_potential()):
            L = model._disc_flow_potential
            L.copy_(0.3 * torch.randn(L.shape, generator=gen))
        if (model.has_circulation()):
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

def test_fixed_group_balanced_potential_energy():
    model = build_model(
        "flow",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6],
        disc_flow_regularize_matrices=False)

    gen = torch.Generator().manual_seed(31)
    x = torch.randn([32, DISC_OBS_DIM], generator=gen)
    energy = model.eval_potential_energy(x)
    group_energy = model.eval_potential_group_energies(x)
    expected = 0.5 * (0.4 * torch.mean(torch.square(x[:, :3]), dim=-1)
                      + 0.6 * torch.mean(torch.square(x[:, 3:]), dim=-1))

    assert torch.allclose(energy, expected, atol=1e-6)
    assert group_energy.shape == (32, 2)
    assert torch.allclose(torch.sum(group_energy, dim=-1), energy, atol=1e-6)
    assert not model._disc_flow_potential.requires_grad

def test_reward_placed_potential_is_absent_from_disc_logit():
    model = build_model(
        "flow",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6],
        disc_flow_regularize_matrices=False)
    randomize_flow_matrices(model)
    disc_obs, disc_obs_prev = rand_inputs(64, seed=32)

    z = model.eval_disc(disc_obs, disc_obs_prev).squeeze(-1)
    f = model.eval_static_score(disc_obs).squeeze(-1)
    q_prog, q_circ = model.eval_flow_scores(disc_obs, disc_obs_prev)

    assert torch.mean(torch.abs(q_prog)) > 1e-3
    assert torch.allclose(z, f + q_circ, atol=1e-5)
    # Flow matrices are not last-layer logit weights in this mode.  In
    # particular, the fixed P cannot be shrunk by disc_logit_reg.
    assert model.get_disc_logit_weights().numel() == 64 + DISC_OBS_DIM * DISC_OBS_DIM

def test_potential_reward_telescopes_exactly():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6],
        disc_flow_regularize_matrices=False)
    gen = torch.Generator().manual_seed(33)
    path = torch.randn([11, DISC_OBS_DIM], generator=gen)
    gamma = 0.99

    shaping = model.eval_potential_shaping(path[1:], path[:-1], gamma)
    discounts = gamma ** torch.arange(shaping.shape[0])
    discounted_sum = torch.sum(discounts * shaping)
    expected = (model.eval_potential_energy(path[:1])[0]
                - gamma ** shaping.shape[0]
                * model.eval_potential_energy(path[-1:])[0])
    assert torch.allclose(discounted_sum, expected, atol=1e-5)

def test_potential_reward_has_no_path_dependent_cycle_bonus():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6])
    gen = torch.Generator().manual_seed(34)
    endpoint = torch.randn([DISC_OBS_DIM], generator=gen)
    path_a = torch.stack([endpoint, torch.randn([DISC_OBS_DIM], generator=gen), endpoint])
    path_b = torch.stack([endpoint, torch.randn([DISC_OBS_DIM], generator=gen), endpoint])
    gamma = 0.99
    discounts = gamma ** torch.arange(2)

    ret_a = torch.sum(discounts * model.eval_potential_shaping(path_a[1:], path_a[:-1], gamma))
    ret_b = torch.sum(discounts * model.eval_potential_shaping(path_b[1:], path_b[:-1], gamma))
    assert torch.allclose(ret_a, ret_b, atol=1e-6)

def test_raw_progress_prefers_low_error_over_oscillation():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6])
    high = torch.ones([DISC_OBS_DIM])
    low = 0.2 * high
    gamma = 0.99

    # All three paths start from the same high error and contain four
    # transitions: reduce-and-hold, oscillate, or stay high.
    paths = [
        torch.stack([high, low, low, low, low]),
        torch.stack([high, low, high, low, high]),
        torch.stack([high, high, high, high, high])
    ]
    returns = []
    discounts = gamma ** torch.arange(4)
    for path in paths:
        progress = model.eval_potential_shaping(
            path[1:], path[:-1], discount=1.0, clip=5.0)
        returns.append(torch.sum(discounts * progress))

    assert returns[0] > returns[1] > returns[2]

def test_fixed_potential_reward_components_are_explicit():
    energy_prev = torch.tensor([0.2, 0.8, 1.5])
    energy = torch.tensor([0.1, 0.8, 1.0])
    progress, absolute = flow_add_agent.calc_fixed_potential_rewards(
        energy=energy,
        energy_prev=energy_prev,
        progress_scale=0.1,
        progress_discount=1.0,
        abs_energy_weight=0.02)

    assert torch.allclose(progress, 0.1 * (energy_prev - energy))
    assert torch.allclose(absolute, -0.02 * energy)
    assert torch.all(absolute <= 0)
    # A constant high-error state receives persistent pressure even though its
    # progress component is exactly zero.
    assert progress[1] == 0
    assert absolute[1] < 0

def test_group_absolute_reward_can_target_root_without_scaling_other_groups():
    group_energy = torch.tensor([
        [0.3, 0.1, 0.2],
        [0.5, 0.4, 0.1]
    ])
    group_reward, total_reward = flow_add_agent.calc_group_abs_rewards(
        group_energy=group_energy,
        base_weight=0.02,
        extra_group_weights=[0.06, 0.0, 0.0])

    expected = -group_energy * torch.tensor([0.08, 0.02, 0.02])
    assert torch.allclose(group_reward, expected)
    assert torch.allclose(total_reward, torch.sum(expected, dim=-1))
    assert torch.all(group_reward <= 0)

def test_absolute_energy_term_cannot_add_a_positive_cycle_bonus():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6])
    high = torch.ones([DISC_OBS_DIM])
    low = 0.2 * high
    cycle = torch.stack([high, low, high, low, high])
    energy = model.eval_potential_energy(cycle[1:], clip=5.0)
    energy_prev = model.eval_potential_energy(cycle[:-1], clip=5.0)
    _, absolute = flow_add_agent.calc_fixed_potential_rewards(
        energy=energy,
        energy_prev=energy_prev,
        progress_scale=0.1,
        progress_discount=1.0,
        abs_energy_weight=0.02)

    assert torch.sum(absolute) < 0
    assert torch.all(absolute <= 0)

def test_fixed_potential_reward_is_bounded_after_energy_clip():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6])
    huge = torch.full([8, DISC_OBS_DIM], 1e6)
    energy = model.eval_potential_energy(huge, clip=5.0)
    _, absolute = flow_add_agent.calc_fixed_potential_rewards(
        energy=energy,
        energy_prev=energy,
        progress_scale=0.1,
        progress_discount=1.0,
        abs_energy_weight=0.02)

    # Sum of normalized group weights is one: E_max=0.5*5^2=12.5.
    assert torch.allclose(energy, torch.full_like(energy, 12.5), atol=1e-5)
    assert torch.allclose(absolute, torch.full_like(absolute, -0.25), atol=1e-6)

def test_potential_energy_clipping_is_symmetric_and_bounded():
    model = build_model(
        "potential",
        disc_flow_potential_in_logit=False,
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6])
    pos = torch.full([4, DISC_OBS_DIM], 100.0)
    neg = -pos
    at_clip = torch.full([4, DISC_OBS_DIM], 5.0)
    e_pos = model.eval_potential_energy(pos, clip=5.0)
    e_neg = model.eval_potential_energy(neg, clip=5.0)
    e_clip = model.eval_potential_energy(at_clip, clip=5.0)
    assert torch.allclose(e_pos, e_neg)
    assert torch.allclose(e_pos, e_clip)

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

def test_eval_static_score_matches_logit_decomposition():
    # z(x_prev, x_t) must decompose exactly as f(x_t) + q_prog + q_circ
    model = build_model("flow")
    randomize_flow_matrices(model)
    disc_obs, disc_obs_prev = rand_inputs(64)

    z = model.eval_disc(disc_obs, disc_obs_prev).squeeze(-1)
    f = model.eval_static_score(disc_obs).squeeze(-1)
    q_prog, q_circ = model.eval_flow_scores(disc_obs, disc_obs_prev)
    assert torch.allclose(z, f + q_prog + q_circ, atol=1e-5)

def test_contract_teacher_gradient_routing():
    # the contraction teacher builds transitions (x, c x) with detached f:
    #   - L must receive a non-zero gradient (the teacher trains S)
    #   - B must receive an exactly-zero gradient: dq_circ/dB at (x, c x) is
    #     x (c x)^T - (c x) x^T = 0, so the teacher cannot bias circulation
    #   - the trunk/f head must receive no gradient (f is detached)
    model = build_model("flow")
    randomize_flow_matrices(model)

    gen = torch.Generator().manual_seed(7)
    x = torch.randn([64, DISC_OBS_DIM], generator=gen)
    c = torch.rand([64, 1], generator=gen)
    x_curr = c * x

    f = model.eval_static_score(x_curr).squeeze(-1).detach()
    q_prog, q_circ = model.eval_flow_scores(x_curr, x)
    contract_logit = f + q_prog + q_circ

    # same BCE-with-positive-label loss the agent uses
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        contract_logit, torch.ones_like(contract_logit))
    loss.backward()

    l_grad = model._disc_flow_potential.grad
    b_grad = model._disc_flow_circulation.grad
    assert l_grad is not None and torch.norm(l_grad) > 1e-6
    assert b_grad is None or torch.all(torch.abs(b_grad) < 1e-6)

    for p in model._disc_layers.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
    assert model._disc_logits.weight.grad is None \
        or torch.all(model._disc_logits.weight.grad == 0)

    # contracting transitions get non-negative progress from a PSD S
    q_prog_val, q_circ_val = model.eval_flow_scores(x_curr.detach(), x.detach())
    assert torch.all(q_prog_val >= -1e-5)
    assert torch.all(torch.abs(q_circ_val) < 1e-4)

def test_centered_log_d_reward_is_non_saturating_and_concave():
    logits = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], requires_grad=True)
    reward = flow_add_agent.calc_flow_disc_reward(
        logits, flow_add_agent.DISC_REWARD_CENTERED_LOG_D, reward_scale=2.0)

    assert torch.all(torch.isfinite(reward))
    assert torch.all(reward[1:] > reward[:-1])
    assert torch.allclose(reward[2], torch.tensor(0.0), atol=1e-6)

    grad = torch.autograd.grad(torch.sum(reward), logits)[0]
    # Bad states retain a strong slope instead of the original softplus
    # reward's vanishing negative-logit slope.
    assert grad[0] > 1.9
    assert grad[0] > 20.0 * grad[-1]

    # Concavity removes the Jensen bonus for a zero-sum +/- temporal flow.
    base = torch.tensor([-3.0, 0.0, 3.0])
    flow = torch.tensor([0.7, 0.7, 0.7])
    r_base = flow_add_agent.calc_flow_disc_reward(
        base, flow_add_agent.DISC_REWARD_CENTERED_LOG_D, reward_scale=2.0)
    r_plus = flow_add_agent.calc_flow_disc_reward(
        base + flow, flow_add_agent.DISC_REWARD_CENTERED_LOG_D, reward_scale=2.0)
    r_minus = flow_add_agent.calc_flow_disc_reward(
        base - flow, flow_add_agent.DISC_REWARD_CENTERED_LOG_D, reward_scale=2.0)
    assert torch.all(r_plus + r_minus <= 2.0 * r_base + 1e-6)

def test_original_softplus_reward_is_preserved():
    logits = torch.linspace(-8.0, 8.0, 33)
    prob = torch.sigmoid(logits)
    expected = -2.0 * torch.log(torch.clamp_min(1.0 - prob, 0.0001))
    actual = flow_add_agent.calc_flow_disc_reward(
        logits, flow_add_agent.DISC_REWARD_SOFTPLUS, reward_scale=2.0)
    assert torch.allclose(actual, expected, atol=1e-6)

def test_radial_rank_teacher_only_trains_static_score():
    model = build_model("flow")
    randomize_flow_matrices(model)

    gen = torch.Generator().manual_seed(19)
    x = torch.randn([64, DISC_OBS_DIM], generator=gen)
    c = torch.rand([64, 1], generator=gen)
    close_score = model.eval_static_score(c * x).squeeze(-1)
    far_score = model.eval_static_score(x).squeeze(-1)

    # A deliberately large margin guarantees an active ordering gradient.
    loss, acc, gap = flow_add_agent.calc_radial_rank_loss(
        close_score=close_score, far_score=far_score,
        contraction=c, error=x, margin=10.0)
    loss.backward()

    static_grad_norm = sum(torch.norm(p.grad) for p in model._disc_layers.parameters()
                           if p.grad is not None)
    static_grad_norm += torch.norm(model._disc_logits.weight.grad)
    assert static_grad_norm > 1e-6
    assert model._disc_flow_potential.grad is None
    assert model._disc_flow_circulation.grad is None
    assert torch.isfinite(loss) and torch.isfinite(acc) and torch.isfinite(gap)

def test_radial_rank_target_vanishes_at_zero_error():
    close_score = torch.zeros(8, requires_grad=True)
    far_score = torch.zeros(8, requires_grad=True)
    contraction = torch.full([8, 1], 0.25)
    zero_error = torch.zeros([8, DISC_OBS_DIM])

    loss, acc, gap = flow_add_agent.calc_radial_rank_loss(
        close_score=close_score, far_score=far_score,
        contraction=contraction, error=zero_error, margin=0.25)
    loss.backward()

    assert loss == 0
    assert acc == 1
    assert gap == 0
    assert torch.count_nonzero(close_score.grad) == 0
    assert torch.count_nonzero(far_score.grad) == 0

def test_radial_rank_target_scales_with_error_radius():
    contraction = torch.full([2, 1], 0.5)
    error = torch.stack([
        torch.full([DISC_OBS_DIM], 0.1),
        torch.full([DISC_OBS_DIM], 0.2)
    ])
    close_score = torch.zeros(2)
    far_score = torch.zeros(2)

    loss, _, _ = flow_add_agent.calc_radial_rank_loss(
        close_score=close_score, far_score=far_score,
        contraction=contraction, error=error, margin=1.0)
    # Per-sample target gaps are 0.05 and 0.10, so the squared-hinge mean is
    # (0.05^2 + 0.10^2) / 2.
    assert torch.allclose(loss, torch.tensor(0.00625), atol=1e-7)

def rand_unit_quat(shape, gen):
    import util.torch_util as torch_util
    axis = torch.randn(list(shape) + [3], generator=gen)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    angle = 2.0 * np.pi * torch.rand(list(shape), generator=gen)
    return torch_util.axis_angle_to_quat(axis, angle)

def make_char_data(n, s, num_joints, num_bodies, gen):
    data = {
        "root_pos": torch.randn([n, s, 3], generator=gen),
        "root_rot": rand_unit_quat([n, s], gen),
        "root_vel": torch.randn([n, s, 3], generator=gen),
        "root_ang_vel": torch.randn([n, s, 3], generator=gen),
        "joint_rot": rand_unit_quat([n, s, num_joints], gen),
        "dof_vel": torch.randn([n, s, num_joints], generator=gen),
        "body_pos": torch.randn([n, s, num_bodies, 3], generator=gen),
    }
    return data

def yaw_translate_char_data(data, yaw_quat, offset):
    # applies a global yaw rotation (about the world origin) plus translation
    # to a whole scene; yaw_quat: [n, 4], offset: [n, 3]
    import util.torch_util as torch_util
    n = yaw_quat.shape[0]

    def expand_to(v, last_dim):
        src = yaw_quat if last_dim == 4 else offset
        shape = [n] + [1] * (v.dim() - 2) + [last_dim]
        return src.reshape(shape).expand(list(v.shape[:-1]) + [last_dim])

    def rot_vec(v):
        return torch_util.quat_rotate(expand_to(v, 4), v)

    out = {
        "root_pos": rot_vec(data["root_pos"]) + expand_to(data["root_pos"], 3),
        "root_rot": torch_util.quat_mul(expand_to(data["root_rot"], 4), data["root_rot"]),
        "root_vel": rot_vec(data["root_vel"]),
        "root_ang_vel": rot_vec(data["root_ang_vel"]),
        "joint_rot": data["joint_rot"].clone(),
        "dof_vel": data["dof_vel"].clone(),
        "body_pos": rot_vec(data["body_pos"]) + expand_to(data["body_pos"], 3),
    }
    return out

def ref_frame_obs(data, ref_root_pos, ref_root_rot):
    import envs.flow_add_disc_obs as flow_add_disc_obs
    return flow_add_disc_obs.compute_ref_frame_disc_obs(
        ref_root_pos=ref_root_pos,
        ref_root_rot=ref_root_rot,
        root_pos=data["root_pos"],
        root_rot=data["root_rot"],
        root_vel=data["root_vel"],
        root_ang_vel=data["root_ang_vel"],
        joint_rot=data["joint_rot"],
        dof_vel=data["dof_vel"],
        body_pos=data["body_pos"])

def test_ref_frame_disc_obs_yaw_translation_invariance():
    # the reference-frame features must be invariant to a global yaw +
    # translation of the whole scene (agent, demo, and reference frame all
    # transformed together), which removes the world-rotation confound from
    # the circulation term; the world-frame (global_obs) features are not
    import util.torch_util as torch_util
    import envs.add_env as add_env

    gen = torch.Generator().manual_seed(11)
    n, s, num_joints, num_bodies = 4, 2, 5, 6
    sim = make_char_data(n, s, num_joints, num_bodies, gen)
    demo = make_char_data(n, s, num_joints, num_bodies, gen)
    # reference frame: demo root at the current (last) step
    ref_pos = demo["root_pos"][:, -1]
    ref_rot = demo["root_rot"][:, -1]

    z_axis = torch.zeros([n, 3])
    z_axis[:, 2] = 1.0
    yaw = torch.pi * (2.0 * torch.rand([n], generator=gen) - 1.0)
    yaw_quat = torch_util.axis_angle_to_quat(z_axis, yaw)
    offset = torch.randn([n, 3], generator=gen)

    sim_w = yaw_translate_char_data(sim, yaw_quat, offset)
    demo_w = yaw_translate_char_data(demo, yaw_quat, offset)
    ref_pos_w = torch_util.quat_rotate(yaw_quat, ref_pos) + offset
    ref_rot_w = torch_util.quat_mul(yaw_quat, ref_rot)

    diff = ref_frame_obs(demo, ref_pos, ref_rot) - ref_frame_obs(sim, ref_pos, ref_rot)
    diff_w = ref_frame_obs(demo_w, ref_pos_w, ref_rot_w) - ref_frame_obs(sim_w, ref_pos_w, ref_rot_w)
    assert torch.allclose(diff, diff_w, atol=1e-4)

    # sanity check of the confound: the same transform changes the world-frame
    # (global_obs = True) differential
    def global_obs(data):
        return add_env.compute_disc_obs(root_pos=data["root_pos"],
                                        root_rot=data["root_rot"],
                                        root_vel=data["root_vel"],
                                        root_ang_vel=data["root_ang_vel"],
                                        joint_rot=data["joint_rot"],
                                        dof_vel=data["dof_vel"],
                                        body_pos=data["body_pos"],
                                        global_obs=True)

    g_diff = global_obs(demo) - global_obs(sim)
    g_diff_w = global_obs(demo_w) - global_obs(sim_w)
    assert not torch.allclose(g_diff, g_diff_w, atol=1e-3)

def test_ref_frame_disc_obs_perfect_tracking_is_zero():
    # identical agent and demo states give a zero differential in the
    # reference frame, preserving ADD's universal ideal point
    gen = torch.Generator().manual_seed(13)
    demo = make_char_data(3, 2, 5, 6, gen)
    ref_pos = demo["root_pos"][:, -1]
    ref_rot = demo["root_rot"][:, -1]

    obs_demo = ref_frame_obs(demo, ref_pos, ref_rot)
    obs_sim = ref_frame_obs({k: v.clone() for k, v in demo.items()}, ref_pos, ref_rot)
    assert torch.allclose(obs_demo - obs_sim, torch.zeros_like(obs_demo), atol=1e-6)

def test_ref_frame_disc_obs_dim_matches_global():
    # the canonical variant must be a drop-in: same feature dimension as the
    # global_obs = True features
    import envs.add_env as add_env

    gen = torch.Generator().manual_seed(17)
    demo = make_char_data(2, 3, 5, 6, gen)
    ref_pos = demo["root_pos"][:, -1]
    ref_rot = demo["root_rot"][:, -1]

    obs_ref = ref_frame_obs(demo, ref_pos, ref_rot)
    obs_global = add_env.compute_disc_obs(root_pos=demo["root_pos"],
                                          root_rot=demo["root_rot"],
                                          root_vel=demo["root_vel"],
                                          root_ang_vel=demo["root_ang_vel"],
                                          joint_rot=demo["joint_rot"],
                                          dof_vel=demo["dof_vel"],
                                          body_pos=demo["body_pos"],
                                          global_obs=True)
    assert obs_ref.shape == obs_global.shape

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
