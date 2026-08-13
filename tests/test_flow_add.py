"""Unit tests for the FlowADD tangent-error flow discriminator (route A).

These tests exercise the discriminator math without a simulator:
  - the discriminator is a strict concat MLP on [x_t, v_t] with
    v_t = x_t - x_t-1, so the previous frame enters only through the tangent
  - perfect tracking (0, 0) stays the single universal positive sample
  - disc_tangent_input = False is a strict falsification switch: identical
    parameters and joint-input graph, tangent channel exactly zero
  - gradients on the joint input [x_t-1, x_t] are finite, including at the
    ideal point (0, 0) (needed for the gradient penalty)
  - the fixed group-balanced potential E(x) and its progress/absolute-energy
    reward components match their definitions and cannot produce a positive
    cycle bonus
  - the reference-frame disc features are invariant to a global yaw +
    translation of the scene, unlike the world-frame (global_obs) features
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

def build_model(**overrides):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units"
    }
    config.update(overrides)
    model = flow_add_model.FlowADDModel(config, FakeEnv())
    return model

def build_potential_model(**overrides):
    return build_model(
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.4, 0.6],
        **overrides)

def rand_inputs(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    disc_obs = torch.randn([n, DISC_OBS_DIM], generator=gen)
    disc_obs_prev = disc_obs + 0.3 * torch.randn([n, DISC_OBS_DIM], generator=gen)
    return disc_obs, disc_obs_prev

def manual_logit(model, disc_obs, disc_flow):
    disc_in = torch.cat([disc_obs, disc_flow], dim=-1)
    return model._disc_logits(model._disc_layers(disc_in))

def test_disc_consumes_tangent_input():
    # the discriminator is a single MLP on the joint input [x_t, v_t]
    model = build_model()
    first_layer = [m for m in model._disc_layers.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first_layer.in_features == 2 * DISC_OBS_DIM

    disc_obs, disc_obs_prev = rand_inputs(16)
    logit = model.eval_disc(disc_obs, disc_obs_prev)
    assert logit.shape == (16, 1)
    assert torch.all(torch.isfinite(logit))

    # the logit must depend on the previous frame through v
    logit_static = model.eval_disc(disc_obs, disc_obs.clone())
    assert not torch.allclose(logit, logit_static)

def test_prev_frame_enters_only_through_tangent():
    # z(x_t-1, x_t) must equal f([x_t, x_t - x_t-1]) exactly
    model = build_model()
    disc_obs, disc_obs_prev = rand_inputs(64, seed=1)

    z = model.eval_disc(disc_obs, disc_obs_prev)
    expected = manual_logit(model, disc_obs, disc_obs - disc_obs_prev)
    assert torch.allclose(z, expected, atol=1e-6)

    # a static transition x_t-1 = x_t reduces to the pointwise scalarizer
    z_static = model.eval_disc(disc_obs, disc_obs.clone())
    expected_static = manual_logit(model, disc_obs, torch.zeros_like(disc_obs))
    assert torch.allclose(z_static, expected_static, atol=1e-6)

def test_perfect_tracking_is_the_single_positive_point():
    # (x_t-1, x_t) = (0, 0) maps to the same fixed input for any motion
    model = build_model()
    zeros = torch.zeros([8, DISC_OBS_DIM])
    z = model.eval_disc(zeros, zeros)
    expected = manual_logit(model, zeros, zeros)
    assert torch.all(torch.isfinite(z))
    assert torch.allclose(z, expected, atol=1e-6)

def test_transition_orientation_sensitivity():
    # reversing a transition flips the tangent sign, so the discriminator can
    # score A -> B and B -> A differently; shuffling x_t-1 across the batch
    # must also change the logit
    model = build_model()
    disc_obs, disc_obs_prev = rand_inputs(64, seed=2)

    z_fwd = model.eval_disc(disc_obs, disc_obs_prev)
    z_bwd = model.eval_disc(disc_obs_prev, disc_obs)
    assert not torch.allclose(z_fwd, z_bwd, atol=1e-3)

    perm = torch.randperm(disc_obs.shape[0])
    z_shuf = model.eval_disc(disc_obs, disc_obs_prev[perm])
    assert not torch.allclose(z_fwd, z_shuf, atol=1e-3)

def test_tangent_off_is_strict_shape_preserving_ablation():
    # disc_tangent_input = False keeps identical parameters and the same
    # joint-input graph while making the tangent channel exactly zero
    model_on = build_model()
    model_off = build_model(disc_tangent_input=False)
    assert model_on.is_tangent_input_enabled()
    assert not model_off.is_tangent_input_enabled()

    params_on = [p.shape for p in model_on.get_disc_params()]
    params_off = [p.shape for p in model_off.get_disc_params()]
    assert params_on == params_off

    disc_obs, disc_obs_prev = rand_inputs(64, seed=3)
    disc_obs.requires_grad_(True)
    disc_obs_prev.requires_grad_(True)

    logit = model_off.eval_disc(disc_obs, disc_obs_prev).squeeze(-1)
    logit_static = model_off.eval_disc(disc_obs, disc_obs.detach().clone()).squeeze(-1)
    assert torch.allclose(logit, logit_static, atol=1e-6)

    # x_t-1 stays in the graph, so the joint-input GP receives a finite zero
    # gradient instead of an unused-tensor error
    grad, grad_prev = torch.autograd.grad(
        torch.sum(logit), [disc_obs, disc_obs_prev])
    assert torch.all(torch.isfinite(grad))
    assert torch.all(torch.isfinite(grad_prev))
    assert torch.count_nonzero(grad_prev) == 0

@pytest.mark.parametrize("tangent_input", [True, False])
def test_grad_penalty_finite(tangent_input):
    # gradient penalty is taken w.r.t. the joint input [x_t-1, x_t], including
    # the ideal point (0, 0); first and second order gradients must be finite
    model = build_model(disc_tangent_input=tangent_input)

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

def test_disc_params_and_logit_weights():
    # the discriminator has no extra matrices: trunk (2 linear layers => 4
    # tensors) + head (weight + bias); logit reg covers only the head weights;
    # the fixed potential buffer is not a trainable parameter
    model = build_potential_model()

    params = model.get_disc_params()
    assert len(params) == 6

    trunk_out = 64  # fc_2layers_128units => [128, 64]
    assert model.get_disc_logit_weights().numel() == trunk_out

    assert model.has_fixed_potential()
    assert not model._potential_diag_weights.requires_grad
    param_ids = {id(p) for p in model.parameters()}
    assert id(model._potential_diag_weights) not in param_ids

def test_fixed_group_balanced_potential_energy():
    model = build_potential_model()

    gen = torch.Generator().manual_seed(31)
    x = torch.randn([32, DISC_OBS_DIM], generator=gen)
    energy = model.eval_potential_energy(x)
    group_energy = model.eval_potential_group_energies(x)
    expected = 0.5 * (0.4 * torch.mean(torch.square(x[:, :3]), dim=-1)
                      + 0.6 * torch.mean(torch.square(x[:, 3:]), dim=-1))

    assert torch.allclose(energy, expected, atol=1e-6)
    assert group_energy.shape == (32, 2)
    assert torch.allclose(torch.sum(group_energy, dim=-1), energy, atol=1e-6)
    assert model.get_num_potential_groups() == 2

    # unnormalized weights give the same energy after normalization
    model_scaled = build_model(
        disc_flow_potential_group_dims=[3, DISC_OBS_DIM - 3],
        disc_flow_potential_group_weights=[0.8, 1.2])
    assert torch.allclose(model_scaled.eval_potential_energy(x), energy, atol=1e-6)

    # without the group config there is no potential
    assert not build_model().has_fixed_potential()

def test_potential_progress_telescopes_exactly():
    # with the policy discount, the discounted sum of E(x_t-1) - gamma E(x_t)
    # telescopes to E(x_0) - gamma^T E(x_T): standard potential shaping
    model = build_potential_model()
    gen = torch.Generator().manual_seed(33)
    path = torch.randn([11, DISC_OBS_DIM], generator=gen)
    gamma = 0.99

    energy = model.eval_potential_energy(path)
    progress, _ = flow_add_agent.calc_fixed_potential_rewards(
        energy=energy[1:],
        energy_prev=energy[:-1],
        progress_scale=1.0,
        progress_discount=gamma,
        abs_energy_weight=0.0)
    discounts = gamma ** torch.arange(progress.shape[0])
    discounted_sum = torch.sum(discounts * progress)
    expected = energy[0] - gamma ** progress.shape[0] * energy[-1]
    assert torch.allclose(discounted_sum, expected, atol=1e-5)

def test_potential_progress_has_no_path_dependent_cycle_bonus():
    # two paths with the same endpoints get identical discounted shaping sums
    model = build_potential_model()
    gen = torch.Generator().manual_seed(34)
    endpoint = torch.randn([DISC_OBS_DIM], generator=gen)
    path_a = torch.stack([endpoint, torch.randn([DISC_OBS_DIM], generator=gen), endpoint])
    path_b = torch.stack([endpoint, torch.randn([DISC_OBS_DIM], generator=gen), endpoint])
    gamma = 0.99
    discounts = gamma ** torch.arange(2)

    def discounted_shaping(path):
        energy = model.eval_potential_energy(path)
        progress, _ = flow_add_agent.calc_fixed_potential_rewards(
            energy=energy[1:], energy_prev=energy[:-1], progress_scale=1.0,
            progress_discount=gamma, abs_energy_weight=0.0)
        return torch.sum(discounts * progress)

    assert torch.allclose(discounted_shaping(path_a), discounted_shaping(path_b), atol=1e-6)

def test_raw_progress_prefers_low_error_over_oscillation():
    # with shaping discount 1.0, reduce-and-hold beats oscillation beats
    # staying at a constant high error
    model = build_potential_model()
    high = torch.ones([DISC_OBS_DIM])
    low = 0.2 * high
    gamma = 0.99

    paths = [
        torch.stack([high, low, low, low, low]),
        torch.stack([high, low, high, low, high]),
        torch.stack([high, high, high, high, high])
    ]
    returns = []
    discounts = gamma ** torch.arange(4)
    for path in paths:
        energy = model.eval_potential_energy(path, clip=5.0)
        progress, _ = flow_add_agent.calc_fixed_potential_rewards(
            energy=energy[1:], energy_prev=energy[:-1], progress_scale=1.0,
            progress_discount=1.0, abs_energy_weight=0.0)
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
    model = build_potential_model()
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
    model = build_potential_model()
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
    model = build_potential_model()
    pos = torch.full([4, DISC_OBS_DIM], 100.0)
    neg = -pos
    at_clip = torch.full([4, DISC_OBS_DIM], 5.0)
    e_pos = model.eval_potential_energy(pos, clip=5.0)
    e_neg = model.eval_potential_energy(neg, clip=5.0)
    e_clip = model.eval_potential_energy(at_clip, clip=5.0)
    assert torch.allclose(e_pos, e_neg)
    assert torch.allclose(e_pos, e_clip)

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

def test_reward_min_clamps_only_low_rewards():
    logits = torch.tensor([-6.0, 0.0, 3.0])
    clamped = flow_add_agent.calc_flow_disc_reward(
        logits, flow_add_agent.DISC_REWARD_CENTERED_LOG_D,
        reward_scale=1.0, reward_min=-0.5)
    unclamped = flow_add_agent.calc_flow_disc_reward(
        logits, flow_add_agent.DISC_REWARD_CENTERED_LOG_D,
        reward_scale=1.0, reward_min=None)

    assert clamped[0] == -0.5
    assert unclamped[0] < -0.5
    assert torch.allclose(clamped[1:], unclamped[1:])

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
    # the tangent features; the world-frame (global_obs) features are not
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
