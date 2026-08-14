"""Unit tests for Structured Trajectory ADD (ST-ADD).

Covers the two pillars of the method without a simulator:

TrajHistory (trajectory differential):
  - winding is accumulated from per-step quaternion-log increments: a full
    2*pi rotation yields 2*pi, not the 0 an endpoint-quaternion comparison
    would give
  - a no-roll shortcut against a rolling reference leaves a ~pi winding
    residual per half-cycle window: the shortcut is no longer a zero of the
    trajectory differential
  - perfect tracking keeps every trajectory feature at exactly zero (the
    single ADD positive sample is preserved)
  - resets clear the rings: nothing leaks across episodes, and partial
    windows accumulate only post-reset motion
  - displacement windows measure anchored accumulated displacement
  - features are invariant to a global yaw (absorbed by the motion anchor)
    and to quaternion hemisphere flips

STADDModel / STADDAgent (structured discriminator):
  - the rotation branch sees only the winding residual segment; the state
    and motion branches cannot see it (structural isolation, both ways)
  - the fused logit is the fixed equal-weight mean of the branch logits
  - per-branch auxiliary BCE reaches exactly its own branch parameters
  - gradients of the fused logit are finite at the zero positive sample
    (gradient-penalty well-posedness)
  - the reward transform softplus keeps rewards strictly positive
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import gymnasium.spaces as spaces

import envs.trajectory_add_obs as trajectory_add_obs
import learning.st_add_model as st_add_model
import util.torch_util as torch_util

DEVICE = "cpu"
WINDOWS = [8, 16, 32]
DOF_DIM = 4
NUM_BODIES = 2

OBS_DIM = 10
ACTION_DIM = 6
STATE_DIM = 12
MOTION_DIM = len(WINDOWS) * (3 + DOF_DIM + 3 * NUM_BODIES)
ROT_DIM = len(WINDOWS) * 3
TOTAL_DIM = STATE_DIM + MOTION_DIM + ROT_DIM

def make_hist(num_envs=1):
    return trajectory_add_obs.TrajHistory(num_envs=num_envs, windows=WINDOWS,
                                          dof_dim=DOF_DIM, num_bodies=NUM_BODIES,
                                          device=DEVICE)

def identity_quat(n=1):
    q = torch.zeros([n, 4])
    q[..., 3] = 1.0
    return q

def yaw_quat(angle, n=1):
    axis = torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1)
    return torch_util.axis_angle_to_quat(axis, torch.full([n], angle))

def axis_quat(axis_xyz, angle, n=1):
    axis = torch.tensor([axis_xyz]).repeat(n, 1)
    return torch_util.axis_angle_to_quat(axis, torch.full([n], angle))

def reset_state(n=1):
    root_pos = torch.zeros([n, 3])
    root_rot = identity_quat(n)
    dof_pos = torch.zeros([n, DOF_DIM])
    body_rel = torch.zeros([n, NUM_BODIES, 3])
    return root_pos, root_rot, dof_pos, body_rel

def fill_reset(hist, n=1, anchor=None):
    if anchor is None:
        anchor = identity_quat(n)
    env_ids = torch.arange(n)
    hist.reset_fill(env_ids, *reset_state(n), anchor)
    return anchor

def split_motion(motion_obs):
    """Split motion features into per-window (delta_pos, delta_dof, delta_body)."""
    per_w = 3 + DOF_DIM + 3 * NUM_BODIES
    out = []
    for i in range(len(WINDOWS)):
        seg = motion_obs[:, i * per_w:(i + 1) * per_w]
        out.append((seg[:, :3], seg[:, 3:3 + DOF_DIM], seg[:, 3 + DOF_DIM:]))
    return out

def split_rot(rot_obs):
    return [rot_obs[:, 3 * i:3 * i + 3] for i in range(len(WINDOWS))]

# ---------------------------------------------------------------------------
# TrajHistory: winding
# ---------------------------------------------------------------------------

def test_winding_full_rotation_not_zero():
    """A full 2*pi roll about the world x-axis accumulates to 2*pi even though
    the endpoint quaternion equals the start quaternion."""
    hist = make_hist()
    anchor = fill_reset(hist)

    steps = 32
    dtheta = 2.0 * np.pi / steps
    for k in range(1, steps + 1):
        rot = axis_quat([1.0, 0.0, 0.0], k * dtheta)
        root_pos, _, dof_pos, body_rel = reset_state()
        hist.push(root_pos, rot, dof_pos, body_rel, anchor)

    _, rot_obs = hist.extract()
    winds = split_rot(rot_obs)

    # endpoint quaternions are identical (2*pi) -> naive comparison gives 0
    end_rot = axis_quat([1.0, 0.0, 0.0], 2.0 * np.pi)
    assert torch.allclose(torch.abs(end_rot), torch.abs(identity_quat()), atol=1e-5)

    # window 32 captures the whole turn; windows 8/16 capture 1/4 and 1/2
    expected = {8: 2.0 * np.pi / 4, 16: np.pi, 32: 2.0 * np.pi}
    for h, wind in zip(WINDOWS, winds):
        assert torch.allclose(wind[0, 0], torch.tensor(expected[h]), atol=1e-4), h
        assert torch.allclose(wind[0, 1:], torch.zeros(2), atol=1e-5)

def test_winding_shortcut_nonzero_residual():
    """Rolling reference vs a policy that stays put: the winding residual is
    ~pi over a half-cycle window, so the shortcut is not a zero of the
    trajectory differential."""
    ref_hist = make_hist()
    sim_hist = make_hist()
    anchor_ref = fill_reset(ref_hist)
    anchor_sim = fill_reset(sim_hist)

    steps = 16
    for k in range(1, steps + 1):
        # reference: rolls pi over the window while translating forward
        ref_rot = axis_quat([0.0, 1.0, 0.0], k * np.pi / steps)
        ref_pos = torch.tensor([[0.05 * k, 0.0, 0.0]])
        _, _, dof_pos, body_rel = reset_state()
        ref_hist.push(ref_pos, ref_rot, dof_pos, body_rel, anchor_ref)
        # shortcut: same forward translation, no rotation
        sim_pos = torch.tensor([[0.05 * k, 0.0, 0.0]])
        sim_hist.push(sim_pos, identity_quat(), dof_pos, body_rel, anchor_sim)

    ref_motion, ref_rot_obs = ref_hist.extract()
    sim_motion, sim_rot_obs = sim_hist.extract()

    rot_resid = split_rot(ref_rot_obs - sim_rot_obs)
    h16 = rot_resid[WINDOWS.index(16)]
    assert torch.allclose(h16[0, 1], torch.tensor(np.pi), atol=1e-4)

    # displacement matches -> motion residual (pos part) is zero; only the
    # winding exposes the shortcut
    motion_resid = split_motion(ref_motion - sim_motion)
    for dp, dd, db in motion_resid:
        assert torch.allclose(dp, torch.zeros_like(dp), atol=1e-5)

def test_perfect_tracking_zero_diff():
    """Identical streams keep the trajectory differential exactly zero, and
    right after a reset all features are exactly zero."""
    ref_hist = make_hist()
    sim_hist = make_hist()
    anchor = fill_reset(ref_hist)
    fill_reset(sim_hist)

    # immediately after reset: exact zeros
    for hist in (ref_hist, sim_hist):
        motion_obs, rot_obs = hist.extract()
        assert torch.all(motion_obs == 0)
        assert torch.all(rot_obs == 0)

    g = torch.Generator().manual_seed(0)
    rot = identity_quat()
    for k in range(40):
        rot = torch_util.quat_mul(axis_quat([0.3, 0.5, 0.8], 0.05), rot)
        root_pos = torch.randn([1, 3], generator=g) * 0.1 + torch.tensor([[0.02 * k, 0.0, 0.9]])
        dof_pos = torch.randn([1, DOF_DIM], generator=g)
        body_rel = torch.randn([1, NUM_BODIES, 3], generator=g)
        ref_hist.push(root_pos, rot, dof_pos, body_rel, anchor)
        sim_hist.push(root_pos, rot, dof_pos, body_rel, anchor)

    ref_motion, ref_rot = ref_hist.extract()
    sim_motion, sim_rot = sim_hist.extract()
    assert torch.allclose(ref_motion, sim_motion, atol=0)
    assert torch.allclose(ref_rot, sim_rot, atol=0)

def test_reset_isolation():
    """reset_fill wipes pre-reset history; post-reset windows accumulate only
    post-reset motion."""
    hist = make_hist()
    anchor = fill_reset(hist)

    # garbage pre-reset episode
    for k in range(40):
        hist.push(torch.full([1, 3], float(k)), axis_quat([1.0, 0.0, 0.0], 0.4 * k),
                  torch.full([1, DOF_DIM], float(k)), torch.full([1, NUM_BODIES, 3], float(k)),
                  anchor)

    # reset: everything must be exactly zero again
    fill_reset(hist)
    motion_obs, rot_obs = hist.extract()
    assert torch.all(motion_obs == 0)
    assert torch.all(rot_obs == 0)

    # one post-reset step: every window sees exactly that one step
    step_pos = torch.tensor([[0.3, 0.0, 0.0]])
    step_angle = 0.2
    hist.push(step_pos, axis_quat([0.0, 0.0, 1.0], step_angle),
              torch.zeros([1, DOF_DIM]), torch.zeros([1, NUM_BODIES, 3]), anchor)
    motion_obs, rot_obs = hist.extract()
    for dp, _, _ in split_motion(motion_obs):
        assert torch.allclose(dp, step_pos, atol=1e-5)
    for wind in split_rot(rot_obs):
        assert torch.allclose(wind[0, 2], torch.tensor(step_angle), atol=1e-5)

def test_displacement_windows():
    """Constant velocity: window h accumulates h * step displacement."""
    hist = make_hist()
    anchor = fill_reset(hist)

    step = torch.tensor([[0.05, -0.02, 0.01]])
    for k in range(1, 40):
        hist.push(k * step, identity_quat(), torch.zeros([1, DOF_DIM]),
                  torch.zeros([1, NUM_BODIES, 3]), anchor)

    motion_obs, _ = hist.extract()
    for h, (dp, dd, db) in zip(WINDOWS, split_motion(motion_obs)):
        assert torch.allclose(dp, h * step, atol=1e-4), h

def test_yaw_invariance():
    """A global yaw of the whole scene is absorbed by the motion anchor:
    trajectory features are unchanged."""
    yaw = 1.1
    q_yaw = yaw_quat(yaw)

    hist_a = make_hist()
    hist_b = make_hist()
    anchor_a = identity_quat()
    anchor_b = torch_util.calc_heading_quat_inv(q_yaw)

    env_ids = torch.arange(1)
    root_pos, root_rot, dof_pos, body_rel = reset_state()
    hist_a.reset_fill(env_ids, root_pos, root_rot, dof_pos, body_rel, anchor_a)
    hist_b.reset_fill(env_ids, torch_util.quat_rotate(q_yaw, root_pos),
                      torch_util.quat_mul(q_yaw, root_rot), dof_pos, body_rel, anchor_b)

    g = torch.Generator().manual_seed(1)
    rot = identity_quat()
    for k in range(36):
        rot = torch_util.quat_mul(axis_quat([0.2, 0.9, 0.1], 0.07), rot)
        pos = torch.randn([1, 3], generator=g) * 0.2
        dof = torch.randn([1, DOF_DIM], generator=g)
        body = torch.randn([1, NUM_BODIES, 3], generator=g)

        hist_a.push(pos, rot, dof, body, anchor_a)
        # body_rel points are world-frame offsets -> also rotate under yaw
        body_yaw = torch_util.quat_rotate(q_yaw.unsqueeze(1).repeat(1, NUM_BODIES, 1).reshape(-1, 4),
                                          body.reshape(-1, 3)).reshape(1, NUM_BODIES, 3)
        hist_b.push(torch_util.quat_rotate(q_yaw, pos), torch_util.quat_mul(q_yaw, rot),
                    dof, body_yaw, anchor_b)

    motion_a, rot_a = hist_a.extract()
    motion_b, rot_b = hist_b.extract()
    assert torch.allclose(motion_a, motion_b, atol=1e-4)
    assert torch.allclose(rot_a, rot_b, atol=1e-4)

def test_hemisphere_invariance():
    """Pushing q or -q gives identical winding increments."""
    hist_a = make_hist()
    hist_b = make_hist()
    anchor_a = fill_reset(hist_a)
    anchor_b = fill_reset(hist_b)

    rot = identity_quat()
    signs = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
    for k in range(32):
        rot = torch_util.quat_mul(axis_quat([0.4, 0.2, 0.88], 0.09), rot)
        args = (torch.zeros([1, 3]), torch.zeros([1, DOF_DIM]), torch.zeros([1, NUM_BODIES, 3]))
        hist_a.push(args[0], rot, args[1], args[2], anchor_a)
        hist_b.push(args[0], signs[k % len(signs)] * rot, args[1], args[2], anchor_b)

    _, rot_a = hist_a.extract()
    _, rot_b = hist_b.extract()
    assert torch.allclose(rot_a, rot_b, atol=1e-5)

# ---------------------------------------------------------------------------
# STADDModel: structured discriminator
# ---------------------------------------------------------------------------

class FakeEnv:
    def get_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(TOTAL_DIM,), dtype=np.float32)

    def get_disc_state_obs_dim(self):
        return STATE_DIM

    def get_disc_traj_motion_obs_dim(self):
        return MOTION_DIM

    def get_disc_traj_rot_obs_dim(self):
        return ROT_DIM

def make_model(fusion="mean", tau=2.0, head="independent"):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_state_net": "fc_2layers_128units",
        "disc_motion_net": "fc_2layers_128units",
        "disc_rot_net": "fc_2layers_128units",
        "disc_fusion": fusion,
        "disc_fusion_tau": tau,
        "disc_head": head,
        "disc_head_dim": 32,
    }
    torch.manual_seed(0)
    return st_add_model.STADDModel(config, FakeEnv())

def make_hetero_model(head):
    """Model with the real heterogeneous encoder widths (1024/512/128) to
    expose the branch logit scale problem."""
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_state_net": "fc_2layers_1024units",
        "disc_motion_net": "fc_2layers_512units",
        "disc_rot_net": "fc_2layers_128units",
        "disc_head": head,
        "disc_head_dim": 128,
    }
    torch.manual_seed(0)
    return st_add_model.STADDModel(config, FakeEnv())

def rand_disc_obs(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn([n, TOTAL_DIM], generator=g)

def test_model_branch_isolation():
    """Each branch only responds to its own segment of the differential."""
    model = make_model()
    x = rand_disc_obs()
    z_s, z_m, z_r, z_f = model.eval_disc_branches(x)

    # perturb state+motion -> rotation logit unchanged
    x2 = x.clone()
    x2[:, :STATE_DIM + MOTION_DIM] += torch.randn_like(x2[:, :STATE_DIM + MOTION_DIM])
    z_s2, z_m2, z_r2, _ = model.eval_disc_branches(x2)
    assert torch.allclose(z_r2, z_r, atol=0)
    assert not torch.allclose(z_s2, z_s)
    assert not torch.allclose(z_m2, z_m)

    # perturb rotation -> state/motion logits unchanged
    x3 = x.clone()
    x3[:, STATE_DIM + MOTION_DIM:] += torch.randn_like(x3[:, STATE_DIM + MOTION_DIM:])
    z_s3, z_m3, z_r3, _ = model.eval_disc_branches(x3)
    assert torch.allclose(z_s3, z_s, atol=0)
    assert torch.allclose(z_m3, z_m, atol=0)
    assert not torch.allclose(z_r3, z_r)

def test_model_fused_is_equal_weight_mean():
    model = make_model()
    x = rand_disc_obs()
    z_s, z_m, z_r, z_f = model.eval_disc_branches(x)
    assert torch.allclose(z_f, (z_s + z_m + z_r) / 3.0, atol=1e-6)
    assert torch.allclose(model.eval_disc(x), z_f, atol=0)

def test_model_logit_weights_and_params():
    model = make_model()
    w = model.get_disc_logit_weights()
    expected = (model._disc_state_logits.weight.numel()
                + model._disc_motion_logits.weight.numel()
                + model._disc_rot_logits.weight.numel())
    assert w.numel() == expected

    disc_params = set(id(p) for p in model.get_disc_params())
    for mod in (model._disc_state_layers, model._disc_motion_layers, model._disc_rot_layers,
                model._disc_state_logits, model._disc_motion_logits, model._disc_rot_logits):
        for p in mod.parameters():
            assert id(p) in disc_params

def test_aux_loss_reaches_only_its_branch():
    """The rotation auxiliary BCE trains exactly the rotation branch."""
    model = make_model()
    x = rand_disc_obs()
    _, _, z_r, _ = model.eval_disc_branches(x)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(z_r.squeeze(-1), torch.zeros(z_r.shape[0]))
    loss.backward()

    rot_grads = [p.grad for p in model._disc_rot_layers.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in rot_grads)
    for mod in (model._disc_state_layers, model._disc_motion_layers):
        for p in mod.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

def test_fused_grad_finite_at_zero():
    """Gradient penalty well-posedness at the single positive sample."""
    model = make_model()
    x = torch.zeros([1, TOTAL_DIM], requires_grad=True)
    z = model.eval_disc(x)
    grad = torch.autograd.grad(z.sum(), x)[0]
    assert torch.all(torch.isfinite(grad))

def test_reward_strictly_positive():
    """r = scale * softplus(z) > 0 for any finite logit."""
    z = torch.linspace(-30.0, 30.0, 101)
    r = 2.0 * torch.nn.functional.softplus(z)
    assert torch.all(r > 0)

# ---------------------------------------------------------------------------
# ZA-STADD: zero-anchored smooth-Tchebycheff fusion
# ---------------------------------------------------------------------------

def test_za_tau_inf_equals_mean():
    """tau -> inf recovers ST-ADD mean fusion exactly (same seed => same
    branch weights, so the fused logits must match)."""
    model_mean = make_model("mean")
    model_za = make_model("za", tau=1e4)
    x = rand_disc_obs()
    _, _, _, z_mean = model_mean.eval_disc_branches(x)
    _, _, _, z_za = model_za.eval_disc_branches(x)
    # residual O(1/tau) bottleneck term + float32 rounding amplified by tau
    assert torch.allclose(z_za, z_mean, atol=1e-2)

def test_za_small_tau_is_bottleneck():
    """tau -> 0 approaches mean(b) - max_i d_i (worst anchored deficit)."""
    model = make_model("za", tau=1e-4)
    x = rand_disc_obs()
    z_s, z_m, z_r, z_f = model.eval_disc_branches(x)
    b = model.eval_zero_anchor()
    d = b - torch.cat([z_s, z_m, z_r], dim=-1)
    expected = torch.mean(b, dim=-1, keepdim=True) - torch.max(d, dim=-1, keepdim=True).values
    # finite-tau residual is tau*log(3) ~= 1e-4
    assert torch.allclose(z_f, expected, atol=1e-3)

def test_za_zero_input_equals_anchor_mean():
    """At the universal ADD ideal point the fused logit equals mean(z(0)) for
    both fusion modes: the positive sample semantics are unchanged."""
    for fusion in ("mean", "za"):
        model = make_model(fusion, tau=1.5)
        zeros = torch.zeros([1, TOTAL_DIM])
        z_f = model.eval_disc(zeros)
        b = model.eval_zero_anchor()
        assert torch.allclose(z_f, torch.mean(b, dim=-1, keepdim=True), atol=1e-6), fusion

def test_za_additive_bias_invariance():
    """Adding a constant bias to one branch head leaves the deficits and the
    bottleneck attention unchanged (the anchor absorbs it); the fused logit
    shifts by exactly bias/3, same as mean fusion."""
    model = make_model("za", tau=2.0)
    x = rand_disc_obs()

    z_s, z_m, z_r, z_f = model.eval_disc_branches(x)
    b = model.eval_zero_anchor()
    d = b - torch.cat([z_s, z_m, z_r], dim=-1)
    w = torch.softmax(d / 2.0, dim=-1)

    bias = 5.0
    with torch.no_grad():
        model._disc_state_logits.bias += bias

    z_s2, z_m2, z_r2, z_f2 = model.eval_disc_branches(x)
    b2 = model.eval_zero_anchor()
    d2 = b2 - torch.cat([z_s2, z_m2, z_r2], dim=-1)
    w2 = torch.softmax(d2 / 2.0, dim=-1)

    assert torch.allclose(d2, d, atol=1e-5)
    assert torch.allclose(w2, w, atol=1e-5)
    assert torch.allclose(z_f2, z_f + bias / 3.0, atol=1e-5)

def test_za_gradient_weights_are_softmax():
    """Local sensitivity of the fused logit to branch logit i is
    w_i = softmax(d/tau)_i (automatic bottleneck attention): a small change
    in one branch logit moves the fused logit by w_i times that change."""
    tau = 1.7
    model = make_model("za", tau=tau)
    x = rand_disc_obs(n=1)

    z_s, z_m, z_r, z_f = model.eval_disc_branches(x)
    b = model.eval_zero_anchor()
    d = b - torch.cat([z_s, z_m, z_r], dim=-1)
    w = torch.softmax(d / tau, dim=-1)

    # perturb only the rotation segment of the input: only z_r moves, so
    # dz_fused ~= w_r * dz_rot
    x2 = x.clone()
    x2[:, STATE_DIM + MOTION_DIM:] += 1e-3
    z_s2, z_m2, z_r2, z_f2 = model.eval_disc_branches(x2)
    assert torch.allclose(z_s2, z_s, atol=0) and torch.allclose(z_m2, z_m, atol=0)
    dz_rot = (z_r2 - z_r).item()
    dz_fused = (z_f2 - z_f).item()
    assert abs(dz_fused - w[0, 2].item() * dz_rot) < 5e-6 + 5e-3 * abs(dz_rot)

# ---------------------------------------------------------------------------
# ST-ADD v2: shared normalized head (single-ruler logits)
# ---------------------------------------------------------------------------

def test_shared_head_scale_identifiability():
    """With independent heads, heterogeneous encoder widths (1024/512/128)
    produce branch logits on visibly different scales at init. The shared
    normalized head puts every branch feature on the same sphere
    (||u_i|| = sqrt(k) exactly, parameter-free LayerNorm), so the achievable
    logit range |z - b| <= ||w||*sqrt(k) is identical across branches: one
    ruler. Across-sample dispersion also gets closer at init."""
    x = rand_disc_obs(n=4096)

    model_ind = make_hetero_model("independent")
    with torch.no_grad():
        z_s, z_m, z_r = model_ind._eval_branch_logits(x)
    stds_ind = torch.tensor([z_s.std(), z_m.std(), z_r.std()])
    ratio_ind = (stds_ind.max() / stds_ind.min()).item()
    assert ratio_ind > 1.5, ratio_ind

    model_shared = make_hetero_model("shared")
    k = model_shared._disc_shared_logits.weight.shape[-1]
    state_obs, motion_obs, rot_obs = model_shared._split_disc_obs(x)
    with torch.no_grad():
        for obs, enc, proj in (
                (state_obs, model_shared._disc_state_layers, model_shared._disc_state_proj),
                (motion_obs, model_shared._disc_motion_layers, model_shared._disc_motion_proj),
                (rot_obs, model_shared._disc_rot_layers, model_shared._disc_rot_proj)):
            u = model_shared._disc_head_norm(proj(enc(obs)))
            norms = torch.linalg.norm(u, dim=-1) / (k ** 0.5)
            assert torch.all(torch.abs(norms - 1.0) < 1e-2), norms

        z_s, z_m, z_r = model_shared._eval_branch_logits(x)
    stds_shared = torch.tensor([z_s.std(), z_m.std(), z_r.std()])
    ratio_shared = (stds_shared.max() / stds_shared.min()).item()
    assert ratio_shared < ratio_ind, (ratio_shared, ratio_ind)

def test_shared_head_is_single_ruler():
    """All three branches are scored by literally the same weight vector, and
    the logit regularization only sees that one head."""
    model = make_model(head="shared")
    w = model.get_disc_logit_weights()
    assert w.numel() == model._disc_shared_logits.weight.numel()
    assert torch.equal(w, torch.flatten(model._disc_shared_logits.weight))

def test_shared_head_gradients_reach_all_encoders():
    """The fused logit trains all three encoders, projections, and the shared
    head."""
    model = make_model(head="shared")
    x = rand_disc_obs()
    _, _, _, z_f = model.eval_disc_branches(x)
    z_f.sum().backward()
    mods = (model._disc_state_layers, model._disc_motion_layers, model._disc_rot_layers,
            model._disc_state_proj, model._disc_motion_proj, model._disc_rot_proj,
            model._disc_shared_logits)
    for mod in mods:
        grads = [p.grad for p in mod.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

def test_shared_head_disc_params_complete():
    """get_disc_params covers every discriminator parameter exactly once in
    shared mode (encoders + projections + shared head)."""
    model = make_model(head="shared")
    params = model.get_disc_params()
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids))
    expected = (list(model._disc_state_layers.parameters())
                + list(model._disc_motion_layers.parameters())
                + list(model._disc_rot_layers.parameters())
                + list(model._disc_state_proj.parameters())
                + list(model._disc_motion_proj.parameters())
                + list(model._disc_rot_proj.parameters())
                + list(model._disc_shared_logits.parameters()))
    assert set(ids) == set(id(p) for p in expected)

def test_shared_head_fusion_and_anchor_still_work():
    """Zero-anchor and mean fusion semantics are unchanged under the shared
    head: eval_disc(0) == mean of the branch anchors."""
    model = make_model(head="shared")
    zeros = torch.zeros([1, TOTAL_DIM])
    z_f = model.eval_disc(zeros)
    b = model.eval_zero_anchor()
    assert torch.allclose(z_f, torch.mean(b, dim=-1, keepdim=True), atol=1e-6)

def test_za_anchor_detached_in_fusion():
    """The fused logit backpropagates through the live branch logits but the
    anchor path is detached; gradients stay finite and reach every branch."""
    model = make_model("za", tau=2.0)
    x = rand_disc_obs()
    x.requires_grad_(True)
    _, _, _, z_f = model.eval_disc_branches(x)
    z_f.sum().backward()
    assert torch.all(torch.isfinite(x.grad))
    # all three branch encoders receive gradient through the fused logit
    for mod in (model._disc_state_layers, model._disc_motion_layers, model._disc_rot_layers):
        grads = [p.grad for p in mod.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)
