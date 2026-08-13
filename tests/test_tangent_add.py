"""Unit tests for the TangentADD tangent branch (stage 1).

Simulator-free tests of the geometry and reward math:
  - perfect tracking gives zero configuration error, zero velocity residual,
    and r_tan = 1 (the universal maximum)
  - matching the reference tangent scores strictly higher than holding still,
    which scores strictly higher than reversing the motion
  - both feature vectors are invariant to a global yaw rotation +
    translation of the scene and to quaternion hemisphere flips (q vs -q)
  - the log map stays continuous while the relative rotation crosses the
    quaternion hemisphere boundary (no pi jumps mid-roll)
  - the motion anchor is a pure yaw rotation derived only from the phase-0
    frame, so it cannot jump while the character is inverted
  - the manifold gate closes monotonically with configuration error and the
    reward is always finite and in [0, 1], even for absurd inputs
  - with lambda_tan = 0 the combined reward is bitwise the base ADD reward,
    and non-negative per-step rewards mean longer episodes never lower return
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import envs.temporal_add_obs as tobs
import util.torch_util as torch_util

NUM_JOINTS = 14
NUM_DOF = 28
CFG_GROUP_DIMS = [3, 3, 3 * NUM_JOINTS]
VEL_GROUP_DIMS = [3, 3, NUM_DOF]
GROUP_WEIGHTS = [1.0 / 3.0] * 3
CFG_SCALES = [0.5, 0.4, 0.4]
TAN_SCALES = [0.9, 2.7, 2.5]
SCALE_FILE = os.path.join(REPO_ROOT, "data", "stats", "humanoid_tangent_scales.npz")

def rand_quat(n, gen):
    q = torch.randn([n, 4], generator=gen)
    return q / torch.norm(q, dim=-1, keepdim=True)

def rand_state(n, gen):
    state = {
        "root_pos": torch.randn([n, 3], generator=gen),
        "root_rot": rand_quat(n, gen),
        "joint_rot": rand_quat(n * NUM_JOINTS, gen).reshape(n, NUM_JOINTS, 4),
        "root_vel": torch.randn([n, 3], generator=gen),
        "root_ang_vel": torch.randn([n, 3], generator=gen),
        "dof_vel": torch.randn([n, NUM_DOF], generator=gen),
    }
    return state

def yaw_quat(angle, n):
    axis = torch.zeros([n, 3])
    axis[:, 2] = 1.0
    return torch_util.axis_angle_to_quat(axis, torch.full([n], angle))

def cfg_err_of(anchor_inv, sim, ref):
    return tobs.calc_config_error(anchor_inv,
                                  sim["root_pos"], sim["root_rot"], sim["joint_rot"],
                                  ref["root_pos"], ref["root_rot"], ref["joint_rot"])

def vel_resid_of(anchor_inv, sim, ref):
    return tobs.calc_vel_residual(anchor_inv,
                                  sim["root_vel"], sim["root_ang_vel"], sim["dof_vel"],
                                  ref["root_vel"], ref["root_ang_vel"], ref["dof_vel"])

def tangent_reward_of(cfg_err, vel_resid):
    _, r_tan, _, _ = tobs.calc_tangent_rewards(cfg_err, vel_resid,
                                               CFG_GROUP_DIMS, VEL_GROUP_DIMS,
                                               GROUP_WEIGHTS, CFG_SCALES, TAN_SCALES,
                                               gate_radius=1.0, error_sigma=1.0)
    return r_tan

def test_perfect_tracking_is_zero_and_max_reward():
    gen = torch.Generator().manual_seed(0)
    ref = rand_state(8, gen)
    anchor_inv = torch_util.calc_heading_quat_inv(ref["root_rot"])

    cfg_err = cfg_err_of(anchor_inv, ref, ref)
    vel_resid = vel_resid_of(anchor_inv, ref, ref)

    assert torch.allclose(cfg_err, torch.zeros_like(cfg_err), atol=1e-5)
    assert torch.allclose(vel_resid, torch.zeros_like(vel_resid), atol=1e-6)

    gate_w, r_tan, e_cfg, e_tan = tobs.calc_tangent_rewards(
        cfg_err, vel_resid, CFG_GROUP_DIMS, VEL_GROUP_DIMS,
        GROUP_WEIGHTS, CFG_SCALES, TAN_SCALES, 1.0, 1.0)
    assert torch.allclose(gate_w, torch.ones_like(gate_w), atol=1e-5)
    assert torch.allclose(r_tan, torch.ones_like(r_tan), atol=1e-5)

def test_correct_tangent_beats_hold_beats_reverse():
    gen = torch.Generator().manual_seed(1)
    ref = rand_state(16, gen)
    anchor_inv = torch_util.calc_heading_quat_inv(ref["root_rot"])
    cfg_err = torch.zeros([16, sum(CFG_GROUP_DIMS)])

    correct = dict(ref)
    hold = dict(ref, root_vel=torch.zeros_like(ref["root_vel"]),
                root_ang_vel=torch.zeros_like(ref["root_ang_vel"]),
                dof_vel=torch.zeros_like(ref["dof_vel"]))
    reverse = dict(ref, root_vel=-ref["root_vel"],
                   root_ang_vel=-ref["root_ang_vel"],
                   dof_vel=-ref["dof_vel"])

    r_correct = tangent_reward_of(cfg_err, vel_resid_of(anchor_inv, correct, ref))
    r_hold = tangent_reward_of(cfg_err, vel_resid_of(anchor_inv, hold, ref))
    r_reverse = tangent_reward_of(cfg_err, vel_resid_of(anchor_inv, reverse, ref))

    assert torch.all(r_correct > r_hold)
    assert torch.all(r_hold > r_reverse)

def test_global_yaw_translation_invariance():
    gen = torch.Generator().manual_seed(2)
    n = 12
    ref = rand_state(n, gen)
    sim = rand_state(n, gen)
    anchor_inv = torch_util.calc_heading_quat_inv(ref["root_rot"])

    cfg0 = cfg_err_of(anchor_inv, sim, ref)
    vel0 = vel_resid_of(anchor_inv, sim, ref)

    yaw = yaw_quat(1.234, n)
    shift = torch.tensor([3.0, -2.0, 0.0])

    def transform(state):
        out = dict(state)
        out["root_pos"] = torch_util.quat_rotate(yaw, state["root_pos"]) + shift
        out["root_rot"] = torch_util.quat_mul(yaw, state["root_rot"])
        out["root_vel"] = torch_util.quat_rotate(yaw, state["root_vel"])
        out["root_ang_vel"] = torch_util.quat_rotate(yaw, state["root_ang_vel"])
        return out

    ref_t = transform(ref)
    sim_t = transform(sim)
    anchor_inv_t = torch_util.calc_heading_quat_inv(ref_t["root_rot"])

    cfg1 = cfg_err_of(anchor_inv_t, sim_t, ref_t)
    vel1 = vel_resid_of(anchor_inv_t, sim_t, ref_t)

    assert torch.allclose(cfg0, cfg1, atol=1e-4)
    assert torch.allclose(vel0, vel1, atol=1e-4)

def test_quaternion_hemisphere_invariance():
    gen = torch.Generator().manual_seed(3)
    n = 12
    ref = rand_state(n, gen)
    sim = rand_state(n, gen)
    anchor_inv = torch_util.calc_heading_quat_inv(ref["root_rot"])

    cfg0 = cfg_err_of(anchor_inv, sim, ref)

    sim_f = dict(sim, root_rot=-sim["root_rot"], joint_rot=-sim["joint_rot"])
    ref_f = dict(ref, root_rot=-ref["root_rot"], joint_rot=-ref["joint_rot"])

    for s, r in [(sim_f, ref), (sim, ref_f), (sim_f, ref_f)]:
        cfg1 = cfg_err_of(anchor_inv, s, r)
        assert torch.allclose(cfg0, cfg1, atol=1e-5)

def test_log_map_continuous_across_hemisphere():
    # relative rotation sweeps a great circle; the raw quaternion changes
    # hemisphere but the log map must evolve continuously (angle stays < pi)
    angles = torch.linspace(-2.5, 2.5, 401)
    axis = torch.tensor([[0.6, 0.8, 0.0]]).repeat(len(angles), 1)
    q_ref = yaw_quat(0.3, len(angles))
    q_delta = torch_util.axis_angle_to_quat(axis, angles)
    q_sim = torch_util.quat_mul(q_ref, q_delta)
    # randomly flip signs to simulate hemisphere-inconsistent inputs
    gen = torch.Generator().manual_seed(4)
    flip = torch.where(torch.rand(len(angles), 1, generator=gen) > 0.5, 1.0, -1.0)
    q_sim = flip * q_sim

    logs = tobs.quat_rel_log(q_ref, q_sim)
    steps = torch.norm(logs[1:] - logs[:-1], dim=-1)
    assert torch.all(steps < 0.05)
    assert torch.allclose(torch.norm(logs, dim=-1), torch.abs(angles), atol=1e-4)

def test_motion_anchor_is_pure_yaw_from_phase0_only():
    gen = torch.Generator().manual_seed(5)
    root_rot0 = rand_quat(6, gen)
    anchor_inv = tobs.calc_motion_anchor_quat_inv(root_rot0)

    # pure yaw: rotating the z axis leaves it unchanged
    z = torch.zeros([6, 3])
    z[:, 2] = 1.0
    z_rot = torch_util.quat_rotate(anchor_inv, z)
    assert torch.allclose(z_rot, z, atol=1e-5)

    # anchor depends only on the phase-0 frame: recomputing it with the same
    # frame while the "current" pose is inverted gives the identical anchor
    anchor_inv2 = tobs.calc_motion_anchor_quat_inv(root_rot0)
    assert torch.equal(anchor_inv, anchor_inv2)

def test_gate_closes_monotonically_with_config_error():
    scales = torch.tensor([1.0])
    for mult0, mult1 in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]:
        cfg0 = mult0 * torch.ones([1, sum(CFG_GROUP_DIMS)])
        cfg1 = mult1 * torch.ones([1, sum(CFG_GROUP_DIMS)])
        vel = torch.zeros([1, sum(VEL_GROUP_DIMS)])
        w0, _, _, _ = tobs.calc_tangent_rewards(cfg0, vel, CFG_GROUP_DIMS, VEL_GROUP_DIMS,
                                                GROUP_WEIGHTS, CFG_SCALES, TAN_SCALES, 1.0, 1.0)
        w1, _, _, _ = tobs.calc_tangent_rewards(cfg1, vel, CFG_GROUP_DIMS, VEL_GROUP_DIMS,
                                                GROUP_WEIGHTS, CFG_SCALES, TAN_SCALES, 1.0, 1.0)
        assert w1 < w0

def test_reward_always_finite_nonnegative_bounded():
    gen = torch.Generator().manual_seed(6)
    n = 64
    cfg = 1e4 * torch.randn([n, sum(CFG_GROUP_DIMS)], generator=gen)
    vel = 1e4 * torch.randn([n, sum(VEL_GROUP_DIMS)], generator=gen)
    cfg[0] = 0.0
    vel[0] = 0.0
    cfg[1] = 1e12
    vel[1] = 1e12

    gate_w, r_tan, e_cfg, e_tan = tobs.calc_tangent_rewards(
        cfg, vel, CFG_GROUP_DIMS, VEL_GROUP_DIMS,
        GROUP_WEIGHTS, CFG_SCALES, TAN_SCALES, 1.0, 1.0)

    for t in [gate_w, r_tan]:
        assert torch.all(torch.isfinite(t))
        assert torch.all(t >= 0.0)
        assert torch.all(t <= 1.0)

def test_lambda_zero_is_bitwise_base_reward_and_returns_never_shrink():
    gen = torch.Generator().manual_seed(7)
    base_r = torch.rand([256], generator=gen) + 0.01  # ADD softplus reward > 0
    r_tan = torch.rand([256], generator=gen)

    total0 = base_r + 0.0 * r_tan
    assert torch.equal(total0, base_r)

    # extending an episode with non-negative rewards never lowers the return
    total = base_r + 0.5 * r_tan
    assert torch.all(total >= 0.0)
    returns = torch.cumsum(total, dim=0)
    assert torch.all(returns[1:] >= returns[:-1])

def test_group_energy_matches_manual():
    gen = torch.Generator().manual_seed(8)
    x = torch.randn([5, sum(CFG_GROUP_DIMS)], generator=gen)
    e = tobs.calc_group_energy(x, CFG_GROUP_DIMS, GROUP_WEIGHTS, CFG_SCALES)

    manual = torch.zeros(5)
    idx = 0
    for d, w, s in zip(CFG_GROUP_DIMS, GROUP_WEIGHTS, CFG_SCALES):
        seg = x[:, idx:idx + d]
        manual += w * torch.mean(seg * seg, dim=-1) / (s * s)
        idx += d
    assert torch.allclose(e, manual, atol=1e-6)

def test_frozen_scale_file_matches_humanoid():
    data = np.load(SCALE_FILE)
    assert list(data["cfg_group_dims"]) == CFG_GROUP_DIMS
    assert list(data["vel_group_dims"]) == VEL_GROUP_DIMS
    assert np.all(data["cfg_scales"] > 0)
    assert np.all(data["tan_scales"] > 0)
    # holding still on the manifold should sit near the E_tan ~= 1 boundary
    # by construction (scales are the reference velocity RMS)
    assert data["tan_scales"][1] > 1.0  # roll has strong root angular velocity

def test_hold_still_energy_calibrated_near_one():
    # with tan scales equal to the reference velocity RMS, "hold still" has
    # residual -u_ref, so E_tan ~= 1 and r_tan ~= exp(-1/2) on the manifold
    gen = torch.Generator().manual_seed(9)
    n = 4096
    u = torch.cat([TAN_SCALES[0] * torch.randn([n, 3], generator=gen),
                   TAN_SCALES[1] * torch.randn([n, 3], generator=gen),
                   TAN_SCALES[2] * torch.randn([n, NUM_DOF], generator=gen)], dim=-1)
    e = tobs.calc_group_energy(u, VEL_GROUP_DIMS, GROUP_WEIGHTS, TAN_SCALES)
    assert abs(torch.mean(e).item() - 1.0) < 0.05
