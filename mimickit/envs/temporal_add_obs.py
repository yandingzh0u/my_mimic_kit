"""Geometry for the TangentADD tangent branch (stage 1).

All quantities are canonicalized by a per-episode fixed motion yaw anchor
H_m: the heading of the reference root at motion time 0. The current root
heading is never used as the anchor, because it is ill posed while the
character is inverted (e.g. mid-roll) and can jump by pi.

Two feature vectors are produced:

  configuration error (pre-step, gates the tangent reward):
      xi_t = [ H_m^-1 (p_sim - p_ref),
               log((R_ref)^-1 R_sim),
               { log((Q_ref_j)^-1 Q_sim_j) }_j ]

  generalized velocity residual (post-step, the tangent error itself):
      e^u_{t+1} = [ H_m^-1 (v_sim - v_ref),
                    H_m^-1 (w_sim - w_ref),
                    qdot_sim - qdot_ref ]

Rotation errors use quaternion relative rotation + log map.
torch_util.quat_to_exp_map forces w >= 0 internally (quat_pos), so the
results are identical for q and -q and the rotation angle is minimal
(in [0, pi]). Relative rotations are frame independent, and the position /
velocity differences are rotated into the anchor frame, so both feature
vectors are invariant to global yaw rotations and translations of the scene.
"""

import torch

import util.torch_util as torch_util

@torch.jit.script
def calc_motion_anchor_quat_inv(ref_root_rot0):
    # type: (Tensor) -> Tensor
    """Inverse yaw anchor H_m^-1 from the reference root rotation at phase 0."""
    return torch_util.calc_heading_quat_inv(ref_root_rot0)

@torch.jit.script
def quat_rel_log(q_ref, q):
    # type: (Tensor, Tensor) -> Tensor
    """Log map of the relative rotation (q_ref)^-1 q, hemisphere safe."""
    q_rel = torch_util.quat_mul(torch_util.quat_conjugate(q_ref), q)
    return torch_util.quat_to_exp_map(q_rel)

@torch.jit.script
def calc_config_error(anchor_inv, root_pos, root_rot, joint_rot,
                      ref_root_pos, ref_root_rot, ref_joint_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Pre-step configuration error xi_t, [n, 3 + 3 + 3 * num_joints]."""
    pos_err = torch_util.quat_rotate(anchor_inv, root_pos - ref_root_pos)
    root_rot_err = quat_rel_log(ref_root_rot, root_rot)

    n = joint_rot.shape[0]
    joint_rot_flat = joint_rot.reshape(-1, 4)
    ref_joint_rot_flat = ref_joint_rot.reshape(-1, 4)
    joint_err = quat_rel_log(ref_joint_rot_flat, joint_rot_flat)
    joint_err = joint_err.reshape(n, -1)

    cfg_err = torch.cat([pos_err, root_rot_err, joint_err], dim=-1)
    return cfg_err

@torch.jit.script
def calc_vel_residual(anchor_inv, root_vel, root_ang_vel, dof_vel,
                      ref_root_vel, ref_root_ang_vel, ref_dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Post-step generalized velocity residual e^u, [n, 3 + 3 + num_dof]."""
    lin_err = torch_util.quat_rotate(anchor_inv, root_vel - ref_root_vel)
    ang_err = torch_util.quat_rotate(anchor_inv, root_ang_vel - ref_root_ang_vel)
    dof_err = dof_vel - ref_dof_vel

    vel_resid = torch.cat([lin_err, ang_err, dof_err], dim=-1)
    return vel_resid

def calc_group_energy(x, group_dims, group_weights, group_scales):
    """Dimension-balanced energy E = sum_g w_g * mean((x_g / s_g)^2).

    group_dims/group_weights/group_scales are python lists of the same
    length; scales are fixed physical scales, never running statistics.
    """
    assert x.shape[-1] == sum(group_dims)
    assert len(group_dims) == len(group_weights) == len(group_scales)

    energy = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    idx = 0
    for dim, weight, scale in zip(group_dims, group_weights, group_scales):
        seg = x[..., idx:idx + dim]
        energy = energy + weight * torch.mean(seg * seg, dim=-1) / (scale * scale)
        idx += dim
    return energy

def calc_tangent_rewards(cfg_err, vel_resid, cfg_group_dims, vel_group_dims,
                         group_weights, cfg_scales, tan_scales,
                         gate_radius, error_sigma):
    """Gated tangent reward r_tan = w_t * exp(-E_tan / (2 sigma^2)),
    with manifold gate w_t = exp(-E_cfg / (2 rho^2)). Both in [0, 1].
    """
    e_cfg = calc_group_energy(cfg_err, cfg_group_dims, group_weights, cfg_scales)
    e_tan = calc_group_energy(vel_resid, vel_group_dims, group_weights, tan_scales)

    gate_w = torch.exp(-e_cfg / (2.0 * gate_radius * gate_radius))
    tangent_r = gate_w * torch.exp(-e_tan / (2.0 * error_sigma * error_sigma))
    return gate_w, tangent_r, e_cfg, e_tan
