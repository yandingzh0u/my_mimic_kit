"""Reference-frame canonicalized discriminator features for FlowADD.

With global_obs = True, ADD's disc features are expressed in the world frame,
so a yaw rotation of the whole scene (e.g. a spinning motion progressing
around the world z-axis) rotates the error differential even when the
tracking error is constant in the character's local frame. That world
rotation shows up as signed area in differential space and confounds the
circulation term.

These functions express both the agent's and the demo's features in a common
frame anchored at the reference motion's current root (heading-only rotation
plus translation). The differential x = phi(demo) - phi(agent) then becomes
invariant to a global yaw + translation of the scene while still retaining
the full relative tracking error, including the heading error (the agent's
root rotation is expressed relative to the demo heading, not its own).

The feature layout matches add_env.compute_disc_obs with global_obs = True
(root pos 3, root rot 6, joint rot 6J, body pos 3B, root vel 3, root ang
vel 3, dof vel D per history step), so the disc obs space is unchanged.
"""

import torch

import util.torch_util as torch_util

@torch.jit.script
def compute_ref_frame_pos_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, body_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # ref_root_pos: [n, 3], ref_root_rot: [n, 4]
    # root_pos: [n, s, 3], root_rot: [n, s, 4]
    # joint_rot: [n, s, J, 4], body_pos: [n, s, B, 3]
    n = root_pos.shape[0]
    s = root_pos.shape[1]

    heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
    heading_inv_steps = heading_inv_rot.unsqueeze(1).repeat(1, s, 1)
    heading_inv_flat = heading_inv_steps.reshape(n * s, 4)

    # root position relative to the reference root, in the reference heading
    # frame; the heading rotation is about z so heights pass through unchanged
    rel_root_pos = root_pos - ref_root_pos.unsqueeze(1)
    rel_root_pos_flat = rel_root_pos.reshape(n * s, 3)
    root_pos_obs = torch_util.quat_rotate(heading_inv_flat, rel_root_pos_flat)
    root_pos_obs = root_pos_obs.reshape(n, s, 3)

    # root rotation in the reference heading frame, keeps the heading error
    root_rot_flat = root_rot.reshape(n * s, 4)
    root_rot_ref = torch_util.quat_mul(heading_inv_flat, root_rot_flat)
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_ref)
    root_rot_obs = root_rot_obs_flat.reshape(n, s, root_rot_obs_flat.shape[-1])

    # joint rotations are parent-relative and already yaw invariant
    num_joints = joint_rot.shape[2]
    joint_rot_flat = joint_rot.reshape(n * s * num_joints, 4)
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = joint_rot_obs_flat.reshape(n, s, num_joints * joint_rot_obs_flat.shape[-1])

    # body positions relative to own root, rotated into the reference heading frame
    num_bodies = body_pos.shape[2]
    body_rel = body_pos - root_pos.unsqueeze(-2)
    heading_inv_body = heading_inv_steps.unsqueeze(2).repeat(1, 1, num_bodies, 1)
    heading_inv_body_flat = heading_inv_body.reshape(n * s * num_bodies, 4)
    body_rel_flat = body_rel.reshape(n * s * num_bodies, 3)
    body_pos_obs = torch_util.quat_rotate(heading_inv_body_flat, body_rel_flat)
    body_pos_obs = body_pos_obs.reshape(n, s, num_bodies * 3)

    obs = [root_pos_obs, root_rot_obs, joint_rot_obs, body_pos_obs]
    return torch.cat(obs, dim=-1)

@torch.jit.script
def compute_ref_frame_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # ref_root_rot: [n, 4], root_vel/root_ang_vel: [n, s, 3], dof_vel: [n, s, D]
    n = root_vel.shape[0]
    s = root_vel.shape[1]

    heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
    heading_inv_steps = heading_inv_rot.unsqueeze(1).repeat(1, s, 1)
    heading_inv_flat = heading_inv_steps.reshape(n * s, 4)

    root_vel_obs = torch_util.quat_rotate(heading_inv_flat, root_vel.reshape(n * s, 3))
    root_vel_obs = root_vel_obs.reshape(n, s, 3)

    root_ang_vel_obs = torch_util.quat_rotate(heading_inv_flat, root_ang_vel.reshape(n * s, 3))
    root_ang_vel_obs = root_ang_vel_obs.reshape(n, s, 3)

    obs = [root_vel_obs, root_ang_vel_obs, dof_vel]
    return torch.cat(obs, dim=-1)

@torch.jit.script
def compute_ref_frame_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel,
                               root_ang_vel, joint_rot, dof_vel, body_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    pos_obs = compute_ref_frame_pos_obs(ref_root_pos=ref_root_pos,
                                        ref_root_rot=ref_root_rot,
                                        root_pos=root_pos,
                                        root_rot=root_rot,
                                        joint_rot=joint_rot,
                                        body_pos=body_pos)

    vel_obs = compute_ref_frame_vel_obs(ref_root_rot=ref_root_rot,
                                        root_vel=root_vel,
                                        root_ang_vel=root_ang_vel,
                                        dof_vel=dof_vel)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs
