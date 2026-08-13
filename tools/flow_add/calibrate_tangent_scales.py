"""One-time calibration of the fixed physical scales for TangentADD.

Produces the frozen scale file consumed by tangent_add_agent (yaml key
tangent_reward.scale_file). The scales are calibrated once from the
reference motion plus a fixed, seeded small-perturbation set, then never
updated (no running normalizers).

  tan_scales  (root lin vel, root ang vel, dof vel): per-coordinate RMS of
      the reference generalized velocities over the motion. A policy that
      simply holds still on the manifold therefore gets E_tan ~= 1 and
      r_tan ~= exp(-1/2), strictly below perfect tracking (r_tan = 1).

  cfg_scales  (root pos, root rot, joint rot): per-coordinate RMS of the
      configuration error produced by perturbing reference frames with the
      given magnitudes. A state perturbed at exactly these magnitudes gets
      E_cfg ~= 1 and manifold gate w ~= exp(-1/2); the gate is effectively
      closed (< 0.15) beyond about twice these magnitudes.

Example:
    python tools/flow_add/calibrate_tangent_scales.py \
        --char_file data/assets/humanoid/humanoid.xml \
        --motion_file data/motions/humanoid/humanoid_roll.pkl \
        --out data/stats/humanoid_tangent_scales.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))

import anim.mjcf_char_model as mjcf_char_model
import anim.motion_lib as motion_lib
import envs.temporal_add_obs as temporal_add_obs
import util.torch_util as torch_util

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fixed tangent/config scales for TangentADD")
    parser.add_argument("--char_file", type=str, default="data/assets/humanoid/humanoid.xml")
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root_pos_mag", type=float, default=0.5, help="m, gate half-trust radius for root position")
    parser.add_argument("--root_rot_mag", type=float, default=0.7, help="rad, gate half-trust radius for root rotation")
    parser.add_argument("--joint_rot_mag", type=float, default=0.7, help="rad, gate half-trust radius for joint rotations")
    parser.add_argument("--vel_scale_floor", type=float, default=0.2, help="minimum per-group velocity scale")
    return parser.parse_args()

def sample_frames(mlib, num_samples, generator, device):
    num_motions = mlib.get_num_motions()
    motion_ids = torch.arange(num_motions, device=device, dtype=torch.long)
    lengths = mlib.get_motion_length(motion_ids)

    # sample motions proportional to length, times uniform within each motion
    probs = lengths / torch.sum(lengths)
    sampled_ids = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    times = torch.rand(num_samples, device=device, generator=generator) * lengths[sampled_ids]
    return sampled_ids, times

def rand_axis_angle_quat(n, num_rots, mag, generator, device):
    axis = torch.randn([n, num_rots, 3], device=device, generator=generator)
    axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-8)
    angle = mag * torch.randn([n, num_rots], device=device, generator=generator)
    quat = torch_util.axis_angle_to_quat(axis.reshape(-1, 3), angle.reshape(-1))
    return quat.reshape(n, num_rots, 4)

def group_rms(x, group_dims):
    out = []
    idx = 0
    for dim in group_dims:
        seg = x[..., idx:idx + dim]
        out.append(float(torch.sqrt(torch.mean(seg * seg)).item()))
        idx += dim
    return out

def main():
    args = parse_args()
    device = "cpu"
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    char_model = mjcf_char_model.MJCFCharModel(device)
    char_model.load(args.char_file)
    mlib = motion_lib.MotionLib(motion_file=args.motion_file, kin_char_model=char_model, device=device)

    num_joints = char_model.get_num_joints()
    dof_size = char_model.get_dof_size()
    cfg_group_dims = [3, 3, 3 * (num_joints - 1)]
    vel_group_dims = [3, 3, dof_size]

    motion_ids, times = sample_frames(mlib, args.num_samples, generator, device)
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, times)

    # per-episode fixed yaw anchor, exactly as used by TangentADDEnv
    num_motions = mlib.get_num_motions()
    all_ids = torch.arange(num_motions, device=device, dtype=torch.long)
    _, root_rot0, _, _, _, _ = mlib.calc_motion_frame(all_ids, torch.zeros(num_motions, device=device))
    anchor_inv = temporal_add_obs.calc_motion_anchor_quat_inv(root_rot0)[motion_ids]

    # ---- tangent (velocity) scales: RMS of the reference generalized velocities
    ref_u = torch.cat([torch_util.quat_rotate(anchor_inv, root_vel),
                       torch_util.quat_rotate(anchor_inv, root_ang_vel),
                       dof_vel], dim=-1)
    tan_scales = group_rms(ref_u, vel_group_dims)
    tan_scales = [max(s, args.vel_scale_floor) for s in tan_scales]

    # ---- config scales: RMS of the config error under the fixed perturbation set
    n = args.num_samples
    pert_root_pos = root_pos + args.root_pos_mag * torch.randn([n, 3], device=device, generator=generator)
    pert_root_rot = torch_util.quat_mul(
        root_rot, rand_axis_angle_quat(n, 1, args.root_rot_mag, generator, device).squeeze(1))
    pert_joint_rot = torch_util.quat_mul(
        joint_rot, rand_axis_angle_quat(n, num_joints - 1, args.joint_rot_mag, generator, device))

    cfg_err = temporal_add_obs.calc_config_error(anchor_inv=anchor_inv,
                                                 root_pos=pert_root_pos,
                                                 root_rot=pert_root_rot,
                                                 joint_rot=pert_joint_rot,
                                                 ref_root_pos=root_pos,
                                                 ref_root_rot=root_rot,
                                                 ref_joint_rot=joint_rot)
    cfg_scales = group_rms(cfg_err, cfg_group_dims)

    print("cfg_scales  (root pos [m], root rot [rad/coord], joint rot [rad/coord]):", cfg_scales)
    print("tan_scales  (root lin [m/s], root ang [rad/s], dof vel [rad/s]):", tan_scales)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             cfg_scales=np.array(cfg_scales, dtype=np.float64),
             tan_scales=np.array(tan_scales, dtype=np.float64),
             cfg_group_dims=np.array(cfg_group_dims, dtype=np.int64),
             vel_group_dims=np.array(vel_group_dims, dtype=np.int64),
             meta_char_file=np.array(args.char_file),
             meta_motion_file=np.array(args.motion_file),
             meta_num_samples=np.array(args.num_samples),
             meta_seed=np.array(args.seed),
             meta_perturb_mags=np.array([args.root_pos_mag, args.root_rot_mag, args.joint_rot_mag]),
             meta_vel_scale_floor=np.array(args.vel_scale_floor))
    print("wrote", args.out)
    return

if __name__ == "__main__":
    main()
