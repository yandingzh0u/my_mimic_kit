"""Behavior-level evaluation for Roll.

Rolls out a trained policy in TEST mode and reports the decisive metrics, which
target behavior rather than average tracking error:

  winding_ratio     directed root rotation about the motion's dominant
                    rotation axis, integrated over the episode, divided by
                    the reference's winding over the same window. ~1 for a
                    correct roll, ~0 for lie-down/get-up or standing still,
                    < 0 for rolling the wrong way.
  disp_ratio        forward root displacement along the motion's travel
                    direction (velocity integral in the anchor frame),
                    divided by the reference's. Roll advances ~per 2 s cycle,
                    so standing in place gives ~0.
  upright_rate      fraction of completed episodes that end upright
                    (root z-axis within 45 deg of vertical, root height
                    > 0.5 m).
  root_height_rmse  RMS of (sim root height - ref root height) over steps.
  shortcut_rate     fraction of completed (non-FAIL) episodes with
                    winding_ratio < 0.5: the episode survived without
                    actually performing the rotation (lie-down/get-up,
                    stand-and-wait, ...).

All directional quantities are expressed in the per-episode fixed motion yaw
anchor frame (reference root heading at motion time 0), the same anchor the
CPMD increments use. The metrics only read root states, so this works for any
ADD-style env.

Example:
    python tools/cpmd/evaluate_roll_behavior.py \
        --env_config data/envs/cpmd_humanoid_roll_eval_env.yaml \
        --engine_config output/cpmd_roll_cycle_1k_seed0/engine_config.yaml \
        --agent_config output/cpmd_roll_cycle_1k_seed0/agent_config.yaml \
        --model_file output/cpmd_roll_cycle_1k_seed0/model.pt \
        --num_envs 64 --episodes 256 --out output/cpmd_roll_cycle_1k_seed0/roll_behavior.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))

import envs.base_env as base_env
import envs.env_builder as env_builder
import envs.cpmd_obs as cpmd_obs
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.util as util

def parse_args():
    parser = argparse.ArgumentParser(description="Roll behavior metrics (winding/displacement/upright/shortcut)")
    parser.add_argument("--env_config", type=str, required=True)
    parser.add_argument("--engine_config", type=str, required=True)
    parser.add_argument("--agent_config", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rand_seed", type=int, default=0)
    parser.add_argument("--upright_cos", type=float, default=0.7071, help="cos threshold for upright root z axis")
    parser.add_argument("--upright_height", type=float, default=0.5, help="minimum root height for upright [m]")
    parser.add_argument("--shortcut_winding", type=float, default=0.5, help="winding ratio below which a completed episode counts as shortcut")
    parser.add_argument("--out", type=str, default="output/roll_behavior.npz")
    return parser.parse_args()

def build(args):
    mp_util.init(0, 1, args.device, 6000)
    util.set_rand_seed(args.rand_seed)

    env = env_builder.build_env(args.env_config, args.engine_config, args.num_envs,
                                args.device, visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    return env, agent

def build_motion_frames(env, device):
    """Per-motion anchor (inverse yaw), dominant rotation axis, and travel
    direction, all in the anchor frame, sampled from the reference motion."""
    mlib = env._motion_lib
    num_motions = mlib.get_num_motions()
    motion_ids = torch.arange(num_motions, device=device, dtype=torch.long)

    _, root_rot0, _, _, _, _ = mlib.calc_motion_frame(motion_ids, torch.zeros(num_motions, device=device))
    anchor_inv = cpmd_obs.calc_motion_anchor_quat_inv(root_rot0)

    lengths = mlib.get_motion_length(motion_ids)
    num_samples = 200
    roll_axis = torch.zeros([num_motions, 3], device=device)
    travel_dir = torch.zeros([num_motions, 3], device=device)
    for m in range(num_motions):
        times = torch.linspace(0.0, float(lengths[m].item()) * (1.0 - 1e-4), num_samples, device=device)
        ids = torch.full([num_samples], m, device=device, dtype=torch.long)
        _, _, root_vel, root_ang_vel, _, _ = mlib.calc_motion_frame(ids, times)
        a_inv = anchor_inv[m].unsqueeze(0).repeat(num_samples, 1)

        mean_ang = torch.mean(torch_util.quat_rotate(a_inv, root_ang_vel), dim=0)
        roll_axis[m] = mean_ang / torch.clamp(torch.norm(mean_ang), min=1e-6)

        mean_vel = torch.mean(torch_util.quat_rotate(a_inv, root_vel), dim=0)
        mean_vel[2] = 0.0
        travel_dir[m] = mean_vel / torch.clamp(torch.norm(mean_vel), min=1e-6)
    return anchor_inv, roll_axis, travel_dir

def evaluate(env, agent, args):
    device = agent._device
    num_envs = env.get_num_envs()
    dt = env._engine.get_timestep()
    char_id = env._get_char_id()

    anchor_inv_tbl, roll_axis_tbl, travel_dir_tbl = build_motion_frames(env, device)

    obs, info = env.reset()

    def env_frames():
        ids = env._motion_ids
        return anchor_inv_tbl[ids], roll_axis_tbl[ids], travel_dir_tbl[ids]

    wind_sim = torch.zeros([num_envs], device=device)
    wind_ref = torch.zeros([num_envs], device=device)
    disp_sim = torch.zeros([num_envs], device=device)
    disp_ref = torch.zeros([num_envs], device=device)
    height_sq = torch.zeros([num_envs], device=device)
    steps = torch.zeros([num_envs], device=device)

    records = {k: [] for k in ["winding_ratio", "disp_ratio", "upright", "height_rmse",
                               "done_flag", "ep_len"]}

    with torch.no_grad():
        while (len(records["done_flag"]) < args.episodes):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)

            anchor_inv, roll_axis, travel_dir = env_frames()

            root_vel = env._engine.get_root_vel(char_id)
            root_ang_vel = env._engine.get_root_ang_vel(char_id)
            root_pos = env._engine.get_root_pos(char_id)
            root_rot = env._engine.get_root_rot(char_id)

            w_sim = torch_util.quat_rotate(anchor_inv, root_ang_vel)
            w_ref = torch_util.quat_rotate(anchor_inv, env._ref_root_ang_vel)
            v_sim = torch_util.quat_rotate(anchor_inv, root_vel)
            v_ref = torch_util.quat_rotate(anchor_inv, env._ref_root_vel)

            wind_sim += dt * torch.sum(w_sim * roll_axis, dim=-1)
            wind_ref += dt * torch.sum(w_ref * roll_axis, dim=-1)
            disp_sim += dt * torch.sum(v_sim * travel_dir, dim=-1)
            disp_ref += dt * torch.sum(v_ref * travel_dir, dim=-1)
            height_sq += torch.square(root_pos[..., 2] - env._ref_root_pos[..., 2])
            steps += 1

            done_ids = torch.flatten((done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False))
            if (len(done_ids) > 0):
                up = torch.zeros([num_envs, 3], device=device)
                up[:, 2] = 1.0
                root_up = torch_util.quat_rotate(root_rot, up)
                upright = torch.logical_and(root_up[..., 2] > args.upright_cos,
                                            root_pos[..., 2] > args.upright_height)

                for env_id in done_ids.tolist():
                    n_steps = float(steps[env_id].item())
                    w_ref_ep = float(wind_ref[env_id].item())
                    d_ref_ep = float(disp_ref[env_id].item())
                    records["winding_ratio"].append(float(wind_sim[env_id].item()) / max(abs(w_ref_ep), 1e-4))
                    records["disp_ratio"].append(float(disp_sim[env_id].item()) / max(abs(d_ref_ep), 1e-4))
                    records["upright"].append(float(upright[env_id].item()))
                    records["height_rmse"].append(float(torch.sqrt(height_sq[env_id] / n_steps).item()))
                    records["done_flag"].append(int(done[env_id].item()))
                    records["ep_len"].append(int(n_steps))

                for buf in [wind_sim, wind_ref, disp_sim, disp_ref, height_sq, steps]:
                    buf[done_ids] = 0.0
                env.reset(done_ids)

    return {k: np.array(v) for k, v in records.items()}

def summarize(rec, args):
    completed = rec["done_flag"] != base_env.DoneFlags.FAIL.value
    winding = rec["winding_ratio"]
    shortcut = np.logical_and(completed, winding < args.shortcut_winding)

    def stats(x):
        return float(np.mean(x)), float(np.std(x)), float(np.median(x))

    print("=" * 64)
    print("episodes: {} (completed: {}, failed: {})".format(
        len(winding), int(np.sum(completed)), int(np.sum(~completed))))
    for key in ["winding_ratio", "disp_ratio", "height_rmse"]:
        m, s, med = stats(rec[key])
        print("{:>16s}: mean {:+.3f}  std {:.3f}  median {:+.3f}".format(key, m, s, med))
    print("{:>16s}: {:.3f}".format("upright_rate", float(np.mean(rec["upright"]))))
    print("{:>16s}: {:.3f}   (completed & winding_ratio < {})".format(
        "shortcut_rate", float(np.mean(shortcut)), args.shortcut_winding))
    print("{:>16s}: {:.1f}".format("mean_ep_len", float(np.mean(rec["ep_len"]))))
    print("=" * 64)
    return

def main():
    args = parse_args()
    env, agent = build(args)
    rec = evaluate(env, agent, args)
    summarize(rec, args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **rec,
             meta_model_file=np.array(args.model_file),
             meta_episodes=np.array(args.episodes),
             meta_shortcut_winding=np.array(args.shortcut_winding))
    print("wrote", args.out)
    return

if __name__ == "__main__":
    main()
