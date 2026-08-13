"""Collects per-episode error-path statistics for a trained FlowADD model.

Rolls out a trained policy in test mode and, for every episode, accumulates
the discrete signed-area matrix of the normalized differential trajectory

    Omega(Gamma) = 0.5 * sum_t (x_t-1 x_t^T - x_t x_t-1^T)

Omega is a model-independent description of the oriented path the tracking
error takes through objective space. Comparing group means of Omega between
completed and failed (or shortcut) episodes tests whether the error paths
differ geometrically at all, independent of any discriminator.

Also records per-episode tangent-error statistics (mean ||v_t|| with
v_t = x_t - x_t-1) and, when the agent uses the fixed potential reward, the
episode's mean energy E(x) and summed raw progress sum(E_prev - E).

Episodes are grouped by their done flag: FAIL = early termination,
TIME/SUCC = ran to completion. Per-episode stats are saved to an npz for
custom grouping (e.g. shortcut detection on completed episodes).

Example:
    python tools/flow_add/collect_omega.py \
        --env_config output/flowadd_run/env_config.yaml \
        --engine_config output/flowadd_run/engine_config.yaml \
        --agent_config output/flowadd_run/agent_config.yaml \
        --model_file output/flowadd_run/model.pt \
        --num_envs 64 --episodes 256 --out output/flowadd_run/omega_stats.npz
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
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as util

def parse_args():
    parser = argparse.ArgumentParser(description="Collect per-episode signed-area (Omega) stats for FlowADD")
    parser.add_argument("--env_config", type=str, required=True)
    parser.add_argument("--engine_config", type=str, required=True)
    parser.add_argument("--agent_config", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rand_seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="output/omega_stats.npz")
    parser.add_argument("--save_omegas", action="store_true",
                        help="also store the full per-episode Omega matrices in the npz")
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

def collect(env, agent, num_episodes, save_omegas):
    device = agent._device
    model = agent._model
    normalizer = agent._disc_obs_norm

    num_envs = env.get_num_envs()
    obs, info = env.reset()
    assert ("disc_obs_prev" in info), "env must be a FlowADD env exposing disc_obs_prev"

    diff_dim = info["disc_obs"].shape[-1]
    omega = torch.zeros([num_envs, diff_dim, diff_dim], device=device)
    flow_norm_sum = torch.zeros([num_envs], device=device)
    energy_sum = torch.zeros([num_envs], device=device)
    prog_sum = torch.zeros([num_envs], device=device)
    ep_len = torch.zeros([num_envs], device=device, dtype=torch.long)

    has_potential = agent._use_external_potential_reward

    records = []

    with torch.no_grad():
        while (len(records) < num_episodes):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)

            # info holds references to env buffers, so compute everything for
            # this step before resetting done envs
            diff = info["disc_obs_demo"] - info["disc_obs"]
            diff_prev = info["disc_obs_demo_prev"] - info["disc_obs_prev"]
            x = normalizer.normalize(diff)
            x_prev = normalizer.normalize(diff_prev)

            omega += 0.5 * (x_prev.unsqueeze(-1) * x.unsqueeze(-2)
                            - x.unsqueeze(-1) * x_prev.unsqueeze(-2))
            flow_norm_sum += torch.norm(x - x_prev, dim=-1)
            if (has_potential):
                clip = agent._disc_flow_potential_obs_clip
                p_obs = agent._disc_potential_norm.normalize(diff)
                p_obs_prev = agent._disc_potential_norm.normalize(diff_prev)
                energy = model.eval_potential_energy(p_obs, clip=clip)
                energy_prev = model.eval_potential_energy(p_obs_prev, clip=clip)
                energy_sum += energy
                prog_sum += energy_prev - energy
            ep_len += 1

            done_ids = torch.flatten((done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False))
            for env_id in done_ids.tolist():
                ep_omega = omega[env_id]
                steps = int(ep_len[env_id].item())
                rec = {
                    "done_flag": int(done[env_id].item()),
                    "ep_len": steps,
                    "motion_id": int(env._motion_ids[env_id].item()),
                    "omega_fro": float(torch.norm(ep_omega).item()),
                    "flow_norm_mean": float(flow_norm_sum[env_id].item()) / steps,
                    "energy_mean": float(energy_sum[env_id].item()) / steps,
                    "prog_sum": float(prog_sum[env_id].item()),
                }
                if (save_omegas):
                    rec["omega"] = ep_omega.cpu().numpy()
                records.append(rec)

                omega[env_id] = 0.0
                flow_norm_sum[env_id] = 0.0
                energy_sum[env_id] = 0.0
                prog_sum[env_id] = 0.0
                ep_len[env_id] = 0

            if (len(done_ids) > 0):
                obs, info = env.reset(done_ids)

    return records

def summarize(records):
    done_flags = np.array([r["done_flag"] for r in records])
    ep_lens = np.array([r["ep_len"] for r in records], dtype=np.float64)
    omega_fros = np.array([r["omega_fro"] for r in records])
    flow_norms = np.array([r["flow_norm_mean"] for r in records])
    energy_means = np.array([r["energy_mean"] for r in records])
    prog_sums = np.array([r["prog_sum"] for r in records])
    has_omegas = "omega" in records[0]

    fail_mask = done_flags == base_env.DoneFlags.FAIL.value
    comp_mask = ~fail_mask

    def group_stats(name, mask):
        n = int(np.sum(mask))
        print("  {:s}: {:d} episodes".format(name, n))
        if (n == 0):
            return
        print("    ep_len:        {:.1f} +/- {:.1f}".format(np.mean(ep_lens[mask]), np.std(ep_lens[mask])))
        print("    ||Omega||_F:   {:.4f} +/- {:.4f}".format(np.mean(omega_fros[mask]), np.std(omega_fros[mask])))
        print("    mean ||v||:    {:.4f} +/- {:.4f}".format(np.mean(flow_norms[mask]), np.std(flow_norms[mask])))
        print("    mean E(x):     {:.4f} +/- {:.4f}".format(np.mean(energy_means[mask]), np.std(energy_means[mask])))
        print("    sum progress:  {:.4f} +/- {:.4f}".format(np.mean(prog_sums[mask]), np.std(prog_sums[mask])))
        return

    print("Episode groups (FAIL = early termination, completed = TIME/SUCC):")
    group_stats("completed", comp_mask)
    group_stats("failed", fail_mask)

    if (has_omegas and np.sum(fail_mask) > 0 and np.sum(comp_mask) > 0):
        omegas = np.stack([r["omega"] for r in records], axis=0)
        omega_gap = np.mean(omegas[comp_mask], axis=0) - np.mean(omegas[fail_mask], axis=0)
        print("Group gap (needs --save_omegas):")
        print("  ||mean Omega_completed - mean Omega_failed||_F: {:.4f}".format(
            float(np.linalg.norm(omega_gap))))
    return

def save(records, out_path):
    out_dir = os.path.dirname(out_path)
    if (out_dir != "" and not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    data = {
        "done_flags": np.array([r["done_flag"] for r in records]),
        "ep_lens": np.array([r["ep_len"] for r in records]),
        "motion_ids": np.array([r["motion_id"] for r in records]),
        "omega_fros": np.array([r["omega_fro"] for r in records]),
        "flow_norm_means": np.array([r["flow_norm_mean"] for r in records]),
        "energy_means": np.array([r["energy_mean"] for r in records]),
        "prog_sums": np.array([r["prog_sum"] for r in records]),
    }
    if ("omega" in records[0]):
        data["omegas"] = np.stack([r["omega"] for r in records], axis=0)

    np.savez_compressed(out_path, **data)
    print("Saved {:d} episode records to {:s}".format(len(records), out_path))
    return

def main():
    args = parse_args()
    env, agent = build(args)

    records = collect(env, agent, args.episodes, args.save_omegas)
    summarize(records)
    save(records, args.out)
    return

if __name__ == "__main__":
    main()
