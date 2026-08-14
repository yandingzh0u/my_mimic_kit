#!/usr/bin/env python3
"""Render a deterministic Roll cycle from an existing checkpoint.

The script is read-only with respect to training artifacts.  It creates a
temporary environment YAML with a phase-zero reset, builds one Isaac Lab
environment with video recording enabled, runs the deterministic policy for
one cycle, and writes an MP4 plus selected PNG frames.

Examples (from the repository root):

  conda run -n env_isaaclab python paper/icra2027_cpmd/scripts/capture_roll_frames.py \
    --env_config data/envs/liesig_st_add_sym_humanoid_roll_eval_env.yaml \
    --engine_config output/liesig_placebo_roll_cycle_1k_seed0/engine_config.yaml \
    --agent_config output/liesig_placebo_roll_cycle_1k_seed0/agent_config.yaml \
    --model_file output/liesig_placebo_roll_cycle_1k_seed0/model.pt \
    --name cpmd

  conda run -n env_isaaclab python paper/icra2027_cpmd/scripts/capture_roll_frames.py \
    --env_config /home/y/.local/share/Trash/files/add_roll_contact_et_1k_seed0/env_config.yaml \
    --engine_config /home/y/.local/share/Trash/files/add_roll_contact_et_1k_seed0/engine_config.yaml \
    --agent_config /home/y/.local/share/Trash/files/add_roll_contact_et_1k_seed0/agent_config.yaml \
    --model_file /home/y/.local/share/Trash/files/add_roll_contact_et_1k_seed0/model.pt \
    --name instantaneous
"""

import argparse
import os
import sys
import tempfile

import imageio.v3 as iio
import numpy as np
import torch
import yaml


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))

import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_config", required=True)
    p.add_argument("--engine_config", required=True)
    p.add_argument("--agent_config", required=True)
    p.add_argument("--model_file", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="paper/icra2027_cpmd/figures/roll_frames")
    return p.parse_args()


def phase_zero_env_file(src):
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["rand_reset"] = False
    cfg["episode_length"] = 2.0
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, handle, sort_keys=False)
    handle.close()
    return handle.name


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mp_util.init(0, 1, args.device, 6000)
    util.set_rand_seed(args.seed)
    temp_env = phase_zero_env_file(args.env_config)
    try:
        env = env_builder.build_env(temp_env, args.engine_config, 1, args.device,
                                    visualize=False, record_video=True)
        agent = agent_builder.build_agent(args.agent_config, env, args.device)
        agent.load(args.model_file)
        agent.eval()
        agent.set_mode(base_agent.AgentMode.TEST)
        obs, info = env.reset()

        with torch.no_grad():
            for _ in range(args.steps):
                action, _ = agent._decide_action(obs, info)
                obs, _, done, info = env.step(action)
                if int(done[0].item()) != base_env.DoneFlags.NULL.value:
                    break

        video = env._engine.get_video_recording()
        frames = list(video.get_frames())
        if not frames:
            raise RuntimeError("Isaac Lab returned no recording frames")

        mp4 = os.path.join(args.out_dir, f"{args.name}_roll.mp4")
        video.save(mp4)

        # Six phases spanning the complete cycle.  Save raw frames and a
        # horizontal strip; the latter is used only after visual inspection.
        ids = np.linspace(0, len(frames) - 1, 6).round().astype(int)
        selected = []
        for rank, frame_id in enumerate(ids):
            frame = np.asarray(frames[frame_id])
            selected.append(frame)
            iio.imwrite(os.path.join(args.out_dir,
                                     f"{args.name}_{rank:02d}_step{frame_id:03d}.png"), frame)
        strip = np.concatenate(selected, axis=1)
        iio.imwrite(os.path.join(args.out_dir, f"{args.name}_strip.png"), strip)
        print(f"wrote {mp4}, {len(frames)} frames, selected {ids.tolist()}")
    finally:
        if os.path.exists(temp_env):
            os.unlink(temp_env)


if __name__ == "__main__":
    main()

