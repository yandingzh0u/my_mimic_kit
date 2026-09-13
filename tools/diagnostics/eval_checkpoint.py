"""Evaluate one checkpoint and report tracking diagnostics as JSON."""

import argparse
import json

import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util


def _to_scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    util.set_rand_seed(args.seed)
    mp_util.init(0, 1, args.device, None)
    env = env_builder.build_env(
        args.env_config, args.engine_config, args.num_envs, args.device,
        visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    result = agent.test_model(args.test_episodes)
    diagnostics = env.record_diagnostics()
    output = {key: _to_scalar(value) for key, value in result.items()}
    for key, value in diagnostics.items():
        if key.endswith("_err"):
            output[key] = _to_scalar(value)
    print("EVAL_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
