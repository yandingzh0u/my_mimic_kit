"""Evaluate actual foot–ball contact for a saved MIRA soccer policy."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))

from envs.base_env import DoneFlags
from learning.base_agent import AgentMode
import run


def contact_views(env):
    from engines.isaac_lab_engine import ENV_PATH_TEMPLATE, OBJ_PATH_TEMPLATE
    from isaacsim.core.simulation_manager import SimulationManager

    eng = env._engine
    sensor = eng._ground_contact_sensors[env._get_char_id()]
    parent = sensor.cfg.prim_path.rsplit("/", 1)[0].replace(".*", "*")
    sim = SimulationManager.get_physics_sim_view()
    mapping = {ENV_PATH_TEMPLATE.format(i): i for i in range(env.get_num_envs())}
    for name in ("right_foot", "left_foot"):
        view = sim.create_rigid_contact_view(
            f"{parent}/{name}", [OBJ_PATH_TEMPLATE.format("*", env._ball_id)])
        indices = [mapping[p.split("/obj_", 1)[0]] for p in view.sensor_paths]
        if sorted(indices) != list(range(len(mapping))) or view.filter_count != 1:
            raise RuntimeError(f"Expected one {name} and one filtered ball per environment")
        yield view, torch.as_tensor(indices, device=env._device).argsort()


@torch.no_grad()
def evaluate(agent):
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env = agent._env
    env.set_test_random_start(False)
    eng = env._engine
    views = list(contact_views(env))
    dt = eng.get_sim_timestep()
    control_dt = eng.get_timestep()
    obs, info = env.reset()
    initial_xy = eng.get_root_pos(env._ball_id)[:, :2].clone()
    impulse = torch.zeros((env.get_num_envs(), 2), device=env._device)
    original = eng._post_sim_step

    def sample_contact():
        original()
        for i, (view, order) in enumerate(views):
            force = view.get_contact_force_matrix(dt=dt)[order, 0]
            impulse[:, i].add_(force.norm(dim=-1) * dt)

    frames = []
    remaining = torch.ones(env.get_num_envs(), dtype=torch.bool, device=env._device)
    eng._post_sim_step = sample_contact
    try:
        for _ in range(math.ceil(env._episode_length / control_dt) + 1):
            impulse.zero_()
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            ball_pos = eng.get_root_pos(env._ball_id)[:, :2]
            speed = eng.get_root_vel(env._ball_id)[:, :2].norm(dim=-1, keepdim=True)
            travel = (ball_pos - initial_xy).norm(dim=-1, keepdim=True)
            frames.append(torch.cat((impulse, speed, travel, done[:, None]), dim=1))
            remaining &= done == DoneFlags.NULL.value
            if not remaining.any():
                break
    finally:
        eng._post_sim_step = original
    if remaining.any():
        raise RuntimeError("Some soccer episodes did not terminate within the reference horizon")

    criteria = dict(impulse_ns_min_exclusive=0.05, ball_xy_speed_mps_min=1.0,
                    ball_xy_displacement_m_min=0.30)
    rows = []
    for trace in torch.stack(frames).transpose(0, 1).cpu().numpy():
        end = int(np.flatnonzero(trace[:, -1])[0])
        trace = trace[:end + 1]
        hits = np.argwhere(trace[:, :2] > criteria["impulse_ns_min_exclusive"])
        hit, foot = map(int, hits[0]) if hits.size else (None, None)
        speed = float(trace[hit:hit + 5, 2].max()) if hit is not None else 0.0
        travel = float(trace[:, 3].max())
        complete = int(trace[-1, -1]) == DoneFlags.SUCC.value
        kick = (foot == 0 and speed >= criteria["ball_xy_speed_mps_min"]
                and travel >= criteria["ball_xy_displacement_m_min"])
        rows.append(dict(complete=complete, steps=end + 1, right_foot_first=foot == 0,
                         first_contact_step=None if hit is None else hit + 1,
                         post_contact_xy_speed_mps=speed, ball_max_xy_displacement_m=travel,
                         peak_right_impulse_ns=float(trace[:, 0].max()), kick=kick))
    return dict(n=len(rows), completed=sum(r["complete"] for r in rows),
                right_foot_first=sum(r["right_foot_first"] for r in rows),
                kick_and_complete=sum(r["kick"] and r["complete"] for r in rows),
                criteria=criteria, per_episode=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    model = args.model or args.run_dir / "model.pt"
    output = args.out or args.run_dir / "soccer_eval.json"

    def test(agent, test_episodes):
        result = evaluate(agent)
        result.update(model=str(model), seed=args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "per_episode"}), flush=True)

    run.test = test
    run.main([
        "soccer_eval", "--mode", "test", "--num_envs", str(args.num_envs),
        "--test_episodes", str(args.num_envs), "--devices", args.device,
        "--visualize", "false", "--rand_seed", str(args.seed), "--model_file", str(model),
        "--out_dir", str(args.run_dir), "--agent_config", str(args.run_dir / "agent_config.yaml"),
        "--engine_config", str(args.run_dir / "engine_config.yaml"),
        "--env_config", str(args.run_dir / "env_config.yaml"),
    ])


if __name__ == "__main__":
    main()
