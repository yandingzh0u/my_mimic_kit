"""Gate 1 for PSI: can a captured physical state be restored reliably?

For a frozen checkpoint the script runs a phase-0 prefix, captures the full
simulator state at given reference frames, then compares three continuations
that replay the *same* action sequence:

  A1 : uninterrupted continuation from the live state
  A2 : independent repeat of the prefix, then the same actions (noise floor)
  B  : fresh reset with the captured state written back, same actions

If B deviates much more than A1 vs A2, the restore (including the PhysX
contact cache that cannot be written back) changes the dynamics.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (ROOT / "output/paper_benchmark/"
           "dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012")


def quat_angle(a, b):
    return 2.0 * torch.arccos((a * b).sum(-1).abs().clamp(0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=1700)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frames", type=int, nargs="+", default=[45, 48, 50])
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "mimickit"))
    sys.path.insert(0, str(ROOT))

    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    from tools.paper_eval.evaluate_checkpoint import _reset_with_protocol
    import util.mp_util as mp_util
    import util.util as util

    mp_util.init(0, 1, "cuda:0", None)
    util.set_rand_seed(0)
    env = env_builder.build_env(str(ARCHIVE / "env_config.yaml"),
                                str(ARCHIVE / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(ARCHIVE / "agent_config.yaml"), env, "cuda:0")
    agent.load(str(ARCHIVE / "int_models/model_{:010d}.pt".format(args.checkpoint)))
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)

    engine = env._engine
    char_id = env._get_char_id()
    env_ids = torch.arange(args.num_envs, device="cuda:0")

    def policy_action(obs):
        with torch.no_grad():
            dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
        return agent._a_norm.unnormalize(dist.mean)

    def capture():
        return {
            "time": env._time_buf.clone(),
            "timestep": env._timestep_buf.clone(),
            "done": env._done_buf.clone(),
            "offset": env._motion_time_offsets.clone(),
            "motion_id": env._motion_ids.clone(),
            "root_pos": engine.get_root_pos(char_id).clone(),
            "root_rot": engine.get_root_rot(char_id).clone(),
            "root_vel": engine.get_root_vel(char_id).clone(),
            "root_ang_vel": engine.get_root_ang_vel(char_id).clone(),
            "dof_pos": engine.get_dof_pos(char_id).clone(),
            "dof_vel": engine.get_dof_vel(char_id).clone(),
            "obs": env._obs_buf.clone(),
        }

    def restore(state):
        """Write the state straight into PhysX.

        The engine setters only touch obj.data and then flag a reset, and the
        reset handler calls obj.reset(), which re-applies the spawn state and
        discards the write.  Writing to the sim directly is the only way to
        make the captured state survive the next step.
        """
        env._time_buf[:] = state["time"]
        env._timestep_buf[:] = state["timestep"]
        env._done_buf[:] = state["done"]
        env._motion_time_offsets[:] = state["offset"]
        env._motion_ids[:] = state["motion_id"]
        env._obs_buf[:] = state["obs"]
        obj = engine._objs[char_id]
        rot_wxyz = state["root_rot"][..., [3, 0, 1, 2]]
        # get_root_pos subtracts the per-env offset; put it back for sim writes.
        pos_w = state["root_pos"].clone()
        pos_w[:, :2] += engine._env_offsets[env_ids]
        root_pose = torch.cat([pos_w, rot_wxyz], dim=-1)
        root_vel = torch.cat([state["root_vel"], state["root_ang_vel"]], dim=-1)
        obj.write_root_link_pose_to_sim(root_pose, env_ids)
        obj.write_root_link_velocity_to_sim(root_vel, env_ids)
        order = engine._dof_order_common2sim[char_id]
        obj.write_joint_position_to_sim(state["dof_pos"][..., order], env_ids=env_ids)
        obj.write_joint_velocity_to_sim(state["dof_vel"][..., order], env_ids=env_ids)
        engine._objs_need_reset[env_ids, char_id] = False

    _reset_with_protocol(env, "phase0")
    S0 = capture()

    def run_prefix(stop_frame):
        _reset_with_protocol(env, "phase0")
        restore(S0)
        obs = env._obs_buf
        for _ in range(stop_frame):
            obs, _, _, _ = env.step(policy_action(obs))
        return obs

    def record(actions):
        out_traj = {"root_pos": [], "root_rot": [], "dof_pos": [], "done": []}
        for a in actions:
            _, _, done, _ = env.step(a)
            out_traj["root_pos"].append(engine.get_root_pos(char_id).clone())
            out_traj["root_rot"].append(engine.get_root_rot(char_id).clone())
            out_traj["dof_pos"].append(engine.get_dof_pos(char_id).clone())
            out_traj["done"].append(done.clone())
        return out_traj

    if args.probe:
        _reset_with_protocol(env, "phase0")
        S = capture()
        obs = env._obs_buf
        for _ in range(50):
            obs, _, _, _ = env.step(policy_action(obs))
        T = capture()
        _reset_with_protocol(env, "phase0")
        restore(T)
        before = engine.get_root_pos(char_id)[0, :3].tolist()
        _, _, _, _ = env.step(torch.zeros_like(acts_dummy) if False else policy_action(env._obs_buf))
        after = engine.get_root_pos(char_id)[0, :3].tolist()
        print(json.dumps({"probe": True, "restored": T["root_pos"][0, :3].tolist(),
                          "before_step": before, "after_step": after}), flush=True)
        env._engine._app_launcher.app.close()
        os._exit(0)
    results = {"checkpoint": args.checkpoint, "num_envs": args.num_envs,
               "frames": args.frames, "rollout": args.rollout, "comparisons": []}
    try:
        for frame in args.frames:
            obs = run_prefix(frame)
            state = capture()
            with torch.no_grad():
                actions = [policy_action(obs)]
                for _ in range(args.rollout - 1):
                    obs, _, _, _ = env.step(actions[-1])
                    actions.append(policy_action(obs))
            a1 = record(actions)

            run_prefix(frame)
            a2 = record(actions)

            _reset_with_protocol(env, "phase0")
            restore(state)
            chk = capture()
            readback = {
                "root_pos": float((chk["root_pos"] - state["root_pos"]).abs().max()),
                "root_rot": float((chk["root_rot"] - state["root_rot"]).abs().max()),
                "dof_pos": float((chk["dof_pos"] - state["dof_pos"]).abs().max()),
                "time": float((chk["time"] - state["time"]).abs().max()),
            }
            print(json.dumps({"frame": frame, "restore_readback": readback}), flush=True)
            b = record(actions)

            entry = {"frame": frame, "restore_readback": readback, "steps": {}}
            for k in (0, 7, args.rollout - 1):
                def dev(t):
                    dp = (t["root_pos"][k] - a1["root_pos"][k]).norm(dim=-1).mean()
                    dr = quat_angle(t["root_rot"][k], a1["root_rot"][k]).mean()
                    dj = (t["dof_pos"][k] - a1["dof_pos"][k]).abs().mean(-1).mean()
                    return float(dp), float(dr), float(dj)
                r2 = dev(a2)
                rb = dev(b)
                entry["steps"][str(k + 1)] = {
                    "repeat_root_pos": r2[0], "repeat_root_rot": r2[1],
                    "repeat_dof_pos": r2[2],
                    "restore_root_pos": rb[0], "restore_root_rot": rb[1],
                    "restore_dof_pos": rb[2]}
            entry["terminated_a1"] = int(torch.stack(a1["done"]).ne(0).any(0).sum())
            entry["terminated_restore"] = int(torch.stack(b["done"]).ne(0).any(0).sum())
            results["comparisons"].append(entry)
            print(json.dumps(entry), flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"complete": True}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env._engine._app_launcher.app.close()
        os._exit(0)


if __name__ == "__main__":
    main()
