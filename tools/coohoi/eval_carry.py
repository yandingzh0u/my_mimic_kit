"""Evaluate pickup, transport, and placement in the carry environment."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))

import torch

from envs.base_env import DoneFlags
from learning.base_agent import AgentMode
import run
from util.torch_util import quat_to_matrix


CRITERIA = {
    "lift_clearance_m": 0.10,
    "lift_time_s": 0.20,
    "airborne_displacement_m": 0.30,
    "final_displacement_m": 1.00,
    "goal_distance_m": 0.15,
    "floor_tolerance_m": 0.04,
    "face_tilt_deg": 15.0,
    "end_speed_m_s": 0.20,
    "end_spin_rad_s": 1.00,
    "hand_contact_n": 5.0,
    "placement_time_s": 0.20,
}


def _box_contact_views(env):
    """Read hand-to-box forces without adding sensors to training."""
    from engines.isaac_lab_engine import ENV_PATH_TEMPLATE, OBJ_PATH_TEMPLATE
    from isaacsim.core.simulation_manager import SimulationManager

    engine = env._engine
    sensor = engine._ground_contact_sensors[env._get_char_id()]
    if sensor is None:
        raise RuntimeError("Carry evaluation requires PhysX contact reporting on the humanoid")
    hand_parent = sensor.cfg.prim_path.rsplit("/", 1)[0].replace(".*", "*")
    box_pattern = OBJ_PATH_TEMPLATE.format("*", env._box_id)
    env_paths = {ENV_PATH_TEMPLATE.format(i): i for i in range(env.get_num_envs())}
    sim_view = SimulationManager.get_physics_sim_view()
    views = []
    for name in ("left_hand", "right_hand"):
        view = sim_view.create_rigid_contact_view(f"{hand_parent}/{name}", [box_pattern])
        # PhysX view order is not guaranteed to match numeric environment order.
        indices = [env_paths[path.split("/obj_", 1)[0]] for path in view.sensor_paths]
        if sorted(indices) != list(range(env.get_num_envs())) or view.filter_count != 1:
            raise RuntimeError(f"Expected one {name} and one filtered box in every environment")
        order = torch.as_tensor(indices, device=env._device).argsort()
        views.append((view, order))
    return views


@torch.no_grad()
def evaluate(agent, start_frame=0):
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env = agent._env
    env.set_test_random_start(False)
    reference = env._reference
    if not 0 <= start_frame < len(reference.box) - 1:
        raise ValueError("Start frame must precede the last reference frame")
    if start_frame == 0:
        obs, info = env.reset()
    else:
        sample_original = env._sample_motion_times

        def sample_frame(count):
            ids = reference.sample_motions(count)
            return ids, reference.time_at(torch.full_like(ids, start_frame))

        env._sample_motion_times = sample_frame
        try:
            obs, info = env.reset()
        finally:
            env._sample_motion_times = sample_original
    dt = env._engine.get_timestep()
    n = env.get_num_envs()
    max_steps = len(reference.box) - start_frame + 1
    min_lift_steps = math.ceil(CRITERIA["lift_time_s"] / dt - 1e-6)
    min_placement_steps = math.ceil(CRITERIA["placement_time_s"] / dt - 1e-6)
    box_half_size = env._box_size / 2
    initial_box = reference.box[start_frame]
    box_start_xy = reference.box[0, :2]
    initial_vertical = quat_to_matrix(initial_box[3:7])[2].abs()
    initial_bottom = initial_box[2] - (initial_vertical * box_half_size).sum()
    starts_on_floor = initial_bottom.abs() <= CRITERIA["floor_tolerance_m"]
    goal = env._box_goal
    contact_views = _box_contact_views(env)
    physics_dt = env._engine.get_sim_timestep()

    active = torch.ones(n, dtype=torch.bool, device=obs.device)
    completed = torch.zeros_like(active)
    hand_contact = torch.zeros_like(active)
    final_steps = torch.zeros(n, dtype=torch.long, device=obs.device)
    lift_steps = torch.zeros_like(final_steps)
    peak_lift_steps = torch.zeros_like(final_steps)
    supported_steps = torch.zeros_like(final_steps)
    peak_supported_steps = torch.zeros_like(final_steps)
    settled_steps = torch.zeros_like(final_steps)
    final_settled_steps = torch.zeros_like(final_steps)
    support_start_xy = torch.zeros((n, 2), device=obs.device, dtype=obs.dtype)
    peak_bottom = torch.zeros(n, device=obs.device, dtype=obs.dtype)
    current_airborne_displacement = torch.zeros_like(peak_bottom)
    peak_airborne_displacement = torch.zeros_like(peak_bottom)
    terminal = {name: torch.zeros_like(peak_bottom) for name in (
        "final_floor_clearance_m", "final_displacement_m", "final_goal_distance_m",
        "final_face_tilt_deg", "final_speed_m_s", "final_spin_rad_s",
        "final_root_height_m", "final_max_body_position_error_m",
        "final_non_support_ground_contact_n",
    )}
    char_id = env._get_char_id()

    for step in range(1, max_steps + 1):
        action, _ = agent._decide_action(obs, info)
        obs, _, done, info = env.step(action)
        position = env._engine.get_root_pos(env._box_id)
        rotation = env._engine.get_root_rot(env._box_id)
        box_rotation = quat_to_matrix(rotation)
        vertical_projections = box_rotation[..., 2, :].abs()
        # The projection of each box half-axis gives the lowest point in world Z.
        bottom = position[:, 2] - (vertical_projections * box_half_size).sum(dim=-1)
        lifted = active & (bottom >= CRITERIA["lift_clearance_m"])
        hand_force = torch.stack([
            view.get_contact_force_matrix(dt=physics_dt)[order, 0].norm(dim=-1)
            for view, order in contact_views
        ], dim=-1)
        touching = hand_force >= CRITERIA["hand_contact_n"]
        hand_contact |= active & touching.any(-1)
        supported = lifted & touching.all(-1)
        new_support = supported & (supported_steps == 0)
        support_start_xy[new_support] = position[new_support, :2]
        lift_steps = torch.where(lifted, lift_steps + 1, 0)
        supported_steps = torch.where(supported, supported_steps + 1, 0)
        peak_lift_steps = torch.maximum(peak_lift_steps, lift_steps)
        peak_supported_steps = torch.maximum(peak_supported_steps, supported_steps)
        airborne_displacement = (position[:, :2] - support_start_xy).norm(dim=-1)
        current_airborne_displacement = torch.where(
            supported, torch.maximum(current_airborne_displacement, airborne_displacement), 0,
        )
        peak_airborne_displacement = torch.where(
            supported_steps >= min_lift_steps,
            torch.maximum(peak_airborne_displacement, current_airborne_displacement),
            peak_airborne_displacement,
        )
        peak_bottom = torch.where(active, torch.maximum(peak_bottom, bottom), peak_bottom)

        displacement = (position[:, :2] - box_start_xy).norm(dim=-1)
        goal_distance = (position - goal).norm(dim=-1)
        # For this cube, any face can rest flat on the floor.
        tilt = torch.rad2deg(vertical_projections.amax(dim=-1).clamp(0, 1).acos())
        speed = env._engine.get_root_vel(env._box_id).norm(dim=-1)
        spin = env._engine.get_root_ang_vel(env._box_id).norm(dim=-1)
        settled = (active & ~touching.any(-1) & (displacement >= CRITERIA["final_displacement_m"])
                   & (goal_distance <= CRITERIA["goal_distance_m"])
                   & (bottom.abs() <= CRITERIA["floor_tolerance_m"])
                   & (tilt <= CRITERIA["face_tilt_deg"])
                   & (speed <= CRITERIA["end_speed_m_s"]) & (spin <= CRITERIA["end_spin_rad_s"]))
        settled_steps = torch.where(settled, settled_steps + 1, 0)

        ended = done != DoneFlags.NULL.value
        first_end = active & ended
        at_end = env.get_motion_phase() >= 1 - 1e-6
        completed |= first_end & at_end & (done != DoneFlags.FAIL.value)
        final_steps[first_end] = step
        final_settled_steps[first_end] = settled_steps[first_end]
        terminal["final_floor_clearance_m"][first_end] = bottom[first_end]
        terminal["final_displacement_m"][first_end] = displacement[first_end]
        terminal["final_goal_distance_m"][first_end] = goal_distance[first_end]
        terminal["final_face_tilt_deg"][first_end] = tilt[first_end]
        terminal["final_speed_m_s"][first_end] = speed[first_end]
        terminal["final_spin_rad_s"][first_end] = spin[first_end]
        if first_end.any():
            body_pos = env._engine.get_body_pos(char_id)[first_end]
            terminal["final_root_height_m"][first_end] = body_pos[:, 0, 2]
            body_error = body_pos - env._ref_body_pos[first_end]
            terminal["final_max_body_position_error_m"][first_end] = body_error.norm(dim=-1).amax(dim=-1)
            ground_forces = env._engine.get_ground_contact_forces(char_id)[first_end]
            ground_forces[:, env._contact_body_ids] = 0
            terminal["final_non_support_ground_contact_n"][first_end] = ground_forces.abs().amax(dim=(1, 2))
        active &= ~ended
        if not active.any():
            break
        if not env.is_running():
            raise RuntimeError("Simulator closed before carry evaluation finished")
        env_ids = ended.nonzero().flatten()
        if env_ids.numel():
            obs, info = env.reset(env_ids)

    if active.any():
        raise RuntimeError("Some carry episodes did not terminate within the evaluation horizon")

    pickup = starts_on_floor & (peak_lift_steps >= min_lift_steps)
    transport = pickup & (peak_airborne_displacement >= CRITERIA["airborne_displacement_m"])
    placement = final_settled_steps >= min_placement_steps
    success = completed & transport & placement
    per_episode = dict(
        episode_length_steps=final_steps, completed=completed, hand_contact=hand_contact, pickup=pickup,
        transport=transport, placement=placement, success=success,
        peak_box_bottom_m=peak_bottom, longest_lift_s=peak_lift_steps * dt,
        longest_bilateral_support_s=peak_supported_steps * dt, final_settled_s=final_settled_steps * dt,
        airborne_displacement_m=peak_airborne_displacement, **terminal,
    )
    return dict(
        n=n, fully_completed=int(completed.sum()), hand_contact=int(hand_contact.sum()), pickup=int(pickup.sum()),
        transport=int(transport.sum()), placement=int(placement.sum()), success=int(success.sum()),
        success_rate=success.float().mean().item(),
        mean_episode_length_steps=final_steps.float().mean().item(),
        mean_peak_box_bottom_m=peak_bottom.mean().item(),
        mean_final_box_to_goal_m=terminal["final_goal_distance_m"].mean().item(),
        policy="deterministic; first episode per environment", start_frame=start_frame,
        final_displacement_origin_frame=0,
        goal_xyz_m=goal.tolist(), criteria=CRITERIA, contact_measurement="PhysX hand-to-box forces",
        per_episode={name: value.tolist() for name, value in per_episode.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    model = args.model or args.run_dir / "model.pt"
    suffix = f"_start{args.start_frame:04d}" if args.start_frame else ""
    output = args.out or args.run_dir / f"carry_eval{suffix}.json"

    def test(agent, test_episodes):
        result = evaluate(agent, start_frame=args.start_frame)
        result.update(model=str(model), seed=args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: val for key, val in result.items() if key != "per_episode"}), flush=True)

    run.test = test
    run.main([
        "eval_carry", "--mode", "test", "--num_envs", str(args.num_envs),
        "--test_episodes", str(args.num_envs), "--devices", args.device,
        "--visualize", str(args.visualize).lower(), "--rand_seed", str(args.seed),
        "--model_file", str(model), "--out_dir", str(args.run_dir),
        "--agent_config", str(args.run_dir / "agent_config.yaml"),
        "--engine_config", str(args.run_dir / "engine_config.yaml"),
        "--env_config", str(args.run_dir / "env_config.yaml"),
    ])


if __name__ == "__main__":
    main()
