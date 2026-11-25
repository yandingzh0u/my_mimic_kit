"""Render actual frozen MIRA policies in Isaac Lab for the paper figure.

The saved PNG files are unretouched RGB simulator camera captures. This script
changes only presentation (lights, material and camera), never the policy or
physics. It performs one deterministic rollout and records its provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
INTERACTION_VIEWS = {
    # Sofa uses a fixed view so the approach and final seat are in one scene.
    'samp_sofa003': {'eye': (-2.09, -1.92, 1.94), 'target': (.42, .53, .74)},
    'continuous_box_transport': {'eye': (1.63, -4.15, 2.17), 'target': (1.63, .06, .72),
                                 'prop_color': (.55, .32, .13)},
    'cmu_10_02_physics': {'eye': (1.62, -5.05, 2.40), 'target': (1.62, -.61, .73),
                         'prop_color': (.84, .55, .12)},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--steps', type=int, nargs='+', help='Capture specific control steps')
    p.add_argument('--current-run', action='store_true',
                   help='Use model.pt and the run config with current asset paths')
    p.add_argument('--resolution', type=int, default=800)
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = run / ('model.pt' if args.current_run else f'int_models/model_{args.updates - 1:010d}.pt')
    assert model.is_file(), model
    manifest_path = run / 'source_manifest.json'
    if args.current_run:
        source = ROOT
        env_config_path, engine_config_path, agent_config_path = (
            run / name for name in ('env_config.yaml', 'engine_config.yaml', 'agent_config.yaml'))
        last_row = json.loads((run / 'train_metrics.jsonl').read_text().splitlines()[-1])
        assert last_row['Iteration'] + 1 == args.updates, last_row['Iteration']
    else:
        source = run / 'source_snapshot'
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            assert sha(source / name) == expected, name
        env_config_path, engine_config_path, agent_config_path = (
            run / f'frozen_{name}.yaml' for name in ('env', 'engine', 'agent'))
    motion = yaml.safe_load(env_config_path.read_text())['motion_name']
    sys.path.insert(0, str(source / 'mimickit'))
    os.chdir(ROOT)
    import torch
    import envs.env_builder as env_builder
    import envs.char_env as char_env
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    # A consistent, readable body material. This changes the renderer only.
    char_env.CharEnv._get_char_color = lambda self: np.array([0.02, 0.48, 0.52])
    util.set_rand_seed(0)
    torch.set_num_threads(4)
    mp_util.init(rank=0, num_procs=1, device='cuda:0', master_port=6394)
    env = env_builder.build_env(str(env_config_path),
                               str(engine_config_path), 1, 'cuda:0',
                               visualize=False, record_video=True)
    agent = agent_builder.build_agent(str(agent_config_path), env, 'cuda:0')
    agent.load(str(model))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env.set_test_random_start(False)
    env._rand_reset = False
    engine = env._engine
    sim = engine.get_sim()

    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    stage = omni.usd.get_context().get_stage()
    def bind_visual_material(root, name, color):
        path = f'/World/{name}'
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.82)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # The default visual bindings leave the physics materials untouched.
    bind_visual_material('/World/ground', 'PaperFloorMaterial', (.72, .75, .79))
    camera = INTERACTION_VIEWS.get(motion)
    prop_id = getattr(env, '_box_id', getattr(env, '_ball_id', None)) if camera else None
    if prop_id is not None and 'prop_color' in camera:
        from engines.isaac_lab_engine import OBJ_PATH_TEMPLATE
        bind_visual_material(OBJ_PATH_TEMPLATE.format(0, prop_id), 'PaperPropMaterial',
                             camera['prop_color'])
    flag_xy = None
    if motion == 'continuous_box_transport':
        # Place the marker just behind the reference drop point. The flag
        # consists only of render primitives and has no physics API.
        goal = env._box_goal.detach().cpu().numpy()
        x, y = float(goal[0]) + .14, float(goal[1]) + .46
        flag_xy = [x, y]
        pole = UsdGeom.Cylinder.Define(stage, '/World/PaperGoalPole')
        pole.CreateHeightAttr(.62)
        pole.CreateRadiusAttr(.012)
        pole.AddTranslateOp().Set(Gf.Vec3d(x, y, .31))
        flag = UsdGeom.Mesh.Define(stage, '/World/PaperGoalFlag')
        flag.CreatePointsAttr([(x, y, .62), (x + .28, y, .55), (x, y, .47)])
        flag.CreateFaceVertexCountsAttr([3])
        flag.CreateFaceVertexIndicesAttr([0, 1, 2])
        flag.CreateSubdivisionSchemeAttr('none')
        flag.CreateDoubleSidedAttr(True)
        bind_visual_material('/World/PaperGoalPole', 'PaperFlagPoleColor', (.17, .18, .20))
        bind_visual_material('/World/PaperGoalFlag', 'PaperFlagColor', (.75, .018, .022))

    resolution = (args.resolution, args.resolution)
    product = rep.create.render_product('/OmniverseKit_Persp', resolution)
    annot = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
    annot.attach([product])
    obs, info = env.reset()
    dt = float(engine.get_timestep())
    horizon = int(math.ceil(float(env._episode_length) / dt))
    # The engine commits a reset's articulated transforms on its first step.
    # Start at step 1, so the first image is the physical initialized pose,
    # rather than the USD asset's uncommitted default T pose.
    steps = args.steps or np.rint(np.linspace(1, horizon, args.frames)).astype(int).tolist()
    if args.steps and (steps != sorted(set(steps)) or not 1 <= steps[0] <= steps[-1] <= horizon):
        p.error(f'--steps must increase strictly and lie in [1, {horizon}]')
    if args.preview:
        steps = [1, horizon // 2]

    # Fixed distances per motion preserve jump height and body proportions.
    # All views use an elevated oblique camera, rather than a flat silhouette.
    directions = {
        'backflip': np.array([.55, -1.0, .27]),
        'climb': np.array([.65, -1.0, .34]),
        'crawl': np.array([.65, -1.0, .45]),
        'roll': np.array([.65, -1.0, .4]),
        'getup_facedown': np.array([.65, -1.0, .35]),
        'spinkick': np.array([.65, -1.0, .3]),
    }
    distances = {'backflip': 3.15, 'climb': 3.8, 'crawl': 2.5,
                 'roll': 2.5, 'getup_facedown': 2.7, 'spinkick': 2.9}
    frames, times = [], []
    poses = []
    # No redundant automatic video recording; selected captures below are RGB.
    engine._recording = False

    def capture(step):
        char = env._get_char_id()
        body = engine.get_body_pos(char)[0].detach().cpu().numpy()
        prop = engine.get_root_pos(prop_id)[0].detach().cpu().numpy() if prop_id is not None else None
        if camera:
            eye, target = np.array(camera['eye']), np.array(camera['target'])
        else:
            target = .5 * (body.min(axis=0) + body.max(axis=0))
            if motion == 'climb':
                target[2] = max(.9, target[2])
            direction = directions[motion]
            eye = target + direction / np.linalg.norm(direction) * distances[motion]
        engine.set_camera_pose(eye, target)
        # Camera/material/render-product initialization can take several frames.
        for _ in range(5 if not frames else 2):
            sim.render()
        raw = annot.get_data()
        if raw is None or raw.size == 0:
            raise RuntimeError('Isaac Lab RGB annotator produced no image')
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(*raw.shape)[:, :, :3].copy()
        import imageio.v2 as iio
        name = f'frame_{len(frames):02d}_step_{step:03d}.png'
        iio.imwrite(out / name, rgb)
        frames.append({'file': name, 'step': step, 'time_s': step * dt,
                       'camera_eye': eye.tolist(),
                       'camera_target': target.tolist(), 'sha256': sha(out / name)})
        if prop is not None:
            frames[-1]['object_position_m'] = prop.tolist()
        print(f'[capture] {motion} step={step} time={step * dt:.3f} -> {name}', flush=True)

    done_value = 0
    with torch.no_grad():
        if 0 in steps:
            capture(0)
        for step in range(1, horizon + 1):
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            poses.append(engine.get_body_pos(env._get_char_id())[0].detach().cpu().numpy())
            times.append(step * dt)
            if step in steps:
                capture(step)
            done_value = int(done[0].item())
            if done_value:
                if step not in steps and not args.steps:
                    capture(step)
                break
            if args.preview and step >= max(steps):
                break
    if args.steps and [frame['step'] for frame in frames] != steps:
        raise RuntimeError(f'Rollout ended on step {len(times)} before all requested captures')
    if done_value == 1:
        raise RuntimeError(f'Policy failed on step {len(times)}')
    metadata = {'motion': motion, 'model': str(model.relative_to(ROOT)),
                'model_sha256': sha(model), 'training_updates': args.updates,
                'source_manifest_sha256': sha(manifest_path) if manifest_path.is_file() else None,
                'env_config': str(env_config_path.relative_to(ROOT)),
                'env_config_sha256': sha(env_config_path),
                'renderer': 'Isaac Lab / Isaac Sim RGB camera',
                'actual_policy_rollout': True, 'deterministic_mean_action': True,
                'training_seed': 0, 'rollout_seed': 0, 'num_envs': 1,
                'reference_start': True, 'automatic_reset': False,
                'visual_changes': 'teal body, neutral floor, camera pose, and prop color where specified',
                'resolution': list(resolution), 'dt': dt, 'requested_steps': horizon,
                'executed_steps': len(times), 'done_flag': done_value, 'frames': frames}
    if flag_xy is not None:
        metadata['visual_goal_flag_xy_m'] = flag_xy
        metadata['reference_box_goal_xy_m'] = goal[:2].tolist()
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez_compressed(out / 'rollout_body_positions.npz',
                        body_pos=np.array(poses), time_s=np.array(times))
    print('[done] ' + str(out / 'metadata.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
