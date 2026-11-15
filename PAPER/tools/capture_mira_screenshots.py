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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--resolution', type=int, default=800)
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = run / f'int_models/model_{args.updates - 1:010d}.pt'
    assert model.is_file(), model
    source = run / 'source_snapshot'
    manifest = json.loads((run / 'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(source / name) == expected, name
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
    env = env_builder.build_env(str(run / 'frozen_env.yaml'),
                               str(run / 'frozen_engine.yaml'), 1, 'cuda:0',
                               visualize=False, record_video=True)
    agent = agent_builder.build_agent(str(run / 'frozen_agent.yaml'), env, 'cuda:0')
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
    # Replace the grid with an uncluttered neutral floor; its collider remains.
    material = UsdShade.Material.Define(stage, '/World/PaperFloorMaterial')
    shader = UsdShade.Shader.Define(stage, '/World/PaperFloorMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.72, .75, .79))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.82)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    ground = stage.GetPrimAtPath('/World/ground')
    for prim in Usd.PrimRange(ground):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

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
    steps = np.rint(np.linspace(1, horizon, args.frames)).astype(int).tolist()
    if args.preview:
        steps = [1, horizon // 2]
    motion = yaml.safe_load((run / 'frozen_env.yaml').read_text())['motion_name']

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
    distances = {'backflip': 2.7, 'climb': 3.8, 'crawl': 2.5,
                 'roll': 2.5, 'getup_facedown': 2.7, 'spinkick': 2.9}
    direction = directions[motion] / np.linalg.norm(directions[motion])
    distance = distances[motion]
    frames, times = [], []
    poses = []
    # No redundant automatic video recording; selected captures below are RGB.
    engine._recording = False

    def capture(step):
        char = env._get_char_id()
        body = engine.get_body_pos(char)[0].detach().cpu().numpy()
        target = .5 * (body.min(axis=0) + body.max(axis=0))
        # Climb includes the ledge, while preserving a near-side view of feet.
        if motion == 'climb':
            target[2] = max(.9, target[2])
        engine.set_camera_pose(target + direction * distance, target)
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
                       'camera_eye': (target + direction * distance).tolist(),
                       'camera_target': target.tolist(), 'sha256': sha(out / name)})
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
                if step not in steps:
                    capture(step)
                break
            if args.preview and step >= max(steps):
                break
    metadata = {'motion': motion, 'model': str(model.relative_to(ROOT)),
                'model_sha256': sha(model), 'training_updates': args.updates,
                'source_manifest_sha256': sha(run / 'source_manifest.json'),
                'renderer': 'Isaac Lab / Isaac Sim RGB camera',
                'actual_policy_rollout': True, 'deterministic_mean_action': True,
                'training_seed': 0, 'rollout_seed': 0, 'num_envs': 1,
                'reference_start': True, 'automatic_reset': False,
                'visual_changes': 'teal body, neutral floor, camera pose only',
                'resolution': list(resolution), 'dt': dt, 'requested_steps': horizon,
                'executed_steps': len(times), 'done_flag': done_value, 'frames': frames}
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
