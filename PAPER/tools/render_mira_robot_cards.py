#!/usr/bin/env python3
"""Capture one of the matched robot card views with Isaac Sim lighting."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--role', choices=('reference', 'simulated'), required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[55])
    parser.add_argument('--distance', type=float, default=4.9)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.steps or min(args.steps) < 1:
        parser.error('steps must be positive')

    sys.path.insert(0, str(ROOT/'mimickit'))
    os.chdir(ROOT)
    import torch
    import envs.env_builder as env_builder
    import envs.deepmimic_env as deepmimic_env
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    # These overrides only enable the renderer's reference visual and change
    # visual colors. Training configs, actions, references and contacts stay intact.
    deepmimic_env.DeepMimicEnv._enable_ref_char = lambda self: True
    robot_color = {'reference': (.28, .31, .35), 'simulated': (.105, .225, .355)}[args.role]
    deepmimic_env.DeepMimicEnv._get_ref_char_color = lambda self: np.array(robot_color)
    util.set_rand_seed(0)
    torch.set_num_threads(4)
    mp_util.init(rank=0, num_procs=1, device='cuda:0', master_port=6784)
    env = env_builder.build_env(str(run/'env_config.yaml'), str(run/'engine_config.yaml'),
                               1, 'cuda:0', visualize=False, record_video=True)
    engine = env._engine
    engine._recording = False
    sim = engine.get_sim()
    agent = agent_builder.build_agent(str(run/'agent_config.yaml'), env, 'cuda:0')
    model = run/'model.pt'
    agent.load(str(model))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env.set_test_random_start(False)

    import carb
    import omni.usd
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from engines.isaac_lab_engine import OBJ_PATH_TEMPLATE
    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()
    settings.set_bool('/rtx/shadows/enabled', True)

    def material(path, name, color, roughness, metallic=0.):
        mat = UsdShade.Material.Define(stage, f'/World/{name}')
        shader = UsdShade.Shader.Define(stage, f'/World/{name}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    material('/World/ground', 'CardFloor', (.59, .65, .73), .81)
    # Keep the native key light and lower the fill so the footprint of the
    # robot is legible in a small frame. No painted or composited shadow.
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            UsdLux.DomeLight(prim).GetIntensityAttr().Set(360.)
        if prim.IsA(UsdLux.DistantLight):
            key = UsdLux.DistantLight(prim)
            key.GetIntensityAttr().Set(2650.)
            key.GetAngleAttr().Set(3.)

    display_id, actor_id = env._get_ref_char_id(), env._get_char_id()
    display_path = OBJ_PATH_TEMPLATE.format(0, display_id)
    actor_path = OBJ_PATH_TEMPLATE.format(0, actor_id)
    material(display_path, 'CardRobot', robot_color, .42, .12)
    UsdGeom.Imageable(stage.GetPrimAtPath(actor_path)).MakeInvisible()
    if hasattr(env, '_ball_id'):
        UsdGeom.Imageable(stage.GetPrimAtPath(OBJ_PATH_TEMPLATE.format(0, env._ball_id))).MakeInvisible()

    resolution = (1200, 455)
    product = rep.create.render_product('/OmniverseKit_Persp', resolution)
    annot = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
    annot.attach([product])
    obs, info = env.reset()
    dt = float(engine.get_timestep())
    direction = np.array([.93, .33, .33])
    direction /= np.linalg.norm(direction)
    frames = []
    with torch.no_grad():
        for step in range(1, max(args.steps) + 1):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            if int(done[0]) != 0:
                raise RuntimeError('The rollout ended before the requested frame')
            if step not in args.steps:
                continue
            engine._update_reset_objs()
            offset = env._ref_char_offset
            if args.role == 'simulated':
                # Reuse the collision-free display mesh with the policy's
                # measured articulated pose. Do not change the physical actor.
                body = engine.get_body_pos(actor_id)[0]
                engine.set_root_pos(None, display_id, engine.get_root_pos(actor_id) + offset)
                engine.set_root_rot(None, display_id, engine.get_root_rot(actor_id))
                engine.set_dof_pos(None, display_id, engine.get_dof_pos(actor_id))
                engine._update_reset_objs()
            else:
                body = env._ref_body_pos[0]
            pose = (body + offset).cpu().numpy()
            sim.forward()
            target = .5 * (pose.min(axis=0) + pose.max(axis=0))
            target[2] = .79
            eye = target + args.distance * direction
            engine.set_camera_pose(eye, target)
            for _ in range(8):
                sim.render()
            raw = annot.get_data()
            if raw is None or not raw.size:
                raise RuntimeError('No RGB pixels from Isaac Sim')
            name = f'{args.role}_{step:03d}.png' if len(args.steps) > 1 else f'{args.role}.png'
            path = out/name
            Image.fromarray(np.asarray(raw)[:, :, :3].copy()).save(path, optimize=True)
            frames.append(dict(file=name, role=args.role, step=step, time_s=step*dt,
                               visible_object=display_path, pose_source=args.role,
                               camera_eye=eye.tolist(), camera_target=target.tolist(),
                               robot_rgb=robot_color, image_sha256=sha(path)))
            print('Captured', args.role, step, path, flush=True)
    provenance = dict(
        renderer='Isaac Lab / Isaac Sim RGB camera',
        embodiment='data/assets/humanoid/humanoid.xml',
        env_config=str((run/'env_config.yaml').relative_to(ROOT)),
        checkpoint=str(model.relative_to(ROOT)), checkpoint_sha256=sha(model),
        simulation_pose='Deterministic policy from the reference start',
        reference_pose='Synchronized motion-library pose from Isaac Lab reference visualization',
        presentation='Shared visualization mesh with cool silver and muted steel blue; same floor, lighting and shadows',
        frames=frames,
    )
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


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
