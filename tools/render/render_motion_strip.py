#!/usr/bin/env python3
"""Render a MimicKit motion into a paper-style strip of evenly spaced frames.

The scene is a pure kinematic visualisation: the character is driven from the
motion file and every prop is spawned as visual-only geometry (no physics), so
no rigid-body setup is needed and the frame poses are exact.

Unlike `render_motion_frames.py` this keeps the ground and the sky visible,
matching the look of the existing `PAPER/figures/dare_*_10.png` strips.

Standalone: nothing under `mimickit/` is modified. `view_motion` forgets to
forward `record_video`, and `IsaacLabEngine` only builds lights/camera when
`visualize or record_video`, so that flag is forced at runtime.

Examples
--------
    python tools/render/render_motion_strip.py \
        --motion data/motions/g1/g1_backflip.pkl \
        --out-dir output/render/strips/backflip --frames 10

    # carrying a box whose pose is stored in a companion trajectory
    python tools/render/render_motion_strip.py \
        --motion data/motions/g1/g1_carry_box.pkl \
        --out-dir output/render/strips/carry_box --frames 10 \
        --prop-traj data/motions/g1/g1_carry_box_object.npz \
        --prop-size 0.55,0.35,0.35
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))


def terminate(code: int = 0) -> None:
    """Isaac Sim hangs in SimulationApp.close(); exit hard instead."""
    print("[shutdown] exiting without SimulationApp.close() (it hangs)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--trim", default="",
                   help="frame range 'A,B' to play; framing is measured on the "
                        "trimmed window, so long travelling clips can be zoomed in")
    p.add_argument("--resolution", default="960,960")
    p.add_argument("--engine-config", default="data/engines/isaac_lab_engine.yaml")
    p.add_argument("--env-config", default="data/envs/view_motion_g1_env.yaml")
    p.add_argument("--cam-dir", default="0,-1,0",
                   help="unit direction from the scene centre to the camera")
    p.add_argument("--cam-elev", type=float, default=0.0,
                   help="camera height above the scene centre, in metres")
    p.add_argument("--coverage", type=float, default=0.72,
                   help="fraction of the image the whole scene should fill")
    p.add_argument("--pad", type=float, default=1.05,
                   help="safety factor applied to the framing")
    p.add_argument("--prop-box", action="append", default=[],
                   metavar="SX,SY,SZ@X,Y,Z[,YAW[,Z]]",
                   help="static visual cuboid; repeatable")
    p.add_argument("--prop-traj", default="",
                   help="npz with keys pos (T,3) and exp (T,3) driving one box")
    p.add_argument("--prop-size", default="0.5,0.4,0.4")
    p.add_argument("--char-color", default="0.2,0.25,0.7",
                   help="'r,g,b' tint. The default matches the blue cast used by "
                        "the paper's dare_*_10.png strips; 'natural' keeps the "
                        "asset materials (the G1 USD has none, so it renders white)")
    p.add_argument("--no-ground", action="store_true")
    p.add_argument("--no-sky", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
def set_visible(path: str, visible: bool) -> bool:
    import omni.usd
    from pxr import UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible() if visible else imageable.MakeInvisible()
    return True


def spawn_box(path: str, size, pos, yaw_deg=0.0, color=(0.45, 0.47, 0.52)):
    """Visual-only cuboid. Returns the prim."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in size),
        collision_props=None,
        rigid_props=None,
        mass_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(color), metallic=0.1, roughness=0.7),
    )
    prim = cfg.func(path, cfg)
    set_pose(path, pos, yaw_deg)
    return prim


def set_pose(path: str, pos, yaw_deg: float = 0.0, quat_wxyz=None) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)

    # `spawn_cuboid` builds a `UsdGeom.Cube` whose side is min(size) and then
    # scales it up to the requested size. That scale op MUST survive, otherwise
    # every prop collapses to a cube of its smallest dimension.
    existing = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
    trans = existing.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
    orient = existing.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()
    scale = existing.get(UsdGeom.XformOp.TypeScale)
    xf.SetXformOpOrder([trans, orient] + ([scale] if scale is not None else []))

    trans.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    if quat_wxyz is None:
        half = np.radians(yaw_deg) / 2.0
        q = Gf.Quatf(float(np.cos(half)), Gf.Vec3f(0.0, 0.0, float(np.sin(half))))
    else:
        q = Gf.Quatf(float(quat_wxyz[0]),
                     Gf.Vec3f(*[float(v) for v in quat_wxyz[1:4]]))
    if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient.Set(Gf.Quatd(float(q.real), Gf.Vec3d(*[float(v) for v in q.imaginary])))
    else:
        orient.Set(q)


def build_annotator(resolution):
    import omni.replicator.core as rep

    rp = rep.create.render_product("/OmniverseKit_Persp", resolution)
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    return annot


def grab(annot):
    data = annot.get_data()
    if data is None or data.size == 0:
        return None
    arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
    return np.ascontiguousarray(arr[:, :, :3])


def content_bbox(rgb, thresh=8, min_px=2):
    """Bounding box of non-background pixels (the ground is flat, so use a
    luminance threshold well above it)."""
    lum = rgb.max(axis=2)
    m = lum >= thresh
    rows = m.sum(axis=1) >= min_px
    cols = m.sum(axis=0) >= min_px
    if not rows.any() or not cols.any():
        return None
    y0 = int(np.argmax(rows)); y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols)); x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.chdir(REPO_DIR)
    res = tuple(int(v) for v in args.resolution.split(","))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- force record_video so lights + the viewport camera get built -------
    import envs.sim_env as sim_env
    _orig = sim_env.SimEnv._build_engine

    def _build_engine(self, engine_config, num_envs, device, visualize, record_video=False):
        return _orig(self, engine_config, num_envs, device, visualize, record_video=True)

    sim_env.SimEnv._build_engine = _build_engine

    import envs.view_motion_env as view_motion_env
    if args.char_color == "natural":
        view_motion_env.ViewMotionEnv._get_char_color = lambda self: None
    else:
        tint = np.array([float(v) for v in args.char_color.split(",")])
        view_motion_env.ViewMotionEnv._get_char_color = lambda self: tint

    import envs.env_builder as env_builder
    import yaml

    with open(args.env_config) as f:
        env_config = yaml.safe_load(f)
    env_config["motion_file"] = args.motion
    tmp_env = os.path.join(out_dir, "_env_config.yaml")
    with open(tmp_env, "w") as f:
        yaml.safe_dump(env_config, f)

    env = env_builder.build_env(tmp_env, args.engine_config, num_envs=1,
                               device=args.device, visualize=False, record_video=True)
    engine = env._engine
    char_id = env._get_char_id()
    sim = engine.get_sim()

    # --- background ---------------------------------------------------------
    hidden = []
    if args.no_ground and set_visible("/World/ground", False):
        hidden.append("ground")
    if args.no_sky and set_visible("/World/Light/dome_light", False):
        hidden.append("sky")
    print(f"[stage] hidden={hidden or 'nothing (ground and sky visible)'}", flush=True)

    # --- props --------------------------------------------------------------
    prop_paths = []          # (path, size) static
    for i, spec in enumerate(args.prop_box):
        body, _, pos_s = spec.partition("@")
        size = [float(v) for v in body.split(",")]
        pos_txt = pos_s.split(",")
        pos = [float(v) for v in pos_txt[:3]]
        yaw = float(pos_txt[3]) if len(pos_txt) > 3 else 0.0
        path = f"/World/props/prop_{i}"
        spawn_box(path, size, pos, yaw)
        prop_paths.append(path)
        print(f"[prop] box size={size} pos={pos} yaw={yaw}")

    traj = None
    if args.prop_traj:
        with np.load(args.prop_traj) as d:
            traj = dict(pos=np.asarray(d["pos"], dtype=np.float64),
                        exp=np.asarray(d["exp"], dtype=np.float64),
                        fps=float(d["fps"]))
        size = [float(v) for v in args.prop_size.split(",")]
        spawn_box("/World/props/traj_box", size, traj["pos"][0])
        prop_paths.append("/World/props/traj_box")
        print(f"[prop] trajectory box size={size} frames={len(traj['pos'])}")

    annot = build_annotator(res)

    # --- timing -------------------------------------------------------------
    dt = engine.get_timestep()
    zeros = np.zeros(env.get_action_space().shape, dtype=np.float32)
    n_motion = int(env._motion_lib.get_motion_length(
        env._get_env_motion_ids())[0].item() / dt)
    total_steps = max(1, n_motion - 1)
    duration = total_steps * dt
    if args.trim:
        lo, hi = (int(v) for v in args.trim.split(","))
    else:
        lo, hi = 0, total_steps
    lo = max(0, min(lo, total_steps))
    hi = max(lo + 1, min(hi, total_steps))
    step_list = np.linspace(lo, hi, args.frames).round().astype(int)
    print(f"[run] motion={os.path.basename(args.motion)} duration={duration:.2f}s "
          f"steps={total_steps} trim=[{lo},{hi}] -> {args.frames} frames at "
          f"steps {step_list.tolist()}")

    def sync_reset(target: int = 0) -> None:
        env.reset()
        sync = getattr(env, "_sync_motion", None)
        if sync is not None:
            sync()
        bp = engine.get_body_pos(char_id)[0].cpu().numpy()
        if float(bp[:, 2].min()) < -0.05:
            print("[warn] kinematic sync after reset did not take; stepping once")
            env.step(zeros)
        # `_sync_motion` only writes the PhysX buffers; the pose reaches the
        # renderer on the next sim step. Zero steps would therefore render the
        # state left over from the previous pass, so always step at least once.
        for _ in range(max(int(target), 1)):
            env.step(zeros)

    # --- pass A: world extent of everything we must frame -------------------
    print("[A] measuring scene extent...", flush=True)
    sync_reset()
    body_pts = []
    for step in range(total_steps + 1):
        if lo <= step <= hi:                     # only the trimmed window is framed
            bp = engine.get_body_pos(char_id)[0].cpu().numpy()
            body_pts.append(bp.reshape(-1, 3))
        if step < total_steps:
            env.step(zeros)
    pts = np.concatenate(body_pts, axis=0)
    if traj is not None:
        pts = np.concatenate([pts, traj["pos"]], axis=0)
    for spec in args.prop_box:
        body, _, pos_s = spec.partition("@")
        size = np.array([float(v) for v in body.split(",")])
        pos = np.array([float(v) for v in pos_s.split(",")[:3]])
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts = np.concatenate(
                        [pts, (pos + 0.5 * size * np.array([sx, sy, sz]))[None]], axis=0)

    centre = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    extent = pts.max(axis=0) - pts.min(axis=0)
    print(f"[A] scene centre={np.round(centre, 3).tolist()} extent={np.round(extent, 3).tolist()}")

    cam_dir = np.array([float(v) for v in args.cam_dir.split(",")])
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    eye_z = centre[2] + args.cam_elev

    def set_cam(dist: float) -> None:
        target = np.array([centre[0], centre[1], eye_z])
        eye = target + cam_dir * dist
        eye[2] += args.cam_elev
        engine.set_camera_pose(eye, target)

    # --- pass B: pick the distance that frames the whole motion -------------
    # The ground spans the whole image, so it is hidden while measuring. The
    # measurement is the union over every captured frame: the robot only fills
    # part of its own trajectory box at any single frame, so measuring one
    # frame would crop the extremes (a backflip apex clips the top edge).
    def measure_union(dist: float):
        set_visible("/World/ground", False)
        set_visible("/World/Light/dome_light", False)
        set_cam(dist)
        u = None
        for st in step_list:
            sync_reset(int(st))
            sim.render()
            bb = content_bbox(grab(annot))
            if bb is None:
                continue
            u = bb if u is None else (min(u[0], bb[0]), max(u[1], bb[1]),
                                      min(u[2], bb[2]), max(u[3], bb[3]))
        set_visible("/World/ground", not args.no_ground)
        set_visible("/World/Light/dome_light", not args.no_sky)
        if u is None:
            raise RuntimeError("nothing visible with the ground hidden")
        return u

    dist = 6.0
    for it in range(6):
        bb = measure_union(dist)
        cov = max((bb[3] - bb[2] + 1) / res[1], (bb[1] - bb[0] + 1) / res[0])
        print(f"[B] iter {it}: dist={dist:.2f}m -> union fills {cov:.3f} "
              f"(target {args.coverage}); content px "
              f"{bb[1] - bb[0] + 1}x{bb[3] - bb[2] + 1}", flush=True)
        if abs(cov - args.coverage) <= 0.02:
            break
        dist *= cov / args.coverage
    dist *= args.pad

    # --- pass C: capture ----------------------------------------------------
    # Each frame is reached with an explicit `sync_reset(step)` rather than by
    # stepping continuously: right after `reset()` the engine still holds the
    # pose left over from the previous pass, so a continuously stepped loop
    # renders a stale state for the very first frame.
    import imageio.v2 as iio
    from util import torch_util
    import torch

    saved = 0
    for st in step_list:
        st = int(st)
        sync_reset(st)
        if traj is not None:
            idx = int(round(st * dt * traj["fps"]))
            idx = min(max(idx, 0), len(traj["pos"]) - 1)
            q = torch_util.exp_map_to_quat(
                torch.from_numpy(traj["exp"][idx][None])).numpy()[0]
            set_pose("/World/props/traj_box", traj["pos"][idx], quat_wxyz=q)
        sim.render()
        frame = grab(annot)
        if frame is None:
            print(f"[warn] empty frame at step {st}")
        else:
            iio.imwrite(os.path.join(out_dir, f"frame_{saved:04d}.png"), frame)
            saved += 1

    print(f"[done] {saved} frames -> {out_dir}")
    terminate()


if __name__ == "__main__":
    main()
