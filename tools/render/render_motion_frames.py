#!/usr/bin/env python3
"""Render a MimicKit motion clip to PNG frames.

Output is RGB on a pure black background (hide the ground plane and the sky,
key the black out afterwards if you need transparency).

Standalone: nothing under `mimickit/` is modified. Two library gaps are worked
around at runtime instead:

  * `envs/env_builder.py` forgets to forward `record_video` for the
    `view_motion` env, and `IsaacLabEngine` only builds lights/camera when
    `visualize or record_video`; without the patch a `visualize=false` run
    renders pure black.
  * Ground and sky are made invisible on the USD stage after the env is built,
    so physics (contacts against the ground plane) is untouched while only the
    character is drawn.

Camera framing is calibrated automatically: the clip is played once without
rendering to find the robot's maximum height, then the camera distance is tuned
until that maximum fills the requested fraction of the image height.

Examples
--------
    python tools/render/render_motion_frames.py \
        --motion data/motions/g1/g1_spinkick.pkl \
        --out-dir output/render/g1_spinkick --interval 0.5 --coverage 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))


def terminate(engine=None, code: int = 0) -> None:
    """Exit hard once every file is on disk.

    Isaac Sim hangs in its own shutdown: `SimulationApp.close()` never returns
    ("Recursive unloadAllPlugins() detected!") and the process then holds
    ~3.3 GB of VRAM forever. That leak is enough to make the *next* run fail
    with "No physics scene created", so everything is flushed and the process
    is killed outright instead of closing the app.
    """
    print("[shutdown] exiting without SimulationApp.close() (it hangs)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-config", default="data/envs/view_motion_g1_env.yaml")
    p.add_argument("--engine-config", default="data/engines/isaac_lab_engine.yaml")
    p.add_argument("--motion", default="data/motions/g1/g1_spinkick.pkl")
    p.add_argument("--out-dir", default="output/render/frames")
    p.add_argument("--interval", type=float, default=0.5,
                   help="seconds between captured frames")
    p.add_argument("--duration", type=float, default=-1.0,
                   help="seconds to play; <0 = exactly one motion pass")
    p.add_argument("--resolution", default="960,960", help="'W,H'")
    p.add_argument("--coverage", type=float, default=0.8,
                   help="fraction of image height the robot reaches at its tallest")
    p.add_argument("--cam-dir", default="0,-1,0",
                   help="unit-ish direction from the robot to the camera, 'x,y,z'")
    p.add_argument("--cam-height", type=float, default=0.0,
                   help="extra z offset relative to the root (0 = level with root)")
    p.add_argument("--fix-cam-z", action="store_true",
                   help="hold the camera at the initial root height instead of "
                        "following the root vertically")
    p.add_argument("--ref", choices=("root", "body_mid"), default="root",
                   help="vertical reference the camera is level with")
    p.add_argument("--char-color", default="natural",
                   help="'natural' keeps the asset's own materials; otherwise "
                        "'r,g,b' in 0..1 to tint the character")
    p.add_argument("--follow", choices=("root", "bbox"), default="bbox",
                   help="what the camera is centred on horizontally. 'bbox' "
                        "keeps a wide kick inside the frame; 'root' is a "
                        "steadier lock on the pelvis")
    p.add_argument("--keep-ground", action="store_true")
    p.add_argument("--keep-sky", action="store_true")
    p.add_argument("--preview", action="store_true",
                   help="stop after calibration and write one preview frame")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# stage surgery
# ---------------------------------------------------------------------------
def hide_background(keep_ground: bool, keep_sky: bool) -> dict:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    report: dict = {"hidden": [], "missing": []}

    targets = []
    if not keep_ground:
        targets.append("/World/ground")
    if not keep_sky:
        targets.append("/World/Light/dome_light")   # renders as the visible sky

    for path in targets:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            report["missing"].append(path)
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        report["hidden"].append(path)
    return report


# ---------------------------------------------------------------------------
# capture helpers
# ---------------------------------------------------------------------------
def build_annotator(resolution: tuple[int, int]):
    import omni.replicator.core as rep

    rp = rep.create.render_product("/OmniverseKit_Persp", resolution)
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    return annot


def grab_frame(annot) -> np.ndarray | None:
    data = annot.get_data()
    if data is None or data.size == 0:
        return None
    arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
    return np.ascontiguousarray(arr[:, :, :3])


def mask_bbox(rgb: np.ndarray, thresh: int = 1,
              min_row_px: int = 2, min_col_px: int = 2):
    """Bounding box of everything that is not exactly black.

    The background is genuinely [0, 0, 0], so even very dark materials survive
    the >= 1 threshold. Row/column pixel counts filter isolated noise.
    """
    lum = rgb.max(axis=2)
    m = lum >= thresh
    rows = m.sum(axis=1) >= min_row_px
    cols = m.sum(axis=0) >= min_col_px
    if not rows.any() or not cols.any():
        return None
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.chdir(REPO_DIR)

    res = tuple(int(v) for v in args.resolution.split(","))
    cam_dir = np.array([float(v) for v in args.cam_dir.split(",")], dtype=np.float64)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- env config: point the viewer at the requested motion ---------------
    with open(args.env_config, "r") as f:
        env_config = yaml.safe_load(f)
    env_config["motion_file"] = args.motion
    if args.duration > 0:
        env_config["episode_length"] = args.duration
    tmp_env = os.path.join(out_dir, "_env_config.yaml")
    with open(tmp_env, "w") as f:
        yaml.safe_dump(env_config, f)

    # --- force record_video so lights + viewport camera get built -----------
    import envs.sim_env as sim_env
    _orig_build_engine = sim_env.SimEnv._build_engine

    def _build_engine(self, engine_config, num_envs, device, visualize, record_video=False):
        return _orig_build_engine(self, engine_config, num_envs, device, visualize,
                                  record_video=True)

    sim_env.SimEnv._build_engine = _build_engine

    # --- character material -------------------------------------------------
    # `view_motion` forces a flat tint on the character. Returning None from
    # `_get_char_color` makes the engine skip its PreviewSurface override, so
    # the asset's own materials are used as-is.
    import envs.view_motion_env as view_motion_env

    if args.char_color == "natural":
        view_motion_env.ViewMotionEnv._get_char_color = lambda self: None
        print("[char] using the asset's own materials (no tint)")
    else:
        tint = np.array([float(v) for v in args.char_color.split(",")], dtype=np.float64)
        view_motion_env.ViewMotionEnv._get_char_color = lambda self: tint
        print(f"[char] tinting character with {tint.tolist()}")

    import envs.env_builder as env_builder

    env = env_builder.build_env(tmp_env, args.engine_config, num_envs=1,
                                device=args.device, visualize=False,
                                record_video=True)
    engine = env._engine
    char_id = env._get_char_id()

    report = hide_background(args.keep_ground, args.keep_sky)
    print(f"[stage] hidden={report['hidden']} missing={report['missing']}", flush=True)

    annot = build_annotator(res)
    sim = engine.get_sim()

    dt = engine.get_timestep()
    interval_steps = max(1, int(round(args.interval / dt)))
    motion_len = float(env._motion_lib.get_motion_length(
        env._get_env_motion_ids())[0].item())
    duration = args.duration if args.duration > 0 else motion_len
    total_steps = int(round(duration / dt))

    zeros = np.zeros(env.get_action_space().shape, dtype=np.float32)

    def sync_reset(step_target: int = 0) -> None:
        """Reset and drive the kinematic state to `step_target`.

        `ViewMotionEnv` only writes the motion frame inside `_update_misc`,
        which runs after a step. Right after `reset()` the character still holds
        its default pose - for g1 that pose sits *below* the ground plane, so
        any measurement taken there is garbage.
        """
        env.reset()
        sync = getattr(env, "_sync_motion", None)
        if sync is not None:
            sync()
        bp = engine.get_body_pos(char_id)[0].cpu().numpy()
        if float(bp[:, 2].min()) < -0.05:
            print("[warn] kinematic sync after reset did not take effect; "
                  "stepping once as a fallback")
            env.step(zeros)
        for _ in range(step_target):
            env.step(zeros)
        return

    # ---------------------------------------------------------------- pass A
    # Play once without rendering: collect the robot's vertical extent so the
    # framing accounts for the jump, not just the standing pose.
    print("[A] measuring robot extent over the whole clip...", flush=True)
    sync_reset()
    heights = np.zeros(total_steps + 1)
    root_z = np.zeros(total_steps + 1)
    body_zmin = np.zeros(total_steps + 1)
    body_zmax = np.zeros(total_steps + 1)
    bbox_cx = np.zeros(total_steps + 1)
    bbox_cy = np.zeros(total_steps + 1)
    root_xy = np.zeros((total_steps + 1, 2))
    diag: list[tuple] = []
    for step in range(total_steps + 1):
        bp = engine.get_body_pos(char_id)[0].cpu().numpy()
        rp = engine.get_root_pos(char_id)[0].cpu().numpy()
        heights[step] = float(bp[:, 2].max() - bp[:, 2].min())
        body_zmin[step] = float(bp[:, 2].min())
        body_zmax[step] = float(bp[:, 2].max())
        bbox_cx[step] = 0.5 * float(bp[:, 0].min() + bp[:, 0].max())
        bbox_cy[step] = 0.5 * float(bp[:, 1].min() + bp[:, 1].max())
        root_xy[step] = rp[:2]
        rz = float(rp[2])
        root_z[step] = rz
        if step in (0, 1, 2, 5, 10, 20, total_steps):
            diag.append((step, step * dt, rz, body_zmin[step], body_zmax[step],
                         heights[step]))
        if step < total_steps:
            env.step(zeros)

    t_star = int(np.argmax(heights))
    h_max = float(heights.max())
    print(f"[A] motion_len={motion_len:.3f}s duration={duration:.3f}s "
          f"steps={total_steps}")
    print("[A]   step      t    root_z  body_zmin  body_zmax   height")
    for s, t, rz, bz0, bz1, h in diag:
        print(f"[A] {s:6d} {t:6.2f}  {rz:8.3f}  {bz0:9.3f}  {bz1:9.3f}  {h:7.3f}")
    print(f"[A] robot height: min={heights.min():.3f}m max={h_max:.3f}m "
          f"at step {t_star} (t={t_star * dt:.2f}s, root_z={root_z[t_star]:.3f}m)")

    # Vertical reference for the camera. `root` is what was asked for; the body
    # bbox centre is the fallback when root_z is unreliable.
    body_mid = 0.5 * (body_zmin + body_zmax)
    if args.ref == "body_mid":
        cam_z_series = body_mid
    else:
        cam_z_series = root_z

    # Horizontal aim. A spinkick throws a leg far to one side, so locking the
    # camera to the pelvis pushes that leg out of frame; centring on the body
    # bbox keeps the whole robot inside without dropping the vertical coverage.
    if args.follow == "bbox":
        cam_xy = np.stack([bbox_cx, bbox_cy], axis=1)
    else:
        cam_xy = root_xy
    drift = np.linalg.norm(cam_xy - root_xy, axis=1)
    print(f"[A] follow={args.follow}: bbox centre drifts up to "
          f"{drift.max():.3f}m horizontally from the root")

    def cam_target(step: int) -> np.ndarray:
        return np.array([cam_xy[step, 0], cam_xy[step, 1], cam_z_series[step]])

    # ---------------------------------------------------------------- pass B
    # Calibrate the camera distance so the tallest pose fills `coverage` of
    # the image height. Rendering is only needed on the tallest frame.
    def render_at(step_target: int, dist: float, cam_z: float) -> np.ndarray:
        sync_reset(step_target)
        target = cam_target(step_target).copy()
        target[2] = cam_z
        eye = target + cam_dir * dist
        eye[2] += args.cam_height
        engine.set_camera_pose(eye, target)
        sim.render()
        return grab_frame(annot)

    cam_z_mode = "fixed at initial root z" if args.fix_cam_z else "follows the root"
    print(f"[B] calibrating distance for coverage={args.coverage:.2f} "
          f"(camera {cam_z_mode})...", flush=True)

    z0 = float(cam_z_series[0])
    below = cam_z_series - body_zmin          # how far the robot reaches under the axis
    above = body_zmax - cam_z_series          # ... and over it
    span = np.maximum(below, above)
    worst_step = int(np.argmax(span))
    h_cover = h_max / (2.0 * args.coverage)   # half-span that yields the target coverage
    h_noclip = float(span.max())
    print(f"[B] vertical half-span required: coverage={h_cover:.3f}m, "
          f"no-clip={h_noclip:.3f}m (worst at step {worst_step}, "
          f"t={worst_step * dt:.2f}s)")

    def cam_z_for(step: int) -> float:
        return z0 if args.fix_cam_z else float(cam_z_series[step])

    def measure(step: int, dist: float):
        frame = render_at(step, dist, cam_z_for(step))
        if frame is None:
            raise RuntimeError("renderer returned an empty frame")
        bb = mask_bbox(frame)
        if bb is None:
            raise RuntimeError("frame is entirely black; nothing to frame")
        return frame, bb

    # One render at a reference distance gives metres-per-pixel, hence the
    # visible half-span as a function of distance.
    ref_dist = 4.0
    frame, bb = measure(t_star, ref_dist)
    cov_ref = (bb[3] - bb[2] + 1) / res[1]
    half_at_ref = h_max / (2.0 * cov_ref)
    span_per_metre = half_at_ref / ref_dist
    print(f"[B]   ref dist={ref_dist:.2f}m -> coverage={cov_ref:.3f}, "
          f"visible half-span={half_at_ref:.3f}m "
          f"({span_per_metre:.3f}m per metre of distance)")

    # Start from the distance that gives exactly the requested coverage, then
    # grow it only as far as the borders demand.
    dist = h_cover / span_per_metre
    if h_noclip > h_cover:
        print(f"[B]   note: the clip-free estimate ({h_noclip:.3f}m) already "
              f"exceeds what {args.coverage:.0%} needs ({h_cover:.3f}m)")

    if args.preview:
        import imageio.v2 as iio
        _frame, bb = measure(t_star, dist)
        iio.imwrite(os.path.join(out_dir, "preview.png"), _frame)
        print(f"[preview] dist={dist:.3f}m coverage="
              f"{(bb[3] - bb[2] + 1) / res[1]:.3f} -> preview.png")
        terminate(engine)

    # ---------------------------------------------------------------- pass C
    # The camera is tilted by `--cam-height`, so which frame clips cannot be
    # predicted from the body z-range alone. Render every captured frame and
    # grow the shot until nothing touches a border; the accepted sweep is also
    # the output, so nothing is rendered twice.
    cap_steps = list(range(0, total_steps + 1, interval_steps))
    print(f"[C] validating {len(cap_steps)} frames at {args.interval}s spacing...",
          flush=True)

    pad = 3                                     # px kept clear on every side
    # The distance is tuned on the *measured* worst-case frame rather than on
    # the analytically tallest one: the tilted camera makes the two disagree.
    best = None                                 # (max_cov, dist, sweep, worst_margin)
    sweep = []
    for it in range(8):
        sweep = [(st,) + measure(st, dist) for st in cap_steps]
        worst = min(min(bb[2], res[1] - 1 - bb[3], bb[0], res[0] - 1 - bb[1])
                    for _st, _f, bb in sweep)
        covs = [(bb[3] - bb[2] + 1) / res[1] for _st, _f, bb in sweep]
        max_cov = max(covs)
        print(f"[C]   iter {it}: dist={dist:.3f}m coverage="
              f"{min(covs):.3f}..{max_cov:.3f} worst margin={worst}px", flush=True)
        if worst >= 1 and (best is None or max_cov > best[0]):
            best = (max_cov, dist, sweep, worst)
        if worst < pad:
            dist *= 1.04                       # too tight, back off
        elif abs(max_cov - args.coverage) <= 0.005:
            break                              # on target: done
        else:
            dist *= max_cov / args.coverage    # pull the worst frame onto target

    if best is not None:
        max_cov_best, dist, sweep, worst = best
        if abs(max_cov_best - max(covs)) > 1e-9:
            print(f"[C]   using the closest clip-free sweep: dist={dist:.3f}m "
                  f"max coverage={max_cov_best:.3f} margin={worst}px")
    else:
        print("[C][warn] no clip-free distance found; writing the last sweep")

    import imageio.v2 as iio

    saved = 0
    covers = []
    margins = []
    for _st, frame, bb in sweep:
        covers.append((bb[3] - bb[2] + 1) / res[1])
        margins.append((bb[2], res[1] - 1 - bb[3], bb[0], res[0] - 1 - bb[1]))
        iio.imwrite(os.path.join(out_dir, f"frame_{saved:04d}.png"), frame)
        saved += 1

    _frame, bbox = measure(t_star, dist)
    cover = (bbox[3] - bbox[2] + 1) / res[1]
    print(f"[B] final: dist={dist:.3f}m coverage at tallest frame={cover:.3f}")

    summary = {
        "motion": args.motion,
        "motion_len_s": motion_len,
        "duration_s": duration,
        "interval_s": args.interval,
        "resolution": list(res),
        "frames": saved,
        "camera": {
            "direction": cam_dir.tolist(),
            "distance_m": dist,
            "height_offset_m": args.cam_height,
            "fixed_z": bool(args.fix_cam_z),
            "follow": args.follow,
            "vertical_ref": args.ref,
            "initial_root_z": z0,
        },
        "robot_height_m": {"max": h_max, "min": float(heights.min())},
        "coverage": {
            "target": args.coverage,
            "at_tallest_step": cover,
            "captured_min": float(min(covers)) if covers else None,
            "captured_max": float(max(covers)) if covers else None,
        },
        "top_bottom_margin_px_min": [int(min(m)) for m in zip(*margins)] if margins else None,
    }
    with open(os.path.join(out_dir, "render_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"[done] {saved} frames -> {out_dir}")
    print(f"[done] coverage over captured frames: "
          f"min={summary['coverage']['captured_min']:.3f} "
          f"max={summary['coverage']['captured_max']:.3f}")
    terminate(engine)


if __name__ == "__main__":
    main()
