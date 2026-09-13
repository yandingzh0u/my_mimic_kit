"""Pure metric definitions shared by all paper baselines.

The functions in this module deliberately do not depend on an agent, reward,
or simulator implementation.  Every method is therefore evaluated from the
same physical simulator and reference states.  Tensors are expected to use the
MimicKit quaternion convention ``(x, y, z, w)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


TRACKING_ERROR_NAMES = (
    "root_pos_err",
    "root_rot_err",
    "body_pos_err",
    "body_rot_err",
    "dof_vel_err",
    "root_vel_err",
    "root_ang_vel_err",
    "paper_pos_err",
)


@dataclass(frozen=True)
class CompletionThresholds:
    """Pre-registered, reference-normalized physical completion thresholds."""

    survival_min: float = 0.95
    progress_min: float = 0.75
    progress_max: float = 1.25
    lateral_ratio_max: float = 0.25
    winding_min: float = 0.75
    winding_max: float = 1.25
    upright_min: float = 0.65
    terminal_orientation_error_max: float = 0.75
    getup_height_ratio_min: float = 0.85
    climb_height_gain_ratio_min: float = 0.80
    climb_final_height_error_max: float = 0.35
    climb_feet_progress_min: float = 0.65
    jump_height_gain_ratio_min: float = 0.75


def _check_last_dim(name: str, value: torch.Tensor, size: int) -> None:
    if value.shape[-1] != size:
        raise ValueError(
            f"{name} has last dimension {value.shape[-1]}, expected {size}"
        )


def quaternion_angle(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Shortest unsigned rotation angle between two xyzw quaternions."""

    _check_last_dim("q0", q0, 4)
    _check_last_dim("q1", q1, 4)
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    # q and -q encode the same rotation.
    cos_half = torch.abs(torch.sum(q0 * q1, dim=-1))
    cos_half = torch.clamp(cos_half, 0.0, 1.0)
    return 2.0 * torch.acos(cos_half)


def quaternion_up_dot(q: torch.Tensor) -> torch.Tensor:
    """Cosine between the body's local +z axis and world +z."""

    _check_last_dim("q", q, 4)
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y = q[..., 0], q[..., 1]
    return 1.0 - 2.0 * (x * x + y * y)


def compute_tracking_errors(
    *,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    dof_vel: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    ref_root_pos: torch.Tensor,
    ref_root_rot: torch.Tensor,
    ref_body_pos: torch.Tensor,
    ref_body_rot: torch.Tensor,
    ref_dof_vel: torch.Tensor,
    ref_root_vel: torch.Tensor,
    ref_root_ang_vel: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the seven stock errors and the paper's position metric.

    Metrics are returned per batch element.  ``paper_pos_err`` implements

      (root displacement + sum of root-relative non-root body errors)
      / number_of_bodies.

    This is kept explicit rather than reconstructed from logged averages so it
    remains correct for other morphologies.
    """

    if body_pos.shape != ref_body_pos.shape:
        raise ValueError("simulated and reference body positions must match")
    if body_rot.shape != ref_body_rot.shape:
        raise ValueError("simulated and reference body rotations must match")
    if body_pos.shape[-1] != 3 or body_rot.shape[-1] != 4:
        raise ValueError("body positions/rotations must end in 3/4 coordinates")
    if body_pos.shape[-2] < 1:
        raise ValueError("at least the root body is required")

    root_pos_err = torch.linalg.vector_norm(ref_root_pos - root_pos, dim=-1)
    root_rot_err = quaternion_angle(root_rot, ref_root_rot)

    local_body_pos = body_pos - root_pos.unsqueeze(-2)
    local_ref_body_pos = ref_body_pos - ref_root_pos.unsqueeze(-2)
    per_body_pos_err = torch.linalg.vector_norm(
        local_ref_body_pos - local_body_pos, dim=-1
    )
    body_pos_err = torch.mean(per_body_pos_err, dim=-1)
    body_rot_err = torch.mean(quaternion_angle(body_rot, ref_body_rot), dim=-1)

    dof_vel_err = torch.mean(torch.abs(ref_dof_vel - dof_vel), dim=-1)
    root_vel_err = torch.mean(torch.abs(ref_root_vel - root_vel), dim=-1)
    root_ang_vel_err = torch.mean(
        torch.abs(ref_root_ang_vel - root_ang_vel), dim=-1
    )

    num_bodies = body_pos.shape[-2]
    nonroot_sum = torch.sum(per_body_pos_err[..., 1:], dim=-1)
    paper_pos_err = (root_pos_err + nonroot_sum) / float(num_bodies)

    return {
        "root_pos_err": root_pos_err,
        "root_rot_err": root_rot_err,
        "body_pos_err": body_pos_err,
        "body_rot_err": body_rot_err,
        "dof_vel_err": dof_vel_err,
        "root_vel_err": root_vel_err,
        "root_ang_vel_err": root_ang_vel_err,
        "paper_pos_err": paper_pos_err,
    }


def projected_motion_metrics(
    sim_delta: torch.Tensor,
    ref_delta: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed progress ratio, lateral distance, and lateral ratio."""

    _check_last_dim("sim_delta", sim_delta, 3)
    _check_last_dim("ref_delta", ref_delta, 3)
    ref_sq = torch.sum(ref_delta * ref_delta, dim=-1)
    ratio = torch.sum(sim_delta * ref_delta, dim=-1) / torch.clamp_min(
        ref_sq, eps
    )
    residual = sim_delta - ratio.unsqueeze(-1) * ref_delta
    lateral = torch.linalg.vector_norm(residual, dim=-1)
    lateral_ratio = lateral / torch.clamp_min(
        torch.linalg.vector_norm(ref_delta, dim=-1), eps
    )
    return ratio, lateral, lateral_ratio


def signed_winding_ratio(
    sim_integrated_ang_vel: torch.Tensor,
    ref_integrated_ang_vel: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reference-axis signed angular traversal ratio."""

    _check_last_dim("sim_integrated_ang_vel", sim_integrated_ang_vel, 3)
    _check_last_dim("ref_integrated_ang_vel", ref_integrated_ang_vel, 3)
    ref_sq = torch.sum(
        ref_integrated_ang_vel * ref_integrated_ang_vel, dim=-1
    )
    return torch.sum(
        sim_integrated_ang_vel * ref_integrated_ang_vel, dim=-1
    ) / torch.clamp_min(ref_sq, eps)


def canonical_motion_name(name: str) -> str:
    """Map filenames and display spellings to a stable evaluator key."""

    key = Path(str(name)).stem.lower().replace("-", "_").replace(" ", "_")
    for prefix in ("humanoid_", "smpl_", "g1_", "go2_"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    aliases = {
        "getup": "getup_facedown",
        "getup_face_down": "getup_facedown",
        "spin_kick": "spinkick",
        "climbing": "climb",
        "climbing_up_down": "climb",
    }
    return aliases.get(key, key)


def _as_bool_component(
    value: torch.Tensor, *, lower: float | None = None, upper: float | None = None
) -> torch.Tensor:
    result = torch.isfinite(value)
    if lower is not None:
        result = torch.logical_and(result, value >= lower)
    if upper is not None:
        result = torch.logical_and(result, value <= upper)
    return result


def compute_completion(
    motion_name: str,
    values: Mapping[str, torch.Tensor],
    thresholds: CompletionThresholds | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a motion-aware physical completion mask.

    All quantities are reference normalized except the explicitly named height
    error.  The returned component masks make every failure auditable instead
    of hiding it in one task-specific scalar.
    """

    th = CompletionThresholds() if thresholds is None else thresholds
    motion = canonical_motion_name(motion_name)
    required = ("survival_ratio",)
    for key in required:
        if key not in values:
            raise KeyError(f"missing completion quantity: {key}")

    survival = _as_bool_component(values["survival_ratio"], lower=th.survival_min)
    components: dict[str, torch.Tensor] = {"survival": survival}

    if motion in ("run", "crawl"):
        progress = _as_bool_component(
            values["displacement_ratio"],
            lower=th.progress_min,
            upper=th.progress_max,
        )
        lateral = _as_bool_component(
            values["lateral_displacement_ratio"], upper=th.lateral_ratio_max
        )
        components.update(progress=progress, lateral=lateral)

    elif motion in ("roll", "backflip", "sideflip"):
        winding = _as_bool_component(
            values["winding_ratio"],
            lower=th.winding_min,
            upper=th.winding_max,
        )
        phase_alignment = _as_bool_component(
            values["final_root_rot_error"],
            upper=th.terminal_orientation_error_max,
        )
        components.update(winding=winding, phase_alignment=phase_alignment)

    elif motion == "spinkick":
        winding = _as_bool_component(
            values["winding_ratio"],
            lower=th.winding_min,
            upper=th.winding_max,
        )
        phase_alignment = _as_bool_component(
            values["final_root_rot_error"],
            upper=th.terminal_orientation_error_max,
        )
        components.update(winding=winding, phase_alignment=phase_alignment)

    elif motion in ("getup_facedown", "getup_faceup"):
        upright = _as_bool_component(values["final_up_dot"], lower=th.upright_min)
        height = _as_bool_component(
            values["final_height_ratio"], lower=th.getup_height_ratio_min
        )
        components.update(upright=upright, recovered_height=height)

    elif motion == "climb":
        progress = _as_bool_component(
            values["displacement_ratio"],
            lower=th.progress_min,
            upper=th.progress_max,
        )
        ascent = _as_bool_component(
            values["max_height_gain_ratio"],
            lower=th.climb_height_gain_ratio_min,
        )
        descent = _as_bool_component(
            values["final_height_error"],
            upper=th.climb_final_height_error_max,
        )
        feet = _as_bool_component(
            values["feet_progress_ratio"],
            lower=th.climb_feet_progress_min,
        )
        components.update(
            progress=progress, ascent=ascent, descent=descent, feet_clear=feet
        )

    elif motion == "jump":
        takeoff = _as_bool_component(
            values["max_height_gain_ratio"],
            lower=th.jump_height_gain_ratio_min,
        )
        upright = _as_bool_component(values["final_up_dot"], lower=th.upright_min)
        height = _as_bool_component(
            values["final_height_ratio"], lower=th.getup_height_ratio_min
        )
        components.update(takeoff=takeoff, landed=upright & height)

    else:
        progress = _as_bool_component(
            values["displacement_ratio"],
            lower=th.progress_min,
            upper=th.progress_max,
        )
        components["progress"] = progress

    complete = torch.ones_like(survival, dtype=torch.bool)
    for component in components.values():
        complete = torch.logical_and(complete, component)
    return complete, components
