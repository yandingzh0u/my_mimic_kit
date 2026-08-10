from __future__ import annotations

import torch

import util.torch_util as torch_util


def make_feature_schema(
    foot_body_names=("right_foot", "left_foot"),
    foot_body_ids=(11, 14),
    ground_height=0.0,
    contact_height_threshold=0.08,
    contact_speed_threshold=0.4,
):
    return {
        "name": "local_motion_dynamics_with_kinematic_contact_v1",
        "feature_dim": 44,
        "groups": [
            {"name": "root_linear_velocity_local", "start": 0, "end": 3},
            {"name": "root_angular_velocity_local", "start": 3, "end": 6},
            {"name": "joint_dof_velocity", "start": 6, "end": 34},
            {"name": "foot_velocity_local", "start": 34, "end": 40},
            {"name": "kinematic_contact_proxy", "start": 40, "end": 42},
            {"name": "delta_kinematic_contact_proxy", "start": 42, "end": 44},
        ],
        "foot_body_names": list(foot_body_names),
        "foot_body_ids": list(foot_body_ids),
        "contact_proxy": {
            "native_contact_label": False,
            "definition": "(foot_z < ground_height + height_threshold) and (world_3d_speed < speed_threshold)",
            "ground_height": float(ground_height),
            "height_threshold": float(contact_height_threshold),
            "speed_threshold": float(contact_speed_threshold),
            "velocity_difference": "forward difference from H+1 kinematic states",
            "delta_definition": "first step zero; subsequent first difference",
        },
        "heading_frame": "per-step root heading",
        "excluded": [
            "root position",
            "root height as an encoder feature",
            "joint pose",
            "target/task observation",
            "phase",
            "native contact labels (not present in mocap)",
            "motion filename or semantic label",
        ],
    }


FEATURE_SCHEMA = make_feature_schema()


def build_motion_dynamic_features(
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    dof_vel: torch.Tensor,
    foot_pos: torch.Tensor,
    timestep: float,
    ground_height: float = 0.0,
    contact_height_threshold: float = 0.08,
    contact_speed_threshold: float = 0.4,
) -> torch.Tensor:
    """Build the exact 44-D per-step training/runtime feature representation.

    Inputs use shapes ``[B,H,4]``, ``[B,H,3]``, ``[B,H,3]``, and
    ``[B,H,28]`` plus foot positions ``[B,H+1,2,3]``. Velocities are
    expressed in each step's root-heading frame. Contact is a deterministic
    kinematic proxy, not a native mocap label.
    """
    if root_rot.ndim != 3 or root_rot.shape[-1] != 4:
        raise ValueError("root_rot must have shape [B,H,4]")
    expected_prefix = root_rot.shape[:2]
    expected_dims = ((root_vel, 3), (root_ang_vel, 3), (dof_vel, 28))
    for tensor, feature_dim in expected_dims:
        if tensor.shape[:2] != expected_prefix or tensor.shape[-1] != feature_dim:
            raise ValueError(
                f"dynamic state tensor must have shape [B,H,{feature_dim}], got {tuple(tensor.shape)}"
            )
    if foot_pos.shape != (root_rot.shape[0], root_rot.shape[1] + 1, 2, 3):
        raise ValueError("foot_pos must have shape [B,H+1,2,3]")
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    tensors = (root_rot, root_vel, root_ang_vel, dof_vel, foot_pos)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise TypeError("dynamic state tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("dynamic state tensors must be finite")

    batch_size, num_steps = expected_prefix
    heading_grid = torch_util.calc_heading_quat_inv(root_rot).reshape(-1, 4)
    local_root_vel = torch_util.quat_rotate(
        heading_grid, root_vel.reshape(-1, 3)
    ).reshape(batch_size, num_steps, 3)
    local_root_ang_vel = torch_util.quat_rotate(
        heading_grid, root_ang_vel.reshape(-1, 3)
    ).reshape(batch_size, num_steps, 3)

    foot_velocity_world = (foot_pos[:, 1:] - foot_pos[:, :-1]) / float(timestep)
    foot_heading_grid = heading_grid.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
    foot_velocity_local = torch_util.quat_rotate(
        foot_heading_grid, foot_velocity_world.reshape(-1, 3)
    ).reshape(batch_size, num_steps, 2, 3)
    foot_height = foot_pos[:, :-1, :, 2]
    foot_speed = foot_velocity_world.norm(dim=-1)
    contact = (
        (foot_height < float(ground_height) + float(contact_height_threshold))
        & (foot_speed < float(contact_speed_threshold))
    ).to(root_vel.dtype)
    delta_contact = torch.zeros_like(contact)
    delta_contact[:, 1:] = contact[:, 1:] - contact[:, :-1]

    features = torch.cat(
        (
            local_root_vel,
            local_root_ang_vel,
            dof_vel,
            foot_velocity_local.reshape(batch_size, num_steps, 6),
            contact,
            delta_contact,
        ),
        dim=-1,
    )
    if features.shape[-1] != 44:
        raise RuntimeError("dynamic feature schema dimension mismatch")
    return features
