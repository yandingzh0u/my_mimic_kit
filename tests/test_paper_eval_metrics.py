import math

import pytest
import torch

from tools.paper_eval.metrics import (
    TRACKING_ERROR_NAMES,
    canonical_motion_name,
    compute_completion,
    compute_tracking_errors,
    projected_motion_metrics,
    quaternion_angle,
    signed_winding_ratio,
)


def test_quaternion_angle_is_sign_invariant():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    same_rotation = -identity
    half_turn_x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert quaternion_angle(identity, same_rotation).item() == pytest.approx(0.0)
    assert quaternion_angle(identity, half_turn_x).item() == pytest.approx(math.pi)


def test_tracking_errors_and_explicit_paper_position_metric():
    # Three bodies: the simulated root is translated by one metre, while one
    # root-relative non-root body is another metre away from its reference.
    root_pos = torch.tensor([[1.0, 0.0, 0.0]])
    ref_root_pos = torch.zeros_like(root_pos)
    body_pos = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    ref_body_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    root_rot = identity.view(1, 4)
    body_rot = identity.view(1, 1, 4).repeat(1, 3, 1)
    dof_vel = torch.tensor([[1.0, -1.0]])
    root_vel = torch.tensor([[1.0, 2.0, 3.0]])
    root_ang_vel = torch.tensor([[2.0, 0.0, -2.0]])

    errors = compute_tracking_errors(
        root_pos=root_pos,
        root_rot=root_rot,
        body_pos=body_pos,
        body_rot=body_rot,
        dof_vel=dof_vel,
        root_vel=root_vel,
        root_ang_vel=root_ang_vel,
        ref_root_pos=ref_root_pos,
        ref_root_rot=root_rot,
        ref_body_pos=ref_body_pos,
        ref_body_rot=body_rot,
        ref_dof_vel=torch.zeros_like(dof_vel),
        ref_root_vel=torch.zeros_like(root_vel),
        ref_root_ang_vel=torch.zeros_like(root_ang_vel),
    )

    assert tuple(errors) == TRACKING_ERROR_NAMES
    assert errors["root_pos_err"].item() == pytest.approx(1.0)
    assert errors["body_pos_err"].item() == pytest.approx(1.0 / 3.0)
    # (root error 1 + root-relative non-root errors [1, 0]) / 3.
    assert errors["paper_pos_err"].item() == pytest.approx(2.0 / 3.0)
    assert errors["dof_vel_err"].item() == pytest.approx(1.0)
    assert errors["root_vel_err"].item() == pytest.approx(2.0)
    assert errors["root_ang_vel_err"].item() == pytest.approx(4.0 / 3.0)


def test_projected_progress_and_signed_winding():
    sim_delta = torch.tensor([[2.0, 1.0, 0.0]])
    ref_delta = torch.tensor([[4.0, 0.0, 0.0]])
    ratio, lateral, lateral_ratio = projected_motion_metrics(sim_delta, ref_delta)
    assert ratio.item() == pytest.approx(0.5)
    assert lateral.item() == pytest.approx(1.0)
    assert lateral_ratio.item() == pytest.approx(0.25)

    ref_turn = torch.tensor([[0.0, 0.0, 2.0 * math.pi]])
    assert signed_winding_ratio(ref_turn, ref_turn).item() == pytest.approx(1.0)
    assert signed_winding_ratio(-ref_turn, ref_turn).item() == pytest.approx(-1.0)


def _base_completion_values(batch_size=2):
    return {
        "survival_ratio": torch.ones(batch_size),
        "displacement_ratio": torch.ones(batch_size),
        "lateral_displacement_ratio": torch.zeros(batch_size),
        "winding_ratio": torch.ones(batch_size),
        "final_up_dot": torch.ones(batch_size),
        "final_root_rot_error": torch.zeros(batch_size),
        "final_height_ratio": torch.ones(batch_size),
        "max_height_gain_ratio": torch.ones(batch_size),
        "final_height_error": torch.zeros(batch_size),
        "feet_progress_ratio": torch.ones(batch_size),
    }


@pytest.mark.parametrize(
    "motion",
    ["run", "crawl", "roll", "backflip", "sideflip", "spinkick", "getup_facedown", "climb", "jump"],
)
def test_motion_aware_completion_accepts_reference_like_outcomes(motion):
    complete, components = compute_completion(motion, _base_completion_values())
    assert torch.all(complete)
    assert "survival" in components


def test_cyclic_completion_uses_reference_phase_not_world_upright():
    values = _base_completion_values(batch_size=1)
    # A 10-second evaluation can end while a wrapped aerial reference is
    # inverted. Perfectly matching that reference must still count as aligned.
    values["final_up_dot"] = torch.tensor([-1.0])
    values["final_root_rot_error"] = torch.tensor([0.0])
    complete, components = compute_completion("backflip", values)
    assert complete.item()
    assert components["phase_alignment"].item()


@pytest.mark.parametrize(
    ("motion", "failed_quantity"),
    [
        ("run", "displacement_ratio"),
        ("crawl", "lateral_displacement_ratio"),
        ("roll", "winding_ratio"),
        ("backflip", "final_root_rot_error"),
        ("spinkick", "winding_ratio"),
        ("getup_facedown", "final_height_ratio"),
        ("climb", "max_height_gain_ratio"),
        ("climb", "final_height_error"),
        ("climb", "feet_progress_ratio"),
    ],
)
def test_motion_aware_completion_rejects_named_failure(motion, failed_quantity):
    values = _base_completion_values(batch_size=1)
    values[failed_quantity] = torch.tensor(
        [
            1.0
            if failed_quantity in (
                "final_height_error",
                "lateral_displacement_ratio",
                "final_root_rot_error",
            )
            else 0.0
        ]
    )
    complete, _ = compute_completion(motion, values)
    assert not complete.item()


def test_canonical_motion_names_cover_paper_files():
    assert canonical_motion_name("data/motions/humanoid/humanoid_run.pkl") == "run"
    assert canonical_motion_name("humanoid_climbing_up_down.pkl") == "climb"
    assert canonical_motion_name("Getup Face Down") == "getup_facedown"
