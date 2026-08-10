from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from learning.skill_encoder.motion_features import build_motion_dynamic_features
from learning.skill_encoder.skill_encoder_model import (
    LabelFreeSkillEncoder,
    embedding_diagnostics,
    vicreg_loss,
)
from tools.skill_encoder.motion_view_data import (
    build_nonoverlapping_view_times,
    required_context_span,
    stratified_family_split,
)


def test_nonoverlapping_h20_view_schedule():
    starts = torch.tensor([0.0, 1.5])
    view_a, view_b = build_nonoverlapping_view_times(starts, 20, 30)

    assert view_a.shape == view_b.shape == (2, 20)
    assert torch.all(view_b[:, 0] > view_a[:, -1])
    torch.testing.assert_close(view_b[:, 0] - view_a[:, -1], torch.full((2,), 1 / 30))
    torch.testing.assert_close(view_b[:, -1] - starts, torch.full((2,), 39 / 30))
    assert required_context_span(20, 30) == 40 / 30


def test_stratified_split_keeps_audit_and_training_families_per_tag():
    family_to_ids = {
        **{f"run-{index}": [index] for index in range(10)},
        **{
            f"walk-{index}": [10 + 2 * index, 11 + 2 * index]
            for index in range(4)
        },
    }
    family_to_tag = {
        family: ("walk" if family.startswith("walk") else "run")
        for family in family_to_ids
    }
    train_ids, heldout_ids, train_families, heldout_families = stratified_family_split(
        family_to_ids, family_to_tag, heldout_fraction=0.2, seed=3
    )

    assert set(train_ids).isdisjoint(heldout_ids)
    assert set(train_families).isdisjoint(heldout_families)
    assert set(train_ids + heldout_ids) == set(range(18))
    for tag in ("run", "walk"):
        assert sum(family_to_tag[family] == tag for family in heldout_families) >= 2
        assert sum(family_to_tag[family] == tag for family in train_families) >= 1


def test_dynamic_feature_schema_uses_only_velocities():
    batch_size, steps = 3, 20
    root_rot = torch.zeros(batch_size, steps, 4)
    root_rot[..., -1] = 1.0
    root_vel = torch.randn(batch_size, steps, 3)
    root_ang_vel = torch.randn(batch_size, steps, 3)
    dof_vel = torch.randn(batch_size, steps, 28)
    foot_pos = torch.zeros(batch_size, steps + 1, 2, 3)
    foot_pos[..., 2] = 0.05

    features = build_motion_dynamic_features(
        root_rot, root_vel, root_ang_vel, dof_vel, foot_pos, timestep=1 / 30
    )

    assert features.shape == (batch_size, steps, 44)
    torch.testing.assert_close(features[..., :3], root_vel)
    torch.testing.assert_close(features[..., 3:6], root_ang_vel)
    torch.testing.assert_close(features[..., 6:34], dof_vel)
    torch.testing.assert_close(features[..., 34:40], torch.zeros_like(features[..., 34:40]))
    torch.testing.assert_close(features[..., 40:42], torch.ones_like(features[..., 40:42]))
    torch.testing.assert_close(features[..., 42:44], torch.zeros_like(features[..., 42:44]))


def test_encoder_trains_raw_y_and_runtime_z_is_unit_norm():
    model = LabelFreeSkillEncoder(44, embedding_dim=8, hidden_dim=32, num_layers=2)
    features = torch.randn(16, 20, 44)
    y = model(features)
    z = model.runtime_z(features)

    assert y.shape == z.shape == (16, 8)
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(16), rtol=1e-5, atol=1e-5)
    loss, parts = vicreg_loss(y, model(features + 0.01 * torch.randn_like(features)))
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"invariance", "variance", "covariance"}
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_noncollapse_diagnostics_detect_rank_zero_and_full_rank():
    collapsed = embedding_diagnostics(torch.ones(32, 8))
    generator = torch.Generator().manual_seed(4)
    diverse = embedding_diagnostics(torch.randn(1024, 8, generator=generator))

    assert collapsed["effective_rank"] == 0.0
    assert collapsed["min_dim_std"] == 0.0
    assert diverse["effective_rank"] > 7.5
    assert diverse["min_dim_std"] > 0.9
