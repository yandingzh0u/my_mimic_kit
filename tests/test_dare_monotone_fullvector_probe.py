import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_monotone_fullvector_probe import VectorControl


GROUPS = [('position', [0, 1, 2]), ('pose', [3, 4, 5, 6]), ('velocity', [7, 8])]


def test_direction_ablation_matches_parameter_count_and_initial_rewards():
    pooled = VectorControl(GROUPS, True, width=8, depth=4)
    full = VectorControl(GROUPS, False, width=8, depth=4)
    full.load_state_dict(pooled.state_dict())
    diff = torch.randn(32, 9)
    assert sum(p.numel() for p in pooled.parameters()) == sum(p.numel() for p in full.parameters())
    torch.testing.assert_close(pooled.reward(diff), full.reward(diff), rtol=0, atol=0)
    torch.testing.assert_close(pooled.features(diff).norm(dim=-1), full.features(diff).norm(dim=-1))


def test_full_vector_can_represent_direction_without_breaking_contraction_bound():
    model = VectorControl(GROUPS, False, width=8, depth=4).double()
    torch.nn.init.normal_(model.embedding.weight)
    diff = torch.randn(64, 9, dtype=torch.double)
    cost = model(diff).squeeze(-1)
    for _, ids in GROUPS:
        changed = diff.clone()
        changed[:, ids] *= .5
        magnitude = diff[:, ids].norm(dim=-1) / len(ids) ** .5
        gain = cost - model(changed).squeeze(-1)
        assert (gain >= .5 * magnitude * model.slope_bounds()[0] - 1e-8).all()
    changed = diff.clone()
    changed[:, GROUPS[1][1]] *= -1
    assert (cost - model(changed).squeeze(-1)).abs().max() > 1e-4


def test_pooled_lift_still_discards_direction_after_training():
    model = VectorControl(GROUPS, True, width=8, depth=4)
    torch.nn.init.normal_(model.embedding.weight)
    diff = torch.randn(16, 9)
    changed = diff.clone()
    changed[:, GROUPS[1][1]] *= -1
    torch.testing.assert_close(model.reward(diff), model.reward(changed), rtol=0, atol=0)
