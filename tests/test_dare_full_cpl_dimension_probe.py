import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_full_cpl_dimension_probe import DimensionBalancedShared
from dare_semantic_heads_probe import SharedCold


def test_dimension_rule_preserves_full_noncontiguous_coordinates_and_parameter_count():
    groups = [('position', [0, 4, 5]), ('pose', [1, 2, 3, 6]), ('velocity', [7, 8])]
    original = SharedCold(groups)
    scaled = DimensionBalancedShared(groups)
    copied = scaled.load_state_dict(original.state_dict(), strict=False)
    assert copied.missing_keys == ['group_input_scale']
    assert not copied.unexpected_keys
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in scaled.parameters())
    diff = torch.randn(16, 9)
    for _, ids in groups:
        torch.testing.assert_close(diff[:, ids] * scaled.group_input_scale[ids],
                                   diff[:, ids] / len(ids) ** .5)
    original.eval()
    scaled.eval()
    torch.testing.assert_close(scaled.logits(diff), original.logits(diff * scaled.group_input_scale),
                               rtol=0, atol=0)
    torch.testing.assert_close(scaled.logits(torch.zeros_like(diff)), original.logits(torch.zeros_like(diff)),
                               rtol=0, atol=0)
