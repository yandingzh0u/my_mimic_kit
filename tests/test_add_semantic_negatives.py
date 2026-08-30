import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mimickit"))

from learning.add_agent import build_radius_balanced_semantic_negatives


GROUPS = (
    ("first", (0, 1)),
    ("second", (2, 3, 4)),
)


def test_semantic_negatives_preserve_full_radius_and_isolate_groups():
    diff = torch.tensor([
        [3.0, 4.0, 0.0, 0.0, 12.0],
        [1.0, 2.0, 2.0, 3.0, 6.0],
    ])

    semantic, valid = build_radius_balanced_semantic_negatives(
        diff, GROUPS)

    assert semantic.shape == (2, 2, 5)
    assert torch.all(valid)
    expected_radius = torch.linalg.vector_norm(diff, dim=-1)
    actual_radius = torch.linalg.vector_norm(semantic, dim=-1)
    torch.testing.assert_close(
        actual_radius, expected_radius.unsqueeze(-1).expand_as(actual_radius))
    assert torch.count_nonzero(semantic[:, 0, 2:]) == 0
    assert torch.count_nonzero(semantic[:, 1, :2]) == 0


def test_zero_group_is_invalid_and_stays_zero():
    diff = torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0]])

    semantic, valid = build_radius_balanced_semantic_negatives(
        diff, GROUPS)

    assert valid.tolist() == [[True, False]]
    torch.testing.assert_close(semantic[0, 1], torch.zeros(5))
    assert torch.isfinite(semantic).all()


def test_semantic_negative_builder_preserves_input_gradient():
    diff = torch.tensor(
        [[3.0, 4.0, 5.0, 0.0, 12.0]], requires_grad=True)

    semantic, _ = build_radius_balanced_semantic_negatives(diff, GROUPS)
    semantic.square().sum().backward()

    assert diff.grad is not None
    assert torch.isfinite(diff.grad).all()
    assert torch.count_nonzero(diff.grad) > 0
