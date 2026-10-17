import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_group_coverage_probe import covered_batch, isolate_groups, reward_contraction


def test_isolation_preserves_selected_group_and_removes_others():
    diff = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    masks = torch.tensor([[True, True, False, False],
                          [False, False, True, True]])
    isolated, valid = isolate_groups(diff, masks, torch.tensor([0, 1]))
    torch.testing.assert_close(isolated, torch.tensor([[1., 2., 0., 0.],
                                                      [0., 0., 7., 8.]]))
    assert valid.all()


def test_isolation_does_not_create_zero_negative():
    diff = torch.tensor([[0., 0., 3., 4.]])
    masks = torch.tensor([[True, True, False, False],
                          [False, False, True, True]])
    isolated, valid = isolate_groups(diff, masks, torch.tensor([0]))
    assert not valid.any()
    torch.testing.assert_close(isolated, diff)


def test_coverage_preserves_half_of_current_and_replay_without_mutation():
    negative = torch.ones(28, 7)
    before = negative.clone()
    masks = torch.eye(7, dtype=torch.bool)
    covered, valid = covered_batch(negative, masks, step=3)
    torch.testing.assert_close(negative, before)
    torch.testing.assert_close(covered[1::2], negative[1::2])
    assert valid.all()
    assert torch.all(covered[0::2].ne(0).sum(-1) == 1)
    assert torch.all(covered[:14:2].ne(0).sum(0) == 1)
    assert torch.all(covered[14::2].ne(0).sum(0) == 1)


class RadialProbe(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias
        self._disc_layers = torch.nn.Module()
        for group in range(7):
            self._disc_layers.register_buffer(
                f"group_indices_{group}", torch.tensor([group]))

    def forward(self, diff):
        return self.bias - diff.abs().sum(-1)


def test_downward_logit_shift_can_reduce_actual_reward_response():
    diff = torch.full((8, 7), .1)
    original = reward_contraction(RadialProbe(-1), diff)
    shifted = reward_contraction(RadialProbe(-2), diff)
    for group in original:
        assert original[group]["correct_fraction"] == 1
        assert shifted[group]["correct_fraction"] == 1
        assert (0 < shifted[group]["mean_reward_gain"]
                < original[group]["mean_reward_gain"])
