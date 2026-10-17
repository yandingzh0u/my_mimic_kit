"""Contracts for fixed-protocol DARE preference learning; no simulator."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_preference_stress import (
    SingleReward, case_records, discounted_score, pack_records,
    physical_label, preference_loss,
)
from temporal_dare_stress import TemporalReward


def test_bradley_terry_gradient_increases_preferred_return():
    winner = torch.tensor([-.3, .2], requires_grad=True)
    loser = torch.tensor([.5, -.1], requires_grad=True)
    loss = preference_loss(winner, loser)
    loss.backward()
    assert (winner.grad < 0).all() and (loser.grad > 0).all()
    torch.testing.assert_close(preference_loss(winner + 10, loser + 10), loss)
    torch.testing.assert_close(preference_loss(winner, loser, torch.ones(2)), loss)


def test_discounted_score_masks_nan_and_averages_replicas_not_time():
    reward = torch.tensor([[[1., 2., float("nan")], [3., 4., 5.]]])
    weights = torch.tensor([[[1., .5, 0.], [1., .5, .25]]])
    torch.testing.assert_close(discounted_score(reward, weights), torch.tensor([4.125]))


def test_physical_label_rejects_tradeoffs_noise_and_shorter_survival():
    base = np.ones(7)
    changed = base * .9
    zeros = np.zeros(7)
    lengths = np.array([10, 10])
    assert physical_label(base, changed, zeros, lengths, lengths) == 1
    assert physical_label(changed, base, zeros, lengths, lengths) == -1
    changed[0] = 1.1
    assert physical_label(base, changed, zeros, lengths, lengths) == 0
    assert physical_label(base, base * .9, np.ones(7), lengths, lengths) == 0
    assert physical_label(base, base * .9, zeros, lengths, lengths - 1) == 0


def test_single_reward_preserves_archive_function_and_has_reward_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(1061640)
    base = TemporalReward([("a", (0, 1, 2))]).eval()
    model = SingleReward(base).eval()
    inputs = torch.randn(8, 172, requires_grad=True)
    pairs = torch.stack((inputs, torch.zeros_like(inputs)), 1)
    torch.testing.assert_close(model.reward(inputs), base.reward(pairs), atol=0., rtol=0.)
    model.reward(inputs).sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0


def test_case_windows_exclude_post_done_and_failed_repeat_controls():
    shape = (5, 4)
    active = np.ones(shape, dtype=bool)
    active[3:, :] = False
    base = dict(active=active, norm_diff=np.ones((*shape, 172), dtype=np.float32),
                errors=np.ones((*shape, 7), dtype=np.float32))
    repeat = {k: v.copy() for k, v in base.items()}
    changed = {k: v.copy() for k, v in base.items()}
    changed["errors"][:, 2:] *= .9
    changed["norm_diff"][:, 2:] *= .9
    # Explicit poison after done must not enter gradients or scoring.
    for value in (base, repeat, changed):
        value["errors"][3:] = np.nan
        value["norm_diff"][3:] = np.nan
    records, excluded = case_records(base, repeat, changed, dict(seed=1), 2,
                                    ["unchanged", "changed"], horizon=4)
    assert len(records) == 1 and not excluded
    packed = pack_records(records)
    # Physical improvement is the winner: do not silently swap BT labels.
    torch.testing.assert_close(packed["inputs"][0, 0, :, :3], torch.full((2, 3, 172), .9))
    torch.testing.assert_close(packed["inputs"][0, 1, :, :3], torch.ones(2, 3, 172))
    assert torch.isfinite(packed["inputs"]).all()
    assert (packed["weights"][..., 3:] == 0).all()
    repeat["norm_diff"][0, 2, 0] += .1
    records, excluded = case_records(base, repeat, changed, dict(seed=1), 2,
                                    ["unchanged", "changed"], horizon=4)
    assert not records and len(excluded) == 1


def test_worse_intervention_keeps_baseline_as_winner():
    active = np.ones((3, 2), dtype=bool)
    base = dict(active=active, norm_diff=np.ones((3, 2, 172), dtype=np.float32),
                errors=np.ones((3, 2, 7), dtype=np.float32))
    repeat = {k: v.copy() for k, v in base.items()}
    changed = {k: v.copy() for k, v in base.items()}
    changed["errors"] *= 1.1
    changed["norm_diff"] *= 1.1
    records, _ = case_records(base, repeat, changed, {}, 2, ["changed"])
    assert len(records) == 1
    torch.testing.assert_close(records[0]["inputs"][0], torch.ones(2, 3, 172))
    torch.testing.assert_close(records[0]["inputs"][1], torch.full((2, 3, 172), 1.1))
