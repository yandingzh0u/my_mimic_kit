import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
if str(MIMICKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_ROOT))

from learning.add_agent import calc_disc_gradient_penalty


def test_add_gradient_penalty_averages_negative_and_positive_sides():
    neg_diff = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                            requires_grad=True)
    pos_diff = torch.tensor([[0.0, 0.0]], requires_grad=True)

    # Negative gradient is [2, 3], positive gradient is [5, 7].
    neg_logit = 2.0 * neg_diff[:, 0] + 3.0 * neg_diff[:, 1]
    pos_logit = 5.0 * pos_diff[:, 0] + 7.0 * pos_diff[:, 1]

    penalty, neg_penalty, pos_penalty = calc_disc_gradient_penalty(
        neg_logit, neg_diff, pos_logit, pos_diff)

    torch.testing.assert_close(neg_penalty, torch.tensor(13.0))
    torch.testing.assert_close(pos_penalty, torch.tensor(74.0))
    torch.testing.assert_close(penalty, torch.tensor(43.5))
