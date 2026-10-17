from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_action_gradient_physical import (
    KINDS, calibration_actions, comparison, coordinate_derivative,
    proposal_vectors, standardized_ascent,
)


def test_action_coordinate_jacobian_and_normalizer_units():
    std = torch.tensor([.05, .1, .2], dtype=torch.double)
    delta = calibration_actions(std, replicas=2)
    jacobian = torch.tensor([[1., 2.], [-1., 3.], [4., .5]], dtype=torch.double)
    output = delta @ jacobian + torch.tensor([10., -5.], dtype=torch.double)
    torch.testing.assert_close(coordinate_derivative(output, std, 2), jacobian,
                               rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(coordinate_derivative(output[:, 0], std, 2), jacobian[:, 0])
    torch.testing.assert_close(delta[:2], torch.zeros(2, 3, dtype=torch.double))


def test_fixed_standardized_ascent_maximizes_local_linear_objective():
    torch.manual_seed(1)
    gradient = torch.randn(28)
    std = torch.linspace(.03, .1, 28)
    step = standardized_ascent(gradient, std)
    torch.testing.assert_close((step / std).norm(), torch.tensor(1.))
    torch.testing.assert_close(step @ gradient, (gradient * std).norm())
    torch.testing.assert_close(standardized_ascent(torch.zeros(28), std), torch.zeros(28))
    with pytest.raises(ValueError):
        standardized_ascent(gradient * float("nan"), std)


def test_original_discounted_value_and_negative_control_direction():
    vectors, result = proposal_vectors(torch.tensor([1., -1.]), torch.eye(2),
        torch.tensor([1., -1.]), torch.tensor([2., 3.]), torch.tensor([4., 5.]),
        torch.tensor([.05, .1]), .99)
    expected = standardized_ascent(torch.tensor([1., -1.]) + .99 * torch.tensor([2., 3.]),
                                   torch.tensor([.05, .1]))
    torch.testing.assert_close(vectors["cpl_value"], expected)
    torch.testing.assert_close(vectors["negative_cpl_value"], -expected)
    torch.testing.assert_close(vectors["cpl_reward"], vectors["fd_reward"])
    assert result["cpl_vs_fd_reward_cosine"] == pytest.approx(1.)
    assert tuple(vectors) == KINDS


def traces(replicas=2):
    n = replicas * len(KINDS)
    return dict(active=np.ones((3, n), dtype=bool), done=np.array([[0]*n, [0]*n, [2]*n]),
                errors=np.ones((3, n, 7)), reward=np.ones((3, n)))


def test_physics_validation_rejects_tradeoffs_and_no_update_drift():
    base, repeat, changed = traces(), traces(), traces()
    changed["errors"][:, 2:4] *= .9
    changed["errors"][:, 4:6] *= .9
    changed["errors"][:, 4:6, 0] = 1.1
    result = comparison(base, repeat, changed, 2, .99, True)
    assert result["cpl_reward"]["all_seven_full_not_worse"]
    assert not result["cpl_value"]["all_seven_full_not_worse"]
    repeat["errors"][:, 2:4] += .01
    result = comparison(base, repeat, changed, 2, .99, True)
    assert result["cpl_reward"]["valid_replicas"] == 0


def test_third_untouched_group_and_initial_state_are_required():
    base, repeat, changed = traces(), traces(), traces()
    changed["errors"][:, :2] += .01
    result = comparison(base, repeat, changed, 2, .99, True)
    assert all(row["valid_replicas"] == 0 for row in result.values())
    result = comparison(base, repeat, repeat, 2, .99, False)
    assert all(row["valid_replicas"] == 0 for row in result.values())


def test_invalid_post_termination_errors_never_enter_comparisons():
    base, repeat, changed = traces(), traces(), traces()
    for value in (base, repeat, changed):
        value["active"][2] = False
        value["done"][1] = 2
        value["errors"][2] = np.nan
        value["reward"][2] = np.nan
    changed["errors"][:2, 2:4] *= .9
    result = comparison(base, repeat, changed, 2, .99, True)
    assert result["cpl_reward"]["all_seven_full_not_worse"]
    assert result["cpl_reward"]["errors"]["full"]["root_pos"]["relative"] == pytest.approx(-.1)
