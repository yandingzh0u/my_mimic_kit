from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_hlgauss_critic_screen import centers, encode, mc_returns, support_from_training


def test_mc_terminal_included_post_terminal_excluded():
    reward = torch.tensor([[1., 2.], [3., 4.], [999., 6.]])
    active = torch.tensor([[True, True], [True, True], [False, True]])
    torch.testing.assert_close(mc_returns(reward, active, .5),
                               torch.tensor([[2.5, 5.5], [3., 7.], [0., 6.]]))


def test_gaussian_mass_is_finite_nonnegative_and_normalized_at_edges():
    edges = torch.linspace(-2, 2, 152)
    labels = torch.tensor([-100., -2., -.1, 0., 2., 100.])
    probabilities = encode(labels, edges)
    assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(len(labels)))
    torch.testing.assert_close(probabilities[0], probabilities[1])
    torch.testing.assert_close(probabilities[-1], probabilities[-2])


def test_training_only_support_has_three_sigma_padding_and_small_decoding_bias():
    training = torch.tensor([0., 10., 20., 30.])
    edges = support_from_training(training)
    sigma = .75 * (edges[1] - edges[0])
    torch.testing.assert_close(training.min() - edges[0], 3 * sigma)
    torch.testing.assert_close(edges[-1] - training.max(), 3 * sigma)
    decoded = (encode(training, edges) * centers(edges)).sum(-1)
    assert (decoded - training).abs().max() < .01


def test_literal_global_support_has_a_nonzero_minimum_decodable_value():
    edges = torch.linspace(0, 1842., 152)
    decoded_zero = (encode(torch.tensor([0.]), edges) * centers(edges)).sum(-1)
    assert float(decoded_zero) >= float(centers(edges)[0]) > 6.


def test_invalid_support_is_rejected():
    for labels in (torch.ones(4), torch.tensor([0., float("nan")]), torch.tensor([1.])):
        with pytest.raises(ValueError):
            support_from_training(labels)
