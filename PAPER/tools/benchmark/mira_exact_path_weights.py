"""Experimental post-update constraint for the scalar nonnegative ReLU energy."""

from collections.abc import Sequence

import torch


def _validate_chain(weights: Sequence[torch.Tensor]) -> None:
    if not weights:
        raise ValueError("Expected at least one linear weight")
    for i, weight in enumerate(weights):
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("Expected nonempty linear weight matrices")
        if i and weights[i - 1].shape[0] != weight.shape[1]:
            raise ValueError("Expected weights in input-to-output order with matching dimensions")
    if weights[-1].shape[0] != 1:
        raise ValueError("The last layer must have one output")


def _path_gain(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    # From the output, every intermediate product is a row, never a matrix.
    path = weights[-1][0].to(torch.float64)
    for i in range(len(weights) - 2, -1, -1):
        path = path @ weights[i].to(torch.float64)
    # Avoid squaring unusually large path contributions before rescaling.
    scale = path.abs().amax().clamp_min(1)
    return torch.linalg.vector_norm(path / scale) * scale


@torch.no_grad()
def exact_positive_path_gain(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the float64 scalar gain of finite nonnegative linear weights.

    Give the matrices in input-to-output order, with one output in the last
    layer. For ReLU between these layers, this is the exact Euclidean energy
    Lipschitz constant on the full nonnegative orthant, for any finite biases.
    It need not be attained within a bounded subset of that orthant.
    """
    _validate_chain(weights)
    return _path_gain(weights)


@torch.no_grad()
def enforce_nonnegative_path_contraction_(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Clip negatives, then repair the path gain by shrinking only the last row.

    The return value is the gain after clipping and before last-layer scaling.
    This is a feasible update, not a Euclidean projection of all the weights.
    Biases and optimizer state are unchanged.
    """
    _validate_chain(weights)
    for weight in weights:
        weight.clamp_min_(0)
    gain = _path_gain(weights)
    last = weights[-1]
    last.copy_(last.to(torch.float64) / gain.clamp_min(1))
    return gain
