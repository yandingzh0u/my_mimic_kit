"""Fixed-covariance specialization of ICLR 2021 trust-region layers.

Independent implementation of Otto et al., sections 4.1 and 4.4. The bound
here is the Gaussian mean KL (including 1/2), matching the official KL-layer
code, not the unscaled Mahalanobis distance written in paper equation (5).
This constrains the projected distribution, NOT the deployed raw network.
"""
import math

import torch


def validate_parameters(kl_bound, regression_weight):
    if not math.isfinite(kl_bound) or kl_bound <= 0:
        raise ValueError("Mean KL bound must be finite and positive")
    if not math.isfinite(regression_weight) or regression_weight <= 0:
        raise ValueError("Projection regression weight must be finite and positive")


def mean_kl(mean, reference_mean, logstd):
    """Exact KL for diagonal Gaussians with an identical fixed covariance."""
    return .5 * ((mean - reference_mean) * torch.exp(-logstd)).square().sum(-1)


def project_mean(mean, old_mean, logstd, kl_bound):
    """Differentiable Mahalanobis-ball projection, with old policy detached.

    The denominator is clamped at the BOUND, so the identity region also has
    a finite derivative at zero. Outside it, only the radial component of the
    policy gradient is removed; tangential learning remains possible.
    """
    if not math.isfinite(kl_bound) or kl_bound <= 0:
        raise ValueError("Mean KL bound must be finite and positive")
    old_mean, logstd = old_mean.detach(), logstd.detach()
    distance = mean_kl(mean, old_mean, logstd)
    scale = torch.sqrt(kl_bound / torch.clamp_min(distance, kl_bound))
    candidate = old_mean + scale.unsqueeze(-1) * (mean - old_mean)
    projected = torch.where((distance <= kl_bound).unsqueeze(-1), mean, candidate)
    return projected, distance


def projection_regression(mean, projected_mean, logstd):
    """Amortization penalty; the projected target MUST be stop-gradient."""
    return mean_kl(mean, projected_mean.detach(), logstd.detach())
