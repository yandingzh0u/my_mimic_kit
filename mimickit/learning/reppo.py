"""Opt-in REPPO building blocks; never imported by the production PPO agent.

Reference: arXiv:2507.11019v4 / reppo-rl/rsl_rl, revision
e0130cd26ef4679c43b2aea477b34ee7a52bbc8d. The diagnostic adapter retains
DARE's Actor initialization, not the paper's from-scratch architecture.
"""
import contextlib
import math

import torch
from torch import nn


def gaussian_labels(target, edges, sigma_ratio=.75):
    """Integrated HL-Gauss labels. Callers must report out-of-support labels."""
    if edges.ndim != 1 or len(edges) < 3 or not torch.all(edges[1:] > edges[:-1]):
        raise ValueError("Strictly increasing categorical support required")
    if sigma_ratio <= 0 or not torch.isfinite(target).all():
        raise ValueError("Finite targets and positive smoothing required")
    sigma = sigma_ratio * (edges[1] - edges[0])
    clipped = target.clamp(edges[0], edges[-1])
    cdf = torch.erf((edges - clipped[..., None]) / (math.sqrt(2) * sigma))
    mass = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)


class CategoricalQ(nn.Module):
    """One action-conditioned critic, LayerNorm/SiLU + HL-Gauss (151 bins)."""
    def __init__(self, obs_dim, action_dim, low, high, width=512, bins=151):
        super().__init__()
        if high <= low:
            raise ValueError("Nondegenerate support required")
        layers = []
        incoming = obs_dim + action_dim
        for _ in range(4):
            layers.extend((nn.Linear(incoming, width), nn.LayerNorm(width), nn.SiLU()))
            incoming = width
        self.net = nn.Sequential(*layers, nn.Linear(width, bins))
        self.register_buffer("edges", torch.linspace(low, high, bins + 1))

    def forward(self, obs, action):
        return self.net(torch.cat((obs, action), -1))

    def value(self, obs, action):
        centers = .5 * (self.edges[1:] + self.edges[:-1])
        return (self(obs, action).softmax(-1) * centers).sum(-1)

    def loss(self, obs, action, target, weights):
        labels = gaussian_labels(target.detach(), self.edges)
        loss = -(labels * self(obs, action).log_softmax(-1)).sum(-1)
        return weighted_mean(loss, weights)


def weighted_mean(value, weights):
    return (value * weights).sum() / weights.sum().clamp_min(1e-12)


def soft_lambda_returns(reward, next_q, next_logp, done, gamma=.99, lam=.95,
                        alpha=0., fail=1, success=2):
    """Entropy belongs to the NEXT sampled action, not twice at this action.

    FAIL/SUCC are absorbing (zero continuation). TIME cuts the trace but
    bootstraps, matching DARE's original truncation contract. Reset episodes
    must never be joined. All inputs are [time, env].
    """
    if not (reward.shape == next_q.shape == next_logp.shape == done.shape):
        raise ValueError("Aligned rollout arrays required")
    terminal = (done == fail) | (done == success)
    bootstrap = torch.where(terminal, 0., next_q - alpha * next_logp)
    result = torch.empty_like(reward)
    result[-1] = reward[-1] + gamma * bootstrap[-1]
    for t in range(len(reward) - 2, -1, -1):
        mix = lam * (done[t] == 0)
        # Future entropy is missing from G[t+1] by definition and must be
        # included in the recursive branch as well as in the one-step branch.
        following = result[t + 1] - alpha * next_logp[t]
        result[t] = reward[t] + gamma * ((1 - mix) * bootstrap[t] + mix * following)
    return result.detach()


class Duals(nn.Module):
    def __init__(self, alpha=.01, beta=.01):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()


@contextlib.contextmanager
def frozen_parameters(module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def actor_objective(dist, old_dist, q, obs, epsilon, duals, weights,
                    kl_limit=.01, target_entropy=-14.):
    """Reparameterized action, pointwise forward-KL clipping, two duals.

    Use exact Gaussian KL instead of the official robot port's four-sample
    estimator. Q parameters must be frozen around this forward AND backward.
    Gradient still propagates through Q with respect to its action input.
    """
    action = dist.mean + dist.stddev * epsilon
    # The robot policy is an unsquashed diagonal Gaussian: both quantities
    # are available exactly, without Monte Carlo noise in multiplier tuning.
    entropy_sample = dist.entropy()
    kl = old_dist.kl(dist)
    policy_q = q.value(obs, action)
    primary = -policy_q - duals.alpha.detach() * entropy_sample
    clipped = torch.where((kl < kl_limit).detach(), primary, duals.beta.detach() * kl)
    entropy = weighted_mean(entropy_sample, weights)
    mean_kl = weighted_mean(kl, weights)
    alpha_loss = duals.alpha * (entropy.detach() - target_entropy)
    beta_loss = duals.beta * (kl_limit - mean_kl.detach())
    return weighted_mean(clipped, weights) + alpha_loss + beta_loss, dict(
        reppo_kl=mean_kl.detach(), reppo_entropy=entropy.detach(),
        reppo_kl_clipped_fraction=weighted_mean((kl >= kl_limit).float(), weights).detach(),
        reppo_alpha=duals.alpha.detach(), reppo_beta=duals.beta.detach(),
        reppo_policy_q=weighted_mean(policy_q.detach(), weights))
