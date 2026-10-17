"""Per-sample policy objectives; reduction and phase weights belong to PPOAgent.

SPO is ICML 2025 Simple Policy Optimization, Eq. 16 (not the older KL-clip
preprint). https://proceedings.mlr.press/v267/xie25m.html
"""

import math

import torch


def validate_policy_objective(name, epsilon):
    if name not in ("ppo", "spo"):
        raise ValueError("actor_objective must be 'ppo' or 'spo'")
    if not math.isfinite(float(epsilon)) or not 0 < epsilon < 1:
        raise ValueError("ppo_clip_ratio must be finite and between zero and one")


def policy_objective(ratio, advantage, epsilon, name="ppo"):
    """Return an objective to MAXIMIZE, without changing probability ratios.

    Deliberately do not clamp ratios for SPO: that would change the published
    algorithm and could hide instability. Caller must detect nonfinite runs.
    """
    if name == "ppo":
        return torch.minimum(ratio * advantage,
                             ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    if name == "spo":
        return (ratio * advantage
                - advantage.abs() * (ratio - 1).square() / (2 * epsilon))
    raise ValueError("Unknown policy objective: {}".format(name))
