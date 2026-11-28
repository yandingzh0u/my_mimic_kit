"""Learnable group-weighted Euclidean norm energy for an isolated MIRA smoke.

For normalized physical errors z>=0 split into the seven audited groups,
E(z)=c + sum_g a_g ||z_g||_2, with a>=0 and ||a||_2<=1.  The parameterized
coefficients are projected from an unconstrained raw vector on every forward.
"""
from __future__ import annotations
import torch
from torch import nn

GROUP_SLICES = ((0, 1), (1, 2), (2, 16), (16, 30), (30, 58), (58, 61), (61, 64))

class GroupWeightedNormEnergy(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        if int(feature_dim) != 64:
            raise ValueError("The Backflip group norm smoke expects 64 features")
        self.feature_dim = int(feature_dim)
        self.raw_group_weights = nn.Parameter(torch.ones(7) / (7 ** 0.5))
        self.output_bias = nn.Parameter(torch.zeros(1))

    @property
    def effective_group_weights(self) -> torch.Tensor:
        positive = self.raw_group_weights.abs()
        return positive / positive.norm().clamp_min(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [N,{self.feature_dim}] normalized errors")
        weights = self.effective_group_weights
        values = [inputs[..., start:stop].norm(dim=-1) for start, stop in GROUP_SLICES]
        norms = torch.stack(values, dim=-1)
        return self.output_bias + (norms * weights).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def enforce_constraints_(self) -> None:
        return None
