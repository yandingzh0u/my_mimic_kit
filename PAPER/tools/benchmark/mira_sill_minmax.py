"""Sill (1997) finite min-max monotone affine energy."""
from __future__ import annotations
import torch
from torch import nn

class SillMinMaxEnergy(nn.Module):
    """E(z)=c+min_k max_j(a_kj^T z+b_kj), K=31,J=11."""
    def __init__(self, in_features: int, plane_count: int = 341, groups: int = 31):
        super().__init__()
        if plane_count != 341 or groups != 31 or plane_count != groups * 11:
            raise ValueError("This pressure test fixes K=31, J=11, and 341 planes")
        self.in_features = int(in_features)
        self.plane_count = int(plane_count)
        self.group_count = int(groups)
        self.raw_slopes = nn.Parameter(torch.empty(plane_count, in_features))
        self.plane_bias = nn.Parameter(torch.empty(plane_count))
        self.output_bias = nn.Parameter(torch.empty(1))
        with torch.no_grad():
            # Keep ordinary affine biases. The raw slopes are free coordinates;
            # effective slopes below are positive unit-row projections.
            nn.init.uniform_(self.raw_slopes, 0, in_features ** -0.5)
            nn.init.uniform_(self.plane_bias, -in_features ** -0.5, in_features ** -0.5)
            nn.init.uniform_(self.output_bias, -plane_count ** -0.5, plane_count ** -0.5)

    @staticmethod
    def effective_from_raw(raw_slopes: torch.Tensor) -> torch.Tensor:
        positive = raw_slopes.abs()
        norms = torch.linalg.vector_norm(positive, dim=-1, keepdim=True)
        return positive / norms.clamp_min(1)

    @property
    def effective_slopes(self) -> torch.Tensor:
        return self.effective_from_raw(self.raw_slopes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs @ self.effective_slopes.transpose(0, 1) + self.plane_bias
        values = values.reshape(*values.shape[:-1], self.group_count, 11)
        return self.output_bias + values.amax(dim=-1).amin(dim=-1, keepdim=True)

    @torch.no_grad()
    def enforce_constraints_(self) -> None:
        # The abs-plus-row projection is applied in forward; nothing is
        # required after SGD, and this hook intentionally changes no parameter.
        return None
