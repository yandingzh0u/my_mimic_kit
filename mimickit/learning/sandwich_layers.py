"""Minimal fully-connected Sandwich layers from the LBDN parameterization.

The formulas follow the BSD-licensed reference implementation at
https://github.com/acfr/LBDN.  Only the two layers required by ADD are kept.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


def cayley(weight):
    """Maps a rectangular parameter matrix to an orthonormal-row matrix."""
    out_dim, in_dim = weight.shape
    if in_dim > out_dim:
        return cayley(weight.transpose(0, 1)).transpose(0, 1)

    u = weight[:in_dim, :]
    v = weight[in_dim:, :]
    identity = torch.eye(in_dim, dtype=weight.dtype, device=weight.device)
    a = u - u.transpose(0, 1) + v.transpose(0, 1) @ v
    inverse = torch.linalg.solve(identity + a, identity)
    return torch.cat((inverse @ (identity - a), -2.0 * v @ inverse), dim=0)


class _SandwichBase(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(
            self.out_features, self.in_features + self.out_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        self.alpha = nn.Parameter(self.weight.detach().norm().reshape(1))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

        self._cache_enabled = False
        self._cached_q = None
        self._last_q = None

    def set_transform_cache(self, enabled):
        self._cache_enabled = bool(enabled)
        self._cached_q = None

    def _orthogonal_matrix(self):
        if self._cache_enabled and self._cached_q is not None:
            return self._cached_q
        normalized = self.alpha * self.weight / self.weight.norm().clamp_min(1e-12)
        q = cayley(normalized)
        self._last_q = q
        if self._cache_enabled:
            self._cached_q = q.detach()
            return self._cached_q
        return q

    def effective_input_weight(self):
        q = self._last_q if self._last_q is not None else self._orthogonal_matrix()
        return self.scale * q[:, self.out_features:]


class SandwichFc(_SandwichBase):
    """A nonlinear 1-Lipschitz fully-connected Sandwich layer."""
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias=bias, scale=scale)
        self.psi = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, inputs):
        q = self._orthogonal_matrix()
        a = q[:, :self.out_features]
        b = q[:, self.out_features:]
        hidden = functional.linear(self.scale * inputs, b)
        hidden = math.sqrt(2.0) * hidden * torch.exp(-self.psi)
        if self.bias is not None:
            hidden = hidden + self.bias
        hidden = functional.relu(hidden) * torch.exp(self.psi)
        return math.sqrt(2.0) * functional.linear(hidden, a.transpose(0, 1))


class SandwichLin(_SandwichBase):
    """A linear 1-Lipschitz Sandwich output layer."""
    def forward(self, inputs):
        q = self._orthogonal_matrix()
        b = q[:, self.out_features:]
        return functional.linear(self.scale * inputs, b, self.bias)
