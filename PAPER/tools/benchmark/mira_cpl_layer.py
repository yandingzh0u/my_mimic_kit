"""CPL ablation: same log-parameterized weights, with a residual step."""

from torch import nn
from torch.nn import functional as F


class CPLLinear(nn.Linear):
    """Apply a CPL step before the shared rectangular readout.

    For the effective unit-bounded weight A, the CPL step is
    x - 2 A.T ReLU(A x + bias) (Araujo et al., Theorem 1, statement 2,
    using A.T as W and I as T). The existing A and bias also serve
    the dimension-changing readout. No additional parameters are used.
    A composition of the two maps is nonexpansive; monotonicity is
    not guaranteed by the residual map.
    """

    def forward(self, x):
        weight = self.weight
        activation = F.relu(F.linear(x, weight, self.bias))
        corrected = x - 2 * F.linear(activation, weight.T)
        return F.linear(corrected, weight, self.bias)
