"""Positive, non-residual diagonal normalization from Araujo et al. (2023).

Theorem 1 and Appendix C.3 of ``A Unified Algebraic Perspective on Lipschitz
Neural Networks`` give W = P T**(-1/2), with T >= P.T P. For P >= 0 and
q > 0, t = P.T(Pq) / q is an analytic diagonal solution. Learning q changes
the relative column scaling without introducing an unbounded output gain.

This is the paper's linear construction, not its residual SLL block. The
log-domain implementation also avoids the asymmetric epsilon ratio corrected
in the paper's erratum. No spectral-normalization module or optimizer hook is
used during training.
"""

import torch
from torch import nn


class PositiveDiagonalRescaling(nn.Module):
    """Preserve positivity and bound the effective L2 operator norm by one.

    A common positive scale of q cancels. The raw log_q parameters therefore
    describe relative column scaling, not a free global amplitude. Floating
    point spectral norms still need a numerical tolerance when checked.
    """

    def __init__(self, in_features, *, device=None, dtype=None):
        super().__init__()
        self.log_q = nn.Parameter(torch.zeros(in_features, device=device, dtype=dtype))

    @torch.no_grad()
    def initialize_from_weight(self, positive):
        """Initialize q to a right Perron vector, reproducing SN initially.

        This deterministic CPU-only initialization uses matrix-vector products;
        it consumes no RNG and is not part of the training forward pass. Dense
        positive initial matrices converge rapidly. Zero columns are allowed,
        but exact SN equivalence only holds for a strictly positive Perron
        vector, as is the case for the ordinary dense initialization.
        """
        if positive.ndim != 2 or positive.shape[1] != self.log_q.numel():
            raise ValueError("Diagonal normalization expects a matching [out,in] matrix")
        matrix = positive.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Diagonal normalization requires finite nonnegative weights")
        maximum = matrix.max()
        if maximum == 0:
            self.log_q.zero_()
            return
        matrix = matrix / maximum
        q = torch.ones(matrix.shape[1], dtype=matrix.dtype)
        q = q / q.norm()
        for _ in range(128):
            updated = matrix.T @ (matrix @ q)
            updated = updated / updated.norm().clamp_min(torch.finfo(matrix.dtype).tiny)
            converged = (updated - q).abs().max() <= 1e-13
            q = updated
            if converged:
                break
        values = q.clamp_min(torch.finfo(q.dtype).tiny).log()
        values -= values.mean()
        self.log_q.copy_(values.to(self.log_q))

    def forward(self, positive):
        # Shared scaling of P cancels as well. Removing it before taking logs
        # avoids overflow/underflow for otherwise representable raw weights.
        if positive.ndim != 2 or positive.shape[1] != self.log_q.numel():
            raise ValueError("Diagonal normalization expects [out,in] weights")
        if not bool(torch.any(positive > 0)):
            # Avoid the undefined all -inf logsumexp path while preserving a
            # zero gradient for an entirely zero effective matrix.
            return positive * 0 + self.log_q.sum() * 0
        tiny = torch.finfo(positive.dtype).tiny
        maximum = positive.detach().amax().clamp_min(tiny)
        scaled = positive / maximum
        # The tiny positive envelope makes both logsumexp reductions finite,
        # including empty rows, zero columns and a completely zero matrix.
        # Its Gram row sums only INCREASE the diagonal bound T. Removing the
        # envelope's zero-weight entries afterwards is conservative for P>=0.
        mask = positive > 0
        log_p = scaled.clamp_min(tiny).log()
        # A finite sentinel avoids NaN derivatives from logsumexp over an
        # all-`-inf` row. It is far below any representable log mass and is
        # removed from the effective output by the masks below.
        sentinel = torch.finfo(positive.dtype).min / 4
        masked_log_p = torch.where(
            mask, log_p, torch.full_like(log_p, sentinel))
        log_q = self.log_q - self.log_q.detach().amax()
        # Zero columns must not receive q mass. Otherwise an extreme learned
        # q on a structurally zero column can underflow every live column,
        # despite being irrelevant to P.T @ (P q).
        log_pq = torch.logsumexp(masked_log_p + log_q.unsqueeze(0), dim=1)
        log_gram_q = torch.logsumexp(
            masked_log_p + log_pq.unsqueeze(1), dim=0)
        column_has_mass = mask.any(dim=0)
        row_has_mass = mask.any(dim=1)
        safe_log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        safe_log_gram_q = torch.where(
            column_has_mass, log_gram_q, torch.zeros_like(log_gram_q))
        raw_log_effective = safe_log_p + 0.5 * (
            log_q - safe_log_gram_q).unsqueeze(0)
        log_effective = torch.where(
            mask, raw_log_effective, torch.zeros_like(raw_log_effective))
        effective = log_effective.exp()
        # A zero output row/column is structurally absent. The explicit mask
        # also prevents its unused q from producing an inf*0 backward path.
        return torch.where(mask & row_has_mass.unsqueeze(1),
                           effective, torch.zeros_like(effective))
