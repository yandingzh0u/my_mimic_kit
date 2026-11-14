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


class SignedDiagonalRescaling(nn.Module):
    """Bound signed linear weights by the same diagonal construction.

    For arbitrary P and positive q, t = abs(P.T @ P) @ q / q gives
    diag(t) >= P.T @ P. Thus W = P diag(t**(-1/2)) has L2 operator
    norm at most one. This reduces to PositiveDiagonalRescaling for P >= 0.

    Unit columns and log-domain column gains avoid overflowing an inverse
    square root when the raw column norms or learned q have extreme scales.
    This implementation computes a Gram matrix, unlike the cheaper positive
    special case. It uses no spectral-normalization state or optimizer hook.
    """

    def __init__(self, in_features, *, device=None, dtype=None):
        super().__init__()
        self.log_q = nn.Parameter(torch.zeros(in_features, device=device, dtype=dtype))

    @torch.no_grad()
    def initialize_from_weight(self, weight):
        """Initialize q deterministically from abs(P.T @ P)'s Perron vector.

        A signed Gram matrix can contain cancellation, so this initialization
        need not reproduce SN. It does reproduce the positive construction
        when the raw weights are nonnegative. Initialization is CPU float64,
        consumes no RNG, and is not part of the training forward pass.
        """
        if weight.ndim != 2 or weight.shape[1] != self.log_q.numel():
            raise ValueError("Diagonal normalization expects a matching [out,in] matrix")
        matrix = weight.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(matrix).all():
            raise ValueError("Diagonal normalization requires finite weights")
        maximum = matrix.abs().max()
        if maximum == 0:
            self.log_q.zero_()
            return
        matrix = matrix / maximum
        gram = (matrix.T @ matrix).abs()
        q = torch.ones(matrix.shape[1], dtype=matrix.dtype)
        q = q / q.norm()
        for _ in range(128):
            updated = gram @ q
            updated = updated / updated.norm().clamp_min(torch.finfo(matrix.dtype).tiny)
            converged = (updated - q).abs().max() <= 1e-13
            q = updated
            if converged:
                break
        values = q.clamp_min(torch.finfo(q.dtype).tiny).log()
        values -= values.mean()
        self.log_q.copy_(values.to(self.log_q))

    def forward(self, weight):
        if weight.ndim != 2 or weight.shape[1] != self.log_q.numel():
            raise ValueError("Diagonal normalization expects [out,in] weights")
        column_maximum = weight.detach().abs().amax(dim=0)
        live = column_maximum > 0
        if not bool(live.any()):
            return weight * 0 + self.log_q.sum() * 0

        # Write P = U diag(c), where each live column of U has unit norm.
        # Safe denominators also keep derivatives of absent columns finite.
        safe_maximum = torch.where(live, column_maximum,
                                   torch.ones_like(column_maximum))
        scaled = weight / safe_maximum.unsqueeze(0)
        column_norm = torch.linalg.vector_norm(scaled, dim=0)
        safe_norm = torch.where(live, column_norm, torch.ones_like(column_norm))
        unit = scaled / safe_norm.unsqueeze(0)
        log_c = safe_maximum.log() + safe_norm.log()

        # The Gram of the unit columns has entries bounded in absolute value
        # by one. Keep exact-zero correlations absent rather than adding an
        # envelope that can dominate under an extreme learned q.
        correlation = (unit.T @ unit).abs()
        tiny = torch.finfo(weight.dtype).tiny
        sentinel = torch.finfo(weight.dtype).min / 4
        log_correlation = torch.where(
            correlation > 0, correlation.clamp_min(tiny).log(),
            torch.full_like(correlation, sentinel))
        live_log_q = torch.where(live, self.log_q,
                                 torch.full_like(self.log_q, sentinel))
        log_q = self.log_q - live_log_q.detach().amax()
        v = torch.where(live, log_c + log_q,
                        torch.full_like(log_q, sentinel))
        log_mass = torch.logsumexp(log_correlation + v.unsqueeze(0), dim=1)
        safe_v = torch.where(live, v, torch.zeros_like(v))
        safe_mass = torch.where(live, log_mass, torch.zeros_like(log_mass))
        gain = torch.exp(0.5 * (safe_v - safe_mass))
        gain = torch.where(live, gain, torch.zeros_like(gain))
        # Keeping the signed numerator differentiable lets a zero element
        # inside a live column learn either sign, as in an ordinary Linear.
        return unit * gain.unsqueeze(0)
