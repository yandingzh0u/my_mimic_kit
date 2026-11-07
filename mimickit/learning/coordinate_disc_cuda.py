"""Fused full-graph coordinate interaction for six-feature CUDA tokens.

Forward and backward evaluate all token pairs without storing the affinity
matrix. The backward includes the derivatives of the affinity weights. An
autograd-recorded VJP supports higher derivatives when explicitly requested.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _interaction_forward(T, H, N: tl.constexpr, Q: tl.constexpr,
                         J: tl.constexpr):
    batch = tl.program_id(0)
    query = tl.program_id(1) * Q + tl.arange(0, Q)
    neighbor = tl.arange(0, J)
    qi = batch * N * 6 + query[:, None] * 6
    nj = batch * N * 6 + neighbor[None, :] * 6
    qm = query[:, None] < N
    nm = neighbor[None, :] < N
    t0 = tl.load(T + qi + 0, qm, other=0)
    t1 = tl.load(T + qi + 1, qm, other=0)
    t2 = tl.load(T + qi + 2, qm, other=0)
    t3 = tl.load(T + qi + 3, qm, other=0)
    t4 = tl.load(T + qi + 4, qm, other=0)
    t5 = tl.load(T + qi + 5, qm, other=0)
    d0 = t0 - tl.load(T + nj + 0, nm, other=0)
    d1 = t1 - tl.load(T + nj + 1, nm, other=0)
    d2 = t2 - tl.load(T + nj + 2, nm, other=0)
    d3 = t3 - tl.load(T + nj + 3, nm, other=0)
    d4 = t4 - tl.load(T + nj + 4, nm, other=0)
    d5 = t5 - tl.load(T + nj + 5, nm, other=0)
    distance2 = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5
    a = tl.where(nm, tl.rsqrt(1.0 + distance2), 0.0)
    factor: tl.constexpr = 0.5 / N
    out = batch * N * 6 + query * 6
    mask = query < N
    tl.store(H + out + 0, tl.sum(t0, 1) - factor * tl.sum(a*d0, 1), mask)
    tl.store(H + out + 1, tl.sum(t1, 1) - factor * tl.sum(a*d1, 1), mask)
    tl.store(H + out + 2, tl.sum(t2, 1) - factor * tl.sum(a*d2, 1), mask)
    tl.store(H + out + 3, tl.sum(t3, 1) - factor * tl.sum(a*d3, 1), mask)
    tl.store(H + out + 4, tl.sum(t4, 1) - factor * tl.sum(a*d4, 1), mask)
    tl.store(H + out + 5, tl.sum(t5, 1) - factor * tl.sum(a*d5, 1), mask)


@triton.jit
def _interaction_backward(T, G, DT, N: tl.constexpr, Q: tl.constexpr,
                          J: tl.constexpr):
    batch = tl.program_id(0)
    query = tl.program_id(1) * Q + tl.arange(0, Q)
    neighbor = tl.arange(0, J)
    qi = batch * N * 6 + query[:, None] * 6
    nj = batch * N * 6 + neighbor[None, :] * 6
    qm = query[:, None] < N
    nm = neighbor[None, :] < N
    d0 = tl.load(T + qi + 0, qm, other=0) - tl.load(T + nj + 0, nm, other=0)
    d1 = tl.load(T + qi + 1, qm, other=0) - tl.load(T + nj + 1, nm, other=0)
    d2 = tl.load(T + qi + 2, qm, other=0) - tl.load(T + nj + 2, nm, other=0)
    d3 = tl.load(T + qi + 3, qm, other=0) - tl.load(T + nj + 3, nm, other=0)
    d4 = tl.load(T + qi + 4, qm, other=0) - tl.load(T + nj + 4, nm, other=0)
    d5 = tl.load(T + qi + 5, qm, other=0) - tl.load(T + nj + 5, nm, other=0)
    g0 = tl.load(G + qi + 0, qm, other=0)
    g1 = tl.load(G + qi + 1, qm, other=0)
    g2 = tl.load(G + qi + 2, qm, other=0)
    g3 = tl.load(G + qi + 3, qm, other=0)
    g4 = tl.load(G + qi + 4, qm, other=0)
    g5 = tl.load(G + qi + 5, qm, other=0)
    dg0 = g0 - tl.load(G + nj + 0, nm, other=0)
    dg1 = g1 - tl.load(G + nj + 1, nm, other=0)
    dg2 = g2 - tl.load(G + nj + 2, nm, other=0)
    dg3 = g3 - tl.load(G + nj + 3, nm, other=0)
    dg4 = g4 - tl.load(G + nj + 4, nm, other=0)
    dg5 = g5 - tl.load(G + nj + 5, nm, other=0)
    distance2 = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5
    a = tl.where(nm, tl.rsqrt(1.0 + distance2), 0.0)
    dot = d0*dg0 + d1*dg1 + d2*dg2 + d3*dg3 + d4*dg4 + d5*dg5
    radial = a*a*a*dot
    factor: tl.constexpr = 0.5 / N
    out = batch * N * 6 + query * 6
    mask = query < N
    tl.store(DT + out + 0, tl.sum(g0, 1) - factor * tl.sum(a*dg0 - d0*radial, 1), mask)
    tl.store(DT + out + 1, tl.sum(g1, 1) - factor * tl.sum(a*dg1 - d1*radial, 1), mask)
    tl.store(DT + out + 2, tl.sum(g2, 1) - factor * tl.sum(a*dg2 - d2*radial, 1), mask)
    tl.store(DT + out + 3, tl.sum(g3, 1) - factor * tl.sum(a*dg3 - d3*radial, 1), mask)
    tl.store(DT + out + 4, tl.sum(g4, 1) - factor * tl.sum(a*dg4 - d4*radial, 1), mask)
    tl.store(DT + out + 5, tl.sum(g5, 1) - factor * tl.sum(a*dg5 - d5*radial, 1), mask)


def _forward(tokens):
    count = tokens.shape[-2]
    batch = tokens.numel() // (count * 6)
    output = torch.empty_like(tokens)
    grid = (batch, triton.cdiv(count, 8))
    _interaction_forward[grid](tokens, output, count, 8,
                               triton.next_power_of_2(count),
                               num_warps=4, enable_fp_fusion=False)
    return output


class _Interaction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens):
        output = _forward(tokens)
        ctx.save_for_backward(tokens)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (tokens,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Only higher-order analysis records this differentiable formula.
            # Ordinary training uses the fused, memory-bounded VJP below.
            difference = tokens.unsqueeze(-2) - tokens.unsqueeze(-3)
            grad_difference = (grad_output.unsqueeze(-2)
                               - grad_output.unsqueeze(-3))
            affinity = torch.rsqrt(1 + difference.square().sum(dim=-1))
            radial = (difference * grad_difference).sum(dim=-1, keepdim=True)
            pair_vjp = (affinity.unsqueeze(-1) * grad_difference
                        - affinity.pow(3).unsqueeze(-1) * difference * radial)
            return grad_output - pair_vjp.sum(dim=-2) / (2 * tokens.shape[-2])
        count = tokens.shape[-2]
        batch = tokens.numel() // (count * 6)
        grad_input = torch.empty_like(tokens)
        grid = (batch, triton.cdiv(count, 8))
        _interaction_backward[grid](tokens, grad_output.contiguous(), grad_input,
                                    count, 8,
                                    triton.next_power_of_2(count),
                                    num_warps=4, enable_fp_fusion=False)
        return grad_input


def interact_cuda(tokens):
    """Apply H(T) to CUDA float32 tensors shaped [..., N, 6]."""
    if (not tokens.is_cuda or tokens.dtype != torch.float32
            or tokens.ndim < 2 or tokens.shape[-1] != 6
            or tokens.shape[-2] < 1):
        raise ValueError("expected CUDA float32 tokens shaped [..., N, 6], N > 0")
    if not tokens.numel():
        return tokens.clone()
    # Keep the contiguous conversion outside Function.forward so autograd
    # retains its relationship to a noncontiguous input for higher gradients.
    tokens = tokens.contiguous()
    if not torch.is_grad_enabled() or not tokens.requires_grad:
        return _forward(tokens)
    return _Interaction.apply(tokens)
