"""Small, independent numerical primitives for a real-parameter TRPO step.

Schulman et al. (ICML 2015). No simulator, reward, or optimizer state is changed
here. Failed or interrupted backtracking restores the EXACT original tensors.
"""
import math

import torch


def flat_gradient(value, parameters, *, create_graph=False):
    gradients = torch.autograd.grad(value, parameters, create_graph=create_graph,
                                    allow_unused=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for p, g in zip(parameters, gradients)])


def dot64(left, right):
    return (left.double() * right.double()).sum()


def conjugate_gradient(operator, rhs, max_steps=15, relative_tolerance=1e-10):
    if max_steps < 1 or not 0 < relative_tolerance < 1:
        raise ValueError("Invalid conjugate-gradient limits")
    if not torch.isfinite(rhs).all():
        raise ValueError("Nonfinite natural-gradient RHS")
    result, residual = torch.zeros_like(rhs), rhs.detach().clone()
    direction = residual.clone()
    original_norm = dot64(rhs, rhs)
    squared_norm = original_norm.clone()
    steps, positive = 0, True
    for _ in range(max_steps):
        if original_norm == 0 or squared_norm <= original_norm * relative_tolerance:
            break
        product = operator(direction)
        if not torch.isfinite(product).all():
            raise RuntimeError("Nonfinite Fisher-vector product")
        curvature = dot64(direction, product)
        if curvature <= 0:
            positive = False
            break
        alpha = squared_norm / curvature
        result.add_(direction * alpha)
        residual.sub_(product * alpha)
        next_norm = dot64(residual, residual)
        direction = residual + direction * (next_norm / squared_norm)
        squared_norm = next_norm
        steps += 1
    ratio = float(squared_norm / original_norm) if original_norm > 0 else 0.
    return result, dict(cg_steps=steps, cg_relative_residual_squared=ratio,
                        cg_positive_curvature=positive)


@torch.no_grad()
def assign_vector(parameters, vector):
    if sum(p.numel() for p in parameters) != vector.numel():
        raise ValueError("Parameter vector length mismatch")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(vector[offset:offset + count].view_as(parameter))
        offset += count


def backtracking(parameters, full_step, evaluate, *, kl_limit=.01,
                 max_trials=10, shrink=.8):
    """Maximize a measured objective, subject to measured nonnegative KL.

    `evaluate` must inspect the actual raw network and return numeric objective
    and kl. Acceptance is on the full supplied evaluation set, not a local
    quadratic prediction. This does NOT establish physical improvement.
    """
    if not math.isfinite(kl_limit) or kl_limit <= 0 or max_trials < 1 or not 0 < shrink < 1:
        raise ValueError("Invalid trust-region line-search limits")
    original = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    if full_step.shape != original.shape or not torch.isfinite(full_step).all():
        raise ValueError("Invalid trust-region parameter proposal")
    baseline = evaluate()
    if not all(math.isfinite(float(baseline[key])) for key in ("objective", "kl")):
        raise ValueError("Nonfinite baseline objective or KL")
    accepted, trials, fraction, final = False, 0, 0., baseline
    try:
        for i in range(max_trials):
            trials = i + 1
            scale = shrink ** i
            assign_vector(parameters, original + scale * full_step)
            measured = evaluate()
            objective, kl = float(measured["objective"]), float(measured["kl"])
            if (math.isfinite(objective) and math.isfinite(kl)
                    and 0 <= kl <= kl_limit and objective > float(baseline["objective"])):
                accepted, fraction, final = True, scale, measured
                break
    finally:
        if not accepted:
            assign_vector(parameters, original)
    return dict(accepted=accepted, trials=trials, step_fraction=fraction,
                objective_gain=float(final["objective"]) - float(baseline["objective"]),
                final=final)
