"""Conditioned positive-definite energy model for anchored imitation.

The module deliberately does not own any observation normalizer.  Callers must
pass an already-normalized residual ``z`` and an already-normalized,
reference-only context ``c`` to :meth:`AnchoredEnergyModel.eval_energy`.

For residual dimension ``D`` and context dimension ``2D``, the conditioner
produces a bounded low-rank factor and a positive diagonal,

    U(c) = tanh(U_raw(c)),
    d(c) = softplus(clamp(d_raw(c))) + diagonal_floor.

Writing ``B = U U^T + diag(d)``, the trace-normalized metric is

    M(c) = eps I + D (1 - eps) B / tr(B),

and the dimension-normalized energy is ``z^T M(c) z / (2D)``.  Consequently,
``M`` has trace ``D`` and is uniformly positive definite whenever
``0 < eps < 1``.  In particular, zero residual is the unique energy minimizer
for every context.  The discriminator logit is ``beta - E``.  ``beta`` is a
global classification threshold; it is intentionally absent from the actor
reward ``1 / (1 + E)``.
"""

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F

import learning.add_model as add_model


def energy_actor_reward(energy):
    """Map a nonnegative energy to a beta-independent actor reward.

    The energy model guarantees nonnegative inputs mathematically.  This
    helper does not clamp them because a clamp would introduce an artificial
    zero-gradient region around the anchored optimum.
    """

    return torch.reciprocal(1.0 + energy)


class ConditionalPositiveDefiniteEnergy(torch.nn.Module):
    """Reference-conditioned, trace-normalized positive-definite energy.

    Parameters
    ----------
    residual_dim:
        Last dimension of the differential residual.
    context_dim:
        Last dimension of the reference-only context.  The anchored-energy
        interface requires this to equal ``2 * residual_dim``.
    hidden_units:
        Widths of the context conditioner.
    rank:
        Rank of the context-dependent low-rank factor ``U``.
    eigen_floor:
        Uniform lower eigenvalue of the normalized metric.
    diagonal_floor:
        Strictly positive numerical floor added after softplus.
    diagonal_logit_clip:
        Symmetric bound applied before softplus for finite extreme-input
        behavior.  This bounds metric construction without clipping energy or
        its residual gradient.
    """

    def __init__(
        self,
        residual_dim,
        context_dim,
        hidden_units=(512, 512),
        rank=16,
        eigen_floor=0.05,
        diagonal_floor=1e-6,
        diagonal_logit_clip=20.0,
    ):
        super().__init__()

        self._residual_dim = int(residual_dim)
        self._context_dim = int(context_dim)
        self._rank = int(rank)
        self._eigen_floor = float(eigen_floor)
        self._diagonal_floor = float(diagonal_floor)
        self._diagonal_logit_clip = float(diagonal_logit_clip)

        if self._residual_dim <= 0:
            raise ValueError("residual_dim must be positive")
        if self._context_dim != 2 * self._residual_dim:
            raise ValueError(
                "context_dim must equal 2 * residual_dim; got "
                f"{self._context_dim} and {self._residual_dim}")
        if self._rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < self._eigen_floor < 1.0:
            raise ValueError("eigen_floor must lie strictly between 0 and 1")
        if not math.isfinite(self._diagonal_floor) or self._diagonal_floor <= 0:
            raise ValueError("diagonal_floor must be finite and positive")
        if (not math.isfinite(self._diagonal_logit_clip)
                or self._diagonal_logit_clip <= 0):
            raise ValueError(
                "diagonal_logit_clip must be finite and positive")

        if isinstance(hidden_units, int):
            hidden_units = (hidden_units,)
        elif not isinstance(hidden_units, Sequence):
            raise TypeError("hidden_units must be an int or a sequence of ints")
        hidden_units = tuple(int(width) for width in hidden_units)
        if any(width <= 0 for width in hidden_units):
            raise ValueError("all hidden_units must be positive")

        layers = []
        input_dim = self._context_dim
        for output_dim in hidden_units:
            layer = torch.nn.Linear(input_dim, output_dim)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            torch.nn.init.zeros_(layer.bias)
            layers.extend((layer, torch.nn.ReLU()))
            input_dim = output_dim
        self._context_layers = torch.nn.Sequential(*layers)

        output_dim = self._residual_dim * (self._rank + 1)
        self._metric_head = torch.nn.Linear(input_dim, output_dim)
        # A small, nonzero initialization keeps the U branch trainable from
        # the first step; U U^T has zero derivative at U == 0.
        torch.nn.init.xavier_uniform_(self._metric_head.weight, gain=0.01)
        torch.nn.init.zeros_(self._metric_head.bias)
        with torch.no_grad():
            low_rank_size = self._residual_dim * self._rank
            self._metric_head.bias[:low_rank_size].uniform_(-0.01, 0.01)

    @property
    def residual_dim(self):
        return self._residual_dim

    @property
    def context_dim(self):
        return self._context_dim

    @property
    def rank(self):
        return self._rank

    @property
    def eigen_floor(self):
        return self._eigen_floor

    def _validate_inputs(self, residual, context=None):
        if residual.shape[-1] != self._residual_dim:
            raise ValueError(
                f"residual last dimension must be {self._residual_dim}, "
                f"got {residual.shape[-1]}")
        if context is not None:
            if context.shape[-1] != self._context_dim:
                raise ValueError(
                    f"context last dimension must be {self._context_dim}, "
                    f"got {context.shape[-1]}")
            if residual.shape[:-1] != context.shape[:-1]:
                raise ValueError(
                    "residual and context must have identical leading shapes; "
                    f"got {residual.shape[:-1]} and {context.shape[:-1]}")

    def metric_factors(self, context):
        """Return ``(U, d, trace_B)`` without materializing a dense metric."""

        if context.shape[-1] != self._context_dim:
            raise ValueError(
                f"context last dimension must be {self._context_dim}, "
                f"got {context.shape[-1]}")

        hidden = self._context_layers(context)
        raw = self._metric_head(hidden)
        low_rank_size = self._residual_dim * self._rank
        u_raw = raw[..., :low_rank_size]
        diagonal_raw = raw[..., low_rank_size:]

        u = torch.tanh(
            u_raw.reshape(*context.shape[:-1], self._residual_dim, self._rank))
        diagonal = F.softplus(torch.clamp(
            diagonal_raw,
            min=-self._diagonal_logit_clip,
            max=self._diagonal_logit_clip,
        )) + self._diagonal_floor
        trace_b = torch.sum(torch.square(u), dim=(-2, -1))
        trace_b = trace_b + torch.sum(diagonal, dim=-1)
        return u, diagonal, trace_b

    def eval_metric(self, context):
        """Materialize ``M(c)`` for diagnostics and mathematical tests."""

        u, diagonal, trace_b = self.metric_factors(context)
        base = torch.matmul(u, u.transpose(-2, -1))
        base = base + torch.diag_embed(diagonal)
        scale = (self._residual_dim * (1.0 - self._eigen_floor)
                 / trace_b)
        metric = base * scale[..., None, None]
        identity = torch.eye(
            self._residual_dim, dtype=metric.dtype, device=metric.device)
        return metric + self._eigen_floor * identity

    def eval_energy(self, residual, context):
        """Evaluate energy for normalized ``residual`` and ``context``.

        Both inputs must have identical leading shapes and final dimensions
        ``D`` and ``2D`` respectively.  The result has shape ``[..., 1]``.
        Dense ``D x D`` metrics are not formed in this training path.
        """

        self._validate_inputs(residual, context)
        u, diagonal, trace_b = self.metric_factors(context)

        # Accumulate in float32 under fp16/bfloat16, while preserving float64
        # for gradcheck and high-precision diagnostics.
        calc_dtype = (torch.float64 if residual.dtype == torch.float64
                      else torch.float32)
        z = residual.to(dtype=calc_dtype)
        u = u.to(dtype=calc_dtype)
        diagonal = diagonal.to(dtype=calc_dtype)
        trace_b = trace_b.to(dtype=calc_dtype)

        projected = torch.matmul(u.transpose(-2, -1), z.unsqueeze(-1))
        low_rank_quadratic = torch.sum(torch.square(projected), dim=(-2, -1))
        diagonal_quadratic = torch.sum(diagonal * torch.square(z), dim=-1)
        base_quadratic = low_rank_quadratic + diagonal_quadratic
        quadratic = self._eigen_floor * torch.sum(torch.square(z), dim=-1)
        quadratic = quadratic + (
            self._residual_dim * (1.0 - self._eigen_floor)
            * base_quadratic / trace_b)
        energy = 0.5 * quadratic / self._residual_dim
        return energy.unsqueeze(-1)

    def forward(self, residual, context):
        return self.eval_energy(residual, context)


class AnchoredEnergyModel(add_model.ADDModel):
    """ADD actor/critic with a conditional anchored-energy discriminator.

    ``eval_energy(residual, context)`` expects agent-normalized tensors with
    last dimensions ``D`` and ``2D``.  The model never computes or updates
    normalization statistics itself.
    """

    def _build_disc(self, config, env):
        disc_obs_space = env.get_disc_obs_space()
        residual_dim = math.prod(disc_obs_space.shape)

        hidden_units = config.get("energy_hidden_units", (512, 512))
        rank = int(config.get("energy_rank", min(16, residual_dim)))
        eigen_floor = float(config.get("energy_eigen_floor", 0.05))
        diagonal_floor = float(config.get("energy_diagonal_floor", 1e-6))
        diagonal_logit_clip = float(
            config.get("energy_diagonal_logit_clip", 20.0))

        self._conditional_energy = ConditionalPositiveDefiniteEnergy(
            residual_dim=residual_dim,
            context_dim=2 * residual_dim,
            hidden_units=hidden_units,
            rank=rank,
            eigen_floor=eigen_floor,
            diagonal_floor=diagonal_floor,
            diagonal_logit_clip=diagonal_logit_clip,
        )
        beta_init = float(config.get("energy_beta_init", 0.0))
        if not math.isfinite(beta_init):
            raise ValueError("energy_beta_init must be finite")
        self._disc_beta = torch.nn.Parameter(
            torch.tensor(beta_init, dtype=torch.float32))

    @property
    def residual_dim(self):
        return self._conditional_energy.residual_dim

    @property
    def context_dim(self):
        return self._conditional_energy.context_dim

    @property
    def disc_beta(self):
        return self._disc_beta

    def get_energy_bias(self):
        """Return the scalar learnable BCE threshold ``beta``."""

        return self._disc_beta

    def get_energy_epsilon(self):
        """Return the fixed global eigenvalue floor used by the metric."""

        return self._conditional_energy.eigen_floor

    def eval_energy(self, residual, context):
        """Return ``E(residual, context)`` with shape ``[..., 1]``."""

        return self._conditional_energy.eval_energy(residual, context)

    def eval_disc(self, residual, context):
        """Return the anchored BCE logit ``beta - E``."""

        return self._disc_beta - self.eval_energy(residual, context)

    def eval_actor_reward(self, residual, context):
        """Return ``1 / (1 + E)``; independent of discriminator beta."""

        return energy_actor_reward(self.eval_energy(residual, context))

    def get_disc_params(self):
        """Return only conditional-energy and global-beta parameters."""

        return (list(self._conditional_energy.parameters())
                + [self._disc_beta])


# Descriptive alias retained for callers that prefer an ADD-specific name.
AnchoredEnergyADDModel = AnchoredEnergyModel
