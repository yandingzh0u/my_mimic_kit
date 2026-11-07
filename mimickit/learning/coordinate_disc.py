"""Coordinate-preserving tokens and one nonexpansive similarity interaction.

The interaction is exactly
    H_i = T_i + (1 / (2 N)) sum_j (T_j - T_i) /
                                      sqrt(1 + ||T_i - T_j||^2).
Fused CUDA evaluation avoids storing pairwise matrices. CPU and other dtypes
use chunking and activation recomputation with the same mathematical map.
"""

import torch
from torch.utils.checkpoint import checkpoint


class CoordinateInteractionEncoder(torch.nn.Module):
    """A unit-gain embedding for every coordinate followed by interaction.

For fixed effective weights, the token embedding is an isometry in input
differences.  The interaction has a global lower distance bound of 1/2 and an
upper distance bound of 1.  Subsequent fusion layers need not preserve the
lower bound.
"""

    # Memory scheduling for the portable PyTorch implementation. CUDA FP32
    # evaluates the complete minibatch with fused forward/backward kernels.
    _interaction_batch_size = 256

    def __init__(self, input_dim, token_dim=6):
        super().__init__()
        if input_dim < 1 or token_dim < 1:
            raise ValueError("input_dim and token_dim must be positive")
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.total_width = self.input_dim * self.token_dim
        self.raw_weight = torch.nn.Parameter(
            torch.randn(self.input_dim, self.token_dim))
        self.bias = torch.nn.Parameter(
            torch.zeros(self.input_dim, self.token_dim))

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        if (self.raw_weight.is_cuda and self.raw_weight.dtype == torch.float32
                and self.token_dim == 6):
            # Initialize the compiled launcher during device transfer. In
            # Isaac Sim, first-use initialization can retain its caller frames;
            # doing it inside a large rollout forward pins those activations.
            # A tiny deterministic input keeps initialization outside that path
            # without changing weights, random state, or model computations.
            from learning.coordinate_disc_cuda import interact_cuda
            with torch.no_grad():
                interact_cuda(self.raw_weight.new_zeros(1, self.input_dim, 6))
        return module

    def unit_weight(self):
        """Return effective unit rows, including a defined exact-zero fallback.

        An epsilon-clamped denominator can produce non-unit vectors.  Instead,
        an exactly zero raw row represents the first basis vector.  Normal
        randomly initialized rows use ordinary differentiable normalization.
        """
        # Scale first so squaring finite very small/large entries cannot
        # underflow/overflow the norm.  This is algebraically w / ||w||.
        scales = self.raw_weight.abs().amax(dim=-1, keepdim=True)
        nonzero = scales > 0
        safe_scales = torch.where(nonzero, scales, torch.ones_like(scales))
        scaled = self.raw_weight / safe_scales
        norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        safe_norms = torch.where(nonzero, norms, torch.ones_like(norms))
        normalized = scaled / safe_norms
        basis = torch.zeros_like(self.raw_weight)
        basis[:, 0] = 1
        return torch.where(nonzero, normalized, basis)

    @staticmethod
    def _interact_chunk(tokens):
        # Centering is algebraically neutral and avoids cancellation when
        # tokens share a large common translation.  Neither pairwise vectors
        # [B,N,N,D] nor an all-minibatch affinity matrix are materialized.
        centered = tokens - tokens.mean(dim=-2, keepdim=True)
        squared_norm = centered.square().sum(dim=-1, keepdim=True)
        distance_squared = (squared_norm + squared_norm.transpose(-1, -2)
                            - 2 * torch.bmm(centered,
                                            centered.transpose(-1, -2)))
        # The exact squared distance is nonnegative; the clamp handles only
        # Gram-product roundoff and does not add a distance offset.
        affinity = torch.rsqrt(1 + distance_squared.clamp_min(0))
        message = (torch.bmm(affinity, centered)
                   - affinity.sum(dim=-1, keepdim=True) * centered)
        return tokens + message / (2 * tokens.shape[-2])

    def interact(self, tokens):
        """Interact tokens shaped [..., input_dim, token_dim], preserving shape."""
        if tokens.shape[-2:] != (self.input_dim, self.token_dim):
            raise ValueError("tokens must end in [input_dim, token_dim]")
        if tokens.is_cuda and tokens.dtype == torch.float32 and self.token_dim == 6:
            from learning.coordinate_disc_cuda import interact_cuda
            return interact_cuda(tokens)
        original_shape = tokens.shape
        flattened = tokens.reshape(-1, self.input_dim, self.token_dim)
        outputs = []
        for chunk in flattened.split(self._interaction_batch_size, dim=0):
            if torch.is_grad_enabled() and chunk.requires_grad:
                # Non-reentrant checkpoint supports gradcheck/autograd.grad
                # and stores only tokens, not every pairwise affinity.
                output = checkpoint(self._interact_chunk, chunk,
                                    use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                output = self._interact_chunk(chunk)
            outputs.append(output)
        if not outputs:
            return tokens.clone()
        return torch.cat(outputs, dim=0).reshape(original_shape)

    def forward(self, inputs):
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("input last dimension must equal input_dim")
        tokens = inputs.unsqueeze(-1) * self.unit_weight() + self.bias
        interacted = self.interact(tokens)
        return interacted.flatten(start_dim=-2)
