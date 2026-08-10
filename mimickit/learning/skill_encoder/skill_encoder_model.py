from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelFreeSkillEncoder(nn.Module):
    """Temporal encoder whose unnormalized output is trained with VICReg.

    ``forward`` returns the raw representation ``y``. Runtime conditioning must
    use ``runtime_z``, which is the L2-normalized form of ``y``.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        if feature_dim <= 0 or embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, embedding_dim, and hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")

        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))

        blocks = []
        in_channels = self.feature_dim
        num_groups = next(
            group for group in range(min(8, hidden_dim), 0, -1) if hidden_dim % group == 0
        )
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        self.temporal_encoder = nn.Sequential(*blocks)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, device=self.feature_mean.device, dtype=torch.float32)
        std = torch.as_tensor(std, device=self.feature_std.device, dtype=torch.float32)
        if mean.shape != (self.feature_dim,) or std.shape != (self.feature_dim,):
            raise ValueError(f"feature statistics must have shape ({self.feature_dim},)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("feature statistics must be finite")
        if torch.any(std <= 0):
            raise ValueError("feature standard deviations must be positive")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B,H,{self.feature_dim}], got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating-point values")
        normalized = (features - self.feature_mean) / self.feature_std
        encoded = self.temporal_encoder(normalized.transpose(1, 2))
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
        return self.projector(pooled)

    def runtime_z(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self(features), p=2, dim=-1, eps=1e-8)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg objective applied directly to raw encoder representations ``y``."""
    if y_a.shape != y_b.shape or y_a.ndim != 2:
        raise ValueError("VICReg inputs must have matching [B,D] shapes")
    if y_a.shape[0] < 2:
        raise ValueError("VICReg requires at least two samples")

    invariance = F.mse_loss(y_a, y_b)
    std_a = torch.sqrt(y_a.var(dim=0, unbiased=True) + eps)
    std_b = torch.sqrt(y_b.var(dim=0, unbiased=True) + eps)
    variance = 0.5 * (
        F.relu(target_std - std_a).mean() + F.relu(target_std - std_b).mean()
    )

    centered_a = y_a - y_a.mean(dim=0)
    centered_b = y_b - y_b.mean(dim=0)
    cov_a = centered_a.T @ centered_a / (y_a.shape[0] - 1)
    cov_b = centered_b.T @ centered_b / (y_b.shape[0] - 1)
    covariance = (
        _off_diagonal(cov_a).square().sum() + _off_diagonal(cov_b).square().sum()
    ) / (2.0 * y_a.shape[1])

    loss = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return loss, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


@torch.no_grad()
def embedding_diagnostics(y: torch.Tensor) -> dict:
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError("embedding diagnostics require a [N,D] tensor with N >= 2")
    centered = y.float() - y.float().mean(dim=0)
    per_dim_std = centered.std(dim=0, unbiased=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if not torch.isfinite(total) or total <= 0:
        effective_rank = 0.0
    else:
        probabilities = (eigenvalues / total).clamp_min(torch.finfo(torch.float32).eps)
        effective_rank = math.exp(float(-(probabilities * probabilities.log()).sum()))
    return {
        "per_dim_std": per_dim_std.cpu().tolist(),
        "min_dim_std": float(per_dim_std.min()),
        "mean_dim_std": float(per_dim_std.mean()),
        "effective_rank": effective_rank,
        "eigenvalues": eigenvalues.cpu().tolist(),
    }
