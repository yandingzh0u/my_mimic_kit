"""Streaming diagnostics for normalized policy-input geometry."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch


class InputGeometryAccumulator:
    """Accumulate exact first/second moments without retaining observations."""

    def __init__(
        self,
        block_slices: Mapping[str, slice],
        device: str | torch.device,
        paired_blocks: tuple[tuple[str, str], ...] = (),
    ):
        self._blocks = dict(block_slices)
        self._count = 0
        self._sum: dict[str, torch.Tensor] = {}
        self._gram: dict[str, torch.Tensor] = {}
        self._paired_blocks = paired_blocks
        self._pair_product: dict[tuple[str, str], torch.Tensor] = {}

        for name, block_slice in self._blocks.items():
            start = 0 if block_slice.start is None else int(block_slice.start)
            stop = int(block_slice.stop)
            dim = stop - start
            if dim <= 0:
                raise ValueError(f"empty input-geometry block: {name}")
            self._sum[name] = torch.zeros(dim, device=device, dtype=torch.float32)
            self._gram[name] = torch.zeros(
                dim, dim, device=device, dtype=torch.float32
            )

        for left, right in paired_blocks:
            if left not in self._blocks or right not in self._blocks:
                raise KeyError(f"unknown paired blocks: {left}, {right}")
            if self._sum[left].numel() != self._sum[right].numel():
                raise ValueError(f"paired block sizes differ: {left}, {right}")
            self._pair_product[(left, right)] = torch.zeros_like(self._sum[left])

    def update(self, normalized_obs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if normalized_obs.ndim != 2:
            raise ValueError("normalized observations must be a 2-D batch")
        rows = normalized_obs if mask is None else normalized_obs[mask]
        if rows.shape[0] == 0:
            return
        self._count += int(rows.shape[0])
        selected: dict[str, torch.Tensor] = {}
        for name, block_slice in self._blocks.items():
            value = rows[:, block_slice].float()
            if value.shape[-1] != self._sum[name].numel():
                raise ValueError(f"observation is too short for block {name}")
            selected[name] = value
            self._sum[name] += torch.sum(value, dim=0)
            self._gram[name] += value.transpose(0, 1) @ value
        for pair in self._paired_blocks:
            left, right = pair
            self._pair_product[pair] += torch.sum(
                selected[left] * selected[right], dim=0
            )

    def finalize(self) -> dict[str, object]:
        if self._count == 0:
            return {"sample_count": 0, "blocks": {}, "paired_correlation": {}}

        blocks: dict[str, dict[str, object]] = {}
        block_moments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self._blocks:
            mean = (self._sum[name] / self._count).detach().cpu().numpy().astype(np.float64)
            gram = (self._gram[name] / self._count).detach().cpu().numpy().astype(np.float64)
            covariance = gram - np.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.T)
            variance = np.maximum(np.diag(covariance), 0.0)
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
            max_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
            tolerance = max(max_eigenvalue * 1e-8, 1e-12)
            retained = eigenvalues[eigenvalues > tolerance]
            rank = int(retained.size)
            condition = (
                float(retained[-1] / retained[0]) if retained.size else None
            )
            trace = float(np.sum(eigenvalues))
            if trace > 0.0:
                probabilities = eigenvalues[eigenvalues > 0.0] / trace
                effective_rank = float(
                    math.exp(-np.sum(probabilities * np.log(probabilities)))
                )
            else:
                effective_rank = 0.0

            scale = np.sqrt(np.maximum(variance, 1e-12))
            correlation = covariance / np.outer(scale, scale)
            if correlation.shape[0] > 1:
                offdiag = correlation[~np.eye(correlation.shape[0], dtype=bool)]
                mean_abs_offdiag = float(np.mean(np.abs(offdiag)))
            else:
                mean_abs_offdiag = 0.0
            blocks[name] = {
                "dimension": int(mean.size),
                "rank": rank,
                "effective_rank": effective_rank,
                "condition_number": condition,
                "eigenvalue_tolerance": tolerance,
                "covariance_trace": trace,
                "mean_abs_offdiag_correlation": mean_abs_offdiag,
                "variance_min": float(np.min(variance)),
                "variance_max": float(np.max(variance)),
                "eigenvalues": eigenvalues.tolist(),
            }
            block_moments[name] = (mean, variance)

        paired: dict[str, dict[str, float | int | None]] = {}
        for (left, right), product_sum in self._pair_product.items():
            left_mean, left_var = block_moments[left]
            right_mean, right_var = block_moments[right]
            product_mean = (
                product_sum / self._count
            ).detach().cpu().numpy().astype(np.float64)
            covariance = product_mean - left_mean * right_mean
            denom = np.sqrt(np.maximum(left_var * right_var, 0.0))
            valid = denom > 1e-12
            correlations = np.full_like(denom, np.nan)
            correlations[valid] = covariance[valid] / denom[valid]
            finite = correlations[np.isfinite(correlations)]
            key = f"{left}__{right}"
            paired[key] = {
                "dimension": int(correlations.size),
                "finite_dimension": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else None,
                "mean_abs": float(np.mean(np.abs(finite))) if finite.size else None,
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }

        return {
            "sample_count": self._count,
            "blocks": blocks,
            "paired_correlation": paired,
        }
