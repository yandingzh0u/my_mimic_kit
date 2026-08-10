from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from learning.normalizer import Normalizer
from learning.tinymdm.EMA import EMA
from learning.tinymdm.arch import TinyStableMotionDiTModel


class FlowMatchingModel(nn.Module):
    """Unconditional rectified-flow prior over flattened motion windows.

    Training pairs standard Gaussian noise ``x0`` with normalized expert motion
    ``x1`` along the straight path ``xt = (1 - t) * x0 + t * x1``.  The DiT
    predicts the constant path velocity ``x1 - x0``.
    """

    def __init__(self, config: dict, device: str | torch.device):
        super().__init__()
        self.config = config
        self._device = torch.device(device)
        self._obs_dtype = torch.float32

        if config.get("arch_name", "DiT") != "DiT":
            raise ValueError("FlowMatchingModel supports only the unconditional DiT architecture")

        self.num_disc_obs_steps = self._infer_num_obs_steps(config)
        self.num_obs_steps = self.num_disc_obs_steps
        self.input_dim = int(config["input_dim"])
        if self.input_dim <= 0 or self.input_dim % self.num_obs_steps != 0:
            raise ValueError(
                f"input_dim ({self.input_dim}) must be divisible by num_disc_obs_steps "
                f"({self.num_obs_steps})"
            )
        self.input_channel = self.input_dim // self.num_obs_steps
        configured_channel = config.get("input_channel")
        if configured_channel is not None and int(configured_channel) != self.input_channel:
            raise ValueError(
                f"input_channel ({configured_channel}) does not match inferred value "
                f"({self.input_channel})"
            )

        self.time_embed_scale = float(config.get("time_embed_scale", 49.0))
        if self.time_embed_scale <= 0:
            raise ValueError("time_embed_scale must be positive")

        self.dmodel = TinyStableMotionDiTModel(
            in_channels=self.input_channel,
            num_layers=int(config["num_layers"]),
            attention_head_dim=int(config.get("attention_head_dim", 64)),
            num_attention_heads=int(config.get("num_attention_heads", 4)),
            out_channels=self.input_channel,
            dropout=float(config.get("dropout", 0.0)),
            max_seq_len=max(32, self.num_obs_steps),
        )
        self.ema_dmodel = EMA(
            self.dmodel,
            beta=float(config.get("model_ema_decay", 0.995)),
            update_every=int(config.get("model_ema_steps", 10)),
            update_after_step=int(config.get("model_ema_update_after", 5_000)),
        )
        # Compatibility with the existing prior trainer/agent convention.
        self.model_ema = True

        self.obs_normalizer = Normalizer(
            self.input_channel,
            device=self._device,
            dtype=self._obs_dtype,
            std_clip=config.get("normalizer_std_clip", 0.2),
        )

    @staticmethod
    def _infer_num_obs_steps(config: dict) -> int:
        configured_steps = config.get("num_disc_obs_steps", config.get("num_obs_steps"))
        if configured_steps is not None:
            steps = int(configured_steps)
        else:
            env_config_path = config.get("env_config")
            if env_config_path is None:
                raise ValueError(
                    "config must define num_disc_obs_steps or point to an env_config"
                )
            with Path(env_config_path).open("r") as stream:
                env_config = yaml.safe_load(stream)
            steps = int(env_config["num_disc_obs_steps"])
        if steps <= 0:
            raise ValueError("num_disc_obs_steps must be positive")
        return steps

    def update_normalizer(self, samples: torch.Tensor) -> None:
        self._validate_motion_samples(samples)
        self.obs_normalizer.record(samples.reshape(-1, self.input_channel))
        self.obs_normalizer.update()

    def normalize(self, samples: torch.Tensor) -> torch.Tensor:
        self._validate_motion_samples(samples)
        shape = samples.shape
        normalized = self.obs_normalizer.normalize(samples.reshape(-1, self.input_channel))
        return normalized.reshape(shape)

    def unnormalize(self, norm_samples: torch.Tensor) -> torch.Tensor:
        self._validate_motion_samples(norm_samples)
        shape = norm_samples.shape
        samples = self.obs_normalizer.unnormalize(
            norm_samples.reshape(-1, self.input_channel)
        )
        return samples.reshape(shape)

    def update_ema(self) -> None:
        self.ema_dmodel.update()

    def forward(
        self,
        x1: torch.Tensor,
        *,
        base_noise: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the standard rectified-flow velocity regression loss."""
        self._validate_flat_samples(x1)
        batch_size = x1.shape[0]
        if base_noise is None:
            x0 = torch.randn_like(x1)
        else:
            if base_noise.shape != x1.shape:
                raise ValueError(
                    f"training base_noise must have shape {tuple(x1.shape)}, got "
                    f"{tuple(base_noise.shape)}"
                )
            x0 = base_noise.to(device=x1.device, dtype=x1.dtype)

        if times is None:
            t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)
        else:
            t = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
            if t.ndim == 0:
                t = t.expand(batch_size)
            if t.shape != (batch_size,):
                raise ValueError(f"training times must have shape ({batch_size},)")
            self._validate_times(t)

        path_t = t.unsqueeze(-1)
        xt = (1.0 - path_t) * x0 + path_t * x1
        target_velocity = x1 - x0
        pred_velocity = self._predict_velocity(self.dmodel, xt, t)
        return torch.mean(torch.square(pred_velocity - target_velocity))

    @torch.no_grad()
    def mismatch_per_time(
        self,
        x1: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Return per-feature velocity MSE with shape ``[batch, noise, time]``."""
        self._validate_flat_samples(x1)
        time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
        if time_values.ndim != 1 or time_values.numel() == 0:
            raise ValueError("times must be a non-empty one-dimensional sequence")
        self._validate_times(time_values)
        noise = self._prepare_inference_noise(base_noise, x1)

        batch_size, num_noise, _ = noise.shape
        num_times = time_values.numel()
        x1_grid = x1[:, None, None, :].expand(-1, num_noise, num_times, -1)
        x0_grid = noise[:, :, None, :].expand(-1, -1, num_times, -1)
        time_grid = time_values[None, None, :, None]
        xt = (1.0 - time_grid) * x0_grid + time_grid * x1_grid
        target_velocity = x1_grid - x0_grid

        denoiser = self.ema_dmodel if use_ema else self.dmodel
        was_training = denoiser.training
        denoiser.eval()
        try:
            pred_velocity = self._predict_velocity(
                denoiser,
                xt.reshape(batch_size * num_noise * num_times, self.input_dim),
                time_values[None, None, :]
                .expand(batch_size, num_noise, -1)
                .reshape(-1),
            ).reshape(batch_size, num_noise, num_times, self.input_dim)
        finally:
            denoiser.train(was_training)

        return torch.square(pred_velocity - target_velocity).mean(dim=-1)

    @torch.no_grad()
    def aggregate_mismatch(
        self,
        x1: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Aggregate mismatch over noise draws and ``t^2``-weighted time points."""
        time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
        weights = torch.square(time_values)
        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("the sum of squared time weights must be positive and finite")
        mismatch = self.mismatch_per_time(
            x1,
            time_values,
            base_noise,
            use_ema=use_ema,
        )
        per_time = mismatch.mean(dim=1)
        return (per_time * weights.unsqueeze(0)).sum(dim=1) / weight_sum

    def _predict_velocity(
        self,
        denoiser: nn.Module,
        xt: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        scaled_times = times.to(dtype=torch.float32) * self.time_embed_scale
        return denoiser(xt, timestep=scaled_times)

    def _prepare_inference_noise(
        self,
        base_noise: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.as_tensor(base_noise, device=x1.device, dtype=x1.dtype)
        batch_size = x1.shape[0]

        if noise.ndim == 1 and noise.shape[0] == self.input_dim:
            noise = noise.reshape(1, 1, self.input_dim).expand(batch_size, -1, -1)
        elif noise.ndim == 2 and noise.shape[1] == self.input_dim:
            noise = noise.unsqueeze(0).expand(batch_size, -1, -1)
        elif noise.ndim == 3 and noise.shape[1:] == (
            self.num_obs_steps,
            self.input_channel,
        ):
            noise = noise.reshape(1, noise.shape[0], self.input_dim).expand(
                batch_size, -1, -1
            )
        elif (
            noise.ndim == 3
            and noise.shape[0] == batch_size
            and noise.shape[2] == self.input_dim
        ):
            pass
        else:
            raise ValueError(
                "base_noise must have shape [D], [K,D], [K,H,F], or [B,K,D]; "
                f"got {tuple(noise.shape)}"
            )

        if noise.shape[1] == 0 or not torch.isfinite(noise).all():
            raise ValueError("base_noise must contain at least one finite noise draw")
        return noise

    def _validate_flat_samples(self, samples: torch.Tensor) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.input_dim:
            raise ValueError(
                f"samples must have shape [B,{self.input_dim}], got {tuple(samples.shape)}"
            )
        if not torch.is_floating_point(samples):
            raise TypeError("samples must use a floating-point dtype")

    def _validate_motion_samples(self, samples: torch.Tensor) -> None:
        is_flat = samples.ndim == 2 and samples.shape[1] == self.input_dim
        is_window = samples.ndim == 3 and samples.shape[1:] == (
            self.num_obs_steps,
            self.input_channel,
        )
        if not (is_flat or is_window):
            raise ValueError(
                f"samples must have shape [B,{self.input_dim}] or "
                f"[B,{self.num_obs_steps},{self.input_channel}], got {tuple(samples.shape)}"
            )
        if not torch.is_floating_point(samples):
            raise TypeError("samples must use a floating-point dtype")

    @staticmethod
    def _validate_times(times: torch.Tensor) -> None:
        if not torch.isfinite(times).all() or torch.any(times < 0) or torch.any(times > 1):
            raise ValueError("flow times must be finite values in [0, 1]")
