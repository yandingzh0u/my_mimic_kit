from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from learning.flow_matching.flow_matching_model import FlowMatchingModel
from learning.normalizer import Normalizer
from learning.tinymdm.EMA import EMA
from learning.tinymdm.arch import TinyStableMotionDiTModel


CONDITIONAL_FLOW_FORMAT_VERSION = 2
CONDITIONAL_FLOW_MODEL_TYPE = "conditional_flow_matching"
DEFAULT_REWARD_TIMES = (0.25, 0.5, 0.75)


class LatentConditionedMotionDiT(TinyStableMotionDiTModel):
    """The existing motion DiT with continuous conditioning in its AdaLN path."""

    def __init__(self, *, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.null_condition = nn.Parameter(torch.zeros(self.latent_dim))
        self.condition_embedder = nn.Sequential(
            nn.Linear(self.latent_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        latent_condition: torch.Tensor | None = None,
        null_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        condition = self._resolve_condition(
            latent_condition,
            null_mask,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.reshape(batch_size, -1, self.in_channels)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = self.proj_in(hidden_states.transpose(1, 2))

        timestep_embedding = self.adaln_single.emb(timestep)
        condition_embedding = self.condition_embedder(condition).unsqueeze(1)
        joint_embedding = timestep_embedding + condition_embedding
        time_hidden_states = self.adaln_single.linear(
            self.adaln_single.silu(joint_embedding)
        )

        hidden_states = self.sequence_pos_encoder(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                time_hidden_states=time_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states).transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        return hidden_states.transpose(1, 2).reshape(batch_size, -1)

    def _resolve_condition(
        self,
        latent_condition: torch.Tensor | None,
        null_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if latent_condition is None:
            if null_mask is not None and not torch.as_tensor(null_mask).bool().all():
                raise ValueError("conditional samples require an explicit latent_condition")
            latent = torch.zeros(
                batch_size, self.latent_dim, device=device, dtype=dtype
            )
            mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        else:
            latent = torch.as_tensor(latent_condition, device=device, dtype=dtype)
            if latent.shape != (batch_size, self.latent_dim):
                raise ValueError(
                    "latent_condition must have shape [{},{}], got {}".format(
                        batch_size, self.latent_dim, tuple(latent.shape)
                    )
                )
            if not torch.isfinite(latent).all():
                raise ValueError("latent_condition contains non-finite values")
            if null_mask is None:
                mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            else:
                mask = torch.as_tensor(null_mask, device=device)
                if mask.dtype != torch.bool or mask.shape != (batch_size,):
                    raise ValueError(
                        "null_mask must be a boolean vector with shape [{}]".format(
                            batch_size
                        )
                    )

        null_condition = self.null_condition.to(dtype=dtype).unsqueeze(0)
        return torch.where(mask.unsqueeze(1), null_condition, latent)


class ConditionalFlowMatchingModel(FlowMatchingModel):
    """Continuous conditional/NULL rectified-flow prior for R2.

    The trainer owns the NULL sampling probability and passes the sampled
    boolean ``null_mask``. The core never drops conditions implicitly.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device):
        nn.Module.__init__(self)
        self.config = config
        self._device = torch.device(device)
        self._obs_dtype = torch.float32

        arch_name = config.get("arch_name", "ConditionalDiT")
        if arch_name not in ("DiT", "ConditionalDiT"):
            raise ValueError(
                "ConditionalFlowMatchingModel supports only the DiT architecture"
            )

        self.latent_dim = int(config.get("latent_dim", 8))
        if self.latent_dim != 8:
            raise ValueError("R2 requires latent_dim=8")
        self.enforce_unit_latent = bool(config.get("enforce_unit_latent", True))

        self.num_disc_obs_steps = self._infer_num_obs_steps(config)
        self.num_obs_steps = self.num_disc_obs_steps
        self.input_dim = int(config["input_dim"])
        if self.input_dim <= 0 or self.input_dim % self.num_obs_steps != 0:
            raise ValueError(
                "input_dim ({}) must be divisible by num_disc_obs_steps ({})".format(
                    self.input_dim, self.num_obs_steps
                )
            )
        self.input_channel = self.input_dim // self.num_obs_steps
        configured_channel = config.get("input_channel")
        if configured_channel is not None and int(configured_channel) != self.input_channel:
            raise ValueError(
                "input_channel ({}) does not match inferred value ({})".format(
                    configured_channel, self.input_channel
                )
            )

        self.time_embed_scale = float(config.get("time_embed_scale", 49.0))
        if self.time_embed_scale <= 0.0:
            raise ValueError("time_embed_scale must be positive")

        dit_args = {
            "latent_dim": self.latent_dim,
            "in_channels": self.input_channel,
            "num_layers": int(config["num_layers"]),
            "attention_head_dim": int(config.get("attention_head_dim", 64)),
            "num_attention_heads": int(config.get("num_attention_heads", 4)),
            "out_channels": self.input_channel,
            "dropout": float(config.get("dropout", 0.0)),
            "max_seq_len": max(32, self.num_obs_steps),
        }
        self.dmodel = LatentConditionedMotionDiT(**dit_args)
        self.ema_dmodel = EMA(
            self.dmodel,
            beta=float(config.get("model_ema_decay", 0.995)),
            update_every=int(config.get("model_ema_steps", 10)),
            update_after_step=int(config.get("model_ema_update_after", 5_000)),
        )
        self.model_ema = True
        self.obs_normalizer = Normalizer(
            self.input_channel,
            device=self._device,
            dtype=self._obs_dtype,
            std_clip=config.get("normalizer_std_clip", 0.2),
        )

    def forward(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor | None,
        *,
        null_mask: torch.Tensor | None = None,
        base_noise: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the standard conditional flow-matching velocity MSE."""
        self._validate_flat_samples(x1)
        self._validate_latent(latent_condition, x1.shape[0], allow_none=True)
        self._validate_null_mask(null_mask, x1.shape[0])
        self._validate_unit_latent(latent_condition, null_mask)

        batch_size = x1.shape[0]
        if base_noise is None:
            x0 = torch.randn_like(x1)
        else:
            if base_noise.shape != x1.shape:
                raise ValueError(
                    "training base_noise must have shape {}, got {}".format(
                        tuple(x1.shape), tuple(base_noise.shape)
                    )
                )
            x0 = base_noise.to(device=x1.device, dtype=x1.dtype)

        if times is None:
            time_values = torch.rand(
                batch_size, device=x1.device, dtype=x1.dtype
            )
        else:
            time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
            if time_values.ndim == 0:
                time_values = time_values.expand(batch_size)
            if time_values.shape != (batch_size,):
                raise ValueError(
                    "training times must have shape ({},)".format(batch_size)
                )
            self._validate_times(time_values)

        path_time = time_values.unsqueeze(-1)
        xt = (1.0 - path_time) * x0 + path_time * x1
        target_velocity = x1 - x0
        pred_velocity = self._predict_conditional_velocity(
            self.dmodel,
            xt,
            time_values,
            latent_condition,
            null_mask,
        )
        return torch.mean(torch.square(pred_velocity - target_velocity))

    @torch.no_grad()
    def mismatch_per_time(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor | None,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        null_mask: torch.Tensor | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Return conditional or NULL mismatch with shape ``[B,K,T]``."""
        self._validate_flat_samples(x1)
        batch_size = x1.shape[0]
        self._validate_latent(latent_condition, batch_size, allow_none=True)
        self._validate_null_mask(null_mask, batch_size)
        self._validate_unit_latent(latent_condition, null_mask)

        time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
        if time_values.ndim != 1 or time_values.numel() == 0:
            raise ValueError("times must be a non-empty one-dimensional sequence")
        self._validate_times(time_values)
        noise = self._prepare_inference_noise(base_noise, x1)
        if noise.shape[1] not in (1, 2):
            raise ValueError("R2 mismatch supports K=1 or K=2 base noises")

        num_noise = noise.shape[1]
        num_times = time_values.numel()
        x1_grid = x1[:, None, None, :].expand(-1, num_noise, num_times, -1)
        x0_grid = noise[:, :, None, :].expand(-1, -1, num_times, -1)
        time_grid = time_values[None, None, :, None]
        xt = (1.0 - time_grid) * x0_grid + time_grid * x1_grid
        target_velocity = x1_grid - x0_grid

        expanded_latent, expanded_mask = self._expand_condition_grid(
            latent_condition,
            null_mask,
            batch_size=batch_size,
            num_noise=num_noise,
            num_times=num_times,
        )
        denoiser = self.ema_dmodel if use_ema else self.dmodel
        was_training = denoiser.training
        denoiser.eval()
        try:
            pred_velocity = self._predict_conditional_velocity(
                denoiser,
                xt.reshape(batch_size * num_noise * num_times, self.input_dim),
                time_values[None, None, :]
                .expand(batch_size, num_noise, -1)
                .reshape(-1),
                expanded_latent,
                expanded_mask,
            ).reshape(batch_size, num_noise, num_times, self.input_dim)
        finally:
            denoiser.train(was_training)

        return torch.square(pred_velocity - target_velocity).mean(dim=-1)

    @torch.no_grad()
    def conditional_mismatch(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> torch.Tensor:
        return self.aggregate_mismatch(
            x1,
            latent_condition,
            times,
            base_noise,
            use_ema=use_ema,
        )

    @torch.no_grad()
    def null_mismatch(
        self,
        x1: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> torch.Tensor:
        return self.aggregate_mismatch(
            x1,
            None,
            times,
            base_noise,
            use_ema=use_ema,
        )

    @torch.no_grad()
    def paired_mismatch_per_time(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Evaluate conditional and NULL branches on identical ``x0`` and ``t``."""
        conditional = self.mismatch_per_time(
            x1,
            latent_condition,
            times,
            base_noise,
            use_ema=use_ema,
        )
        null = self.mismatch_per_time(
            x1,
            None,
            times,
            base_noise,
            use_ema=use_ema,
        )
        return {"conditional": conditional, "null": null}

    @torch.no_grad()
    def paired_mismatch(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        use_ema: bool = True,
    ) -> dict[str, torch.Tensor]:
        time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
        weights = torch.square(time_values)
        if not torch.isfinite(weights).all() or weights.sum() <= 0.0:
            raise ValueError("the sum of squared time weights must be positive and finite")
        per_time = self.paired_mismatch_per_time(
            x1,
            latent_condition,
            time_values,
            base_noise,
            use_ema=use_ema,
        )
        return {
            key: self._aggregate_per_time(value, weights)
            for key, value in per_time.items()
        }

    @torch.no_grad()
    def aggregate_mismatch(
        self,
        x1: torch.Tensor,
        latent_condition: torch.Tensor | None,
        times: Sequence[float] | torch.Tensor,
        base_noise: torch.Tensor,
        *,
        null_mask: torch.Tensor | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        time_values = torch.as_tensor(times, device=x1.device, dtype=x1.dtype)
        weights = torch.square(time_values)
        if not torch.isfinite(weights).all() or weights.sum() <= 0.0:
            raise ValueError("the sum of squared time weights must be positive and finite")
        mismatch = self.mismatch_per_time(
            x1,
            latent_condition,
            time_values,
            base_noise,
            null_mask=null_mask,
            use_ema=use_ema,
        )
        return self._aggregate_per_time(mismatch, weights)

    def condition_schema(self) -> dict[str, Any]:
        return {
            "type": "continuous_latent_with_learned_null",
            "latent_dim": self.latent_dim,
            "injection": "timestep_adaln_sum",
            "null_token": "learned",
            "null_sampling": "trainer_supplied_boolean_mask",
            "runtime_embedding": "l2_normalize(y)",
            "conditional_latent_norm": "unit_l2",
            "reward_times": list(DEFAULT_REWARD_TIMES),
            "aggregation": "t_squared_weighted_mean",
            "reward_noise_samples": [1, 2],
        }

    def checkpoint_metadata(
        self, encoder_schema: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(encoder_schema, Mapping):
            raise TypeError("encoder_schema must be a mapping")
        encoder_schema = deepcopy(dict(encoder_schema))
        encoder_latent_dim = encoder_schema.get(
            "latent_dim", encoder_schema.get("output_dim")
        )
        if encoder_latent_dim != self.latent_dim:
            raise ValueError(
                "encoder schema latent dimension must equal {}".format(self.latent_dim)
            )
        metadata = {
            "input_dim": self.input_dim,
            "frame_dim": self.input_channel,
            "window_steps": self.num_obs_steps,
            "time_embed_scale": self.time_embed_scale,
            "latent_dim": self.latent_dim,
            "condition_mode": "continuous_or_null",
            "runtime_embedding": "l2_normalize(y)",
            "encoder_schema": encoder_schema,
            "condition_schema": self.condition_schema(),
        }
        if "dataset_manifest" in encoder_schema:
            metadata["dataset_manifest"] = deepcopy(
                encoder_schema["dataset_manifest"]
            )
        return metadata

    def _predict_conditional_velocity(
        self,
        denoiser: nn.Module,
        xt: torch.Tensor,
        times: torch.Tensor,
        latent_condition: torch.Tensor | None,
        null_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scaled_times = times.to(dtype=torch.float32) * self.time_embed_scale
        return denoiser(
            xt,
            timestep=scaled_times,
            latent_condition=latent_condition,
            null_mask=null_mask,
        )

    def _expand_condition_grid(
        self,
        latent_condition: torch.Tensor | None,
        null_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_noise: int,
        num_times: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if latent_condition is None:
            return None, None
        latent = latent_condition[:, None, None, :].expand(
            batch_size, num_noise, num_times, self.latent_dim
        )
        expanded_latent = latent.reshape(-1, self.latent_dim)
        if null_mask is None:
            return expanded_latent, None
        mask = null_mask[:, None, None].expand(batch_size, num_noise, num_times)
        return expanded_latent, mask.reshape(-1)

    @staticmethod
    def _aggregate_per_time(
        mismatch: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        per_time = mismatch.mean(dim=1)
        return (per_time * weights.unsqueeze(0)).sum(dim=1) / weights.sum()

    def _validate_latent(
        self,
        latent_condition: torch.Tensor | None,
        batch_size: int,
        *,
        allow_none: bool,
    ) -> None:
        if latent_condition is None:
            if allow_none:
                return
            raise ValueError("latent_condition is required")
        if latent_condition.shape != (batch_size, self.latent_dim):
            raise ValueError(
                "latent_condition must have shape [{},{}], got {}".format(
                    batch_size, self.latent_dim, tuple(latent_condition.shape)
                )
            )
        if not torch.is_floating_point(latent_condition):
            raise TypeError("latent_condition must use a floating-point dtype")
        if not torch.isfinite(latent_condition).all():
            raise ValueError("latent_condition contains non-finite values")

    @staticmethod
    def _validate_null_mask(null_mask: torch.Tensor | None, batch_size: int) -> None:
        if null_mask is None:
            return
        if null_mask.dtype != torch.bool or null_mask.shape != (batch_size,):
            raise ValueError(
                "null_mask must be a boolean vector with shape [{}]".format(batch_size)
            )

    def _validate_unit_latent(
        self,
        latent_condition: torch.Tensor | None,
        null_mask: torch.Tensor | None,
    ) -> None:
        if latent_condition is None or not self.enforce_unit_latent:
            return
        conditional = latent_condition if null_mask is None else latent_condition[~null_mask]
        if conditional.numel() == 0:
            return
        norms = torch.linalg.vector_norm(conditional.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-3):
            raise ValueError("non-NULL latent_condition rows must have unit L2 norm")


def conditional_checkpoint_payload(
    model: ConditionalFlowMatchingModel,
    encoder: nn.Module,
    *,
    encoder_schema: Mapping[str, Any],
    calibration: Mapping[str, Any],
    iteration: int,
    offline_validation: Mapping[str, Any] | None = None,
    encoder_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the self-describing R2 model+encoder checkpoint payload."""
    if not isinstance(model, ConditionalFlowMatchingModel):
        raise TypeError("model must be a ConditionalFlowMatchingModel")
    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping")
    if offline_validation is not None and not isinstance(offline_validation, Mapping):
        raise TypeError("offline_validation must be a mapping")
    if encoder_gate is not None and not isinstance(encoder_gate, Mapping):
        raise TypeError("encoder_gate must be a mapping")
    payload = {
        "format_version": CONDITIONAL_FLOW_FORMAT_VERSION,
        "model_type": CONDITIONAL_FLOW_MODEL_TYPE,
        "iteration": int(iteration),
        "model_config": deepcopy(dict(model.config)),
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "metadata": model.checkpoint_metadata(encoder_schema),
        "calibration": deepcopy(dict(calibration)),
    }
    if offline_validation is not None:
        payload["offline_validation"] = deepcopy(dict(offline_validation))
    if encoder_gate is not None:
        payload["encoder_gate"] = deepcopy(dict(encoder_gate))
    return payload
