from __future__ import annotations

import math
import os

import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.ppo_agent as ppo_agent
import learning.rl_util as rl_util
import learning.skill_conditioned_ppo_model as skill_ppo_model
import util.mp_util as mp_util
import util.torch_util as torch_util
from learning.flow_matching.conditional_flow_matching_model import (
    CONDITIONAL_FLOW_FORMAT_VERSION,
    CONDITIONAL_FLOW_MODEL_TYPE,
    ConditionalFlowMatchingModel,
)
from learning.skill_conditioned_runtime import assert_dataset_manifest_equal
from learning.skill_encoder.skill_encoder_model import LabelFreeSkillEncoder


class SkillConditionedFlowAgent(ppo_agent.PPOAgent):
    """R2 PPO with one frozen encoder command and conditional-flow reward only."""

    def _load_params(self, config):
        super()._load_params(config)
        if config.get("enable_gsi", False):
            raise ValueError("R2 forbids GSI")
        self._prior_path = config["conditional_prior_model"]
        self._flow_reward_alpha = float(config.get("flow_reward_alpha", 0.003))
        self._smp_eval_batch_size = int(config["smp_eval_batch_size"])
        if not math.isfinite(self._flow_reward_alpha) or self._flow_reward_alpha <= 0:
            raise ValueError("flow_reward_alpha must be finite and positive")
        if self._smp_eval_batch_size <= 0:
            raise ValueError("smp_eval_batch_size must be positive")

    def _build_model(self, config):
        self._model = skill_ppo_model.SkillConditionedPPOModel(
            config["model"], self._env, latent_dim=8
        )
        self._build_frozen_prior()
        self.register_buffer(
            "_current_latent",
            torch.zeros(self.get_num_envs(), 8, device=self._device),
            persistent=False,
        )

    def _build_frozen_prior(self):
        for method in ("get_skill_dataset_manifest", "get_skill_reset_context"):
            if not callable(getattr(self._env, method, None)):
                raise TypeError(
                    "R2 requires an environment with public {}()".format(method)
                )
        if not os.path.isfile(self._prior_path):
            raise FileNotFoundError("Missing R2 conditional prior: {}".format(self._prior_path))
        checkpoint = torch.load(self._prior_path, map_location=self._device, weights_only=True)
        model_config, metadata, encoder_schema, calibration = self._validate_artifact(
            checkpoint
        )

        self._prior_model = ConditionalFlowMatchingModel(model_config, self._device)
        incompatible = self._prior_model.load_state_dict(checkpoint["model_state_dict"])
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "conditional prior state mismatch: missing={}, unexpected={}".format(
                    incompatible.missing_keys, incompatible.unexpected_keys
                )
            )
        self._skill_encoder = LabelFreeSkillEncoder(
            feature_dim=int(encoder_schema["feature_dim"]),
            embedding_dim=int(encoder_schema["embedding_dim"]),
            hidden_dim=int(encoder_schema["hidden_dim"]),
            num_layers=int(encoder_schema["num_layers"]),
        ).to(self._device)
        incompatible = self._skill_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "skill encoder state mismatch: missing={}, unexpected={}".format(
                    incompatible.missing_keys, incompatible.unexpected_keys
                )
            )

        self._encoder_feature_schema = encoder_schema["feature_schema"]
        self.register_buffer(
            "_flow_times",
            torch.as_tensor(calibration["times"], device=self._device, dtype=torch.float32),
        )
        self.register_buffer(
            "_flow_base_noise",
            torch.as_tensor(
                calibration["base_noise"], device=self._device, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "_conditional_expert_scale",
            torch.as_tensor(
                calibration["conditional_expert_scale"],
                device=self._device,
                dtype=torch.float32,
            ).reshape(()),
        )
        self._freeze_priors()
        print(
            "Loaded R2 conditional prior: model={}, format=v{}, K={}, alpha={}".format(
                self._prior_path,
                checkpoint["format_version"],
                self._flow_base_noise.shape[0],
                self._flow_reward_alpha,
            )
        )

    def _validate_artifact(self, checkpoint):
        required = {
            "format_version",
            "model_type",
            "model_config",
            "model_state_dict",
            "encoder_state_dict",
            "metadata",
            "calibration",
            "offline_validation",
            "encoder_gate",
        }
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError(
                "R2 artifact is not self-describing; missing {}".format(
                    sorted(required.difference(checkpoint if isinstance(checkpoint, dict) else {}))
                )
            )
        if checkpoint["format_version"] != CONDITIONAL_FLOW_FORMAT_VERSION:
            raise ValueError("R2 artifact format_version must be 2")
        if checkpoint["model_type"] != CONDITIONAL_FLOW_MODEL_TYPE:
            raise ValueError("R2 artifact model_type must be conditional_flow_matching")
        for gate_name in ("offline_validation", "encoder_gate"):
            gate = checkpoint[gate_name]
            if not isinstance(gate, dict) or not gate.get("gate_passed", False):
                raise ValueError("R2 artifact failed {} gate".format(gate_name))

        model_config = checkpoint["model_config"]
        metadata = checkpoint["metadata"]
        calibration = checkpoint["calibration"]
        if not all(isinstance(item, dict) for item in (model_config, metadata, calibration)):
            raise ValueError("R2 model_config, metadata, and calibration must be dictionaries")
        encoder_schema = metadata.get("encoder_schema")
        if not isinstance(encoder_schema, dict):
            raise ValueError("R2 metadata is missing encoder_schema")

        disc_space = self._env.get_disc_obs_space()
        input_dim = int(disc_space.shape[-1])
        expected_metadata = {
            "input_dim": input_dim,
            "window_steps": 10,
            "latent_dim": 8,
            "condition_mode": "continuous_or_null",
            "runtime_embedding": "l2_normalize(y)",
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    "R2 metadata mismatch for {}: artifact={}, runtime={}".format(
                        key, metadata.get(key), expected
                    )
                )
        if int(model_config.get("input_dim", -1)) != input_dim:
            raise ValueError("R2 model_config input_dim does not match the environment")
        if int(model_config.get("latent_dim", -1)) != 8:
            raise ValueError("R2 model_config latent_dim must be 8")
        if model_config.get("enforce_unit_latent", True) is not True:
            raise ValueError("R2 runtime forbids disabling the unit-latent contract")
        configured_steps = model_config.get(
            "num_disc_obs_steps", model_config.get("num_obs_steps")
        )
        if int(configured_steps) != 10:
            raise ValueError("R2 model_config must describe H=10 motion windows")
        if float(model_config.get("time_embed_scale", -1)) != float(
            metadata.get("time_embed_scale", -2)
        ):
            raise ValueError("R2 time_embed_scale differs between config and metadata")
        frame_dim = input_dim // 10
        if input_dim % 10 or metadata.get("frame_dim") != frame_dim:
            raise ValueError("R2 frame/window dimensions do not match the environment")
        if metadata.get("aggregation") != "t_squared_weighted_mean":
            raise ValueError("R2 metadata must use t_squared_weighted_mean aggregation")
        condition_schema = metadata.get("condition_schema")
        expected_condition = {
            "type": "continuous_latent_with_learned_null",
            "latent_dim": 8,
            "runtime_embedding": "l2_normalize(y)",
            "conditional_latent_norm": "unit_l2",
            "aggregation": "t_squared_weighted_mean",
        }
        if not isinstance(condition_schema, dict):
            raise ValueError("R2 metadata is missing condition_schema")
        for key, expected in expected_condition.items():
            if condition_schema.get(key) != expected:
                raise ValueError("R2 condition_schema mismatch for {}".format(key))

        expected_encoder = {
            "feature_dim": 44,
            "view_steps": 20,
            "embedding_dim": 8,
        }
        for key, expected in expected_encoder.items():
            if encoder_schema.get(key) != expected:
                raise ValueError(
                    "R2 encoder schema mismatch for {}: {} != {}".format(
                        key, encoder_schema.get(key), expected
                    )
                )
        for key in ("hidden_dim", "num_layers"):
            if not isinstance(encoder_schema.get(key), int) or encoder_schema[key] <= 0:
                raise ValueError("R2 encoder schema requires positive {}".format(key))
        feature_schema = encoder_schema.get("feature_schema")
        if not isinstance(feature_schema, dict) or feature_schema.get("feature_dim") != 44:
            raise ValueError("R2 encoder feature_schema must describe the frozen 44-D features")
        if len(feature_schema.get("foot_body_names", [])) != 2 or len(
            feature_schema.get("foot_body_ids", [])
        ) != 2:
            raise ValueError("R2 feature_schema is missing bilateral foot names/ids")
        proxy = feature_schema.get("contact_proxy")
        if not isinstance(proxy, dict) or not {
            "ground_height",
            "height_threshold",
            "speed_threshold",
        }.issubset(proxy):
            raise ValueError("R2 feature_schema is missing frozen contact thresholds")

        top_manifest = metadata.get("dataset_manifest")
        encoder_manifest = encoder_schema.get("dataset_manifest")
        if not isinstance(top_manifest, dict) or top_manifest != encoder_manifest:
            raise ValueError("top-level and encoder dataset_manifest differ")
        runtime_manifest = self._env.get_skill_dataset_manifest()
        assert_dataset_manifest_equal(top_manifest, runtime_manifest, length_tol=1e-5)

        needed_calibration = {"conditional_expert_scale", "times", "base_noise"}
        if not needed_calibration.issubset(calibration):
            raise ValueError("R2 calibration is missing {}".format(
                sorted(needed_calibration.difference(calibration))
            ))
        scale = float(torch.as_tensor(calibration["conditional_expert_scale"]).item())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("conditional_expert_scale must be finite and positive")
        times = torch.as_tensor(calibration["times"], dtype=torch.float32)
        expected_times = torch.tensor(
            [0.25, 0.5, 0.75], device=times.device, dtype=times.dtype
        )
        if not torch.equal(times, expected_times):
            raise ValueError("R2 calibration times must be exactly [.25,.5,.75]")
        noise = torch.as_tensor(calibration["base_noise"], dtype=torch.float32)
        valid_shapes = {(1, input_dim), (2, input_dim), (1, 10, frame_dim), (2, 10, frame_dim)}
        if tuple(noise.shape) not in valid_shapes or not torch.isfinite(noise).all():
            raise ValueError("R2 base_noise has an invalid shape or non-finite values")
        if metadata.get("reward_noise_samples", metadata.get("K")) != noise.shape[0]:
            raise ValueError("R2 reward noise count differs between metadata and calibration")
        if noise.shape[0] == 2 and not torch.allclose(
            noise[0], -noise[1], rtol=1e-5, atol=1e-6
        ):
            raise ValueError("R2 K=2 base_noise must be antithetic")
        return model_config, metadata, encoder_schema, calibration

    def _freeze_priors(self):
        self._prior_model.eval()
        self._skill_encoder.eval()
        for module in (self._prior_model, self._skill_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "_prior_model"):
            self._freeze_priors()
        return self

    @torch.no_grad()
    def _reset_envs(self, env_ids=None):
        obs, info = super()._reset_envs(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long).flatten()
        if env_ids.numel() > 0:
            context = self._env.get_skill_reset_context(
                env_ids=env_ids,
                steps=20,
                feature_schema=self._encoder_feature_schema,
            )
            latent = self._skill_encoder.runtime_z(context["features"])
            self._assert_runtime_latent(latent)
            self._current_latent[env_ids] = latent
        return obs, info

    def set_skill_command(self, *, motion_path=None, clip_sha256=None, context_start_sec):
        setter = getattr(self._env, "set_skill_command", None)
        if not callable(setter):
            raise TypeError("the R2 environment does not support external skill commands")
        command = setter(
            motion_path=motion_path,
            clip_sha256=clip_sha256,
            context_start_sec=context_start_sec,
        )
        if hasattr(self, "_curr_obs"):
            self._curr_obs, self._curr_info = self._reset_envs()
        return command

    @torch.no_grad()
    def set_evaluation_skill_context(
        self, *, motion_path=None, clip_sha256=None, context_start_sec
    ):
        """Set the current test episode's z from a legal expert context only.

        The command is deliberately non-persistent: the next environment reset
        restores the latent derived from that reset's own expert context.  This
        keeps the API useful for paired counterfactual evaluation without
        creating an arbitrary-z training path.
        """
        context, latent = self._encode_evaluation_skill_context(
            motion_path=motion_path,
            clip_sha256=clip_sha256,
            context_start_sec=context_start_sec,
        )
        self._current_latent[:] = latent.expand(self.get_num_envs(), -1)
        return {
            **context,
            "latent": latent[0].detach().cpu().clone(),
        }

    @torch.no_grad()
    def score_evaluation_windows(
        self, disc_obs, *, motion_path=None, clip_sha256=None, context_start_sec
    ):
        """Score observed W under a manifest-backed expert condition."""
        context, latent = self._encode_evaluation_skill_context(
            motion_path=motion_path,
            clip_sha256=clip_sha256,
            context_start_sec=context_start_sec,
        )
        windows = torch.as_tensor(disc_obs, device=self._device)
        if windows.ndim == 1:
            windows = windows.unsqueeze(0)
        windows = windows.reshape(windows.shape[0], -1).to(dtype=torch.float32)
        expected_dim = int(self._prior_model.input_dim)
        if (
            windows.shape[0] == 0
            or windows.shape[1] != expected_dim
            or not torch.isfinite(windows).all()
        ):
            raise ValueError(
                "evaluation windows must be finite [B,{}] tensors".format(expected_dim)
            )
        normalized = self._prior_model.normalize(windows)
        condition = latent.expand(normalized.shape[0], -1)
        mismatch_parts = []
        for start in range(0, normalized.shape[0], self._smp_eval_batch_size):
            stop = start + self._smp_eval_batch_size
            mismatch_parts.append(
                self._prior_model.aggregate_mismatch(
                    normalized[start:stop],
                    condition[start:stop],
                    self._flow_times,
                    self._flow_base_noise,
                    use_ema=True,
                )
            )
        raw = torch.cat(mismatch_parts)
        if raw.ndim != 1 or not torch.isfinite(raw).all():
            raise FloatingPointError("evaluation mismatch must be finite [B]")
        return {
            **context,
            "latent": latent[0].detach().cpu().clone(),
            "raw_mismatch": raw.detach().cpu(),
            "scaled_mismatch": (raw / self._conditional_expert_scale).detach().cpu(),
        }

    def _encode_evaluation_skill_context(
        self, *, motion_path=None, clip_sha256=None, context_start_sec
    ):
        if self._mode != base_agent.AgentMode.TEST:
            raise ValueError("expert-context injection is available only in test mode")
        getter = getattr(self._env, "get_expert_skill_context", None)
        if not callable(getter):
            raise TypeError("the R2 environment lacks the expert-context evaluation API")
        payload = getter(
            motion_path=motion_path,
            clip_sha256=clip_sha256,
            context_start_sec=context_start_sec,
            steps=20,
            feature_schema=self._encoder_feature_schema,
        )
        if not isinstance(payload, dict) or "features" not in payload:
            raise TypeError("expert-context evaluation payload is malformed")
        latent = self._skill_encoder.runtime_z(payload["features"])
        self._assert_runtime_latent(latent)
        if latent.shape[0] != 1:
            raise ValueError("expert-context evaluation must encode exactly one context")
        context = {
            key: payload[key]
            for key in (
                "motion_id",
                "motion_path",
                "clip_sha256",
                "context_start_sec",
            )
        }
        return context, latent

    @staticmethod
    def _assert_runtime_latent(latent):
        if latent.ndim != 2 or latent.shape[-1] != 8 or not torch.isfinite(latent).all():
            raise ValueError("runtime encoder must return finite [B,8] latents")
        norms = torch.linalg.vector_norm(latent.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-3):
            raise ValueError("runtime encoder latents must have unit L2 norm")

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs, self._current_latent)
        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode
            exp_prob = torch.full(
                [norm_a_rand.shape[0], 1],
                self._get_exp_prob(),
                device=self._device,
                dtype=torch.float,
            )
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)
        elif self._mode == base_agent.AgentMode.TEST:
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
        else:
            raise ValueError("unsupported agent mode")
        norm_a_logp = norm_action_dist.log_prob(norm_a)
        norm_a = norm_a.detach()
        return self._a_norm.unnormalize(norm_a), {
            "a_logp": norm_a_logp.detach(),
            "rand_action_mask": rand_action_mask,
        }

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("latent", self._current_latent)

    def _record_data_post_step(self, next_obs, reward, done, next_info):
        super()._record_data_post_step(next_obs, reward, done, next_info)
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])

    def _build_train_data(self):
        reward_info = self._compute_rewards()
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        reward = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        latent = self._exp_buffer.get_data("latent")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_inputs = {"obs": norm_next_obs, "latent": latent}
        next_vals = torch_util.eval_minibatch(
            self._model.eval_critic, next_inputs, self._critic_eval_batch_size
        ).squeeze(-1).detach()
        next_vals[done == base_env.DoneFlags.SUCC.value] = self._compute_succ_val()
        next_vals[done == base_env.DoneFlags.FAIL.value] = self._compute_fail_val()
        new_vals = rl_util.compute_td_lambda_return(
            reward, next_vals, done, self._discount, self._td_lambda
        )

        critic_inputs = {"obs": self._obs_norm.normalize(obs), "latent": latent}
        vals = torch_util.eval_minibatch(
            self._model.eval_critic, critic_inputs, self._critic_eval_batch_size
        ).squeeze(-1).detach()
        adv = new_vals - vals
        random_mask = (rand_action_mask == 1.0).flatten()
        random_adv = adv.flatten()[random_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(random_adv)
        norm_adv = torch.clamp(
            (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5),
            -self._norm_adv_clip,
            self._norm_adv_clip,
        )
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        return {
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            **reward_info,
            **self._latent_diagnostics(),
        }

    def _compute_critic_loss(self, batch):
        pred = self._model.eval_critic(
            self._obs_norm.normalize(batch["obs"]), batch["latent"]
        ).squeeze(-1)
        loss = torch.mean(torch.square(batch["tar_val"] - pred))
        return {"critic_loss": loss}

    def _compute_actor_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        latent = batch["latent"]
        norm_a = self._a_norm.normalize(batch["action"])
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        mask = batch["rand_action_mask"] == 1.0
        norm_obs, latent, norm_a = norm_obs[mask], latent[mask], norm_a[mask]
        old_a_logp, adv = old_a_logp[mask], adv[mask]
        a_dist = self._model.eval_actor(norm_obs, latent)
        a_logp = a_dist.log_prob(norm_a)
        ratio = torch.exp(a_logp - old_a_logp)
        actor_loss = -torch.minimum(
            adv * ratio,
            adv * torch.clamp(
                ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio
            ),
        ).mean()
        info = {
            "actor_loss": actor_loss,
            "clip_frac": (torch.abs(ratio - 1.0) > self._ppo_clip_ratio).float().mean().detach(),
            "imp_ratio": ratio.mean().detach(),
        }
        if self._action_bound_weight != 0:
            value = self._compute_action_bound_loss(a_dist)
            if value is not None:
                value = value.mean()
                actor_loss = actor_loss + self._action_bound_weight * value
                info["action_bound_loss"] = value.detach()
        if self._action_entropy_weight != 0:
            value = a_dist.entropy().mean()
            actor_loss = actor_loss - self._action_entropy_weight * value
            info["action_entropy"] = value.detach()
        if self._action_reg_weight != 0:
            value = a_dist.param_reg().mean()
            actor_loss = actor_loss + self._action_reg_weight * value
            info["action_reg_loss"] = value.detach()
        info["actor_loss"] = actor_loss
        return info

    @torch.no_grad()
    def _compute_rewards(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs").reshape(
            self._exp_buffer.get_data_flat("disc_obs").shape[0], -1
        )
        latent = self._exp_buffer.get_data_flat("latent")
        self._assert_runtime_latent(latent)
        norm_disc_obs = self._prior_model.normalize(disc_obs)
        mismatch_parts = []
        for start in range(0, norm_disc_obs.shape[0], self._smp_eval_batch_size):
            stop = start + self._smp_eval_batch_size
            mismatch_parts.append(
                self._prior_model.aggregate_mismatch(
                    norm_disc_obs[start:stop],
                    latent[start:stop],
                    self._flow_times,
                    self._flow_base_noise,
                    use_ema=True,
                )
            )
        raw = torch.cat(mismatch_parts)
        if raw.ndim != 1 or not torch.isfinite(raw).all():
            raise FloatingPointError("conditional mismatch must be finite [B]")
        scaled = raw / self._conditional_expert_scale
        reward = torch.exp(-self._flow_reward_alpha * scaled)
        self._exp_buffer.set_data_flat("reward", reward)
        q = torch.quantile(scaled, torch.tensor([0.05, 0.5, 0.95], device=scaled.device))
        return {
            "conditional_mismatch_scaled_mean": scaled.mean(),
            "conditional_mismatch_scaled_std": scaled.std(unbiased=False),
            "conditional_mismatch_scaled_p05": q[0],
            "conditional_mismatch_scaled_p50": q[1],
            "conditional_mismatch_scaled_p95": q[2],
            "conditional_reward_mean": reward.mean(),
            "conditional_reward_std": reward.std(unbiased=False),
        }

    @torch.no_grad()
    def _latent_diagnostics(self):
        latent = self._exp_buffer.get_data("latent")
        norms = torch.linalg.vector_norm(latent, dim=-1)
        if latent.shape[0] > 1:
            active = self._exp_buffer.get_data("done")[:-1] == base_env.DoneFlags.NULL.value
            delta = torch.linalg.vector_norm(latent[1:] - latent[:-1], dim=-1)
            stability = (delta[active] <= 1e-6).float().mean() if active.any() else latent.new_tensor(1.0)
        else:
            stability = latent.new_tensor(1.0)
        return {
            "latent_norm_mean": norms.mean(),
            "latent_norm_std": norms.std(unbiased=False),
            "latent_component_std_mean": latent.reshape(-1, 8).std(dim=0, unbiased=False).mean(),
            "latent_command_stability": stability,
        }
