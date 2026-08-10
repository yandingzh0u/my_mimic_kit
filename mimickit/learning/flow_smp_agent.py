import math
import os

import torch
import yaml

import learning.ppo_agent as ppo_agent
import learning.smp_model as smp_model

from learning.flow_matching.flow_matching_model import FlowMatchingModel


_FLOW_FORMAT_VERSION = 1
_FLOW_MODEL_TYPE = "unconditional_flow_matching"


class FlowSMPAgent(ppo_agent.PPOAgent):
    """PPO with a frozen, unconditional flow-matching motion prior reward."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

    def _load_params(self, config):
        super()._load_params(config)

        if config.get("enable_gsi", False):
            raise ValueError("Flow-SMP R1 does not support GSI")

        self._smp_eval_batch_size = int(config["smp_eval_batch_size"])
        self._flow_reward_alpha = float(config.get("flow_reward_alpha", 1.0))
        self._smp_reward_scale = float(config.get("smp_reward_scale", 1.0))
        self._task_reward_weight = float(config["task_reward_weight"])
        self._smp_reward_weight = float(config["smp_reward_weight"])

        if self._smp_eval_batch_size <= 0:
            raise ValueError("smp_eval_batch_size must be positive")
        if self._flow_reward_alpha <= 0.0:
            raise ValueError("flow_reward_alpha must be positive")
        if self._smp_reward_scale <= 0.0:
            raise ValueError("smp_reward_scale must be positive")

    def _build_model(self, config):
        self._model = smp_model.SMPModel(config["model"], self._env)
        self._build_prior_model(config)

    def _build_prior_model(self, agent_config):
        prior_cfg_path = agent_config["flow_prior_cfg"]
        prior_model_path = agent_config["flow_prior_model"]
        if not os.path.isfile(prior_cfg_path):
            raise FileNotFoundError("Missing Flow-SMP prior config: {}".format(prior_cfg_path))
        if not os.path.isfile(prior_model_path):
            raise FileNotFoundError("Missing Flow-SMP prior model: {}".format(prior_model_path))

        with open(prior_cfg_path, "r") as stream:
            prior_config = yaml.safe_load(stream)

        disc_obs_space = self._env.get_disc_obs_space()
        input_dim = int(disc_obs_space.shape[-1])
        window_steps = int(self._env._num_disc_obs_steps)
        if input_dim % window_steps != 0:
            raise ValueError(
                "Discriminator observation size {} is not divisible by {} history steps".format(
                    input_dim, window_steps
                )
            )
        frame_dim = input_dim // window_steps

        self._check_prior_env_config(prior_config)
        prior_config["input_dim"] = input_dim
        prior_config["input_channel"] = frame_dim

        checkpoint = torch.load(
            prior_model_path,
            map_location=self._device,
            weights_only=True,
        )
        metadata, calibration = self._validate_checkpoint(
            checkpoint=checkpoint,
            prior_config=prior_config,
            input_dim=input_dim,
            frame_dim=frame_dim,
            window_steps=window_steps,
        )

        self._prior_model = FlowMatchingModel(prior_config, self._device)
        incompatible_keys = self._prior_model.load_state_dict(checkpoint["model_state_dict"])
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Flow-SMP checkpoint state mismatch: missing={}, unexpected={}".format(
                    incompatible_keys.missing_keys, incompatible_keys.unexpected_keys
                )
            )

        self.register_buffer(
            "_flow_times",
            torch.as_tensor(
                calibration["times"], device=self._device, dtype=torch.float32
            ).detach(),
        )
        self.register_buffer(
            "_flow_base_noise",
            torch.as_tensor(
                calibration["base_noise"], device=self._device, dtype=torch.float32
            ).detach(),
        )
        self.register_buffer(
            "_flow_expert_scale",
            torch.as_tensor(
                calibration["expert_scale"], device=self._device, dtype=torch.float32
            ).reshape(()).detach(),
        )

        self._prior_model.eval()
        for param in self._prior_model.parameters():
            param.requires_grad = False

        print(
            "Loaded Flow-SMP prior:",
            "cfg={}, model={}, format=v{}, K={}, times={}".format(
                prior_cfg_path,
                prior_model_path,
                checkpoint["format_version"],
                self._flow_base_noise.shape[0],
                self._flow_times.tolist(),
            ),
        )

    def _validate_checkpoint(
        self, checkpoint, prior_config, input_dim, frame_dim, window_steps
    ):
        required_sections = {
            "model_state_dict",
            "metadata",
            "calibration",
            "offline_validation",
        }
        if not isinstance(checkpoint, dict) or not required_sections.issubset(checkpoint):
            raise ValueError(
                "Flow-SMP checkpoint must contain model_state_dict, metadata, calibration, "
                "and offline_validation"
            )

        metadata = checkpoint["metadata"]
        calibration = checkpoint["calibration"]
        if not isinstance(metadata, dict) or not isinstance(calibration, dict):
            raise ValueError("Flow-SMP metadata and calibration must be dictionaries")

        offline_validation = checkpoint["offline_validation"]
        if not isinstance(offline_validation, dict) or not offline_validation.get(
            "gate_passed", False
        ):
            raise ValueError("Flow-SMP checkpoint did not pass the offline mismatch gate")

        expected_header = {
            "format_version": _FLOW_FORMAT_VERSION,
            "model_type": _FLOW_MODEL_TYPE,
        }
        for key, expected in expected_header.items():
            actual = checkpoint.get(key)
            if actual != expected:
                raise ValueError(
                    "Flow-SMP checkpoint mismatch for {}: checkpoint={}, expected={}".format(
                        key, actual, expected
                    )
                )

        expected_metadata = {
            "input_dim": input_dim,
            "frame_dim": frame_dim,
            "window_steps": window_steps,
            "aggregation": "t_squared_weighted_mean",
        }
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    "Flow-SMP metadata mismatch for {}: checkpoint={}, expected={}".format(
                        key, actual, expected
                    )
                )

        expected_time_scale = float(prior_config["time_embed_scale"])
        actual_time_scale = metadata.get("time_embed_scale")
        if actual_time_scale is None or not math.isclose(
            float(actual_time_scale), expected_time_scale, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                "Flow-SMP metadata mismatch for time_embed_scale: checkpoint={}, config={}".format(
                    actual_time_scale, expected_time_scale
                )
            )

        required_calibration = {"expert_scale", "times", "base_noise"}
        if not required_calibration.issubset(calibration):
            missing = sorted(required_calibration.difference(calibration))
            raise ValueError("Flow-SMP checkpoint calibration is missing {}".format(missing))

        expert_scale = float(
            torch.as_tensor(calibration["expert_scale"], dtype=torch.float64).item()
        )
        if not math.isfinite(expert_scale) or expert_scale <= 0.0:
            raise ValueError("Flow-SMP expert_scale must be finite and positive")

        times = torch.as_tensor(calibration["times"], dtype=torch.float32)
        expected_times = torch.as_tensor(
            prior_config.get("reward_times", [0.25, 0.5, 0.75]),
            device=times.device,
            dtype=torch.float32,
        )
        if times.shape != expected_times.shape or not torch.equal(times, expected_times):
            raise ValueError(
                "Flow-SMP calibration times do not match the prior config: checkpoint={}, config={}".format(
                    times.tolist(), expected_times.tolist()
                )
            )
        if not torch.isfinite(times).all() or not ((times > 0.0) & (times < 1.0)).all():
            raise ValueError("Flow-SMP calibration times must be finite and lie in (0, 1)")

        base_noise = torch.as_tensor(calibration["base_noise"], dtype=torch.float32)
        valid_noise_shapes = {
            (base_noise.shape[0], input_dim),
            (base_noise.shape[0], window_steps, frame_dim),
        } if base_noise.ndim >= 2 else set()
        if tuple(base_noise.shape) not in valid_noise_shapes:
            raise ValueError(
                "Flow-SMP base_noise must have shape [K, {}] or [K, {}, {}], got {}".format(
                    input_dim, window_steps, frame_dim, tuple(base_noise.shape)
                )
            )
        if base_noise.shape[0] not in (1, 2):
            raise ValueError("Flow-SMP R1 supports K=1 or K=2 fixed base noises")
        if metadata.get("reward_noise_samples") != base_noise.shape[0]:
            raise ValueError(
                "Flow-SMP reward_noise_samples mismatch: metadata={}, calibration={}".format(
                    metadata.get("reward_noise_samples"), base_noise.shape[0]
                )
            )
        if not torch.isfinite(base_noise).all():
            raise ValueError("Flow-SMP base_noise contains non-finite values")
        if base_noise.shape[0] == 2 and not torch.allclose(
            base_noise[0], -base_noise[1], rtol=1e-5, atol=1e-6
        ):
            raise ValueError("Flow-SMP K=2 calibration noise must be antithetic")

        return metadata, calibration

    def _check_prior_env_config(self, prior_config):
        with open(prior_config["env_config"], "r") as stream:
            prior_env_config = yaml.safe_load(stream)

        env_checks = [
            ("global_obs", self._env._global_obs, False),
            ("root_height_obs", self._env._root_height_obs, False),
            ("enable_tar_obs", self._env._enable_tar_obs, False),
            ("num_disc_obs_steps", self._env._num_disc_obs_steps, None),
            ("disc_dof_vel_obs", self._env._disc_dof_vel_obs, False),
        ]
        for key, env_value, default_value in env_checks:
            prior_value = prior_env_config.get(key, default_value)
            if prior_value != env_value:
                raise ValueError(
                    "Flow-SMP prior env mismatch for {}: prior={}, env={}".format(
                        key, prior_value, env_value
                    )
                )

        prior_key_bodies = prior_env_config.get("key_bodies", [])
        env_num_key_bodies = len(self._env._key_body_ids)
        if len(prior_key_bodies) != env_num_key_bodies:
            raise ValueError(
                "Flow-SMP prior env mismatch for key_bodies: prior={}, env={}".format(
                    len(prior_key_bodies), env_num_key_bodies
                )
            )

        prior_control_freq = int(prior_config["control_freq"])
        env_control_freq = int(round(1.0 / self._env._engine.get_timestep()))
        if prior_control_freq != env_control_freq:
            raise ValueError(
                "Flow-SMP prior config mismatch for control_freq: prior={}, env={}".format(
                    prior_control_freq, env_control_freq
                )
            )

    def _record_data_post_step(self, next_obs, reward, done, next_info):
        super()._record_data_post_step(next_obs, reward, done, next_info)
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])

    def _build_train_data(self):
        reward_info = self._compute_rewards()
        train_info = super()._build_train_data()
        return {**train_info, **reward_info}

    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")

        batch_size = disc_obs.shape[0]
        disc_obs = disc_obs.reshape(batch_size, -1)
        norm_disc_obs = self._prior_model.normalize(disc_obs)
        smp_reward, flow_info = self._calc_smp_rewards(norm_disc_obs)

        reward = (
            self._task_reward_weight * task_reward
            + self._smp_reward_weight * smp_reward
        )
        self._exp_buffer.set_data_flat("reward", reward)

        flow_info["smp_reward_mean"] = torch.mean(smp_reward)
        flow_info["smp_reward_std"] = torch.std(smp_reward, unbiased=False)
        return flow_info

    @torch.no_grad()
    def _calc_smp_rewards(self, norm_disc_obs):
        mismatch_parts = []
        for start in range(0, norm_disc_obs.shape[0], self._smp_eval_batch_size):
            stop = start + self._smp_eval_batch_size
            mismatch = self._prior_model.aggregate_mismatch(
                norm_disc_obs[start:stop],
                times=self._flow_times,
                base_noise=self._flow_base_noise,
                use_ema=True,
            )
            if mismatch.ndim != 1:
                raise RuntimeError(
                    "FlowMatchingModel.aggregate_mismatch must return [batch], got {}".format(
                        tuple(mismatch.shape)
                    )
                )
            mismatch_parts.append(mismatch)

        raw_mismatch = torch.cat(mismatch_parts, dim=0)
        if not torch.isfinite(raw_mismatch).all():
            raise FloatingPointError("Flow-SMP mismatch contains non-finite values")

        scaled_mismatch = raw_mismatch / self._flow_expert_scale
        smp_reward = self._smp_reward_scale * torch.exp(
            -self._flow_reward_alpha * scaled_mismatch
        )

        quantiles = torch.tensor(
            [0.05, 0.50, 0.95], device=raw_mismatch.device, dtype=raw_mismatch.dtype
        )
        raw_q = torch.quantile(raw_mismatch, quantiles)
        scaled_q = torch.quantile(scaled_mismatch, quantiles)
        flow_info = {
            "flow_mismatch_raw_mean": torch.mean(raw_mismatch),
            "flow_mismatch_raw_std": torch.std(raw_mismatch, unbiased=False),
            "flow_mismatch_raw_p05": raw_q[0],
            "flow_mismatch_raw_p50": raw_q[1],
            "flow_mismatch_raw_p95": raw_q[2],
            "flow_mismatch_scaled_mean": torch.mean(scaled_mismatch),
            "flow_mismatch_scaled_std": torch.std(scaled_mismatch, unbiased=False),
            "flow_mismatch_scaled_p05": scaled_q[0],
            "flow_mismatch_scaled_p50": scaled_q[1],
            "flow_mismatch_scaled_p95": scaled_q[2],
            "flow_reward_saturated_high_frac": torch.mean(
                (smp_reward >= 0.95 * self._smp_reward_scale).float()
            ),
            "flow_reward_saturated_low_frac": torch.mean(
                (smp_reward <= 0.05 * self._smp_reward_scale).float()
            ),
        }
        return smp_reward, flow_info
