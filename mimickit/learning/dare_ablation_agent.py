"""Isolated A/B experiment agent; production DAREMonotone stays unchanged."""

import torch

from learning.add_agent import ADDAgent
from learning.dare_monotone_agent import DAREMonotoneAgent
from learning.dare_ablation_model import (
    ABLATION_SCHEMA_VERSION, ABLATION_VARIANTS, DAREAblationModel)
from learning.dare_monotone_model import (
    FEATURE_SLICES, HIDDEN_WIDTHS, METRIC_NAMES, aggregate_error_metrics)
from learning.diff_normalizer import DiffNormalizer
from learning.physical_error_normalizer import PhysicalErrorNormalizer
from learning.ppo_agent import PPOAgent
import util.torch_util as torch_util


class DAREAblationAgent(DAREMonotoneAgent):
    def _load_params(self, config):
        self._ablation_variant = config.get("ablation_variant")
        if self._ablation_variant not in ABLATION_VARIANTS:
            raise ValueError("ablation_variant must be no_a, no_b or no_ab")
        # Retain the production guards: no GP, structured negatives or reward
        # multiplier, and no additional loss or optimization coefficients.
        super()._load_params(config)

    def _build_model(self, config):
        model_config = dict(config["model"], ablation_variant=self._ablation_variant)
        self._model = DAREAblationModel(model_config, self._env)

    def _build_normalizers(self):
        PPOAgent._build_normalizers(self)
        space = self._env.get_disc_obs_space()
        dtype = torch_util.numpy_dtype_to_torch(space.dtype)
        if self._ablation_variant == "no_b":
            self._disc_obs_norm = PhysicalErrorNormalizer(device=self._device, dtype=dtype)
        else:
            self._disc_obs_norm = DiffNormalizer(space.shape, device=self._device, dtype=dtype)

    def _build_pos_diff(self):
        dtype = torch_util.numpy_dtype_to_torch(self._env.get_disc_obs_space().dtype)
        return torch.zeros(self._model.input_dim, device=self._device, dtype=dtype)

    @torch.no_grad()
    def _pair_disc_inputs(self, policy, reference):
        if self._model.physical_errors:
            return self._pair_error_metrics(policy, reference)
        return self._model.pair_to_input(policy, reference)

    @torch.no_grad()
    def calc_disc_reward_from_obs(self, policy, reference):
        inputs = self._pair_disc_inputs(policy, reference)
        if inputs.shape[0] == 0:
            return inputs.new_zeros((0,))
        return self._calc_disc_rewards(self._disc_obs_norm.normalize(inputs))

    @torch.no_grad()
    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        policy = self._exp_buffer.get_data_flat("disc_obs")
        reference = self._exp_buffer.get_data_flat("disc_obs_demo")
        inputs = self._pair_disc_inputs(policy, reference)
        initialized_normalizer = (
            self._need_normalizer_update() and int(self._disc_obs_norm.get_count()) == 0)
        if initialized_normalizer:
            self._disc_obs_norm.record(inputs)
            self._disc_obs_norm.update()
        disc_reward = self._calc_disc_rewards(self._disc_obs_norm.normalize(inputs))
        reward = self._task_reward_weight * task_reward + self._disc_reward_weight * disc_reward
        self._exp_buffer.set_data_flat("reward", reward)
        if self._need_normalizer_update() and not initialized_normalizer:
            self._disc_obs_norm.record(inputs)
        std, mean = torch.std_mean(disc_reward)
        return {"disc_reward_mean": mean, "disc_reward_std": std}

    def _compute_disc_loss(self, batch):
        replay = self._disc_buffer.sample(batch["disc_obs"].shape[0])
        policy = torch.cat((batch["disc_obs"], replay["disc_obs"]), 0)
        reference = torch.cat((batch["disc_obs_demo"], replay["disc_obs_demo"]), 0)
        inputs = self._pair_disc_inputs(policy, reference)
        normalized = self._disc_obs_norm.normalize(inputs)
        scores = self._model.eval_disc(torch.cat((self._pos_diff[None], normalized), 0)).squeeze(-1)
        positive, negative = scores[:1], scores[1:]
        cls_loss = 0.5 * (self._disc_loss_pos(positive) + self._disc_loss_neg(negative))
        logit_loss = self._model.get_disc_logit_weights().square().sum()
        logit_reg_loss = self._disc_logit_reg * logit_loss
        neg_acc, pos_acc = self._compute_disc_acc(negative, positive)
        zero = scores.new_zeros(())
        info = {
            "disc_loss": cls_loss + logit_reg_loss,
            "disc_cls_loss": cls_loss.detach(),
            "disc_grad_penalty": zero, "disc_neg_grad_penalty": zero,
            "disc_pos_grad_penalty": zero,
            "disc_logit_loss": logit_loss.detach(),
            "disc_logit_reg_loss": logit_reg_loss.detach(),
            "disc_pos_acc": pos_acc.detach(), "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": positive.mean().detach(),
            "disc_neg_logit": negative.mean().detach(),
        }
        physical = inputs if self._model.physical_errors else self._pair_error_metrics(policy, reference)
        tracking_metrics = aggregate_error_metrics(physical)
        info.update({"disc_metric_" + name: tracking_metrics[:, index].mean().detach()
                     for index, name in enumerate(METRIC_NAMES)})
        return info

    def _ablation_metadata(self):
        model = self._model
        return {
            "version": ABLATION_SCHEMA_VERSION, "variant": model.variant,
            "input_dim": model.input_dim, "physical_errors": model.physical_errors,
            "diagonal_enabled": model.diagonal_enabled,
            "initialization": "matched_effective_diagonal_initialization_no_pretraining",
            "hidden_widths": HIDDEN_WIDTHS, "activation": "MixedELU_unchanged",
            "input": ("same_frame_unpooled_64_physical_errors" if model.physical_errors
                      else "same_frame_signed_reference_minus_policy_172"),
            "input_scaling": ("seven_metric_history_broadcast_with_fixed_group_rms"
                              if model.physical_errors else "ADD_coordinate_mean_absolute_history"),
            "features": ({name: (part.start, part.stop) for name, part in FEATURE_SLICES.items()}
                         if model.physical_errors else {}),
            "weight_sign": "nonnegative" if model.physical_errors else "free_signed",
            "normalization": "SLL_nonresidual_absolute_Gram_diagonal" if model.diagonal_enabled else "none",
            "score": "learned_bias_minus_anchored_energy",
            "reward": "original_ADD_clipped_mapping_no_multiplier",
            "disc_logit_reg": self._disc_logit_reg,
        }

    def _get_checkpoint_extra_state(self):
        # Bypass the production-only monotone metadata, retaining ADD/AMP
        # replay state and every other inherited checkpoint component.
        state = ADDAgent._get_checkpoint_extra_state(self)
        state["dare_reward"] = {
            "version": 2, "logits": "raw_discriminator_output",
            "mapping": "original_ADD_without_reward_multiplier"}
        state["ablation_disc"] = self._ablation_metadata()
        return state

    def _load_checkpoint_extra_state(self, state):
        expected = self._ablation_metadata()
        saved = dict(state.get("ablation_disc", {}))
        if "hidden_widths" in saved:
            saved["hidden_widths"] = tuple(saved["hidden_widths"])
        if "features" in saved:
            saved["features"] = {name: tuple(part) for name, part in saved["features"].items()}
        if saved != expected:
            raise ValueError("Cannot resume with different DARE ablation semantics")
        if state.get("dare_reward") != {
                "version": 2, "logits": "raw_discriminator_output",
                "mapping": "original_ADD_without_reward_multiplier"}:
            raise ValueError("Cannot resume a different DARE ablation reward mapping")
        ADDAgent._load_checkpoint_extra_state(self, state)
