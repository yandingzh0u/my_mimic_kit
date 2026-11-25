"""Single monotone physical-error discriminator; original BCE/ADD/PPO mapping."""

import torch

from learning.add_agent import ADDAgent, calc_unscaled_disc_reward
from learning.ppo_agent import PPOAgent
from learning.mira_monotone_model import (
    MIRAMonotoneModel, HIDDEN_WIDTHS, physical_feature_slices)
from learning.physical_error_normalizer import PhysicalErrorNormalizer, aggregate_physical_groups
import util.torch_util as torch_util


class MIRAMonotoneAgent(ADDAgent):
    """One CMNN scorer, zero-error BCE, original ADD reward and uniform PPO."""

    CHECKPOINT_AGENT_ALIASES = ("DAREMonotoneAgent",)

    def _load_params(self, config):
        self._disc_group_size_scaling = config.get("disc_group_size_scaling", True)
        if not isinstance(self._disc_group_size_scaling, bool):
            raise ValueError("disc_group_size_scaling must be a boolean")
        if config.get("disc_group_negatives", False):
            raise ValueError("MIRAMonotone uses only real paired observations, not structured negatives")
        if "disc_reward_scale" in config:
            raise ValueError("MIRAMonotone uses the original ADD reward without a multiplier")
        if config.get("disc_grad_penalty", 0) != 0:
            raise ValueError("MIRAMonotone requires disc_grad_penalty=0")
        super()._load_params(config)
        del self._disc_reward_scale

    def _build_model(self, config):
        self._model = MIRAMonotoneModel(config["model"], self._env)

    def _build_pos_diff(self):
        return torch.zeros(self._model._pair_metrics.feature_dim, device=self._device)

    def _build_normalizers(self):
        # Keep the baseline actor/action normalizers and scale each physical
        # group with its own historical statistics.
        PPOAgent._build_normalizers(self)
        dtype = torch_util.numpy_dtype_to_torch(self._env.get_disc_obs_space().dtype)
        self._disc_obs_norm = PhysicalErrorNormalizer(
            device=self._device, dtype=dtype, feature_slices=physical_feature_slices(self._env),
            group_size_scaling=self._disc_group_size_scaling)

    @torch.no_grad()
    def _pair_error_metrics(self, policy, reference):
        # Bound temporary world-rotation tensors using the existing observation
        # batching setting, not a new learning/regularization hyperparameter.
        layout = self._model._pair_metrics
        if policy.ndim != 2 or policy.shape[-1] != layout.obs_dim or reference.shape != policy.shape:
            raise ValueError(f"Expected matching [N,{layout.obs_dim}] policy and reference observations")
        chunk = max(1, self._disc_demo_batch_size)
        if policy.shape[0] == 0:
            return policy.new_zeros((0, layout.feature_dim))
        return torch.cat([self._model.pair_to_error_metrics(p, r)
                          for p, r in zip(policy.split(chunk), reference.split(chunk))], 0)

    @torch.no_grad()
    def calc_disc_reward_from_obs(self, policy, reference):
        """Evaluate raw paired ADD observations, including standalone evaluators."""
        metrics = self._pair_error_metrics(policy, reference)
        if metrics.shape[0] == 0:
            return metrics.new_zeros((0,))
        return self._calc_disc_rewards(self._disc_obs_norm.normalize(metrics))

    @torch.no_grad()
    def _calc_disc_rewards(self, normalized_errors):
        logits = torch_util.eval_minibatch(
            self._model.eval_disc, {"disc_obs": normalized_errors}, self._disc_eval_batch_size)
        return calc_unscaled_disc_reward(logits.squeeze(-1))

    @torch.no_grad()
    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        policy = self._exp_buffer.get_data_flat("disc_obs")
        reference = self._exp_buffer.get_data_flat("disc_obs_demo")
        metrics = self._pair_error_metrics(policy, reference)
        # Initialize the first rollout's physical scales before scoring it.
        # Otherwise the 64-D RMS geometry would be evaluated with seven unit
        # scales, creating an artificial first-iteration reward collapse.
        initialized_normalizer = (
            self._need_normalizer_update()
            and int(self._disc_obs_norm.get_count()) == 0)
        if initialized_normalizer:
            self._disc_obs_norm.record(metrics)
            self._disc_obs_norm.update()
        normalized = self._disc_obs_norm.normalize(metrics)
        disc_reward = self._calc_disc_rewards(normalized)
        reward = self._task_reward_weight * task_reward + self._disc_reward_weight * disc_reward
        self._exp_buffer.set_data_flat("reward", reward)
        if self._need_normalizer_update() and not initialized_normalizer:
            self._disc_obs_norm.record(metrics)
        std, mean = torch.std_mean(disc_reward)
        return {"disc_reward_mean": mean, "disc_reward_std": std}

    def _compute_disc_loss(self, batch):
        replay = self._disc_buffer.sample(batch["disc_obs"].shape[0])
        policy = torch.cat((batch["disc_obs"], replay["disc_obs"]), 0)
        reference = torch.cat((batch["disc_obs_demo"], replay["disc_obs_demo"]), 0)
        metrics = self._pair_error_metrics(policy, reference)
        normalized = self._disc_obs_norm.normalize(metrics)
        # The zero positive and all negatives share one effective forward.
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
            "disc_grad_penalty": zero,
            "disc_neg_grad_penalty": zero,
            "disc_pos_grad_penalty": zero,
            "disc_logit_loss": logit_loss.detach(),
            "disc_logit_reg_loss": logit_reg_loss.detach(),
            "disc_pos_acc": pos_acc.detach(), "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": positive.mean().detach(),
            "disc_neg_logit": negative.mean().detach(),
        }
        tracking_metrics = aggregate_physical_groups(metrics)
        info.update({"disc_metric_" + name: tracking_metrics[:, index].mean().detach()
                     for index, name in enumerate(self._model._pair_metrics.metric_names)})
        return info

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        state["mira_reward"] = {
            "version": 2,
            "logits": "raw_discriminator_output",
            "mapping": "original_ADD_without_reward_multiplier",
        }
        layout = self._model._pair_metrics
        metric_count = {7: "seven", 11: "eleven", 12: "twelve"}[len(layout.metric_names)]
        state["monotone_disc"] = {
            "version": self._model.disc_schema_version,
            "metrics": layout.metric_names, "hidden_widths": HIDDEN_WIDTHS,
            "input": f"same_frame_unpooled_{layout.feature_dim}_physical_errors",
            "features": {name: (part.start, part.stop)
                         for name, part in layout.feature_slices.items()},
            "body_positions": "nonroot_root_relative_world_axes_L2",
            "body_rotations": "nonroot_world_geodesic",
            "input_scaling": (metric_count + "_metric_history_broadcast"
                              + ("_with_fixed_group_rms" if self._disc_group_size_scaling else "")),
            "first_layer": "free_positive_dense_no_group_budget",
            "score": "learned_bias_minus_anchored_positive_diagonal_CMNN_energy",
            "normalization": "SLL_nonresidual_positive_diagonal_logdomain_Perron_init",
            "learned_diagonal_parameters": sum(
                layer.parametrizations.weight[-1].log_q.numel()
                for layer in self._model._disc_energy
                if isinstance(layer, torch.nn.Linear)),
            "reward": "original_ADD_clipped_mapping_no_multiplier",
        }
        if self._model.disc_signed_weights:
            state["monotone_disc"].update({
                "first_layer": "signed_dense_no_group_budget",
                "score": "learned_bias_minus_anchored_signed_diagonal_energy",
                "normalization": "same_positive_diagonal_scales_on_weight_magnitudes",
            })
        if not self._model.disc_learn_anchor_logit:
            score = state["monotone_disc"]["score"]
            state["monotone_disc"]["score"] = score.replace("learned_bias", "fixed_zero_bias", 1)
        if layout.grip_release_aware:
            state["monotone_disc"]["carry_grip"] = "through_last_reference_contact"
        return state

    def _load_checkpoint_extra_state(self, state):
        expected = self._get_checkpoint_extra_state()["monotone_disc"]
        saved = dict(state.get("monotone_disc", {}))
        for name in ("metrics", "hidden_widths"):
            if name in saved:
                saved[name] = tuple(saved[name])
        if "features" in saved:
            saved["features"] = {name: tuple(part) for name, part in saved["features"].items()}
        if saved != expected:
            raise ValueError("Cannot resume with different monotone physical-error semantics")
        if state.get("mira_reward", state.get("dare_reward")) != {
                "version": 2, "logits": "raw_discriminator_output",
                "mapping": "original_ADD_without_reward_multiplier"}:
            raise ValueError("Cannot resume a legacy scaled MIRA reward checkpoint")
        super()._load_checkpoint_extra_state(state)
