"""Single monotone physical-error discriminator; original BCE/ADD/PPO mapping."""

import torch

from learning.amp_agent import AMPAgent
from learning.dare_agent import DAREAgent
from learning.dare_monotone_model import (
    DAREMonotoneModel, HIDDEN_WIDTHS, METRIC_NAMES, MONOTONE_SCHEMA_VERSION)
from learning.diff_normalizer import DiffNormalizer
import util.torch_util as torch_util


class DAREMonotoneAgent(DAREAgent):
    def _load_params(self, config):
        if config.get("disc_group_negatives", False):
            raise ValueError("DAREMonotone uses only real paired observations, not structured negatives")
        super()._load_params(config)

    def _build_model(self, config):
        self._model = DAREMonotoneModel(config["model"], self._env)

    def _build_pos_diff(self):
        return torch.zeros(7, device=self._device)

    def _build_normalizers(self):
        # Keep baseline actor/action normalizers. The discriminator now scales
        # seven nonnegative physical magnitudes, never the raw signed172 input.
        super(AMPAgent, self)._build_normalizers()
        dtype = torch_util.numpy_dtype_to_torch(self._env.get_disc_obs_space().dtype)
        self._disc_obs_norm = DiffNormalizer((7,), device=self._device, dtype=dtype)

    @torch.no_grad()
    def _pair_error_metrics(self, policy, reference):
        # Bound temporary world-rotation tensors using the existing observation
        # batching setting, not a new learning/regularization hyperparameter.
        if policy.ndim != 2 or policy.shape[-1] != 172 or reference.shape != policy.shape:
            raise ValueError("Physical ERR conversion requires matching [N,172] policy/reference observations")
        chunk = max(1, self._disc_demo_batch_size)
        if policy.shape[0] == 0:
            return policy.new_zeros((0, 7))
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
    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        policy = self._exp_buffer.get_data_flat("disc_obs")
        reference = self._exp_buffer.get_data_flat("disc_obs_demo")
        metrics = self._pair_error_metrics(policy, reference)
        normalized = self._disc_obs_norm.normalize(metrics)
        disc_reward = self._calc_disc_rewards(normalized)
        reward = self._task_reward_weight * task_reward + self._disc_reward_weight * disc_reward
        self._exp_buffer.set_data_flat("reward", reward)
        if self._need_normalizer_update():
            self._disc_obs_norm.record(metrics)
        std, mean = torch.std_mean(disc_reward)
        return {"disc_reward_mean": mean, "disc_reward_std": std}

    def _compute_disc_loss(self, batch):
        replay = self._disc_buffer.sample(batch["disc_obs"].shape[0])
        policy = torch.cat((batch["disc_obs"], replay["disc_obs"]), 0)
        reference = torch.cat((batch["disc_obs_demo"], replay["disc_obs_demo"]), 0)
        metrics = self._pair_error_metrics(policy, reference)
        normalized = self._disc_obs_norm.normalize(metrics)
        # The zero positive and all negatives share one effective SN forward.
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
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }
        info.update({"disc_metric_" + name: metrics[:, index].mean().detach()
                     for index, name in enumerate(METRIC_NAMES)})
        return info

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        state["monotone_disc"] = {
            "version": MONOTONE_SCHEMA_VERSION,
            "metrics": METRIC_NAMES, "hidden_widths": HIDDEN_WIDTHS,
            "input": "same_frame_physical_ERR_mean_absolute_scale_no_centering",
            "score": "learned_bias_minus_anchored_positive_SN_CMNN_energy",
            "reward": "original_ADD_clipped_mapping_no_multiplier",
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        expected = self._get_checkpoint_extra_state()["monotone_disc"]
        saved = dict(state.get("monotone_disc", {}))
        for name in ("metrics", "hidden_widths"):
            if name in saved:
                saved[name] = tuple(saved[name])
        if saved != expected:
            raise ValueError("Cannot resume with different monotone physical-error semantics")
        super()._load_checkpoint_extra_state(state)
