import torch

import learning.add_agent as add_agent
import learning.dare_model as dare_model
from util.logger import Logger
import util.torch_util as torch_util


def logit_standardization(model, norm, pos_diff, current_diff, replay_diff,
                          batch_size):
    """Estimate balanced affine statistics for the raw discriminator logit.

    Returns (center, std, gap) of the balanced calibration distribution built
    from the zero-differential anchor (weight 1/2) and the current/replay
    policy residuals (weight 1/4 each):
        c_f = 1/2 [f(0) + 1/2 (mu_cur + mu_rep)]
        s_f^2 = 1/2 [f(0) - c_f]^2 + 1/4 E_cur[(f - c_f)^2]
             + 1/4 E_rep[(f - c_f)^2]
    so that the standardized logit z = (f - c_f) / s_f has zero mean and unit
    variance over that distribution, placing it in the softplus transition
    region without any hand-picked response band.
    """
    def class_stats(raw_diff):
        def eval_raw(disc_obs):
            return model.eval_disc_raw(norm.normalize(disc_obs))

        logits = torch_util.eval_minibatch(
            eval_raw, {"disc_obs": raw_diff}, batch_size)
        return logits.mean(), logits.square().mean()

    pos_logit = model.eval_disc_raw(pos_diff.unsqueeze(0)).mean()
    cur_mean, cur_sq = class_stats(current_diff)
    rep_mean, rep_sq = class_stats(replay_diff)
    neg_mean = 0.5 * (cur_mean + rep_mean)
    center = 0.5 * (pos_logit + neg_mean)
    var = (0.5 * (pos_logit - center).square()
           + 0.25 * (cur_sq - 2.0 * cur_mean * center + center.square())
           + 0.25 * (rep_sq - 2.0 * rep_mean * center + center.square()))
    gap = float(pos_logit - neg_mean)
    return float(center), float(torch.sqrt(var)), gap


class DAREAgent(add_agent.ADDAgent):
    """DARE with raw Full-SN BCE and reward-side logit calibration.

    ``one_shot`` preserves the historical v6 behavior.  ``rollout`` updates
    the same balanced affine statistics once per rollout after the input
    normalizer has frozen.  The latter prevents a deep discriminator's raw
    logit scale from drifting away from a scale measured at freeze time.
    """

    CALIBRATION_BATCH = 16384
    def __init__(self, config, env, device):
        self._normalizer_freeze_mode = config.get(
            "normalizer_freeze_mode", "fixed_samples")
        self._normalizer_min_samples = int(
            config.get("normalizer_min_samples", 0))
        self._normalizer_stability_window = int(
            config.get("normalizer_stability_window", 10))
        self._normalizer_stability_tol = float(
            config.get("normalizer_stability_tol", 0.01))
        self._normalizer_stability_patience = int(
            config.get("normalizer_stability_patience", 3))
        self._normalizer_max_samples = int(config.get(
            "normalizer_max_samples", config.get("normalizer_samples", 100000000)))
        self._normalizer_frozen = False
        self._normalizer_stable_count = 0
        self._normalizer_freeze_iteration = -1
        self._normalizer_freeze_samples = -1
        self._normalizer_stability_history = []
        self._normalizer_stability_prev = None
        self._normalizer_stability_score = float("nan")
        self._logit_reg_warned = False
        self._calibration_deferred = False
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("DARE requires disc_grad_penalty=0")
        self._enable_anchor_calibration = bool(
            config.get("disc_anchor_calibration", True))
        self._calibration_mode = config.get(
            "disc_anchor_calibration_mode", "one_shot")
        if self._calibration_mode not in ("one_shot", "rollout"):
            raise ValueError(
                "disc_anchor_calibration_mode must be 'one_shot' or "
                "'rollout', got {}".format(self._calibration_mode))
        self._disc_reward_mode = config.get("disc_reward_mode", "absolute")
        if self._disc_reward_mode not in ("absolute", "anchor_relative"):
            raise ValueError(
                "disc_reward_mode must be 'absolute' or 'anchor_relative', "
                "got {}".format(self._disc_reward_mode))
        self._calibration_updates = 0
        self._calibration_gap_raw = float("nan")
        self._calibration_center_raw = float("nan")
        self._calibration_std_raw = float("nan")
        if not self._enable_anchor_calibration:
            Logger.print("DARE anchor calibration disabled: fixed kappa=1.0000")

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _get_disc_normalizer_groups(self):
        return None

    def _need_normalizer_update(self):
        if self._normalizer_freeze_mode != "stability":
            need = super()._need_normalizer_update()
            if not need and not self._normalizer_frozen:
                self._normalizer_frozen = True
                self._normalizer_freeze_iteration = self._iter
                self._normalizer_freeze_samples = int(self._sample_count)
            return need
        return (not self._normalizer_frozen
                and self._sample_count < self._normalizer_max_samples)

    def _update_normalizers(self):
        super()._update_normalizers()
        if self._normalizer_freeze_mode == "stability":
            self._update_normalizer_stability()

    def _update_normalizer_stability(self):
        count = int(self._disc_obs_norm.get_count().item())
        mean = self._disc_obs_norm.get_abs_mean().detach().clone()
        self._normalizer_stability_history.append((count, mean.cpu()))
        window = max(1, self._normalizer_stability_window)
        if len(self._normalizer_stability_history) > window + 1:
            self._normalizer_stability_history.pop(0)
        if self._normalizer_stability_prev is not None:
            delta = torch.abs(torch.log(torch.clamp_min(mean, 1e-8))
                              - torch.log(torch.clamp_min(
                                  self._normalizer_stability_prev, 1e-8)))
            self._normalizer_stability_score = float(torch.quantile(
                delta, 0.95).item())
            self._normalizer_stable_count = (
                self._normalizer_stable_count + 1
                if (count >= self._normalizer_min_samples
                    and self._normalizer_stability_score
                    < self._normalizer_stability_tol) else 0)
        self._normalizer_stability_prev = mean
        if (count >= self._normalizer_min_samples
                and (self._normalizer_stable_count
                     >= self._normalizer_stability_patience
                     or count >= self._normalizer_max_samples)):
            self._normalizer_frozen = True
            self._normalizer_freeze_iteration = self._iter
            self._normalizer_freeze_samples = count

    def _compute_rewards(self):
        normalizer_updating = self._need_normalizer_update()
        if (self._enable_anchor_calibration and not normalizer_updating):
            if (self._calibration_mode == "rollout"
                    or not self._model.is_disc_logit_calibrated()):
                self._calibrate_disc_logit_scale()
        info = super()._compute_rewards()
        info.update(self._reward_log_info())
        return info

    @torch.no_grad()
    def _calibrate_disc_logit_scale(self):
        was_training = self._model.training
        self._model.eval()
        try:
            current_diff = (self._exp_buffer.get_data_flat("disc_obs_demo")
                            - self._exp_buffer.get_data_flat("disc_obs"))
            replay_count = self._disc_buffer.get_sample_count()
            replay_diff = (self._disc_buffer.get_data_flat("disc_obs_demo")[
                :replay_count] - self._disc_buffer.get_data_flat("disc_obs")[
                    :replay_count])
            center, std, gap = logit_standardization(
                self._model, self._disc_obs_norm, self._pos_diff,
                current_diff, replay_diff, self.CALIBRATION_BATCH)
        finally:
            self._model.train(was_training)
        if not gap > 0.0:
            # A positive anchor gap is required to preserve the intended
            # reward ordering.  During rollout calibration, retain the last
            # valid transform rather than replacing it with an invalid one.
            if (self._calibration_mode == "rollout"
                    and self._model.is_disc_logit_calibrated()):
                Logger.print(
                    "DARE rollout calibration skipped: separation gap "
                    "{:.4f} is not positive; retaining the last valid "
                    "transform.".format(gap))
                return
            if not self._calibration_deferred:
                self._calibration_deferred = True
                Logger.print(
                    "DARE classifier calibration deferred: separation gap "
                    "{:.4f} is not positive yet; retrying next iteration."
                    .format(gap))
            return
        if not (std > 0.0 and std == std and std < float("inf")):
            raise RuntimeError(
                "Logit standardization requires a finite positive spread, "
                "got s_f={}".format(std))
        scale = 1.0 / std
        self._model.set_disc_logit_calibration(center, scale)
        self._calibration_gap_raw = gap
        self._calibration_center_raw = center
        self._calibration_std_raw = std
        self._calibration_updates += 1
        Logger.print("DARE classifier calibration at iter {}: M_f={:.4f} "
                     "c_f={:.4f} s_f={:.4f} kappa_D={:.4f}".format(
                         self._iter, gap, center, std, scale))

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            if self._disc_reward_mode == "anchor_relative":
                # Subtract the raw positive-anchor logit before softplus.  This
                # removes the discriminator's arbitrary additive logit offset
                # while preserving the residual ordering and the BCE-trained
                # raw classifier.  The anchor is exactly the zero differential.
                logits = self._model.eval_disc_raw(norm_diff).squeeze(-1)
                anchor = self._model.eval_disc_raw(
                    self._pos_diff.unsqueeze(0)).squeeze(-1)
                logits = logits - anchor
            else:
                logits = self._model.eval_disc(norm_diff).squeeze(-1)
            return self._disc_reward_scale * add_agent.calc_unscaled_disc_reward(
                logits)

    def _compute_disc_loss(self, batch):
        """Original v6 zero-vs-residual BCE objective (without GP)."""
        # Positive first preserves a30's spectral-normalization update order.
        # NOTE: discriminator training uses the RAW logit f_raw. The affine
        # calibration (z = (f - c) / s) is a reward-side transform only; feeding
        # it into the BCE would make the calibration scale kappa = 1/s_f change
        # the classifier's effective loss scale as well.
        pos_logit = self._model.eval_disc_raw(
            self._pos_diff.unsqueeze(0)).squeeze(-1)
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        norm_diff = self._disc_obs_norm.normalize(
            torch.cat((current_diff, replay_diff), dim=0))
        neg_logit = self._model.eval_disc_raw(norm_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logit)
        neg_loss = self._disc_loss_neg(neg_logit)
        cls_loss = 0.5 * (pos_loss + neg_loss)
        # `disc_logit_reg` is structurally inert for DARE, so it is NOT added to
        # the objective.  DARE spectral-normalizes `_disc_logits` (see
        # DAREModel._build_disc), which makes the effective weight matrix
        # scale-invariant in the raw parameter: ||W_eff||_2 == 1 holds after
        # every optimizer step, so sum(W_eff^2) is a constant whose gradient is
        # exactly zero.  Dropping the term is therefore bit-exact with respect
        # to the previous objective; reporting it as zero plus the one-time
        # warning keeps the dead knob visible instead of silently training with
        # a regularizer that cannot act.  (AMP/ADD do not normalize this layer,
        # so their term is genuine - this only affects DARE.)
        if self._disc_logit_reg != 0 and not self._logit_reg_warned:
            self._logit_reg_warned = True
            Logger.print(
                "WARNING: disc_logit_reg={} has no effect for DARE - the logit "
                "layer is spectral-normalized, so sum(W_eff^2)==1 is a "
                "constant with zero gradient. Any ablation arm that only "
                "toggles this value compares two identical models. Set it to 0 "
                "or redesign the arm.".format(self._disc_logit_reg))
        logit_loss = torch.zeros((), device=self._device)
        logit_reg_loss = torch.zeros((), device=self._device)
        disc_loss = cls_loss
        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        zero = torch.zeros((), device=self._device)
        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": cls_loss.detach(),
            "disc_grad_penalty": zero,
            "disc_neg_grad_penalty": zero,
            "disc_pos_grad_penalty": zero,
            "disc_logit_loss": logit_loss.detach(),
            "disc_logit_reg_loss": logit_reg_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logit.mean().detach(),
            "disc_neg_logit": neg_logit.mean().detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
            "disc_group_embedding_enabled": torch.tensor(
                float(self._model.uses_disc_group_embedding()),
                device=self._device),
            "disc_anchor_calibration_enabled": torch.tensor(
                float(self._enable_anchor_calibration), device=self._device),
            "disc_logit_scale": self._model.get_disc_logit_scale(),
            # logits above are raw now, so "gap" and "raw gap" coincide; keep
            # both keys for downstream log tooling compatibility.
            "disc_anchor_gap": (pos_logit.mean() - neg_logit.mean()).detach(),
            "disc_anchor_gap_raw": (
                pos_logit.mean() - neg_logit.mean()).detach(),
        }

    def _reward_log_info(self):
        return {
            "disc_classifier_scale": self._model.get_disc_logit_scale(),
            # Explicit name for the reward-side affine gain.  Keep the
            # historical classifier-scale key for existing log tooling.
            "disc_kappa": self._model.get_disc_logit_scale(),
            "disc_logit_center": self._model.get_disc_logit_center(),
            "disc_logit_std": torch.tensor(
                self._calibration_std_raw, device=self._device),
            "disc_anchor_gap_raw": torch.tensor(
                self._calibration_gap_raw, device=self._device),
            "norm_stability_q95": torch.tensor(
                self._normalizer_stability_score, device=self._device),
            "norm_stability_count": torch.tensor(
                self._normalizer_stable_count, device=self._device),
            "norm_frozen": torch.tensor(float(self._normalizer_frozen),
                                         device=self._device),
            "disc_calibration_updates": torch.tensor(
                self._calibration_updates, device=self._device),
            "disc_reward_anchor_relative": torch.tensor(
                float(self._disc_reward_mode == "anchor_relative"),
                device=self._device),
        }

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        state["dare"] = {
            "normalizer_freeze_mode": self._normalizer_freeze_mode,
            "normalizer_frozen": self._normalizer_frozen,
            "normalizer_stable_count": self._normalizer_stable_count,
            "normalizer_freeze_iteration": self._normalizer_freeze_iteration,
            "normalizer_freeze_samples": self._normalizer_freeze_samples,
            "normalizer_stability_score": self._normalizer_stability_score,
            "normalizer_stability_prev": (
                None if self._normalizer_stability_prev is None
                else self._normalizer_stability_prev.cpu()),
            "normalizer_stability_history": [
                (int(c), m) for c, m in self._normalizer_stability_history],
            "calibration_updates": self._calibration_updates,
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        super()._load_checkpoint_extra_state(state)
        saved = state.get("dare", {})
        self._normalizer_frozen = bool(saved.get("normalizer_frozen", False))
        self._normalizer_stable_count = int(saved.get(
            "normalizer_stable_count", 0))
        self._normalizer_freeze_iteration = int(saved.get(
            "normalizer_freeze_iteration", -1))
        self._normalizer_freeze_samples = int(saved.get(
            "normalizer_freeze_samples", -1))
        self._normalizer_stability_score = float(saved.get(
            "normalizer_stability_score", float("nan")))
        prev = saved.get("normalizer_stability_prev", None)
        self._normalizer_stability_prev = (
            None if prev is None else prev.to(self._device))
        self._normalizer_stability_history = [
            (int(c), m) for c, m in saved.get("normalizer_stability_history", [])]
        self._calibration_updates = int(saved.get("calibration_updates", 0))
