import torch

import learning.add_agent as add_agent
import learning.dare_model as dare_model
from util.logger import Logger
import util.torch_util as torch_util


def logit_standardization(model, norm, pos_diff, current_diff, replay_diff,
                          batch_size):
    """Balanced anchor/current/replay statistics for reward-side logits."""
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
    """DARE with raw BCE and rollout-calibrated reward logits."""

    CALIBRATION_BATCH = 16384

    def __init__(self, config, env, device):
        # DARE uses the same fixed-sample observation-normalizer schedule as
        # the reproduced CPL baseline.  There is no second stability policy.
        self._normalizer_frozen = False
        self._normalizer_freeze_iteration = -1
        self._normalizer_freeze_samples = -1
        self._calibration_deferred = False
        super().__init__(config, env, device)
        self._enable_anchor_calibration = bool(
            config.get("disc_anchor_calibration", True))
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
        need = super()._need_normalizer_update()
        if not need and not self._normalizer_frozen:
            self._normalizer_frozen = True
            self._normalizer_freeze_iteration = self._iter
            self._normalizer_freeze_samples = int(self._sample_count)
        return need

    def _update_normalizers(self):
        super()._update_normalizers()

    def _compute_rewards(self):
        normalizer_updating = self._need_normalizer_update()
        if self._enable_anchor_calibration and not normalizer_updating:
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
            if self._model.is_disc_logit_calibrated():
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
            logits = self._model.eval_disc(norm_diff).squeeze(-1)
            return self._disc_reward_scale * add_agent.calc_unscaled_disc_reward(
                logits)

    def _compute_disc_loss(self, batch):
        """Original v6 zero-vs-residual BCE objective (without GP)."""
        # Positive first preserves a30's spectral-normalization update order.
        # Discriminator training uses the raw logit and the fixed BCE objective.
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
        disc_loss = cls_loss
        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": cls_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logit.mean().detach(),
            "disc_neg_logit": neg_logit.mean().detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
            "disc_anchor_calibration_enabled": torch.tensor(
                float(self._enable_anchor_calibration), device=self._device),
            "disc_logit_scale": self._model.get_disc_logit_scale(),
            "disc_anchor_gap": (pos_logit.mean() - neg_logit.mean()).detach(),
            "disc_anchor_gap_raw": (
                pos_logit.mean() - neg_logit.mean()).detach(),
        }

    def _reward_log_info(self):
        return {
            "disc_classifier_scale": self._model.get_disc_logit_scale(),
            "disc_kappa": self._model.get_disc_logit_scale(),
            "disc_logit_center": self._model.get_disc_logit_center(),
            "disc_logit_std": torch.tensor(
                self._calibration_std_raw, device=self._device),
            "disc_anchor_gap_raw": torch.tensor(
                self._calibration_gap_raw, device=self._device),
            "norm_frozen": torch.tensor(float(self._normalizer_frozen),
                                         device=self._device),
            "disc_calibration_updates": torch.tensor(
                self._calibration_updates, device=self._device),
        }

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        state["dare"] = {
            "normalizer_frozen": self._normalizer_frozen,
            "normalizer_freeze_iteration": self._normalizer_freeze_iteration,
            "normalizer_freeze_samples": self._normalizer_freeze_samples,
            "calibration_updates": self._calibration_updates,
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        super()._load_checkpoint_extra_state(state)
        saved = state.get("dare", {})
        self._normalizer_frozen = bool(saved.get("normalizer_frozen", False))
        self._normalizer_freeze_iteration = int(saved.get(
            "normalizer_freeze_iteration", -1))
        self._normalizer_freeze_samples = int(saved.get(
            "normalizer_freeze_samples", -1))
        self._calibration_updates = int(saved.get("calibration_updates", 0))
