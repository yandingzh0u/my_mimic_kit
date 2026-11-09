import torch
import torch.nn.functional as F

import learning.add_agent as add_agent
import learning.dare_model as dare_model
from learning.dare_model import ANCHOR_GAP_TARGET
from learning.group_negatives import GroupNegativeSampler
from util.logger import Logger
import util.torch_util as torch_util


def anchor_gap(model, norm, pos_diff, current_diff, replay_diff,
               batch_size):
    """Measure raw-logit separation from the zero differential anchor."""
    def mean_raw(raw_diff):
        def eval_raw(disc_obs):
            return model.eval_disc_raw(norm.normalize(disc_obs))

        return torch_util.eval_minibatch(
            eval_raw, {"disc_obs": raw_diff}, batch_size).mean()

    pos_logit = model.eval_disc_raw(pos_diff.unsqueeze(0)).mean()
    current_logit = mean_raw(current_diff)
    replay_logit = mean_raw(replay_diff)
    return float((pos_logit - 0.5 * (current_logit + replay_logit)).item())


class DAREAgent(add_agent.ADDAgent):
    """DARE with one-shot output-scale calibration after input normalization."""

    CALIBRATION_BATCH = 16384

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("DARE requires disc_grad_penalty=0")
        self._calibration_gap_raw = float("nan")
        self._disc_group_negative_sampler = None
        if self._disc_group_negatives:
            self._disc_group_negative_sampler = GroupNegativeSampler(
                groups=env.get_disc_error_groups(),
                input_dim=int(env.get_disc_obs_space().shape[0]),
                device=device, seed=torch.initial_seed())

    def _load_params(self, config):
        self._disc_group_negatives = config.get("disc_group_negatives", False)
        if not isinstance(self._disc_group_negatives, bool):
            raise ValueError("disc_group_negatives must be a boolean")
        super()._load_params(config)

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        sampler = getattr(self, "_disc_group_negative_sampler", None)
        if sampler is not None:
            state["disc_group_negatives"] = sampler.state_dict()
        return state

    def _load_checkpoint_extra_state(self, state):
        super()._load_checkpoint_extra_state(state)
        sampler = getattr(self, "_disc_group_negative_sampler", None)
        saved = state.get("disc_group_negatives")
        if (sampler is None) != (saved is None):
            raise ValueError(
                "Cannot resume with a different structured-negative mode; "
                "use a weights-only load to start a separate experiment")
        if sampler is not None:
            sampler.load_state_dict(saved)

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _compute_rewards(self):
        if (not self._need_normalizer_update()
                and not self._model.is_disc_logit_calibrated()):
            self._calibrate_disc_logit_scale()
        return super()._compute_rewards()

    @torch.no_grad()
    def _calibrate_disc_logit_scale(self):
        was_training = self._model.training
        self._model.eval()
        try:
            current_diff = (
                self._exp_buffer.get_data_flat("disc_obs_demo")
                - self._exp_buffer.get_data_flat("disc_obs"))
            replay_count = self._disc_buffer.get_sample_count()
            replay_diff = (
                self._disc_buffer.get_data_flat("disc_obs_demo")[:replay_count]
                - self._disc_buffer.get_data_flat("disc_obs")[:replay_count])
            gap = anchor_gap(
                self._model, self._disc_obs_norm, self._pos_diff,
                current_diff, replay_diff, self.CALIBRATION_BATCH)
        finally:
            self._model.train(was_training)

        if not gap > 0.0:
            raise RuntimeError(
                "Anchor-gap calibration requires positive separation, got {}"
                .format(gap))

        scale = ANCHOR_GAP_TARGET / gap
        self._model.set_disc_logit_scale(scale)
        self._calibration_gap_raw = gap
        Logger.print(
            "DARE anchor calibration at iter {}: M_f={:.4f} kappa={:.4f}"
            .format(self._iter, gap, scale))

    def _compute_disc_loss(self, batch):
        # Keep a30's forward order. PyTorch spectral_norm updates its power
        # iteration buffers on every training-mode forward, so order matters.
        pos_logit = self._model.eval_disc(
            self._pos_diff.unsqueeze(0)).squeeze(-1)

        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)

        structured = None
        sampler = getattr(self, "_disc_group_negative_sampler", None)
        if sampler is not None:
            structured = sampler.sample(
                norm_diff, (current_diff.shape[0], replay_diff.shape[0]))
            norm_diff = structured.values

        neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logit)
        if structured is None:
            neg_loss = self._disc_loss_neg(neg_logit)
        else:
            negative_losses = F.binary_cross_entropy_with_logits(
                neg_logit, torch.zeros_like(neg_logit), reduction="none")
            neg_loss = self._masked_mean(negative_losses, structured.valid)
        cls_loss = 0.5 * (pos_loss + neg_loss)

        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))
        logit_reg_loss = self._disc_logit_reg * logit_loss
        disc_loss = cls_loss + logit_reg_loss
        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        zero = torch.zeros((), device=self._device)
        info = {
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
            "disc_group_total_width": (
                self._model.get_disc_group_total_width()),
            "disc_logit_scale": self._model.get_disc_logit_scale(),
            "disc_anchor_gap": (
                pos_logit.mean() - neg_logit.mean()).detach(),
            "disc_anchor_gap_raw": (
                (pos_logit.mean() - neg_logit.mean())
                / self._model.get_disc_logit_scale()).detach(),
        }
        if structured is not None:
            info.update(self._group_negative_info(
                structured, neg_logit, negative_losses))
            info["disc_neg_acc"] = self._masked_mean(
                (neg_logit < 0).float(), structured.valid).detach()
        return info

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1)

    def _group_negative_info(self, structured, logits, losses):
        applied = structured.group_ids >= 0
        requested = structured.requested_group_ids >= 0
        real = ~applied & structured.valid
        info = {
            "disc_group_negative_fraction": applied.float().mean(),
            "disc_group_negative_requested_fraction": requested.float().mean(),
            "disc_group_negative_zero_fallback_fraction": (
                requested & ~applied).float().mean(),
            "disc_neg_valid_fraction": structured.valid.float().mean(),
            "disc_real_neg_loss": self._masked_mean(losses, real),
            "disc_real_neg_acc": self._masked_mean((logits < 0).float(), real),
            "disc_real_neg_logit": self._masked_mean(logits, real),
            "disc_group_neg_loss": self._masked_mean(losses, applied),
            "disc_group_neg_acc": self._masked_mean((logits < 0).float(), applied),
            "disc_group_neg_logit": self._masked_mean(logits, applied),
        }
        for group_id, (name, _) in enumerate(self._disc_group_negative_sampler.groups):
            mask = structured.group_ids == group_id
            info["disc_group_negative_{}_fraction".format(name)] = mask.float().mean()
            info["disc_group_negative_{}_requested_fraction".format(name)] = (
                structured.requested_group_ids == group_id).float().mean()
            info["disc_group_negative_{}_acc".format(name)] = self._masked_mean(
                (logits < 0).float(), mask)
        return {name: value.detach() for name, value in info.items()}
