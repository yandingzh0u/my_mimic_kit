"""Three independent zero-anchor classifiers; ADD reward of their mean logit."""

import torch
import torch.nn.functional as F

from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_agent import DAREAgent
from learning.dare_multi_model import DAREMultiModel, MULTI_DISC_SCHEMA_VERSION
import util.torch_util as torch_util


class DAREMultiAgent(DAREAgent):

    def _load_params(self, config):
        if config.get("disc_group_negatives", False):
            raise ValueError("DAREMulti uses real group errors, not structured negatives")
        super()._load_params(config)

    def _build_model(self, config):
        self._model = DAREMultiModel(config["model"], self._env)

    def _get_checkpoint_extra_state(self):
        state = super()._get_checkpoint_extra_state()
        state["multi_disc"] = {
            "version": MULTI_DISC_SCHEMA_VERSION,
            "groups": self._model.get_disc_error_groups(),
            "reward_aggregation": "mapped_mean_logit",
            "loss_aggregation": "sum_of_independent_losses",
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        super()._load_checkpoint_extra_state(state)
        saved = state.get("multi_disc", {})
        groups = tuple((name, tuple(indices)) for name, indices in saved.get("groups", ()))
        if (saved.get("version") != MULTI_DISC_SCHEMA_VERSION
                or groups != self._model.get_disc_error_groups()
                or saved.get("reward_aggregation") != "mapped_mean_logit"
                or saved.get("loss_aggregation") != "sum_of_independent_losses"):
            raise ValueError("Cannot resume with different multi-discriminator semantics")

    def _compute_disc_loss(self, batch):
        # Preserve baseline SN forward ordering: positive, negative, head weights.
        pos_logit = self._model.eval_disc(self._pos_diff.unsqueeze(0))
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        neg_logit = self._model.eval_disc(norm_diff)
        groups = self._model.get_disc_error_groups()
        # A zero local error is the positive anchor, even if other groups err.
        valid = torch.stack([raw_diff[:, indices].ne(0).any(dim=-1)
                             for _, indices in groups], dim=-1)
        neg_losses = F.binary_cross_entropy_with_logits(
            neg_logit, torch.zeros_like(neg_logit), reduction="none")
        counts = valid.sum(dim=0).clamp_min(1)
        neg_loss = (neg_losses * valid).sum(dim=0) / counts
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logit, torch.ones_like(pos_logit), reduction="none").mean(dim=0)
        cls_losses = .5 * (pos_loss + neg_loss)
        logit_loss = self._model.get_disc_logit_weights().square().sum()
        reg_loss = self._disc_logit_reg * logit_loss
        # Independent parameter blocks each receive their full baseline gradient.
        disc_loss = cls_losses.sum() + reg_loss
        zero = neg_logit.new_zeros(())
        info = {
            "disc_loss": disc_loss,
            "disc_loss_mean": disc_loss.detach() / len(groups),
            "disc_cls_loss": cls_losses.mean().detach(),
            "disc_grad_penalty": zero,
            "disc_neg_grad_penalty": zero,
            "disc_pos_grad_penalty": zero,
            "disc_logit_loss": logit_loss.detach(),
            "disc_logit_reg_loss": reg_loss.detach(),
            "disc_pos_acc": (pos_logit > 0).float().mean().detach(),
            "disc_neg_acc": self._masked_mean((neg_logit < 0).float(), valid).detach(),
            "disc_pos_logit": pos_logit.mean().detach(),
            "disc_neg_logit": neg_logit.mean().detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }
        for group_id, (name, _) in enumerate(groups):
            prefix = "disc_" + name + "_"
            info.update({
                prefix + "cls_loss": cls_losses[group_id].detach(),
                prefix + "pos_acc": (pos_logit[:, group_id] > 0).float().mean().detach(),
                prefix + "neg_acc": self._masked_mean(
                    (neg_logit[:, group_id] < 0).float(), valid[:, group_id]).detach(),
                prefix + "neg_valid_fraction": valid[:, group_id].float().mean().detach(),
                prefix + "pos_logit": pos_logit[:, group_id].mean().detach(),
                prefix + "neg_logit": neg_logit[:, group_id].mean().detach(),
            })
        return info

    @torch.no_grad()
    def _calc_disc_rewards(self, norm_diff):
        logits = torch_util.eval_minibatch(
            self._model.eval_disc, {"disc_obs": norm_diff}, self._disc_eval_batch_size)
        joint_logit = logits.mean(dim=-1)
        return calc_unscaled_disc_reward(joint_logit)
