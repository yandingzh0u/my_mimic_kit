import torch

import learning.add_agent as add_agent
import learning.dare_model as dare_model
import util.torch_util as torch_util


class DAREAgent(add_agent.ADDAgent):
    """Zero-differential adversarial learning with a CPL discriminator."""

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _compute_disc_loss(self, batch):
        """Classify the zero anchor against current and replay residuals."""
        # Preserve the positive-first spectral-normalization update order.
        pos_logits = self._model.eval_disc_raw(
            self._pos_diff.unsqueeze(0)).squeeze(-1)
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = (replay_data["disc_obs_demo"]
                       - replay_data["disc_obs"])

        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        neg_logits = self._model.eval_disc_raw(norm_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logits)
        neg_loss = self._disc_loss_neg(neg_logits)
        disc_loss = 0.5 * (pos_loss + neg_loss)
        neg_acc, pos_acc = self._compute_disc_acc(neg_logits, pos_logits)

        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logits.mean().detach(),
            "disc_neg_logit": neg_logits.mean().detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }

    def _calc_disc_rewards(self, norm_diff):
        """Use the bounded ADD reward paired with the BCE discriminator."""
        with torch.no_grad():
            logits = torch_util.eval_minibatch(
                self._model.eval_disc_raw,
                {"disc_obs": norm_diff}, self._disc_eval_batch_size,
            ).squeeze(-1)
            return self._disc_reward_scale * add_agent.calc_unscaled_disc_reward(
                logits)
