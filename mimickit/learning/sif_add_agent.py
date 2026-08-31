import torch

import learning.add_agent as add_agent
import learning.sif_add_model as sif_add_model


class SIFADDAgent(add_agent.ADDAgent):
    """Official ADD objective with a GP-free SIF discriminator."""

    def _build_model(self, config):
        self._model = sif_add_model.SIFADDModel(
            config["model"], self._env)

    def _update_model(self):
        info = super()._update_model()
        info.update(self._model.get_disc_geometry_info())
        return info

    def _compute_disc_loss(self, batch):
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)

        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        pos_diff = self._pos_diff.unsqueeze(0)
        neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)
        pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)

        pos_loss = self._disc_loss_pos(pos_logit)
        neg_loss = self._disc_loss_neg(neg_logit)
        cls_loss = 0.5 * (pos_loss + neg_loss)

        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))
        logit_reg_loss = self._disc_logit_reg * logit_loss
        disc_loss = cls_loss + logit_reg_loss

        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        zero = cls_loss.detach().new_zeros(())
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
        }
