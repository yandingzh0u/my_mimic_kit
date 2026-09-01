import torch

import learning.add_agent as add_agent
import learning.sbe_fsn_add_model as sbe_fsn_add_model


class SBEFSNADDAgent(add_agent.ADDAgent):
    """ADD with semantic block-equalized Full Spectral Normalization."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("SBE-FSN ADD requires disc_grad_penalty=0")

    def _build_model(self, config):
        self._model = sbe_fsn_add_model.SBEFSNADDModel(
            config["model"], self._env)

    def _compute_disc_loss(self, batch):
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)

        inputs = torch.cat((self._pos_diff.unsqueeze(0), norm_diff), dim=0)
        logits = self._model.eval_disc(inputs).squeeze(-1)
        pos_logit = logits[:1]
        neg_logit = logits[1:]
        pos_loss = self._disc_loss_pos(pos_logit)
        neg_loss = self._disc_loss_neg(neg_logit)
        cls_loss = 0.5 * (pos_loss + neg_loss)

        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))
        logit_reg_loss = self._disc_logit_reg * logit_loss
        disc_loss = cls_loss + logit_reg_loss
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
            "disc_lipschitz_bound": (
                self._model.get_disc_lipschitz_bound()),
            "disc_semantic_gain_mean": (
                self._model.get_disc_semantic_gain_mean().detach()),
            "disc_semantic_gain_spread": (
                self._model.get_disc_semantic_gain_spread().detach()),
            "disc_semantic_energy_ratio": (
                self._model.get_disc_semantic_energy_ratio().detach()),
        }
