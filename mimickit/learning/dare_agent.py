import torch
import torch.nn.functional as F

import learning.add_agent as add_agent
import learning.dare_model as dare_model
import util.torch_util as torch_util


class DAREAgent(add_agent.ADDAgent):
    """DARE's single path: normalized differential-quality learning with CPL."""

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _compute_disc_loss(self, batch):
        """Fit a fixed quality target computed from the normalized residual."""
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = (replay_data["disc_obs_demo"]
                       - replay_data["disc_obs"])

        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        pos_logits = self._model.eval_disc_raw(
            self._pos_diff.unsqueeze(0)).squeeze(-1)
        neg_logits = self._model.eval_disc_raw(norm_diff).squeeze(-1)

        flat_diff = norm_diff.detach().reshape(norm_diff.shape[0], -1)
        residual_energy = torch.mean(torch.square(flat_diff), dim=-1)
        quality_target = torch.reciprocal(1.0 + residual_energy)
        pos_prob = torch.sigmoid(pos_logits)
        neg_prob = torch.sigmoid(neg_logits)
        anchor_loss = F.mse_loss(pos_prob, torch.ones_like(pos_prob))
        quality_loss = F.mse_loss(neg_prob, quality_target)
        disc_loss = 0.5 * (anchor_loss + quality_loss)

        permutation = torch.randperm(quality_target.shape[0],
                                     device=quality_target.device)
        quality_delta = quality_target - quality_target[permutation]
        valid = quality_delta.abs() > 1e-6
        if torch.any(valid):
            score_delta = neg_prob - neg_prob[permutation]
            pair_acc = (
                (torch.sign(quality_delta[valid])
                 == torch.sign(score_delta[valid])).float().mean())
        else:
            pair_acc = neg_prob.new_zeros(())
        anchor_acc = (pos_prob[:, None] > neg_prob[None, :]).float().mean()

        return {
            "disc_loss": disc_loss,
            "disc_anchor_quality_loss": anchor_loss.detach(),
            "disc_quality_loss": quality_loss.detach(),
            "disc_anchor_acc": anchor_acc.detach(),
            "disc_pair_acc": pair_acc.detach(),
            "disc_pos_logit": pos_logits.mean().detach(),
            "disc_neg_logit": neg_logits.mean().detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }

    def _calc_disc_rewards(self, norm_diff):
        """Return the bounded learned quality probability."""
        with torch.no_grad():
            logits = torch_util.eval_minibatch(
                self._model.eval_disc_raw,
                {"disc_obs": norm_diff}, self._disc_eval_batch_size,
            ).squeeze(-1)
            return torch.sigmoid(logits)
