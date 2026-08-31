import torch

import learning.add_agent as add_agent
import learning.rd_fsn_add_model as rd_fsn_add_model
import util.torch_util as torch_util


class RDFSNADDAgent(add_agent.ADDAgent):
    """Reward-Decoupled Full Spectral Normalization ADD."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("RD-FSN ADD requires disc_grad_penalty=0")

    def _build_model(self, config):
        self._model = rd_fsn_add_model.RDFSNADDModel(
            config["model"], self._env)

    def _compute_disc_loss(self, batch):
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)

        pos_diff = self._pos_diff.unsqueeze(0)
        all_inputs = torch.cat((pos_diff, norm_diff), dim=0)
        all_scores = self._model.eval_disc_score(all_inputs).squeeze(-1)
        pos_score = all_scores[:1]
        neg_score = all_scores[1:]

        class_scale = self._model.get_disc_class_scale()
        pos_logit = class_scale * pos_score
        neg_logit = class_scale * neg_score
        pos_loss = self._disc_loss_pos(pos_logit)
        neg_loss = self._disc_loss_neg(neg_logit)
        cls_loss = 0.5 * (pos_loss + neg_loss)

        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))
        logit_reg_loss = self._disc_logit_reg * logit_loss
        disc_loss = cls_loss + logit_reg_loss

        score_gap = torch.abs(neg_score - pos_score[0])
        radius = torch.linalg.vector_norm(norm_diff, dim=-1)
        reward_violation = torch.relu(score_gap - radius)
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
            # Standard logit fields retain their classifier meaning.
            "disc_pos_logit": pos_logit.mean().detach(),
            "disc_neg_logit": neg_logit.mean().detach(),
            # These are the values actually consumed by PPO reward.
            "disc_reward_pos_score": pos_score.mean().detach(),
            "disc_reward_neg_score": neg_score.mean().detach(),
            "disc_class_scale": class_scale.detach(),
            "disc_reward_lipschitz_bound": (
                self._model.get_disc_reward_lipschitz_bound()),
            "disc_class_lipschitz_bound": (
                self._model.get_disc_class_lipschitz_bound().detach()),
            "disc_score_gap_mean": score_gap.mean().detach(),
            "disc_score_gap_p99": torch.quantile(score_gap, 0.99).detach(),
            "disc_reward_lip_violation_max": reward_violation.max().detach(),
            "disc_reward_lip_violation_frac": (
                (reward_violation > 1e-4).float().mean().detach()),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            scores = torch_util.eval_minibatch(
                self._model.eval_disc_score,
                {"diff": norm_diff},
                self._disc_eval_batch_size).squeeze(-1)
            return (self._disc_reward_scale
                    * add_agent.calc_unscaled_disc_reward(scores))
