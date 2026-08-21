import torch
import torch.nn.functional as functional

import learning.aligned_add_agent as aligned_add_agent
import util.torch_util as torch_util


def calc_relative_logit(negative_logit, anchor_logit):
    """Compare every policy differential with the fixed zero differential."""
    return negative_logit - anchor_logit


def calc_relative_logistic_loss(relative_logit):
    """Match the class-balanced gradient scale of stock ADD's BCE objective."""
    return 0.5 * torch.mean(functional.softplus(relative_logit))


def calc_symmetric_relative_reward(relative_logit, reward_scale):
    """Bound reward by reward_scale and peak at discriminator indifference."""
    return 2.0 * reward_scale * torch.sigmoid(-torch.abs(relative_logit))


class RelativeLogisticAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Aligned ADD with a zero-anchor pairwise logistic discriminator.

    The policy conditioning, differential representation, normalizer, replay
    buffer, network, optimizers, gradient penalty, PPO, and architecture are
    inherited unchanged.  Only the absolute BCE discriminator objective and
    its reward mapping are replaced.
    """

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        target_disc_obs = batch["disc_obs_demo"]

        anchor_diff = self._pos_diff.unsqueeze(dim=0)
        anchor_logit = self._model.eval_disc(anchor_diff).squeeze(-1)

        diff_obs = target_disc_obs - disc_obs
        replay_data = self._disc_buffer.sample(diff_obs.shape[0])
        replay_diff = (
            replay_data["disc_obs_demo"] - replay_data["disc_obs"])
        diff_obs = torch.cat([diff_obs, replay_diff], dim=0)

        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_diff_obs.requires_grad_(True)
        negative_logit = self._model.eval_disc(norm_diff_obs).squeeze(-1)
        relative_logit = calc_relative_logit(negative_logit, anchor_logit)

        relative_loss = calc_relative_logistic_loss(relative_logit)

        # Keep stock ADD's output-weight regularizer and coefficient.
        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))

        # Since f(0) is independent of delta, grad_delta q(delta) equals
        # grad_delta f(delta).  This is exactly stock ADD's negative GP.
        relative_grad = torch.autograd.grad(
            relative_logit, norm_diff_obs,
            grad_outputs=torch.ones_like(relative_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = torch.mean(
            torch.sum(torch.square(relative_grad), dim=-1))

        # Build a new tensor instead of mutating relative_loss in place.  The
        # minibatch logger accumulates every metric independently, so aliased
        # tensors would corrupt both reported values even though backprop uses
        # the correct graph.
        disc_loss = (
            relative_loss
            + self._disc_logit_reg * logit_loss
            + self._disc_grad_penalty * grad_penalty)

        order_acc = torch.mean((relative_logit < 0).float())
        info = {
            "disc_loss": disc_loss,
            "disc_relative_loss": relative_loss.detach(),
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_logit_loss": logit_loss.detach(),
            "disc_order_acc": order_acc.detach(),
            "disc_anchor_logit": torch.mean(anchor_logit).detach(),
            "disc_neg_logit": torch.mean(negative_logit).detach(),
            "disc_relative_logit": torch.mean(relative_logit).detach(),
            "disc_relative_abs_logit": (
                torch.mean(torch.abs(relative_logit)).detach()),
        }
        return info

    def _calc_disc_rewards(self, norm_disc_obs):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_disc_obs}
            negative_logit = torch_util.eval_minibatch(
                self._model.eval_disc, disc_inputs,
                self._disc_eval_batch_size).squeeze(-1)

            anchor_diff = self._pos_diff.unsqueeze(dim=0)
            anchor_logit = self._model.eval_disc(anchor_diff).squeeze(-1)
            relative_logit = calc_relative_logit(
                negative_logit, anchor_logit)
            reward = calc_symmetric_relative_reward(
                relative_logit, self._disc_reward_scale)
        return reward
