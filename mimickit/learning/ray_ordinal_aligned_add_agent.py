import torch
import torch.nn.functional as functional

import learning.aligned_add_agent as aligned_add_agent


def build_ray_samples(norm_diff_obs, alpha=None):
    """Contract each normalized differential toward the exact zero anchor."""
    if alpha is None:
        alpha = torch.rand(
            (norm_diff_obs.shape[0], 1),
            device=norm_diff_obs.device,
            dtype=norm_diff_obs.dtype)
    if alpha.shape != (norm_diff_obs.shape[0], 1):
        raise ValueError("alpha must have shape [batch_size, 1]")
    return alpha * norm_diff_obs, alpha


def calc_ray_ordinal_objective(anchor_logit, ray_logit, negative_logit):
    """Balanced absolute and ray-order logistic objective.

    The one-quarter scale matches stock ADD's aggregate endpoint gradients at
    zero-logit initialization while adding f(0) > f(alpha*d) > f(d).
    """
    absolute_pos = torch.mean(functional.softplus(-anchor_logit))
    absolute_neg = torch.mean(functional.softplus(negative_logit))
    ordinal_near = torch.mean(functional.softplus(ray_logit - anchor_logit))
    ordinal_far = torch.mean(functional.softplus(negative_logit - ray_logit))
    objective = 0.25 * (
        absolute_pos + absolute_neg + ordinal_near + ordinal_far)
    return objective, {
        "absolute_pos": absolute_pos,
        "absolute_neg": absolute_neg,
        "ordinal_near": ordinal_near,
        "ordinal_far": ordinal_far,
    }


class RayOrdinalAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Aligned ADD with parameter-free ordering along residual rays.

    Policy conditioning, the differential representation and normalizer,
    replay, model architecture, optimizers, negative gradient penalty, PPO,
    and the stock ADD reward are inherited unchanged.  Only the discriminator
    objective receives one synthetic point alpha*d on every residual ray.
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
        ray_diff_obs, alpha = build_ray_samples(norm_diff_obs)

        ray_logit = self._model.eval_disc(ray_diff_obs).squeeze(-1)
        negative_logit = self._model.eval_disc(norm_diff_obs).squeeze(-1)
        ray_objective, terms = calc_ray_ordinal_objective(
            anchor_logit, ray_logit, negative_logit)

        # Keep stock ADD's output-weight regularizer and coefficient.
        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))

        # Keep stock ADD's gradient penalty on policy-generated negatives only.
        negative_grad = torch.autograd.grad(
            negative_logit, norm_diff_obs,
            grad_outputs=torch.ones_like(negative_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = torch.mean(
            torch.sum(torch.square(negative_grad), dim=-1))

        disc_loss = (
            ray_objective
            + self._disc_logit_reg * logit_loss
            + self._disc_grad_penalty * grad_penalty)

        anchor_above_ray = anchor_logit > ray_logit
        ray_above_negative = ray_logit > negative_logit
        anchor_negative_gap = anchor_logit - negative_logit
        anchor_ray_gap = anchor_logit - ray_logit
        ray_negative_gap = ray_logit - negative_logit
        negative_acc, positive_acc = self._compute_disc_acc(
            negative_logit, anchor_logit)

        info = {
            "disc_loss": disc_loss,
            "disc_ray_objective": ray_objective.detach(),
            "disc_absolute_pos_loss": terms["absolute_pos"].detach(),
            "disc_absolute_neg_loss": terms["absolute_neg"].detach(),
            "disc_ordinal_near_loss": terms["ordinal_near"].detach(),
            "disc_ordinal_far_loss": terms["ordinal_far"].detach(),
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_logit_loss": logit_loss.detach(),
            "disc_pos_acc": positive_acc.detach(),
            "disc_neg_acc": negative_acc.detach(),
            "disc_order_anchor_ray_acc": (
                torch.mean(anchor_above_ray.float()).detach()),
            "disc_order_ray_neg_acc": (
                torch.mean(ray_above_negative.float()).detach()),
            "disc_order_full_acc": torch.mean(
                (anchor_above_ray & ray_above_negative).float()).detach(),
            "disc_anchor_logit": torch.mean(anchor_logit).detach(),
            "disc_ray_logit": torch.mean(ray_logit).detach(),
            "disc_neg_logit": torch.mean(negative_logit).detach(),
            "disc_gap_anchor_ray": torch.mean(anchor_ray_gap).detach(),
            "disc_gap_ray_neg": torch.mean(ray_negative_gap).detach(),
            "disc_gap_anchor_neg": torch.mean(anchor_negative_gap).detach(),
            "disc_ray_alpha_mean": torch.mean(alpha).detach(),
            "disc_ray_alpha_std": torch.std(alpha, unbiased=False).detach(),
            "disc_ray_alpha_min": torch.min(alpha).detach(),
            "disc_ray_alpha_max": torch.max(alpha).detach(),
        }
        return info
