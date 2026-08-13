import torch

import learning.add_agent as add_agent
import learning.st_add_model as st_add_model

class STADDAgent(add_agent.ADDAgent):
    """Structured Trajectory ADD agent.

    Identical to ADD except for the discriminator loss:

        L_D = L_fused + alpha * (L_state + L_motion + L_rot) + logit reg + GP

    where L_fused is the standard ADD zero-vs-policy BCE on the fused logit
    and each branch gets the same zero-vs-policy BCE on its own logit so the
    fused network cannot learn to ignore a branch (in particular the winding
    branch). The gradient penalty is applied to the fused logit w.r.t. the
    full differential, as in ADD.

    The policy reward path is fully inherited: ONE fused logit ->
    disc_reward_scale * softplus -> ONE always-positive PPO reward.
    """

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_aux_branch_weight = config.get("disc_aux_branch_weight", 0.5)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = st_add_model.STADDModel(model_config, self._env)
        return

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]

        pos_diff = self._pos_diff.clone()
        pos_diff = pos_diff.unsqueeze(dim=0)
        pos_diff.requires_grad_(True)
        pos_state_logit, pos_motion_logit, pos_rot_logit, pos_fused_logit = self._model.eval_disc_branches(pos_diff)
        pos_state_logit = pos_state_logit.squeeze(-1)
        pos_motion_logit = pos_motion_logit.squeeze(-1)
        pos_rot_logit = pos_rot_logit.squeeze(-1)
        pos_fused_logit = pos_fused_logit.squeeze(-1)

        diff_obs = tar_disc_obs - disc_obs

        replay_data = self._disc_buffer.sample(diff_obs.shape[0])
        replay_disc_obs = replay_data["disc_obs"]
        replay_tar_disc_obs = replay_data["disc_obs_demo"]
        replay_diff = replay_tar_disc_obs - replay_disc_obs
        diff_obs = torch.cat([diff_obs, replay_diff], dim=0)

        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_diff_obs.requires_grad_(True)
        neg_state_logit, neg_motion_logit, neg_rot_logit, neg_fused_logit = self._model.eval_disc_branches(norm_diff_obs)
        neg_state_logit = neg_state_logit.squeeze(-1)
        neg_motion_logit = neg_motion_logit.squeeze(-1)
        neg_rot_logit = neg_rot_logit.squeeze(-1)
        neg_fused_logit = neg_fused_logit.squeeze(-1)

        # fused zero-vs-policy BCE (standard ADD objective)
        disc_loss_pos = self._disc_loss_pos(pos_fused_logit)
        disc_loss_neg = self._disc_loss_neg(neg_fused_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # auxiliary per-branch zero-vs-policy BCE: keeps every branch (most
        # importantly the winding branch) individually discriminative so the
        # fused logit cannot ignore it
        aux_state_loss = 0.5 * (self._disc_loss_pos(pos_state_logit) + self._disc_loss_neg(neg_state_logit))
        aux_motion_loss = 0.5 * (self._disc_loss_pos(pos_motion_logit) + self._disc_loss_neg(neg_motion_logit))
        aux_rot_loss = 0.5 * (self._disc_loss_pos(pos_rot_logit) + self._disc_loss_neg(neg_rot_logit))
        disc_aux_loss = aux_state_loss + aux_motion_loss + aux_rot_loss
        disc_loss += self._disc_aux_branch_weight * disc_aux_loss

        # logit reg over all branch heads
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty on the fused logit w.r.t. the full differential
        disc_neg_grad = torch.autograd.grad(neg_fused_logit, norm_diff_obs,
                                            grad_outputs=torch.ones_like(neg_fused_logit),
                                            create_graph=True, retain_graph=True, only_inputs=True)
        disc_neg_grad = disc_neg_grad[0]
        disc_neg_grad_squared = torch.sum(torch.square(disc_neg_grad), dim=-1)

        disc_pos_grad = torch.autograd.grad(pos_fused_logit, pos_diff,
                                            grad_outputs=torch.ones_like(pos_fused_logit),
                                            create_graph=True, retain_graph=True, only_inputs=True)
        disc_pos_grad = disc_pos_grad[0]
        disc_pos_grad_squared = torch.sum(torch.square(disc_pos_grad), dim=-1)

        disc_grad_penalty = 0.5 * (torch.mean(disc_neg_grad_squared) + torch.mean(disc_pos_grad_squared))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(neg_fused_logit, pos_fused_logit)
        rot_neg_acc, rot_pos_acc = self._compute_disc_acc(neg_rot_logit, pos_rot_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_aux_loss": disc_aux_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(pos_fused_logit).detach(),
            "disc_neg_logit": torch.mean(neg_fused_logit).detach(),
            "disc_state_neg_logit": torch.mean(neg_state_logit).detach(),
            "disc_motion_neg_logit": torch.mean(neg_motion_logit).detach(),
            "disc_rot_neg_logit": torch.mean(neg_rot_logit).detach(),
            "disc_rot_pos_logit": torch.mean(pos_rot_logit).detach(),
            "disc_rot_neg_acc": rot_neg_acc.detach(),
            "disc_rot_pos_acc": rot_pos_acc.detach(),
        }
        return disc_info
