import torch

import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util


def calc_unscaled_disc_reward(logits):
    probability = torch.sigmoid(logits)
    return -torch.log(torch.clamp_min(1 - probability, 0.0001))


def calc_disc_gradient_penalty(disc_neg_logit, neg_diff,
                               disc_pos_logit, pos_diff):
    """Official release ADD penalty on negative and zero-positive inputs."""
    disc_neg_grad = torch.autograd.grad(
        disc_neg_logit, neg_diff,
        grad_outputs=torch.ones_like(disc_neg_logit),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    neg_penalty = torch.mean(torch.sum(
        torch.square(disc_neg_grad), dim=-1))

    disc_pos_grad = torch.autograd.grad(
        disc_pos_logit, pos_diff,
        grad_outputs=torch.ones_like(disc_pos_logit),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    pos_penalty = torch.mean(torch.sum(
        torch.square(disc_pos_grad), dim=-1))

    penalty = 0.5 * (neg_penalty + pos_penalty)
    return penalty, neg_penalty, pos_penalty


class ADDAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._pos_diff = self._build_pos_diff()
        return

    def _build_model(self, config):
        self._model = add_model.ADDModel(config["model"], self._env)
        return

    def _build_pos_diff(self):
        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        return torch.zeros(
            disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)

    def _build_normalizers(self):
        super(amp_agent.AMPAgent, self)._build_normalizers()
        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(
            disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super(amp_agent.AMPAgent, self)._record_data_post_step(
            next_obs, r, done, next_info)
        self._exp_buffer.record("disc_obs_demo", next_info["disc_obs_demo"])
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])
        return

    def _record_disc_demo_data(self):
        return

    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        idx = self._sample_disc_replay_indices(disc_obs.shape[0])
        disc_data = {
            "disc_obs": disc_obs[idx].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[idx].unsqueeze(1)
        }
        self._disc_buffer.push(disc_data)
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        disc_r = self._calc_disc_rewards(norm_obs_diff)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        reward = (self._task_reward_weight * task_r
                  + self._disc_reward_weight * disc_r)
        self._exp_buffer.set_data_flat("reward", reward)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(obs_diff)

        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std
        }

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        disc_obs_demo = batch["disc_obs_demo"]

        current_diff = disc_obs_demo - disc_obs
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = (replay_data["disc_obs_demo"]
                       - replay_data["disc_obs"])
        diff = torch.cat([current_diff, replay_diff], dim=0)
        norm_diff = self._disc_obs_norm.normalize(diff)

        use_grad_penalty = self._disc_grad_penalty > 0
        if use_grad_penalty:
            norm_diff.requires_grad_(True)

        pos_diff = self._pos_diff.clone().unsqueeze(0)
        if use_grad_penalty:
            pos_diff.requires_grad_(True)

        disc_pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)
        disc_neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_cls_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_logit_reg_loss = self._disc_logit_reg * disc_logit_loss
        disc_loss = disc_cls_loss + disc_logit_reg_loss

        if use_grad_penalty:
            (disc_grad_penalty,
             disc_grad_penalty_neg,
             disc_grad_penalty_pos) = calc_disc_gradient_penalty(
                 disc_neg_logit, norm_diff, disc_pos_logit, pos_diff)
            disc_loss += self._disc_grad_penalty * disc_grad_penalty
        else:
            disc_grad_penalty = torch.zeros((), device=self._device)
            disc_grad_penalty_neg = torch.zeros((), device=self._device)
            disc_grad_penalty_pos = torch.zeros((), device=self._device)

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_cls_loss.detach(),
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_grad_penalty_neg": disc_grad_penalty_neg.detach(),
            "disc_grad_penalty_pos": disc_grad_penalty_pos.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_logit_reg_loss": disc_logit_reg_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach()
        }

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            if self._disc_eval_batch_size <= 0:
                disc_logits = self._model.eval_disc(norm_diff)
            else:
                logits = []
                for start in range(0, norm_diff.shape[0],
                                   self._disc_eval_batch_size):
                    end = min(start + self._disc_eval_batch_size,
                              norm_diff.shape[0])
                    logits.append(self._model.eval_disc(norm_diff[start:end]))
                disc_logits = torch.cat(logits, dim=0)
            disc_r = calc_unscaled_disc_reward(disc_logits.squeeze(-1))
            return self._disc_reward_scale * disc_r
