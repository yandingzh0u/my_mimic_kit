import torch

import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util


def calc_unscaled_disc_reward(logits):
    probability = torch.sigmoid(logits)
    return -torch.log(torch.clamp_min(1 - probability, 0.0001))


class ADDAgent(amp_agent.AMPAgent):
    """Official ADD: direct differential classification with Both-GP."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._pos_diff = self._build_pos_diff()

    def _build_model(self, config):
        self._model = add_model.ADDModel(config["model"], self._env)

    def _build_pos_diff(self):
        space = self._env.get_disc_obs_space()
        dtype = torch_util.numpy_dtype_to_torch(space.dtype)
        return torch.zeros(space.shape, device=self._device, dtype=dtype)

    def _build_normalizers(self):
        super(amp_agent.AMPAgent, self)._build_normalizers()
        space = self._env.get_disc_obs_space()
        dtype = torch_util.numpy_dtype_to_torch(space.dtype)
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(
            space.shape, device=self._device, dtype=dtype)

    def _record_data_post_step(self, next_obs, reward, done, next_info):
        super(amp_agent.AMPAgent, self)._record_data_post_step(
            next_obs, reward, done, next_info)
        self._exp_buffer.record("disc_obs_demo", next_info["disc_obs_demo"])
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])

    def _record_disc_demo_data(self):
        return

    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        indices = self._sample_disc_replay_indices(disc_obs.shape[0])
        self._disc_buffer.push({
            "disc_obs": disc_obs[indices].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[indices].unsqueeze(1),
        })

    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        raw_diff = disc_obs_demo - disc_obs
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        disc_reward = self._calc_disc_rewards(norm_diff)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_reward)

        reward = (self._task_reward_weight * task_reward
                  + self._disc_reward_weight * disc_reward)
        self._exp_buffer.set_data_flat("reward", reward)
        if self._need_normalizer_update():
            self._disc_obs_norm.record(raw_diff)
        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
        }

    def _compute_disc_loss(self, batch):
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)

        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        norm_diff.requires_grad_(True)
        pos_diff = self._pos_diff.unsqueeze(0).requires_grad_(True)

        neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)
        pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logit)
        neg_loss = self._disc_loss_neg(neg_logit)
        cls_loss = 0.5 * (pos_loss + neg_loss)

        neg_grad = torch.autograd.grad(
            neg_logit, norm_diff, torch.ones_like(neg_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        pos_grad = torch.autograd.grad(
            pos_logit, pos_diff, torch.ones_like(pos_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        neg_gp = torch.mean(torch.sum(torch.square(neg_grad), dim=-1))
        pos_gp = torch.mean(torch.sum(torch.square(pos_grad), dim=-1))
        grad_penalty = 0.5 * (neg_gp + pos_gp)

        logit_weights = self._model.get_disc_logit_weights()
        logit_loss = torch.sum(torch.square(logit_weights))
        logit_reg_loss = self._disc_logit_reg * logit_loss
        disc_loss = (cls_loss
                     + self._disc_grad_penalty * grad_penalty
                     + logit_reg_loss)

        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": cls_loss.detach(),
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_neg_grad_penalty": neg_gp.detach(),
            "disc_pos_grad_penalty": pos_gp.detach(),
            "disc_logit_loss": logit_loss.detach(),
            "disc_logit_reg_loss": logit_reg_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logit.mean().detach(),
            "disc_neg_logit": neg_logit.mean().detach(),
        }

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            logits = torch_util.eval_minibatch(
                self._model.eval_disc, {"disc_obs": norm_diff},
                self._disc_eval_batch_size).squeeze(-1)
            return self._disc_reward_scale * calc_unscaled_disc_reward(logits)
