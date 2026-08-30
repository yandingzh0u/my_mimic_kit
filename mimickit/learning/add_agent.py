import torch

import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util


def calc_unscaled_disc_reward(logits):
    return torch.nn.functional.softplus(logits)


def build_semantic_contractions(diff, groups, tau):
    """Scale exactly one semantic group per counterfactual sample."""
    num_groups = len(groups)
    if tau.shape != (num_groups, diff.shape[0], 1):
        raise ValueError("Semantic contraction tau has an invalid shape")

    contractions = diff.unsqueeze(0).expand(
        num_groups, *diff.shape).clone()
    for group_id, (_, indices) in enumerate(groups):
        contractions[group_id, :, indices] *= tau[group_id]
    return contractions


def calc_pc_loss(pos_logit, neg_logit, contraction_logit):
    """Joint likelihood for zero, policy, and semantic-order events."""
    pos_loss = torch.nn.functional.softplus(-pos_logit).mean()
    neg_loss = torch.nn.functional.softplus(neg_logit).mean()
    preference_loss = torch.nn.functional.softplus(
        neg_logit.unsqueeze(0) - contraction_logit).mean(dim=1)
    loss = (pos_loss + neg_loss + preference_loss.sum()) \
        / (preference_loss.shape[0] + 2)
    return loss, pos_loss, neg_loss, preference_loss


class ADDAgent(amp_agent.AMPAgent):
    """Pareto-consistent ADD with semantic contraction supervision."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("PC-ADD requires GP=0")
        if self._disc_logit_reg != 0:
            raise ValueError("PC-ADD requires logit regularization=0")
        self._pos_diff = self._build_pos_diff()
        self._disc_error_groups = self._env.get_disc_error_groups()

    def _build_model(self, config):
        self._model = add_model.ADDModel(config["model"], self._env)

    def _build_pos_diff(self):
        disc_obs_space = self._env.get_disc_obs_space()
        dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        return torch.zeros(
            disc_obs_space.shape, device=self._device, dtype=dtype)

    def _build_normalizers(self):
        super(amp_agent.AMPAgent, self)._build_normalizers()
        disc_obs_space = self._env.get_disc_obs_space()
        dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(
            disc_obs_space.shape, device=self._device, dtype=dtype)

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super(amp_agent.AMPAgent, self)._record_data_post_step(
            next_obs, r, done, next_info)
        self._exp_buffer.record("disc_obs_demo", next_info["disc_obs_demo"])
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])

    def _record_disc_demo_data(self):
        return

    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        idx = self._sample_disc_replay_indices(disc_obs.shape[0])
        self._disc_buffer.push({
            "disc_obs": disc_obs[idx].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[idx].unsqueeze(1)
        })

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        obs_diff = disc_obs_demo - disc_obs
        norm_diff = self._disc_obs_norm.normalize(obs_diff)
        disc_r = self._calc_disc_rewards(norm_diff)
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
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        raw_diff = torch.cat((current_diff, replay_diff), dim=0)

        num_groups = len(self._disc_error_groups)
        tau = torch.rand(
            (num_groups, raw_diff.shape[0], 1),
            device=raw_diff.device, dtype=raw_diff.dtype)
        contractions = build_semantic_contractions(
            raw_diff, self._disc_error_groups, tau)

        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        flat_contractions = contractions.flatten(0, 1)
        norm_contractions = self._disc_obs_norm.normalize(flat_contractions)
        pos_diff = self._pos_diff.clone().unsqueeze(0)

        # A single forward keeps every comparison on the same SN weights.
        all_inputs = torch.cat(
            (pos_diff, norm_diff, norm_contractions), dim=0)
        all_logits = self._model.eval_disc(all_inputs).squeeze(-1)
        disc_pos_logit = all_logits[:1]
        neg_end = 1 + norm_diff.shape[0]
        disc_neg_logit = all_logits[1:neg_end]
        contraction_logit = all_logits[neg_end:].reshape(
            num_groups, norm_diff.shape[0])

        disc_loss, disc_loss_pos, disc_loss_neg, preference_loss = \
            calc_pc_loss(
                disc_pos_logit, disc_neg_logit, contraction_logit)
        disc_cls_loss = 0.5 * (disc_loss_pos + disc_loss_neg)
        preference_margin = contraction_logit - disc_neg_logit.unsqueeze(0)
        preference_acc = torch.mean(
            (preference_margin > 0).float(), dim=1)

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        zero = torch.zeros((), device=self._device)
        info = {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_cls_loss.detach().clone(),
            "disc_pc_loss": preference_loss.mean().detach(),
            "disc_pc_acc": preference_acc.mean().detach(),
            "disc_pc_margin": preference_margin.mean().detach(),
            "disc_pc_tau": tau.mean().detach(),
            "disc_grad_penalty": zero,
            "disc_logit_loss": zero,
            "disc_logit_reg_loss": zero,
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width()
        }
        for group_id, (name, _) in enumerate(self._disc_error_groups):
            prefix = "disc_pc_{}".format(name)
            info[prefix + "_loss"] = preference_loss[group_id].detach()
            info[prefix + "_acc"] = preference_acc[group_id].detach()
            info[prefix + "_margin"] = \
                preference_margin[group_id].mean().detach()
        return info

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            logits = self._model.eval_disc(norm_diff).squeeze(-1)
            return self._disc_reward_scale * calc_unscaled_disc_reward(logits)
