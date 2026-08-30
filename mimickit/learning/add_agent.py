import torch

import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util


def calc_unscaled_disc_reward(logits):
    return torch.nn.functional.softplus(logits)


class ADDAgent(amp_agent.AMPAgent):
    """ADD with a structurally Pareto-dominant residual discriminator."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("PDR-ADD requires GP=0")
        if self._disc_logit_reg != 0:
            raise ValueError("PDR-ADD requires logit regularization=0")
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

        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        pos_diff = self._pos_diff.clone().unsqueeze(0)

        # Fixed contraction probes are diagnostics only and never enter loss.
        probe_count = min(64, norm_diff.shape[0])
        probe_tau = 0.5
        num_groups = len(self._disc_error_groups)
        probes = norm_diff[:probe_count].unsqueeze(0).expand(
            num_groups, probe_count, norm_diff.shape[-1]).clone()
        for group_id, (_, indices) in enumerate(self._disc_error_groups):
            probes[group_id, :, indices] *= probe_tau

        all_inputs = torch.cat((
            pos_diff,
            norm_diff,
            probes.flatten(0, 1),
        ), dim=0)
        all_logits, all_residuals, all_cores, all_radii = \
            self._model.eval_disc_components(all_inputs)
        all_logits = all_logits.squeeze(-1)
        disc_pos_logit = all_logits[:1]
        neg_end = 1 + norm_diff.shape[0]
        disc_neg_logit = all_logits[1:neg_end]
        probe_logits = all_logits[neg_end:].reshape(
            num_groups, probe_count)
        neg_residuals = all_residuals[1:neg_end]
        neg_cores = all_cores[1:neg_end]
        neg_radii = all_radii[1:neg_end]

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_cls_loss = 0.5 * (disc_loss_pos + disc_loss_neg)
        disc_loss = disc_cls_loss

        probe_gain = probe_logits - disc_neg_logit[:probe_count].unsqueeze(0)
        probe_scale = ((1.0 - probe_tau)
                       * neg_radii[:probe_count].transpose(0, 1))
        probe_slopes = probe_gain / torch.clamp(probe_scale, min=1e-8)

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        zero = torch.zeros((), device=self._device)
        info = {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_cls_loss.detach().clone(),
            "disc_grad_penalty": zero,
            "disc_logit_loss": zero,
            "disc_logit_reg_loss": zero,
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach(),
            "disc_residual_logit_mean": neg_residuals.mean().detach(),
            "disc_residual_logit_std": neg_residuals.std().detach(),
            "disc_semantic_core_mean": neg_cores.mean().detach(),
            "disc_semantic_core_std": neg_cores.std().detach(),
            "disc_theoretical_floor":
                self._model.get_disc_theoretical_floor(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width()
        }
        for group_id, (name, _) in enumerate(self._disc_error_groups):
            info["disc_rho_{}".format(name)] = \
                neg_radii[:, group_id].mean().detach()
            info["disc_slope_{}_mean".format(name)] = \
                probe_slopes[group_id].mean().detach()
            info["disc_slope_{}_min".format(name)] = \
                probe_slopes[group_id].min().detach()
        return info

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            logits = self._model.eval_disc(norm_diff).squeeze(-1)
            return self._disc_reward_scale * calc_unscaled_disc_reward(logits)
