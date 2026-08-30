import torch

import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util


_SEMANTIC_RADIUS_EPS = 1e-6


def build_radius_balanced_semantic_negatives(diff, groups,
                                             eps=_SEMANTIC_RADIUS_EPS):
    """Keep one semantic group at a time while preserving total L2 radius.

    Returns counterfactuals with shape ``[..., num_groups, diff_dim]`` and a
    boolean mask identifying groups with a nonzero direction.  Invalid groups
    must not participate in hard-negative selection: a zero group has no
    direction that can be rescaled to the radius of the complete residual.
    """
    full_radius = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
    semantic_negatives = []
    valid_groups = []

    for _, group_indices in groups:
        indices = torch.as_tensor(
            group_indices, device=diff.device, dtype=torch.long)
        group_diff = torch.index_select(diff, -1, indices)
        group_radius = torch.linalg.vector_norm(
            group_diff, dim=-1, keepdim=True)
        valid = group_radius > eps
        scale = torch.where(
            valid, full_radius / torch.clamp_min(group_radius, eps),
            torch.zeros_like(group_radius))
        scaled_group = group_diff * scale
        semantic = torch.zeros_like(diff).index_copy(
            -1, indices, scaled_group)

        semantic_negatives.append(semantic)
        valid_groups.append(valid.squeeze(-1))

    return (torch.stack(semantic_negatives, dim=-2),
            torch.stack(valid_groups, dim=-1))


def calc_unscaled_disc_reward(logits):
    probability = torch.sigmoid(logits)
    return -torch.log(torch.clamp_min(1 - probability, 0.0001))


class ADDAgent(amp_agent.AMPAgent):
    """Exact a30 training path."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if self._disc_grad_penalty != 0:
            raise ValueError("a30 requires GP=0")
        self._disc_error_groups = self._env.get_disc_error_groups()
        self._pos_diff = self._build_pos_diff()

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
        disc_pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)

        semantic_negatives, valid_groups = \
            build_radius_balanced_semantic_negatives(
                norm_diff, self._disc_error_groups)
        all_negatives = torch.cat(
            (norm_diff.unsqueeze(-2), semantic_negatives), dim=-2)
        all_neg_logits = self._model.eval_disc(all_negatives).squeeze(-1)
        disc_neg_logit = all_neg_logits[..., 0]
        semantic_logits = all_neg_logits[..., 1:]

        masked_semantic_logits = semantic_logits.masked_fill(
            ~valid_groups, -torch.inf)
        semantic_hard_logit, semantic_hard_group = torch.max(
            masked_semantic_logits, dim=-1)
        has_semantic_direction = torch.any(valid_groups, dim=-1)
        semantic_hard_logit = torch.where(
            has_semantic_direction, semantic_hard_logit, disc_neg_logit)
        semantic_hard_group = torch.where(
            has_semantic_direction, semantic_hard_group,
            torch.full_like(semantic_hard_group, -1))

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg_full = self._disc_loss_neg(disc_neg_logit)
        disc_loss_neg_semantic = self._disc_loss_neg(semantic_hard_logit)
        disc_loss_neg = 0.5 * (
            disc_loss_neg_full + disc_loss_neg_semantic)
        disc_cls_loss = 0.5 * (disc_loss_pos + disc_loss_neg)
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_logit_reg_loss = self._disc_logit_reg * disc_logit_loss
        disc_loss = disc_cls_loss + disc_logit_reg_loss

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        disc_info = {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_cls_loss.detach().clone(),
            "disc_neg_full_loss": disc_loss_neg_full.detach(),
            "disc_neg_semantic_loss": disc_loss_neg_semantic.detach(),
            "disc_grad_penalty": torch.zeros((), device=self._device),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_logit_reg_loss": disc_logit_reg_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach(),
            "disc_semantic_hard_logit": torch.mean(
                semantic_hard_logit).detach(),
            "disc_semantic_hard_gain": torch.mean(
                semantic_hard_logit - disc_neg_logit).detach(),
            "disc_semantic_radius": torch.mean(torch.linalg.vector_norm(
                norm_diff, dim=-1)).detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width()
        }
        for group_id, (group_name, _) in enumerate(
                self._disc_error_groups):
            disc_info["disc_hard_{}_frac".format(group_name)] = torch.mean(
                (semantic_hard_group == group_id).float()).detach()
        return disc_info

    def _calc_disc_rewards(self, norm_diff):
        with torch.no_grad():
            logits = self._model.eval_disc(norm_diff).squeeze(-1)
            return self._disc_reward_scale * calc_unscaled_disc_reward(logits)
