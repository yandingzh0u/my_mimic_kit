import torch

import learning.add_agent as add_agent
import learning.dare_model as dare_model
import util.torch_util as torch_util


class DAREAgent(add_agent.ADDAgent):
    """Zero-differential adversarial learning with a CPL discriminator."""

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_pairwise_weight = float(
            config.get("disc_pairwise_weight", 0.0))

    def _init_iter(self):
        super()._init_iter()
        if self._disc_pairwise_weight > 0.0:
            self._rank_episode_start = torch.zeros(
                self.get_num_envs(), device=self._device, dtype=torch.long)

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        if self._disc_pairwise_weight <= 0.0:
            return

        sentinel = torch.full(
            (self.get_num_envs(),), -1.0, device=self._device)
        self._exp_buffer.record("disc_outcome", sentinel)
        done_ids = torch.nonzero(done != 0, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return
        head = int(self._exp_buffer._buffer_head)
        outcome_buffer = self._exp_buffer.get_data("disc_outcome")
        for env_id in done_ids.detach().cpu().tolist():
            start = int(self._rank_episode_start[env_id].item())
            cursor = start
            label = 1.0 if int(done[env_id].item()) == 2 else 0.0
            while True:
                outcome_buffer[cursor, env_id] = label
                if cursor == head:
                    break
                cursor = (cursor + 1) % self._exp_buffer._buffer_length
            self._rank_episode_start[env_id] = (head + 1) % self._exp_buffer._buffer_length

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _compute_disc_loss(self, batch):
        """Classify the zero anchor against current and replay residuals."""
        # Preserve the positive-first spectral-normalization update order.
        pos_logits = self._model.eval_disc_raw(
            self._pos_diff.unsqueeze(0)).squeeze(-1)
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = (replay_data["disc_obs_demo"]
                       - replay_data["disc_obs"])

        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        neg_logits = self._model.eval_disc_raw(norm_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logits)
        neg_loss = self._disc_loss_neg(neg_logits)
        disc_loss = 0.5 * (pos_loss + neg_loss)
        neg_acc, pos_acc = self._compute_disc_acc(neg_logits, pos_logits)

        pairwise_loss = torch.zeros((), device=current_diff.device)
        pair_count = torch.zeros((), device=current_diff.device)
        pairwise_weight = getattr(self, "_disc_pairwise_weight", 0.0)
        if pairwise_weight > 0.0 and "disc_outcome" in batch:
            current_norm_diff = norm_diff[:current_diff.shape[0]]
            current_logits = self._model.eval_disc_raw(
                current_norm_diff).squeeze(-1)
            outcome = batch["disc_outcome"].flatten()
            phase = batch.get("diag_phase", torch.zeros_like(outcome)).flatten()
            phase_bin = torch.clamp((phase * 8).long(), 0, 7)
            pos_indices = []
            neg_indices = []
            for bin_id in range(8):
                ids = torch.nonzero(phase_bin == bin_id, as_tuple=False).flatten()
                pos_ids = ids[outcome[ids] > 0.5]
                neg_ids = ids[outcome[ids] < 0.5]
                count = min(pos_ids.numel(), neg_ids.numel())
                if count:
                    pos_indices.append(pos_ids[:count])
                    neg_indices.append(neg_ids[:count])
            if pos_indices:
                pos_ids = torch.cat(pos_indices)
                neg_ids = torch.cat(neg_indices)
                pairwise_loss = torch.nn.functional.softplus(
                    current_logits[neg_ids] - current_logits[pos_ids]).mean()
                pair_count = torch.tensor(
                    float(pos_ids.numel()), device=current_diff.device)
                disc_loss = disc_loss + pairwise_weight * pairwise_loss

        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logits.mean().detach(),
            "disc_neg_logit": neg_logits.mean().detach(),
            "disc_pairwise_loss": pairwise_loss.detach(),
            "disc_pairwise_pairs": pair_count.detach(),
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }

    def _calc_disc_rewards(self, norm_diff):
        """Use the bounded ADD reward paired with the BCE discriminator."""
        with torch.no_grad():
            logits = torch_util.eval_minibatch(
                self._model.eval_disc_raw,
                {"disc_obs": norm_diff}, self._disc_eval_batch_size,
            ).squeeze(-1)
            return self._disc_reward_scale * add_agent.calc_unscaled_disc_reward(
                logits)
