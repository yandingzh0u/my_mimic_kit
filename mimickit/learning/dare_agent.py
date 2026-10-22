import math

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
        self._disc_pair_bins = int(config.get("disc_pair_bins", 64))
        self._disc_pair_mag_tol = float(config.get("disc_pair_mag_tol", 0.05))
        self._disc_pair_pool_max = int(config.get("disc_pair_pool_max", 20000))
        self._disc_pair_batch = int(config.get("disc_pair_batch", 256))
        self._disc_rank_label = str(config.get("disc_rank_label", "env_succ"))
        if self._disc_rank_label != "env_succ":
            raise ValueError(
                "Only env SUCC labels are available online; offline strict "
                "completion is a different label and must not be mixed in")
        self._rank_pool = None

    def _update_model(self):
        # Build the rollout-level pair pool once per iteration, before the
        # discriminator updates consume it.
        self._rank_pool = self._build_rank_pool()
        return super()._update_model()

    @torch.no_grad()
    def _build_rank_pool(self):
        """Matched (success, failure) pairs over the whole rollout.

        Matching mirrors the offline screen: 64 phase bins and a magnitude
        tolerance of disc_pair_mag_tol. Unresolved episodes (-1) are excluded.
        """
        empty = {"pos": None, "neg": None, "pairs": 0, "pos_count": 0,
                 "neg_count": 0, "unknown": 0, "cells": 0}
        if self._disc_pairwise_weight <= 0.0:
            return empty
        buf = self._exp_buffer
        for name in ("disc_outcome", "diag_phase", "disc_obs", "disc_obs_demo"):
            if not buf.has_buffer(name):
                return empty
        outcome = buf.get_data_flat("disc_outcome").flatten().to(torch.float64)
        phase = buf.get_data_flat("diag_phase").flatten()
        obs = buf.get_data_flat("disc_obs")
        demo = buf.get_data_flat("disc_obs_demo")
        samples = int(outcome.numel())
        is_pos = outcome == 1.0
        is_neg = outcome == 0.0
        unknown = int((~(is_pos | is_neg)).sum())
        pos_idx = torch.nonzero(is_pos, as_tuple=False).flatten()
        neg_idx = torch.nonzero(is_neg, as_tuple=False).flatten()
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return {**empty, "pos_count": int(pos_idx.numel()),
                    "neg_count": int(neg_idx.numel()), "unknown": unknown}
        magnitude = torch.empty(samples, device=self._device, dtype=torch.float32)
        chunk = 32768
        for start in range(0, samples, chunk):
            end = min(start + chunk, samples)
            diff = demo[start:end] - obs[start:end]
            norm = self._disc_obs_norm.normalize(diff)
            magnitude[start:end] = torch.linalg.vector_norm(norm, dim=-1)
        bins = max(1, self._disc_pair_bins)
        tol = max(self._disc_pair_mag_tol, 1e-6)
        phase_bin = torch.clamp((phase * bins).long(), 0, bins - 1)
        mag_bin = torch.floor(magnitude / tol).long()
        cell = phase_bin * (int(mag_bin.max()) + 1) + mag_bin
        pool_pos, pool_neg = self._matched_rank_pairs(pos_idx, neg_idx, cell)
        if pool_pos is None or pool_pos.numel() == 0:
            return {**empty, "pos_count": int(pos_idx.numel()),
                    "neg_count": int(neg_idx.numel()), "unknown": unknown,
                    "cells": int(torch.unique(cell).numel())}
        if pool_pos.numel() > self._disc_pair_pool_max:
            stride = int(math.ceil(pool_pos.numel() / self._disc_pair_pool_max))
            pool_pos = pool_pos[::stride]
            pool_neg = pool_neg[::stride]
        pos_diff = self._disc_obs_norm.normalize(demo[pool_pos] - obs[pool_pos])
        neg_diff = self._disc_obs_norm.normalize(demo[pool_neg] - obs[pool_neg])
        return {"pos": pos_diff, "neg": neg_diff, "pairs": int(pool_pos.numel()),
                "pos_count": int(pos_idx.numel()), "neg_count": int(neg_idx.numel()),
                "unknown": unknown, "cells": int(torch.unique(cell).numel())}

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
        cls_loss = 0.5 * (pos_loss + neg_loss)
        # Separate storage: disc_loss is accumulated in place by the logger.
        disc_loss = cls_loss.clone()
        neg_acc, pos_acc = self._compute_disc_acc(neg_logits, pos_logits)

        pairwise_loss = torch.zeros((), device=current_diff.device)
        pair_count = torch.zeros((), device=current_diff.device)
        pos_count = torch.zeros((), device=current_diff.device)
        neg_count = torch.zeros((), device=current_diff.device)
        unknown_count = torch.zeros((), device=current_diff.device)
        pool_pairs = torch.zeros((), device=current_diff.device)
        pairwise_weight = getattr(self, "_disc_pairwise_weight", 0.0)
        pool = getattr(self, "_rank_pool", None) or {}
        if pool:
            pos_count = torch.tensor(float(pool.get("pos_count", 0)),
                                     device=current_diff.device)
            neg_count = torch.tensor(float(pool.get("neg_count", 0)),
                                     device=current_diff.device)
            unknown_count = torch.tensor(float(pool.get("unknown", 0)),
                                         device=current_diff.device)
            pool_pairs = torch.tensor(float(pool.get("pairs", 0)),
                                      device=current_diff.device)
        if (pairwise_weight > 0.0 and pool.get("pos") is not None
                and int(pool.get("pairs", 0)) > 0):
            take = min(int(self._disc_pair_batch), int(pool["pairs"]))
            sel = torch.randint(0, int(pool["pairs"]), (take,), device=self._device)
            pair_pos = pool["pos"][sel]
            pair_neg = pool["neg"][sel]
            # The auxiliary forward must not advance spectral-norm power
            # iteration, otherwise an empty/zero ranking term would still
            # perturb the CPL state. eval() freezes it; with no pairs we skip
            # the forward entirely, so the term is a true no-op.
            was_training = self._model.training
            self._model.eval()
            try:
                logits = self._model.eval_disc_raw(
                    torch.cat((pair_pos, pair_neg), dim=0)).squeeze(-1)
            finally:
                self._model.train(was_training)
            pairwise_loss = torch.nn.functional.softplus(
                logits[take:] - logits[:take]).mean()
            pair_count = torch.tensor(float(take), device=current_diff.device)
            disc_loss = disc_loss + pairwise_weight * pairwise_loss

        return {
            "disc_loss": disc_loss,
            # Keep a distinct storage: add_torch_dict accumulates in place, and
            # a shared storage here would double-count disc_loss in the log.
            "disc_cls_loss": disc_loss.detach().clone(),
            "disc_bce_loss": cls_loss.detach().clone(),
            "disc_pairwise_weighted": (pairwise_weight * pairwise_loss).detach().clone(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logits.mean().detach(),
            "disc_neg_logit": neg_logits.mean().detach(),
            "disc_pairwise_loss": pairwise_loss.detach(),
            "disc_pairwise_pairs": pair_count.detach(),
            "disc_rank_pos": pos_count,
            "disc_rank_neg": neg_count,
            "disc_rank_unknown": unknown_count,
            "disc_rank_pool_pairs": pool_pairs,
            "disc_group_width": self._model.get_disc_group_width(),
            "disc_group_total_width": self._model.get_disc_group_total_width(),
        }

    @staticmethod
    def _matched_rank_pairs(pos_ids, neg_ids, cell):
        """Aligned pos/neg indices with equal counts inside every cell."""
        if pos_ids.numel() == 0 or neg_ids.numel() == 0:
            return None, None
        pos_cell = cell[pos_ids]
        neg_cell = cell[neg_ids]
        pos_ids = pos_ids[torch.argsort(pos_cell, stable=True)]
        pos_cell = cell[pos_ids]
        neg_ids = neg_ids[torch.argsort(neg_cell, stable=True)]
        neg_cell = cell[neg_ids]
        uniq = torch.unique(cell)
        pos_slot = torch.searchsorted(uniq, pos_cell)
        neg_slot = torch.searchsorted(uniq, neg_cell)
        pos_counts = torch.bincount(pos_slot, minlength=uniq.numel())
        neg_counts = torch.bincount(neg_slot, minlength=uniq.numel())
        keep = torch.minimum(pos_counts, neg_counts)
        # Global start of each cell inside the already sorted arrays; using the
        # kept counts here would shift ranks whenever a cell keeps nothing.
        pos_start = torch.cumsum(pos_counts, 0) - pos_counts
        neg_start = torch.cumsum(neg_counts, 0) - neg_counts
        pos_rank = torch.arange(pos_cell.numel(), device=pos_cell.device) - pos_start[pos_slot]
        neg_rank = torch.arange(neg_cell.numel(), device=neg_cell.device) - neg_start[neg_slot]
        return pos_ids[pos_rank < keep[pos_slot]], neg_ids[neg_rank < keep[neg_slot]]

    def _calc_disc_rewards(self, norm_diff):
        """Use the bounded ADD reward paired with the BCE discriminator."""
        with torch.no_grad():
            logits = torch_util.eval_minibatch(
                self._model.eval_disc_raw,
                {"disc_obs": norm_diff}, self._disc_eval_batch_size,
            ).squeeze(-1)
            return self._disc_reward_scale * add_agent.calc_unscaled_disc_reward(
                logits)
