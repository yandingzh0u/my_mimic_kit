"""ADD with a separately optimized bilinear motion-context correction."""

import torch

import learning.add_agent as add_agent
import learning.cpmd_residual_model as cpmd_residual_model
import learning.diff_normalizer as diff_normalizer
import learning.mp_optimizer as mp_optimizer
import util.torch_util as torch_util
from util.logger import Logger


class CPMDResidualAgent(add_agent.ADDAgent):
    """Stock ADD base plus ``u^T h + 1/4 h^T A s``.

    The ADD discriminator and its optimizer remain parameter-disjoint from the
    contextual correction. A single scale-only normalizer is shared by the
    difference memory ``h`` and common-motion memory ``s`` so the symmetric
    interaction keeps its intended coordinate geometry.
    """

    def _build_model(self, config):
        self._model = cpmd_residual_model.CPMDResidualModel(
            config["model"], self._env)
        return

    def _build_optimizer(self, config):
        super()._build_optimizer(config)

        ctx_params = [p for p in self._model.get_context_params()
                      if p.requires_grad]
        base_params = [p for p in self._model.get_disc_params()
                       if p.requires_grad]
        assert len(ctx_params) > 0
        assert set(map(id, ctx_params)).isdisjoint(set(map(id, base_params)))

        self._ctx_optimizer = mp_optimizer.MPOptimizer(
            config["context_optimizer"], ctx_params)
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._ctx_optimizer.sync()
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        motion_dim = self._env.get_cpmd_motion_dim()
        self._motion_norm = diff_normalizer.DiffNormalizer(
            [motion_dim], device=self._device)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record(
            "cpmd_delta_motion", next_info["cpmd_delta_motion"])
        self._exp_buffer.record(
            "cpmd_sum_motion", next_info["cpmd_sum_motion"])
        return

    def _store_disc_replay_data(self):
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        delta_motion = self._exp_buffer.get_data_flat("cpmd_delta_motion")
        sum_motion = self._exp_buffer.get_data_flat("cpmd_sum_motion")
        assert disc_diff.shape[0] == delta_motion.shape[0] == sum_motion.shape[0]

        n = disc_diff.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if self._disc_buffer.is_full():
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = min(n, self._disc_buffer.get_capacity())
        idx = rand_idx[:num_samples]

        replay = {
            "disc_diff": disc_diff[idx].unsqueeze(1),
            "cpmd_delta_motion": delta_motion[idx].unsqueeze(1),
            "cpmd_sum_motion": sum_motion[idx].unsqueeze(1),
        }
        self._disc_buffer.push(replay)
        return

    @staticmethod
    def _recover_side_summaries(delta_motion, sum_motion):
        motion_ref = 0.5 * (sum_motion + delta_motion)
        motion_sim = 0.5 * (sum_motion - delta_motion)
        return motion_ref, motion_sim

    def _record_motion_scale(self, delta_motion, sum_motion):
        motion_ref, motion_sim = self._recover_side_summaries(
            delta_motion, sum_motion)
        self._motion_norm.record(torch.cat([motion_ref, motion_sim], dim=0))
        return

    def _ensure_motion_scale_initialized(self, delta_motion, sum_motion):
        """Calibrate before the first unbounded context update.

        MimicKit normally updates normalizers after the first model update.
        Bilinear products are more scale-sensitive, so their shared scale is
        initialized from the first complete rollout before reward/loss use.
        """
        if int(self._motion_norm.get_count().item()) == 0:
            self._record_motion_scale(delta_motion, sum_motion)
            self._motion_norm.update()
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        delta_motion = self._exp_buffer.get_data_flat("cpmd_delta_motion")
        sum_motion = self._exp_buffer.get_data_flat("cpmd_sum_motion")

        self._ensure_motion_scale_initialized(delta_motion, sum_motion)
        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_delta = self._motion_norm.normalize(delta_motion)
        norm_sum = self._motion_norm.normalize(sum_motion)

        with torch.no_grad():
            base_logits = self._model.eval_disc(norm_diff).squeeze(-1)
            correction = self._model.eval_context_residual(
                norm_delta, norm_sum)
            final_logits = base_logits + correction
            disc_r = self._reward_from_logits(final_logits)
            base_r = self._reward_from_logits(base_logits)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_diff)
            self._record_motion_scale(delta_motion, sum_motion)

        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "disc_base_reward_mean": torch.mean(base_r),
            "ctx_reward_delta_mean": torch.mean(disc_r - base_r),
        }

    def _reward_from_logits(self, logits):
        prob = torch.sigmoid(logits)
        complement = torch.clamp_min(1.0 - prob, 1e-4)
        return -self._disc_reward_scale * torch.log(complement)

    def _update_normalizers(self):
        super()._update_normalizers()
        self._motion_norm.update()
        return

    def _compute_disc_loss(self, batch):
        # Positive ADD anchor.
        pos_diff = self._pos_diff.clone().unsqueeze(0)
        pos_diff.requires_grad_(True)
        base_pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)

        # Current/replay rows remain aligned across all three fields.
        diff_obs = batch["disc_diff"]
        delta_motion = batch["cpmd_delta_motion"]
        sum_motion = batch["cpmd_sum_motion"]
        replay = self._disc_buffer.sample(diff_obs.shape[0])

        diff_obs = torch.cat([diff_obs, replay["disc_diff"]], dim=0)
        delta_motion = torch.cat([
            delta_motion, replay["cpmd_delta_motion"]], dim=0)
        sum_motion = torch.cat([
            sum_motion, replay["cpmd_sum_motion"]], dim=0)

        norm_diff = self._disc_obs_norm.normalize(diff_obs)
        norm_diff.requires_grad_(True)
        norm_delta = self._motion_norm.normalize(delta_motion)
        norm_sum = self._motion_norm.normalize(sum_motion)

        base_neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)

        # Stock ADD base loss, final-head regularizer and two-endpoint logit GP.
        base_loss_pos = self._disc_loss_pos(base_pos_logit)
        base_loss_neg = self._disc_loss_neg(base_neg_logit)
        base_loss = 0.5 * (base_loss_pos + base_loss_neg)

        base_logit_weights = self._model.get_disc_logit_weights()
        base_logit_loss = torch.sum(torch.square(base_logit_weights))
        base_loss = base_loss + self._disc_logit_reg * base_logit_loss

        neg_grad = torch.autograd.grad(
            base_neg_logit, norm_diff,
            grad_outputs=torch.ones_like(base_neg_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        pos_grad = torch.autograd.grad(
            base_pos_logit, pos_diff,
            grad_outputs=torch.ones_like(base_pos_logit),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = 0.5 * (
            torch.mean(torch.sum(torch.square(neg_grad), dim=-1))
            + torch.mean(torch.sum(torch.square(pos_grad), dim=-1)))
        base_loss = base_loss + self._disc_grad_penalty * grad_penalty

        correction, linear, bilinear = self._model.eval_context(
            norm_delta, norm_sum)
        detached_base = base_neg_logit.detach()
        final_neg_logit = detached_base + correction

        # This is the negative half of a zero-vs-policy BCE. The positive
        # context feature is exactly zero and contributes no context gradient.
        ctx_class_loss = self._disc_loss_neg(final_neg_logit)
        ctx_weights = self._model.get_context_logit_weights()
        ctx_logit_loss = torch.sum(torch.square(ctx_weights))
        ctx_loss = (0.5 * ctx_class_loss
                    + self._disc_logit_reg * ctx_logit_loss)

        context_grads = torch.autograd.grad(
            ctx_loss, self._model.get_context_params(), retain_graph=True)
        ctx_grad_norm = torch.sqrt(torch.sum(torch.stack([
            torch.sum(torch.square(g)) for g in context_grads
        ])))

        neg_acc, pos_acc = self._compute_disc_acc(
            base_neg_logit, base_pos_logit)
        final_neg_acc = torch.mean((final_neg_logit < 0.0).float())
        abs_correction = torch.abs(correction)
        motion_scale = self._motion_norm.get_abs_mean()

        return {
            "disc_loss": base_loss,
            "ctx_loss": ctx_loss,
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_logit_loss": base_logit_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": torch.mean(base_pos_logit).detach(),
            "disc_neg_logit": torch.mean(base_neg_logit).detach(),
            "ctx_class_loss": ctx_class_loss.detach(),
            "ctx_logit_loss": ctx_logit_loss.detach(),
            "ctx_correction_rms": torch.sqrt(
                torch.mean(torch.square(correction))).detach(),
            "ctx_correction_abs_p95": torch.quantile(
                abs_correction, 0.95).detach(),
            "ctx_correction_abs_max": torch.max(abs_correction).detach(),
            "ctx_linear_rms": torch.sqrt(
                torch.mean(torch.square(linear))).detach(),
            "ctx_bilinear_rms": torch.sqrt(
                torch.mean(torch.square(bilinear))).detach(),
            "ctx_final_neg_logit": torch.mean(final_neg_logit).detach(),
            "ctx_final_neg_acc": final_neg_acc.detach(),
            "ctx_delta_rms": torch.sqrt(
                torch.mean(torch.square(delta_motion))).detach(),
            "ctx_delta_rms_norm": torch.sqrt(
                torch.mean(torch.square(norm_delta))).detach(),
            "ctx_sum_rms": torch.sqrt(
                torch.mean(torch.square(sum_motion))).detach(),
            "ctx_sum_rms_norm": torch.sqrt(
                torch.mean(torch.square(norm_sum))).detach(),
            "ctx_linear_weight_norm": (
                self._model.get_context_linear_norm().detach()),
            "ctx_bilinear_weight_norm": (
                self._model.get_context_bilinear_norm().detach()),
            "ctx_grad_norm": ctx_grad_norm.detach(),
            "ctx_motion_scale_mean": torch.mean(motion_scale).detach(),
            "ctx_motion_scale_min": torch.min(motion_scale).detach(),
        }

    def _update_disc(self, batch_size, steps):
        info = {}
        device_type = torch.device(self._device).type

        for _ in range(steps):
            batch = self._exp_buffer.sample(batch_size)
            with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self._use_mixed_precision,
                    dtype=torch.bfloat16):
                loss_info = self._compute_disc_loss(batch)

            self._disc_optimizer.step(loss_info["disc_loss"])
            self._ctx_optimizer.step(loss_info["ctx_loss"])
            torch_util.add_torch_dict(loss_info, info)

        torch_util.scale_torch_dict(1.0 / steps, info)
        return info

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        Logger.print(
            "Bilinear CPMD: ADD {}D + h {}D + {} symmetric pairs "
            "(shared motion scale, zero initialized)".format(
                env.get_disc_obs_space().shape[0],
                env.get_cpmd_motion_dim(),
                self._model.get_context_num_pairs()))
        return
