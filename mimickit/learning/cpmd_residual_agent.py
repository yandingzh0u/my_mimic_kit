import numpy as np
import torch

import learning.add_agent as add_agent
import learning.cpmd_residual_model as cpmd_residual_model
import learning.diff_normalizer as diff_normalizer
import learning.mp_optimizer as mp_optimizer
import util.torch_util as torch_util
from util.logger import Logger


class CPMDResidualAgent(add_agent.ADDAgent):
    """ADD with a small, separately optimized contextual logit residual.

    The native ADD discriminator remains a 172-D zero-vs-policy classifier.
    The contextual path sees only a 34-D discounted motion mismatch and the
    aligned 34-D reference motion.  Its parameters never receive gradients
    from the ADD loss, and the ADD parameters never receive gradients from the
    contextual loss.

    A bounded residual and an ADD-only warm-up are deliberate safeguards.  At
    initialization (and whenever the history mismatch is zero), the final
    logit is exactly the ADD logit.
    """

    def _load_params(self, config):
        super()._load_params(config)
        self._ctx_warmup_iters = int(config.get("cpmd_residual_warmup_iters", 100))
        self._ctx_output_reg = float(config.get("cpmd_residual_output_reg", 0.01))
        assert self._ctx_warmup_iters >= 0
        assert self._ctx_output_reg >= 0.0
        return

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

        hist_dim = self._env.get_cpmd_history_dim()
        ref_dim = self._env.get_cpmd_ref_motion_dim()
        self._hist_err_norm = diff_normalizer.DiffNormalizer(
            [hist_dim], device=self._device)
        self._ref_motion_norm = diff_normalizer.DiffNormalizer(
            [ref_dim], device=self._device)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("cpmd_hist_err", next_info["cpmd_hist_err"])
        self._exp_buffer.record("cpmd_ref_motion", next_info["cpmd_ref_motion"])
        return

    def _store_disc_replay_data(self):
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_err = self._exp_buffer.get_data_flat("cpmd_hist_err")
        ref_motion = self._exp_buffer.get_data_flat("cpmd_ref_motion")
        assert disc_diff.shape[0] == hist_err.shape[0] == ref_motion.shape[0]

        n = disc_diff.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if self._disc_buffer.is_full():
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = min(n, self._disc_buffer.get_capacity())
        idx = rand_idx[:num_samples]

        replay = {
            "disc_diff": disc_diff[idx].unsqueeze(1),
            "cpmd_hist_err": hist_err[idx].unsqueeze(1),
            "cpmd_ref_motion": ref_motion[idx].unsqueeze(1),
        }
        self._disc_buffer.push(replay)
        return

    def _context_enabled(self):
        return self._iter >= self._ctx_warmup_iters

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_err = self._exp_buffer.get_data_flat("cpmd_hist_err")
        ref_motion = self._exp_buffer.get_data_flat("cpmd_ref_motion")

        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_hist = self._hist_err_norm.normalize(hist_err)
        norm_ref = self._ref_motion_norm.normalize(ref_motion)

        with torch.no_grad():
            base_logits = self._model.eval_disc(norm_diff).squeeze(-1)
            residual, _ = self._model.eval_context(norm_hist, norm_ref)
            residual = residual.squeeze(-1)
            gate = torch.sigmoid(base_logits)
            enabled = float(self._context_enabled())
            correction = enabled * gate * residual
            final_logits = base_logits + correction

            disc_r = self._reward_from_logits(final_logits)
            base_r = self._reward_from_logits(base_logits)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_diff)
            self._hist_err_norm.record(hist_err)
            self._ref_motion_norm.record(ref_motion)

        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "disc_base_reward_mean": torch.mean(base_r),
            "ctx_reward_delta_mean": torch.mean(disc_r - base_r),
            "ctx_enabled": torch.tensor(enabled, device=self._device),
        }

    def _reward_from_logits(self, logits):
        # Equivalent to the repository ADD reward, including its numerical cap.
        prob = torch.sigmoid(logits)
        complement = torch.clamp_min(1.0 - prob, 1e-4)
        return -self._disc_reward_scale * torch.log(complement)

    def _update_normalizers(self):
        super()._update_normalizers()
        self._hist_err_norm.update()
        self._ref_motion_norm.update()
        return

    def _compute_disc_loss(self, batch):
        # Positive ADD anchor.
        pos_diff = self._pos_diff.clone().unsqueeze(0)
        pos_diff.requires_grad_(True)
        base_pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)

        # Current and replay negatives remain aligned across all three fields.
        diff_obs = batch["disc_diff"]
        hist_err = batch["cpmd_hist_err"]
        ref_motion = batch["cpmd_ref_motion"]
        replay = self._disc_buffer.sample(diff_obs.shape[0])

        diff_obs = torch.cat([diff_obs, replay["disc_diff"]], dim=0)
        hist_err = torch.cat([hist_err, replay["cpmd_hist_err"]], dim=0)
        ref_motion = torch.cat([ref_motion, replay["cpmd_ref_motion"]], dim=0)

        norm_diff = self._disc_obs_norm.normalize(diff_obs)
        norm_diff.requires_grad_(True)
        norm_hist = self._hist_err_norm.normalize(hist_err)
        norm_ref = self._ref_motion_norm.normalize(ref_motion)

        base_neg_logit = self._model.eval_disc(norm_diff).squeeze(-1)

        # Stock ADD base loss, regularizer and two-endpoint logit GP.
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

        # Context loss cannot alter the ADD discriminator.  Its only effective
        # supervision is on policy negatives; h=0 makes the positive residual
        # identically zero and therefore supplies no context gradient.
        residual, raw_residual = self._model.eval_context(norm_hist, norm_ref)
        residual = residual.squeeze(-1)
        raw_residual = raw_residual.squeeze(-1)
        detached_base = base_neg_logit.detach()
        gate = torch.sigmoid(detached_base).detach()
        correction = gate * residual
        final_neg_logit = detached_base + correction

        ctx_class_loss = self._disc_loss_neg(final_neg_logit)
        ctx_output_penalty = torch.mean(torch.square(correction))
        ctx_logit_weights = self._model.get_context_logit_weights()
        ctx_logit_loss = torch.sum(torch.square(ctx_logit_weights))
        ctx_loss = (ctx_class_loss
                    + self._ctx_output_reg * ctx_output_penalty
                    + self._disc_logit_reg * ctx_logit_loss)

        neg_acc, pos_acc = self._compute_disc_acc(
            base_neg_logit, base_pos_logit)

        bound = self._model.get_context_residual_bound()
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
            "ctx_output_penalty": ctx_output_penalty.detach(),
            "ctx_logit_loss": ctx_logit_loss.detach(),
            "ctx_raw_rms": torch.sqrt(torch.mean(torch.square(raw_residual))).detach(),
            "ctx_residual_rms": torch.sqrt(torch.mean(torch.square(residual))).detach(),
            "ctx_correction_rms": torch.sqrt(torch.mean(torch.square(correction))).detach(),
            "ctx_correction_abs_max": torch.max(torch.abs(correction)).detach(),
            "ctx_saturated_frac": torch.mean((torch.abs(residual) > 0.95 * bound).float()).detach(),
            "ctx_gate_mean": torch.mean(gate).detach(),
            "ctx_gate_p95": torch.quantile(gate, 0.95).detach(),
            "ctx_gate_active_frac": torch.mean((gate > 0.5).float()).detach(),
            "ctx_final_neg_logit": torch.mean(final_neg_logit).detach(),
            "ctx_hist_rms": torch.sqrt(torch.mean(torch.square(hist_err))).detach(),
            "ctx_hist_rms_norm": torch.sqrt(torch.mean(torch.square(norm_hist))).detach(),
            "ctx_ref_rms": torch.sqrt(torch.mean(torch.square(ref_motion))).detach(),
            "ctx_ref_rms_norm": torch.sqrt(torch.mean(torch.square(norm_ref))).detach(),
            "ctx_weight_norm": self._model.get_context_weight_norm().detach(),
        }

    def _update_disc(self, batch_size, steps):
        info = {}
        device_type = torch.device(self._device).type
        context_enabled = self._context_enabled()

        for _ in range(steps):
            batch = self._exp_buffer.sample(batch_size)
            with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self._use_mixed_precision,
                    dtype=torch.bfloat16):
                loss_info = self._compute_disc_loss(batch)

            self._disc_optimizer.step(loss_info["disc_loss"])
            if context_enabled:
                self._ctx_optimizer.step(loss_info["ctx_loss"])

            torch_util.add_torch_dict(loss_info, info)

        torch_util.scale_torch_dict(1.0 / steps, info)
        return info

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        Logger.print(
            "CPMD residual: ADD {}D + rank-{} bounded context "
            "(warmup {} iters, |residual| <= {:.3f})".format(
                env.get_disc_obs_space().shape[0],
                self._model.get_context_rank(),
                self._ctx_warmup_iters,
                self._model.get_context_residual_bound()))
        return
