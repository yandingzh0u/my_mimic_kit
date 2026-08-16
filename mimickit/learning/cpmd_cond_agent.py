"""Context-paired adversarial differential agent."""

import torch

import learning.add_agent as add_agent
import learning.cpmd_cond_model as cpmd_cond_model
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util
from util.logger import Logger


class CPMDConditionalAgent(add_agent.ADDAgent):
    """Train one discriminator on paired ``(0, c)`` and ``(e, c)`` rows."""

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_pair_microbatch_size = int(
            config.get("disc_pair_microbatch_size", 4096))
        assert self._disc_pair_microbatch_size > 0
        return

    def _build_model(self, config):
        self._model = cpmd_cond_model.CPMDConditionalModel(
            config["model"], self._env)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        self._hist_error_norm = diff_normalizer.DiffNormalizer(
            [self._env.get_cpmd_error_dim()], device=self._device)
        self._ref_context_norm = diff_normalizer.DiffNormalizer(
            [self._env.get_cpmd_context_dim()], device=self._device)
        self._context_scales_bootstrapped_this_iter = False
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record(
            "cpmd_error_memory", next_info["cpmd_error_memory"])
        self._exp_buffer.record(
            "cpmd_ref_context", next_info["cpmd_ref_context"])
        return

    def _store_disc_replay_data(self):
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_error = self._exp_buffer.get_data_flat("cpmd_error_memory")
        ref_context = self._exp_buffer.get_data_flat("cpmd_ref_context")
        assert (disc_diff.shape[0] == hist_error.shape[0]
                == ref_context.shape[0])

        n = disc_diff.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if self._disc_buffer.is_full():
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = min(n, self._disc_buffer.get_capacity())
        idx = rand_idx[:num_samples]

        # One index vector preserves the exact differential/history/context
        # pairing in replay.
        replay = {
            "disc_diff": disc_diff[idx].unsqueeze(1),
            "cpmd_error_memory": hist_error[idx].unsqueeze(1),
            "cpmd_ref_context": ref_context[idx].unsqueeze(1),
        }
        self._disc_buffer.push(replay)
        return

    def _record_context_scales(self, hist_error, ref_context):
        self._hist_error_norm.record(hist_error)
        self._ref_context_norm.record(ref_context)
        return

    def _ensure_context_scales_initialized(self, hist_error, ref_context):
        """Calibrate added coordinates before their first discriminator use."""
        if int(self._hist_error_norm.get_count().item()) == 0:
            assert int(self._ref_context_norm.get_count().item()) == 0
            self._record_context_scales(hist_error, ref_context)
            self._hist_error_norm.update()
            self._ref_context_norm.update()
            self._context_scales_bootstrapped_this_iter = True
        return

    def _normalize_conditional_inputs(self, disc_diff, hist_error,
                                      ref_context):
        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_hist = self._hist_error_norm.normalize(hist_error)
        norm_context = self._ref_context_norm.normalize(ref_context)
        error_obs = torch.cat([norm_diff, norm_hist], dim=-1)
        return error_obs, norm_context

    def _eval_cond_logits(self, error_obs, ref_context):
        inputs = {"error_obs": error_obs, "ref_context": ref_context}
        logits = torch_util.eval_minibatch(
            self._model.eval_cond,
            inputs,
            self._disc_eval_batch_size,
        )
        return logits.squeeze(-1)

    def _reward_from_logits(self, logits):
        prob = torch.sigmoid(logits)
        complement = torch.clamp_min(1.0 - prob, 1e-4)
        return -self._disc_reward_scale * torch.log(complement)

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_error = self._exp_buffer.get_data_flat("cpmd_error_memory")
        ref_context = self._exp_buffer.get_data_flat("cpmd_ref_context")

        self._ensure_context_scales_initialized(hist_error, ref_context)
        error_obs, norm_context = self._normalize_conditional_inputs(
            disc_diff, hist_error, ref_context)

        with torch.no_grad():
            final_logits = self._eval_cond_logits(error_obs, norm_context)
            disc_r = self._reward_from_logits(final_logits)

            # Diagnostic only: the exact embedded ADD specialization.
            base_error = error_obs.clone()
            base_error[..., self._model.get_disc_state_dim():] = 0.0
            base_context = torch.zeros_like(norm_context)
            base_logits = self._eval_cond_logits(base_error, base_context)
            base_r = self._reward_from_logits(base_logits)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_diff)
            if not self._context_scales_bootstrapped_this_iter:
                self._record_context_scales(hist_error, ref_context)

        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "disc_add_reward_mean": torch.mean(base_r),
            "disc_cond_reward_delta": torch.mean(disc_r - base_r),
        }

    def _update_normalizers(self):
        super()._update_normalizers()
        if self._context_scales_bootstrapped_this_iter:
            # The first rollout was already committed before its first use.
            self._context_scales_bootstrapped_this_iter = False
        else:
            self._hist_error_norm.update()
            self._ref_context_norm.update()
        return

    def _build_paired_rows(self, disc_diff, hist_error, ref_context):
        neg_error, context = self._normalize_conditional_inputs(
            disc_diff, hist_error, ref_context)
        pos_error = torch.zeros_like(neg_error)
        # The exact same context tensor is returned for both labels.
        return pos_error, neg_error, context

    def _compute_paired_chunk(self, disc_diff, hist_error, ref_context):
        pos_error, neg_error, context = self._build_paired_rows(
            disc_diff, hist_error, ref_context)
        pos_error = pos_error.detach().requires_grad_(True)
        neg_error = neg_error.detach().requires_grad_(True)
        context = context.detach().requires_grad_(True)

        pos_logit = self._model.eval_cond(pos_error, context).squeeze(-1)
        neg_logit = self._model.eval_cond(neg_error, context).squeeze(-1)

        class_loss = 0.5 * (
            self._disc_loss_pos(pos_logit) + self._disc_loss_neg(neg_logit))

        # Smooth only the tracking-error coordinates. Context is a condition,
        # not a label-bearing input; its sensitivity is measured separately.
        pos_error_grad = torch.autograd.grad(
            pos_logit,
            pos_error,
            grad_outputs=torch.ones_like(pos_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        neg_error_grad = torch.autograd.grad(
            neg_logit,
            neg_error,
            grad_outputs=torch.ones_like(neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = 0.5 * (
            torch.mean(torch.sum(torch.square(pos_error_grad), dim=-1))
            + torch.mean(torch.sum(torch.square(neg_error_grad), dim=-1)))
        loss = class_loss + self._disc_grad_penalty * grad_penalty

        pos_context_grad = torch.autograd.grad(
            pos_logit,
            context,
            grad_outputs=torch.ones_like(pos_logit),
            retain_graph=True,
            only_inputs=True,
        )[0]
        neg_context_grad = torch.autograd.grad(
            neg_logit,
            context,
            grad_outputs=torch.ones_like(neg_logit),
            retain_graph=True,
            only_inputs=True,
        )[0]

        state_dim = self._model.get_disc_state_dim()
        neg_state_grad_sq = torch.sum(
            torch.square(neg_error_grad[..., :state_dim]), dim=-1)
        neg_hist_grad_sq = torch.sum(
            torch.square(neg_error_grad[..., state_dim:]), dim=-1)
        neg_grad_total = torch.clamp_min(
            neg_state_grad_sq + neg_hist_grad_sq, 1e-12)
        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)

        info = {
            "disc_loss": loss,
            "disc_class_loss": class_loss.detach(),
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": torch.mean(pos_logit).detach(),
            "disc_pos_logit_std": torch.std(
                pos_logit, unbiased=False).detach(),
            "disc_neg_logit": torch.mean(neg_logit).detach(),
            "disc_error_state_grad_share": torch.mean(
                neg_state_grad_sq / neg_grad_total).detach(),
            "disc_error_hist_grad_share": torch.mean(
                neg_hist_grad_sq / neg_grad_total).detach(),
            "disc_pos_context_grad_rms": torch.sqrt(torch.mean(
                torch.square(pos_context_grad))).detach(),
            "disc_neg_context_grad_rms": torch.sqrt(torch.mean(
                torch.square(neg_context_grad))).detach(),
            "disc_hist_error_rms": torch.sqrt(torch.mean(
                torch.square(hist_error))).detach(),
            "disc_ref_context_rms": torch.sqrt(torch.mean(
                torch.square(ref_context))).detach(),
        }
        return info

    def _update_disc(self, batch_size, steps):
        total_info = {}
        device_type = torch.device(self._device).type

        for _ in range(steps):
            current = self._exp_buffer.sample(batch_size)
            replay = self._disc_buffer.sample(batch_size)
            disc_diff = torch.cat(
                [current["disc_diff"], replay["disc_diff"]], dim=0)
            hist_error = torch.cat([
                current["cpmd_error_memory"],
                replay["cpmd_error_memory"],
            ], dim=0)
            ref_context = torch.cat([
                current["cpmd_ref_context"],
                replay["cpmd_ref_context"],
            ], dim=0)
            sample_count = disc_diff.shape[0]

            self._disc_optimizer.zero_grad()
            step_info = {}
            for start in range(0, sample_count,
                               self._disc_pair_microbatch_size):
                end = min(start + self._disc_pair_microbatch_size,
                          sample_count)
                weight = float(end - start) / float(sample_count)
                with torch.amp.autocast(
                        device_type=device_type,
                        enabled=self._use_mixed_precision,
                        dtype=torch.bfloat16):
                    chunk_info = self._compute_paired_chunk(
                        disc_diff[start:end],
                        hist_error[start:end],
                        ref_context[start:end],
                    )
                    chunk_loss = weight * chunk_info["disc_loss"]
                self._disc_optimizer.backward(chunk_loss)

                for key, value in chunk_info.items():
                    if key == "disc_loss":
                        continue
                    weighted = weight * value.detach()
                    step_info[key] = step_info.get(key, 0.0) + weighted

            logit_weights = self._model.get_disc_logit_weights()
            logit_loss = torch.sum(torch.square(logit_weights))
            logit_reg_loss = self._disc_logit_reg * logit_loss
            self._disc_optimizer.backward(logit_reg_loss)

            first_grad = self._model._disc_layers[0].weight.grad
            assert first_grad is not None
            state_dim = self._model.get_disc_state_dim()
            error_dim = self._model.get_error_dim()
            step_info["disc_hist_column_grad_norm"] = torch.linalg.vector_norm(
                first_grad[:, state_dim:error_dim]).detach()
            step_info["disc_context_column_grad_norm"] = (
                torch.linalg.vector_norm(
                    first_grad[:, error_dim:]).detach())
            step_info["disc_added_weight_norm"] = torch.linalg.vector_norm(
                self._model.get_added_input_weights()).detach()
            step_info["disc_logit_loss"] = logit_loss.detach()
            step_info["disc_loss"] = (
                step_info["disc_class_loss"]
                + self._disc_grad_penalty * step_info["disc_grad_penalty"]
                + logit_reg_loss.detach())

            self._disc_optimizer.apply_step()
            torch_util.add_torch_dict(step_info, total_info)

        torch_util.scale_torch_dict(1.0 / steps, total_info)
        return total_info

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        Logger.print(
            "Conditional CPMD: paired (0,c)/(e,c), error {}D, context {}D, "
            "single {}D discriminator".format(
                self._model.get_error_dim(),
                self._model.get_context_dim(),
                self._model.get_conditional_dim(),
            ))
        return
