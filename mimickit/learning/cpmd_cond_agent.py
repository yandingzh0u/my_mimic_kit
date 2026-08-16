"""ADD with an isolated paired temporal-context veto."""

import torch

import learning.add_agent as add_agent
import learning.cpmd_cond_model as cpmd_cond_model
import learning.diff_normalizer as diff_normalizer
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import util.torch_util as torch_util
from util.logger import Logger


class CPMDConditionalAgent(add_agent.ADDAgent):
    """Train stock ADD and a disjoint paired context critic.

    ADD scores the instantaneous 172-D differential.  The context critic sees
    paired ``(0, c)`` and ``(h, c)`` rows and may only reduce ADD's reward:
    ``r = r_add / (1 + relu(p(0,c) - p(h,c)))``.
    """

    def _load_params(self, config):
        super()._load_params(config)
        self._context_buffer_size = int(
            config.get("context_buffer_size", config["disc_buffer_size"]))
        self._context_replay_samples = int(config.get(
            "context_replay_samples", config["disc_replay_samples"]))
        self._context_grad_penalty = float(config.get(
            "context_grad_penalty", self._disc_grad_penalty))
        self._context_logit_reg = float(config.get(
            "context_logit_reg", self._disc_logit_reg))
        self._context_pair_microbatch_size = int(
            config.get("context_pair_microbatch_size", 4096))
        assert self._context_buffer_size > 0
        assert self._context_replay_samples > 0
        assert self._context_pair_microbatch_size > 0
        return

    def _build_model(self, config):
        self._model = cpmd_cond_model.CPMDConditionalModel(
            config["model"], self._env)
        return

    def _build_optimizer(self, config):
        super()._build_optimizer(config)
        context_config = config.get(
            "context_optimizer", config["disc_optimizer"])
        context_params = [
            p for p in self._model.get_context_params() if p.requires_grad]
        self._context_optimizer = mp_optimizer.MPOptimizer(
            context_config, context_params)

        base_ids = {id(p) for p in self._model.get_disc_params()}
        context_ids = {id(p) for p in context_params}
        assert base_ids.isdisjoint(context_ids)
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._context_optimizer.sync()
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        self._context_buffer = experience_buffer.ExperienceBuffer(
            buffer_length=self._context_buffer_size,
            batch_size=1,
            device=self._device,
        )
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

    @staticmethod
    def _num_replay_rows(buffer, rollout_rows, replay_rows):
        if buffer.is_full():
            return min(rollout_rows, replay_rows)
        return min(rollout_rows, buffer.get_capacity())

    def _store_disc_replay_data(self):
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_error = self._exp_buffer.get_data_flat("cpmd_error_memory")
        ref_context = self._exp_buffer.get_data_flat("cpmd_ref_context")
        assert (disc_diff.shape[0] == hist_error.shape[0]
                == ref_context.shape[0])

        n = disc_diff.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)

        base_n = self._num_replay_rows(
            self._disc_buffer, n, self._disc_replay_samples)
        base_idx = rand_idx[:base_n]
        self._disc_buffer.push({
            "disc_diff": disc_diff[base_idx].unsqueeze(1),
        })

        context_n = self._num_replay_rows(
            self._context_buffer, n, self._context_replay_samples)
        context_idx = rand_idx[:context_n]
        self._context_buffer.push({
            "cpmd_error_memory": hist_error[context_idx].unsqueeze(1),
            "cpmd_ref_context": ref_context[context_idx].unsqueeze(1),
        })
        return

    def _record_context_scales(self, hist_error, ref_context):
        self._hist_error_norm.record(hist_error)
        self._ref_context_norm.record(ref_context)
        return

    def _ensure_context_scales_initialized(self, hist_error, ref_context):
        """Calibrate context coordinates before their first critic update."""
        if int(self._hist_error_norm.get_count().item()) == 0:
            assert int(self._ref_context_norm.get_count().item()) == 0
            self._record_context_scales(hist_error, ref_context)
            self._hist_error_norm.update()
            self._ref_context_norm.update()
            self._context_scales_bootstrapped_this_iter = True
        return

    def _normalize_context_inputs(self, hist_error, ref_context):
        return (
            self._hist_error_norm.normalize(hist_error),
            self._ref_context_norm.normalize(ref_context),
        )

    def _eval_context_logits(self, hist_error, ref_context):
        inputs = {
            "error_memory": hist_error,
            "ref_context": ref_context,
        }
        logits = torch_util.eval_minibatch(
            self._model.eval_context,
            inputs,
            self._disc_eval_batch_size,
        )
        return logits.squeeze(-1)

    @staticmethod
    def _apply_context_veto(add_reward, pos_logit, neg_logit):
        pos_prob = torch.sigmoid(pos_logit)
        neg_prob = torch.sigmoid(neg_logit)
        veto = torch.clamp_min(pos_prob - neg_prob, 0.0)
        ratio = torch.reciprocal(1.0 + veto)
        return add_reward * ratio, veto, ratio

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        hist_error = self._exp_buffer.get_data_flat("cpmd_error_memory")
        ref_context = self._exp_buffer.get_data_flat("cpmd_ref_context")

        self._ensure_context_scales_initialized(hist_error, ref_context)
        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_hist, norm_context = self._normalize_context_inputs(
            hist_error, ref_context)

        with torch.no_grad():
            add_reward = self._calc_disc_rewards(norm_diff)
            pos_logit = self._eval_context_logits(
                torch.zeros_like(norm_hist), norm_context)
            neg_logit = self._eval_context_logits(norm_hist, norm_context)
            disc_reward, veto, ratio = self._apply_context_veto(
                add_reward, pos_logit, neg_logit)

        reward_std, reward_mean = torch.std_mean(disc_reward)
        reward = (self._task_reward_weight * task_r
                  + self._disc_reward_weight * disc_reward)
        self._exp_buffer.set_data_flat("reward", reward)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_diff)
            if not self._context_scales_bootstrapped_this_iter:
                self._record_context_scales(hist_error, ref_context)

        return {
            "disc_reward_mean": reward_mean,
            "disc_reward_std": reward_std,
            "disc_add_reward_mean": torch.mean(add_reward),
            "ctx_veto_mean": torch.mean(veto),
            "ctx_veto_max": torch.max(veto),
            "ctx_veto_active_frac": torch.mean((veto > 0.0).float()),
            "ctx_veto_saturation_frac": torch.mean((veto > 0.95).float()),
            "ctx_reward_ratio_mean": torch.mean(ratio),
            "ctx_reward_ratio_min": torch.min(ratio),
            "ctx_pos_prob_mean": torch.mean(torch.sigmoid(pos_logit)),
            "ctx_neg_prob_mean": torch.mean(torch.sigmoid(neg_logit)),
        }

    def _update_normalizers(self):
        super()._update_normalizers()
        if self._context_scales_bootstrapped_this_iter:
            self._context_scales_bootstrapped_this_iter = False
        else:
            self._hist_error_norm.update()
            self._ref_context_norm.update()
        return

    def _compute_context_chunk(self, hist_error, ref_context):
        neg_hist, context = self._normalize_context_inputs(
            hist_error, ref_context)
        pos_hist = torch.zeros_like(neg_hist)
        pos_hist = pos_hist.detach().requires_grad_(True)
        neg_hist = neg_hist.detach().requires_grad_(True)
        context = context.detach().requires_grad_(True)

        pos_logit = self._model.eval_context(
            pos_hist, context).squeeze(-1)
        neg_logit = self._model.eval_context(
            neg_hist, context).squeeze(-1)
        class_loss = 0.5 * (
            self._disc_loss_pos(pos_logit) + self._disc_loss_neg(neg_logit))

        pos_hist_grad = torch.autograd.grad(
            pos_logit,
            pos_hist,
            grad_outputs=torch.ones_like(pos_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        neg_hist_grad = torch.autograd.grad(
            neg_logit,
            neg_hist,
            grad_outputs=torch.ones_like(neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = 0.5 * (
            torch.mean(torch.sum(torch.square(pos_hist_grad), dim=-1))
            + torch.mean(torch.sum(torch.square(neg_hist_grad), dim=-1)))
        loss = class_loss + self._context_grad_penalty * grad_penalty

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

        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)
        pos_prob = torch.sigmoid(pos_logit.detach())
        neg_prob = torch.sigmoid(neg_logit.detach())
        veto = torch.clamp_min(pos_prob - neg_prob, 0.0)
        return {
            "ctx_loss": loss,
            "ctx_class_loss": class_loss.detach(),
            "ctx_grad_penalty": grad_penalty.detach(),
            "ctx_pos_acc": pos_acc.detach(),
            "ctx_neg_acc": neg_acc.detach(),
            "ctx_pos_logit": torch.mean(pos_logit).detach(),
            "ctx_neg_logit": torch.mean(neg_logit).detach(),
            "ctx_train_veto_mean": torch.mean(veto),
            "ctx_pos_context_grad_rms": torch.sqrt(torch.mean(
                torch.square(pos_context_grad))).detach(),
            "ctx_neg_context_grad_rms": torch.sqrt(torch.mean(
                torch.square(neg_context_grad))).detach(),
            "ctx_hist_error_rms": torch.sqrt(torch.mean(
                torch.square(hist_error))).detach(),
            "ctx_ref_context_rms": torch.sqrt(torch.mean(
                torch.square(ref_context))).detach(),
        }

    def _update_context(self, batch_size, steps):
        total_info = {}
        device_type = torch.device(self._device).type

        for _ in range(steps):
            current = self._exp_buffer.sample(batch_size)
            replay = self._context_buffer.sample(batch_size)
            hist_error = torch.cat([
                current["cpmd_error_memory"],
                replay["cpmd_error_memory"],
            ], dim=0)
            ref_context = torch.cat([
                current["cpmd_ref_context"],
                replay["cpmd_ref_context"],
            ], dim=0)
            sample_count = hist_error.shape[0]

            self._context_optimizer.zero_grad()
            step_info = {}
            for start in range(
                    0, sample_count, self._context_pair_microbatch_size):
                end = min(
                    start + self._context_pair_microbatch_size, sample_count)
                weight = float(end - start) / float(sample_count)
                with torch.amp.autocast(
                        device_type=device_type,
                        enabled=self._use_mixed_precision,
                        dtype=torch.bfloat16):
                    chunk_info = self._compute_context_chunk(
                        hist_error[start:end], ref_context[start:end])
                    chunk_loss = weight * chunk_info["ctx_loss"]
                self._context_optimizer.backward(chunk_loss)

                for key, value in chunk_info.items():
                    if key == "ctx_loss":
                        continue
                    weighted = weight * value.detach()
                    step_info[key] = step_info.get(key, 0.0) + weighted

            logit_weights = self._model.get_context_logit_weights()
            logit_loss = torch.sum(torch.square(logit_weights))
            logit_reg_loss = self._context_logit_reg * logit_loss
            self._context_optimizer.backward(logit_reg_loss)

            grad_sq = torch.zeros([], device=self._device)
            for param in self._model.get_context_params():
                if param.grad is not None:
                    grad_sq = grad_sq + torch.sum(torch.square(param.grad))
            step_info["ctx_grad_norm"] = torch.sqrt(grad_sq).detach()
            step_info["ctx_weight_norm"] = torch.linalg.vector_norm(
                torch.nn.utils.parameters_to_vector(
                    self._model.get_context_params())).detach()
            step_info["ctx_logit_loss"] = logit_loss.detach()
            step_info["ctx_loss"] = (
                step_info["ctx_class_loss"]
                + self._context_grad_penalty * step_info["ctx_grad_penalty"]
                + logit_reg_loss.detach())

            self._context_optimizer.apply_step()
            torch_util.add_torch_dict(step_info, total_info)

        torch_util.scale_torch_dict(1.0 / steps, total_info)
        return total_info

    def _update_disc(self, batch_size, steps):
        # These updates are sequential but their parameter sets and losses are
        # disjoint.  The ADD branch uses the inherited stock implementation.
        base_info = super()._update_disc(batch_size, steps)
        context_info = self._update_context(batch_size, steps)
        return {**base_info, **context_info}

    def _compute_disc_input_diagnostics(self, norm_diff_obs, disc_neg_grad,
                                        disc_neg_logit):
        """Attribute the stock ADD input gradient to semantic state groups."""
        del norm_diff_obs, disc_neg_logit
        with torch.no_grad():
            grad_sq = torch.square(disc_neg_grad.detach())
            total = torch.clamp_min(torch.sum(grad_sq), 1e-12)
            group_dims = self._env.get_disc_feature_group_dims()
            group_energy = {
                name: torch.zeros([], device=grad_sq.device)
                for name in group_dims
            }

            offset = 0
            for _ in range(self._env.get_num_disc_obs_steps()):
                for name, dim in group_dims.items():
                    group_energy[name] += torch.sum(
                        grad_sq[..., offset:offset + dim])
                    offset += dim
            assert offset == grad_sq.shape[-1]

            return {
                "disc_grad_{}_frac".format(name): energy / total
                for name, energy in group_energy.items()
            }

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        Logger.print(
            "CPMD veto: isolated ADD {}D + paired context critic {}D".format(
                env.get_disc_state_obs_dim(),
                self._model.get_context_input_dim(),
            ))
        return
