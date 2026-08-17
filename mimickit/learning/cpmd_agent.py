import torch

import learning.add_agent as add_agent
import learning.cpmd_model as cpmd_model
import learning.normalizer as normalizer
import util.torch_util as torch_util
from util.logger import Logger


class CPMDAgent(add_agent.ADDAgent):
    """ADD with a fixed-budget, reference-conditioned error metric.

    The policy and critic remain reference-blind. The reward model receives
    the normalized ADD differential and an intrinsic synchronized reference
    context. Context can only reallocate a trace-one PSD metric; it cannot add
    a logit by itself and every context shares the same zero anchor.
    """

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._disc_dim = env.get_disc_state_obs_dim()
        self._context_dim = env.get_cpmd_context_dim()
        assert self._disc_dim == env.get_disc_obs_space().shape[0]

        Logger.print(
            "CPMD fixed-budget metric: differential {} + intrinsic context {}, "
            "rank {}, base {:.6f}, budget {:.6f}".format(
                self._disc_dim,
                self._context_dim,
                self._model.get_metric_rank(),
                self._model.get_metric_base_weight(),
                self._model.get_metric_context_budget(),
            )
        )
        return

    def _build_model(self, config):
        self._model = cpmd_model.CPMDModel(config["model"], self._env)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        context_space = self._env.get_cpmd_context_space()
        context_dtype = torch_util.numpy_dtype_to_torch(context_space.dtype)
        self._context_norm = normalizer.Normalizer(
            context_space.shape,
            clip=10.0,
            device=self._device,
            dtype=context_dtype,
        )
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("cpmd_context", next_info["cpmd_context"])
        return

    def _store_disc_replay_data(self):
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        context = self._exp_buffer.get_data_flat("cpmd_context")
        assert disc_diff.shape[0] == context.shape[0]

        n = disc_diff.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if self._disc_buffer.is_full():
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = min(n, self._disc_buffer.get_capacity())

        idx = rand_idx[:num_samples]
        self._disc_buffer.push({
            "disc_diff": disc_diff[idx].unsqueeze(1),
            "cpmd_context": context[idx].unsqueeze(1),
        })
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_diff = self._exp_buffer.get_data_flat("disc_diff")
        context = self._exp_buffer.get_data_flat("cpmd_context")

        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_context = self._context_norm.normalize(context)

        with torch.no_grad():
            inputs = {"disc_obs": norm_diff, "context": norm_context}
            logits = torch_util.eval_minibatch(
                self._model.eval_disc, inputs, self._disc_eval_batch_size
            ).squeeze(-1)
            prob = torch.sigmoid(logits)
            disc_r = -torch.log(torch.clamp_min(1.0 - prob, 1.0e-4))
            disc_r *= self._disc_reward_scale
            if not torch.isfinite(logits).all() or not torch.isfinite(disc_r).all():
                raise FloatingPointError(
                    "Non-finite CPMD metric logit or discriminator reward")

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_diff)
            self._context_norm.record(context)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "disc_reward_logit_slope_mean": torch.mean(
                self._disc_reward_scale * torch.sigmoid(logits)
            ),
        }

    def _update_normalizers(self):
        super()._update_normalizers()
        self._context_norm.update()
        return

    def _compute_disc_loss(self, batch):
        disc_diff = batch["disc_diff"]
        context = batch["cpmd_context"]

        replay = self._disc_buffer.sample(disc_diff.shape[0])
        disc_diff = torch.cat([disc_diff, replay["disc_diff"]], dim=0)
        context = torch.cat([context, replay["cpmd_context"]], dim=0)

        norm_diff = self._disc_obs_norm.normalize(disc_diff)
        norm_diff.requires_grad_(True)
        norm_context = self._context_norm.normalize(context).detach()

        terms = self._model.eval_metric_terms(norm_diff, norm_context)
        neg_logit = terms["logit"].squeeze(-1)
        pos_logit = self._model.eval_zero_logit(
            batch_size=1, device=norm_diff.device, dtype=norm_diff.dtype
        ).squeeze(-1)

        disc_loss_pos = self._disc_loss_pos(pos_logit)
        disc_loss_neg = self._disc_loss_neg(neg_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # The metric has a fixed trace budget; only the shared zero-anchor
        # logit remains as an unconstrained output scalar.
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # For a centered quadratic energy, grad z / grad delta is exactly zero
        # at the positive anchor. Smoothness is therefore defined only on the
        # normalized policy differential.
        neg_grad = torch.autograd.grad(
            neg_logit,
            norm_diff,
            grad_outputs=torch.ones_like(neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_grad_penalty = torch.mean(torch.sum(torch.square(neg_grad), dim=-1))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        neg_acc, pos_acc = self._compute_disc_acc(neg_logit, pos_logit)

        base_energy = terms["base_energy"]
        context_energy = terms["context_energy"]
        total_energy = torch.clamp_min(base_energy + context_energy, 1.0e-12)
        metric_diag = terms["metric_diag"]
        metric_diag_mean = torch.mean(metric_diag, dim=0)

        info = {
            "disc_loss": disc_loss,
            "disc_class_loss": 0.5 * (disc_loss_pos + disc_loss_neg).detach(),
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": torch.mean(pos_logit).detach(),
            "disc_neg_logit": torch.mean(neg_logit).detach(),
            "metric_base_energy": torch.mean(base_energy).detach(),
            "metric_context_energy": torch.mean(context_energy).detach(),
            "metric_context_energy_frac": torch.mean(context_energy / total_energy).detach(),
            "metric_trace_mean": torch.mean(terms["trace"]).detach(),
            "metric_trace_min": torch.min(terms["trace"]).detach(),
            "metric_trace_max": torch.max(terms["trace"]).detach(),
            "metric_v_norm_mean": torch.mean(torch.sqrt(terms["v_norm_sq"])).detach(),
            "metric_diag_min": torch.min(metric_diag_mean).detach(),
            "metric_diag_max": torch.max(metric_diag_mean).detach(),
            "metric_diag_context_std": torch.mean(
                torch.std(metric_diag, dim=0, unbiased=False)
            ).detach(),
            "disc_neg_logit_lt_m5_frac": torch.mean((neg_logit < -5.0).float()).detach(),
        }

        finite_tensors = [disc_loss, neg_logit, base_energy, context_energy,
                          terms["trace"], terms["v_norm_sq"]]
        all_finite = torch.stack([
            torch.isfinite(x).all() for x in finite_tensors
        ]).all()
        if not bool(all_finite.detach().cpu().item()):
            raise FloatingPointError("Non-finite CPMD metric training value")
        info["metric_all_finite"] = all_finite.float().detach()
        return info
