import numpy as np
import torch

import learning.amp_agent as amp_agent
import learning.add_model as add_model
import util.torch_util as torch_util
import learning.diff_normalizer as diff_normalizer
import learning.normalizer as normalizer


def calc_influence_allocation_loss(gains, margin, target):
    desired_gain = margin.detach() * target.detach()
    loss = 0.5 * torch.sum(torch.square(gains - desired_gain))
    return loss, desired_gain


class ADDAgent(amp_agent.AMPAgent):
    _ALLOC_EPS = 1e-6

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._pos_diff = self._build_pos_diff()
        self._disc_error_groups = tuple()
        self._disc_group_indices = tuple()
        if self._use_influence_allocation:
            self._disc_error_groups = self._env.get_disc_error_groups()
            self._disc_group_indices = tuple(
                torch.tensor(indices, device=self._device, dtype=torch.long)
                for _, indices in self._disc_error_groups)
            num_groups = len(self._disc_error_groups)
            self.register_buffer(
                "_alloc_initial_error",
                torch.zeros(num_groups, device=self._device))
            self.register_buffer(
                "_alloc_baseline_ready",
                torch.zeros((), device=self._device, dtype=torch.bool))
            self._alloc_current_error = torch.zeros(
                num_groups, device=self._device)
            self._alloc_target = torch.full(
                (num_groups,), 1.0 / num_groups, device=self._device)
        return
    
    def _build_model(self, config):
        model_config = dict(config["model"])
        model_config["disc_geometry"] = self._disc_geometry
        model_config["disc_spectral_norm"] = self._disc_spectral_norm
        self._model = add_model.ADDModel(model_config, self._env)
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_geometry = config.get("disc_geometry", "add")
        self._disc_spectral_norm = bool(
            config.get("disc_spectral_norm", False))
        self._use_influence_allocation = bool(
            config.get("disc_influence_allocation", False))
        if self._disc_geometry not in {"add", "ref_concat"}:
            raise ValueError(
                "disc_geometry must be 'add' or 'ref_concat'")
        if self._use_influence_allocation:
            if self._disc_geometry != "add":
                raise ValueError(
                    "Influence allocation requires the direct ADD geometry")
            if not self._disc_spectral_norm:
                raise ValueError(
                    "Influence allocation requires discriminator spectral "
                    "normalization")
            if self._disc_grad_penalty != 0:
                raise ValueError(
                    "Influence allocation replaces the ADD gradient penalty")
        return
    
    def _build_pos_diff(self):
        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        pos_diff = torch.zeros(disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)
        return pos_diff
    
    def _build_normalizers(self):
        super(amp_agent.AMPAgent, self)._build_normalizers()

        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)
        if self._uses_context():
            self._disc_context_norm = normalizer.Normalizer(
                disc_obs_space.shape, clip=10.0, device=self._device,
                dtype=disc_obs_dtype)
        return

    def _uses_context(self):
        return self._disc_geometry == "ref_concat"

    def _update_normalizers(self):
        super()._update_normalizers()
        if self._uses_context():
            self._disc_context_norm.update()
        return

    def _normalize_context(self, context):
        if not self._uses_context():
            return None
        return self._disc_context_norm.normalize(context)
    
    def _record_data_post_step(self, next_obs, r, done, next_info):
        super(amp_agent.AMPAgent, self)._record_data_post_step(next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        disc_obs_demo = next_info["disc_obs_demo"]
        self._exp_buffer.record("disc_obs_demo", disc_obs_demo)
        self._exp_buffer.record("disc_obs", disc_obs)
        return
    
    def _record_disc_demo_data(self):
        return
    
    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        n = disc_obs.shape[0]
        idx = self._sample_disc_replay_indices(n)
        replay_disc_obs = disc_obs[idx]
        replay_disc_obs_demo = disc_obs_demo[idx]
        disc_data = {
            "disc_obs": replay_disc_obs.unsqueeze(1),
            "disc_obs_demo": replay_disc_obs_demo.unsqueeze(1)
        }
        self._disc_buffer.push(disc_data)
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        norm_context = self._normalize_context(disc_obs_demo)
        disc_r = self._calc_disc_rewards(norm_obs_diff, norm_context)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        alloc_info = {}
        if self._use_influence_allocation:
            alloc_info = self._update_allocation_target(obs_diff)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)
        
        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(obs_diff)
            if self._uses_context():
                self._disc_context_norm.record(disc_obs_demo)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std
        }
        info.update(alloc_info)
        return info

    def _update_allocation_target(self, raw_diff):
        with torch.no_grad():
            errors = []
            for indices in self._disc_group_indices:
                group_diff = torch.index_select(raw_diff, -1, indices)
                errors.append(torch.sqrt(torch.mean(torch.square(group_diff))))
            errors = torch.stack(errors)

            if not bool(self._alloc_baseline_ready.item()):
                self._alloc_initial_error.copy_(
                    torch.clamp_min(errors, self._ALLOC_EPS))
                self._alloc_baseline_ready.fill_(True)

            ratios = errors / torch.clamp_min(
                self._alloc_initial_error, self._ALLOC_EPS)
            target = ratios / torch.clamp_min(
                torch.sum(ratios), self._ALLOC_EPS)
            self._alloc_current_error = errors.detach()
            self._alloc_target = target.detach()

            info = {}
            for group_id, (name, _) in enumerate(self._disc_error_groups):
                info["alloc_error_{}".format(name)] = errors[group_id]
                info["alloc_error_ratio_{}".format(name)] = ratios[group_id]
                info["alloc_target_{}".format(name)] = target[group_id]
            info["alloc_target_entropy"] = -torch.sum(
                target * torch.log(torch.clamp_min(target, self._ALLOC_EPS)))
            info["alloc_target_max"] = torch.max(target)
            info["alloc_target_min"] = torch.min(target)
            return info
    
    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]

        if not self._uses_context():
            pos_diff = self._pos_diff.clone().unsqueeze(dim=0)
            disc_pos_logit = self._model.eval_disc(pos_diff).squeeze(-1)
        
        current_diff_obs = tar_disc_obs - disc_obs
        current_count = current_diff_obs.shape[0]
        
        replay_data = self._disc_buffer.sample(current_count)
        replay_disc_obs = replay_data["disc_obs"]
        replay_tar_disc_obs = replay_data["disc_obs_demo"]
        replay_diff = replay_tar_disc_obs - replay_disc_obs
        diff_obs = torch.cat([current_diff_obs, replay_diff], dim=0)
        context = torch.cat([tar_disc_obs, replay_tar_disc_obs], dim=0)

        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_context = self._normalize_context(context)

        if self._uses_context():
            pos_diff = torch.zeros_like(norm_diff_obs)
            disc_pos_logit = self._model.eval_disc(pos_diff, norm_context)
            disc_pos_logit = disc_pos_logit.squeeze(-1)

        if not self._use_influence_allocation:
            norm_diff_obs.requires_grad_(True)
        disc_neg_logit = self._model.eval_disc(
            norm_diff_obs, norm_context)
        disc_neg_logit = disc_neg_logit.squeeze(-1)
        
        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_cls_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # logit reg
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_logit_reg_loss = self._disc_logit_reg * disc_logit_loss
        disc_loss = disc_cls_loss + disc_logit_reg_loss

        if self._use_influence_allocation:
            allocation_info = self._compute_influence_allocation(
                norm_diff_obs[:current_count],
                disc_neg_logit[:current_count], disc_pos_logit)
            allocation_loss = allocation_info["alloc_loss"]
            disc_loss = disc_loss + allocation_loss
            disc_grad_penalty = torch.zeros_like(allocation_loss)
        else:
            disc_neg_grad = torch.autograd.grad(
                disc_neg_logit, norm_diff_obs,
                grad_outputs=torch.ones_like(disc_neg_logit),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
            disc_grad_penalty = torch.mean(torch.sum(
                torch.square(disc_neg_grad), dim=-1))
            disc_loss = (disc_loss
                         + self._disc_grad_penalty * disc_grad_penalty)
            allocation_info = {}
        
        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(disc_neg_logit, disc_pos_logit)
        disc_pos_logit_mean = torch.mean(disc_pos_logit)
        disc_neg_logit_mean = torch.mean(disc_neg_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_cls_loss.detach(),
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_logit_reg_loss": disc_logit_reg_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": disc_pos_logit_mean.detach(),
            "disc_neg_logit": disc_neg_logit_mean.detach()
        }
        disc_info.update(allocation_info)
        if self._disc_spectral_norm:
            disc_info.update(self._model.get_disc_scale_stats())
        return disc_info

    def _compute_influence_allocation(self, current_norm_diff,
                                      current_logit, pos_logit):
        target = self._alloc_target
        margin = torch.mean(torch.abs(
            pos_logit.detach().mean() - current_logit.detach()))
        gains = []
        negative_fractions = []
        for group_id, indices in enumerate(self._disc_group_indices):
            counterfactual = current_norm_diff.clone()
            counterfactual.index_fill_(-1, indices, 0.0)
            counterfactual_logit = self._model.eval_disc(
                counterfactual).squeeze(-1)
            sample_gain = counterfactual_logit - current_logit
            gain = torch.mean(sample_gain)
            gains.append(gain)
            negative_fractions.append(torch.mean(
                (sample_gain < 0).to(dtype=torch.float32)))

        gains = torch.stack(gains)
        negative_fractions = torch.stack(negative_fractions)
        allocation_loss, desired_gain = calc_influence_allocation_loss(
            gains, margin, target)
        residual = gains - desired_gain
        info = {
            "alloc_loss": allocation_loss,
            "alloc_margin": margin,
            "alloc_gain_sum": torch.sum(gains).detach(),
            "alloc_desired_gain_sum": torch.sum(desired_gain).detach(),
            "alloc_gain_residual_rms": torch.sqrt(
                torch.mean(torch.square(residual))).detach(),
            "alloc_negative_fraction": torch.mean(
                negative_fractions).detach(),
        }
        for group_id, (name, _) in enumerate(self._disc_error_groups):
            info["alloc_gain_{}".format(name)] = gains[group_id].detach()
            info["alloc_desired_gain_{}".format(name)] = \
                desired_gain[group_id].detach()
            info["alloc_negative_fraction_{}".format(name)] = \
                negative_fractions[group_id].detach()
        return info

    def _calc_disc_rewards(self, norm_diff_obs, norm_context=None):
        with torch.no_grad():
            if self._disc_eval_batch_size <= 0:
                disc_logits = self._model.eval_disc(
                    norm_diff_obs, norm_context)
            else:
                logits = []
                for start in range(0, norm_diff_obs.shape[0],
                                   self._disc_eval_batch_size):
                    end = min(start + self._disc_eval_batch_size,
                              norm_diff_obs.shape[0])
                    curr_context = (None if norm_context is None
                                    else norm_context[start:end])
                    logits.append(self._model.eval_disc(
                        norm_diff_obs[start:end], curr_context))
                disc_logits = torch.cat(logits, dim=0)
            disc_logits = disc_logits.squeeze(-1)
            prob = torch.sigmoid(disc_logits)
            disc_r = -torch.log(torch.clamp_min(1 - prob, 0.0001))
            return self._disc_reward_scale * disc_r
