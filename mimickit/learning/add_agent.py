import torch

import learning.amp_agent as amp_agent
import learning.add_model as add_model
import util.torch_util as torch_util
import learning.diff_normalizer as diff_normalizer
import learning.normalizer as normalizer


def calc_group_balanced_gp(grad, group_indices, group_weights):
    raw = []
    for indices in group_indices:
        raw.append(torch.mean(torch.sum(
            torch.square(torch.index_select(grad, -1, indices)), dim=-1)))
    raw = torch.stack(raw)
    weighted = raw * group_weights
    return torch.sum(weighted), raw, weighted


def calc_unscaled_disc_reward(logits):
    probability = torch.sigmoid(logits)
    return -torch.log(torch.clamp_min(1 - probability, 0.0001))


class ADDAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._pos_diff = self._build_pos_diff()
        self._disc_error_groups = tuple()
        self._disc_group_indices = tuple()
        self._disc_group_weights = torch.empty(0, device=self._device)
        if self._use_group_balanced_gp:
            self._disc_error_groups = self._env.get_disc_error_groups()
            self._disc_group_indices = tuple(
                torch.tensor(indices, device=self._device, dtype=torch.long)
                for _, indices in self._disc_error_groups)
            dims = torch.tensor(
                [len(indices) for _, indices in self._disc_error_groups],
                device=self._device, dtype=torch.float32)
            calibration_dim = torch.sum(torch.square(dims)) / torch.sum(dims)
            self._disc_group_weights = dims / calibration_dim
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
        self._use_group_balanced_gp = bool(
            config.get("disc_group_balanced_gp", False))
        if self._disc_geometry not in {"add", "ref_concat"}:
            raise ValueError(
                "disc_geometry must be 'add' or 'ref_concat'")
        if self._use_group_balanced_gp:
            if self._disc_geometry != "add":
                raise ValueError(
                    "Group-balanced GP requires the direct ADD geometry")
            if self._disc_grad_penalty <= 0:
                raise ValueError(
                    "Group-balanced GP requires a positive GP coefficient")
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

        if self._disc_grad_penalty > 0:
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

        if self._disc_grad_penalty > 0:
            disc_neg_grad = torch.autograd.grad(
                disc_neg_logit, norm_diff_obs,
                grad_outputs=torch.ones_like(disc_neg_logit),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
            if self._use_group_balanced_gp:
                disc_grad_penalty, group_gp_raw, group_gp_weighted = \
                    calc_group_balanced_gp(
                        disc_neg_grad, self._disc_group_indices,
                        self._disc_group_weights)
            else:
                disc_grad_penalty = torch.mean(torch.sum(
                    torch.square(disc_neg_grad), dim=-1))
            disc_loss = (disc_loss
                         + self._disc_grad_penalty * disc_grad_penalty)
        else:
            disc_grad_penalty = torch.zeros((), device=self._device)
        
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
        if self._use_group_balanced_gp:
            weighted_total = torch.clamp_min(
                torch.sum(group_gp_weighted.detach()), 1e-12)
            for group_id, (name, indices) in enumerate(
                    self._disc_error_groups):
                disc_info["disc_gp_raw_{}".format(name)] = \
                    group_gp_raw[group_id].detach()
                disc_info["disc_gp_weight_{}".format(name)] = \
                    self._disc_group_weights[group_id].detach().clone()
                disc_info["disc_gp_weighted_{}".format(name)] = \
                    group_gp_weighted[group_id].detach()
                disc_info["disc_gp_fraction_{}".format(name)] = \
                    group_gp_weighted[group_id].detach() / weighted_total
        return disc_info

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
            disc_r = calc_unscaled_disc_reward(disc_logits)
            return self._disc_reward_scale * disc_r
