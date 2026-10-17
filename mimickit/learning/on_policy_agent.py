import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.distribution_gaussian_diag as distribution_gaussian_diag
import learning.mp_optimizer as mp_optimizer
import learning.ppo_model as ppo_model
import util.mp_util as mp_util
import util.torch_util as torch_util

class OnPolicyAgent(base_agent.BaseAgent):
    """Rollout, optimizer scheduling and diagnostics; objectives are composed."""
    _phase_balance_bins = 8

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        
        self._actor_epochs = config["actor_epochs"]
        self._actor_batch_size = config["actor_batch_size"]
        self._critic_epochs = config["critic_epochs"]
        self._critic_batch_size = config["critic_batch_size"]

        self._algorithm = self._build_algorithm(config)
        self._td_lambda = config["td_lambda"]

        self._action_bound_weight = config["action_bound_weight"]
        self._action_entropy_weight = config["action_entropy_weight"]
        self._action_reg_weight = config["action_reg_weight"]

        self._critic_eval_batch_size = int(config.get("critic_eval_batch_size", 0))
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)
        return

    def _build_algorithm(self, config):
        from learning.algorithms import build_algorithm
        return build_algorithm(config)

    def _build_model(self, config):
        model_config = config["model"]
        self._model = ppo_model.PPOModel(model_config, self._env)
        return
    
    def _build_optimizer(self, config):
        actor_config = config["actor_optimizer"]
        actor_params = list(self._model.get_actor_params())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(actor_config, actor_params)
        
        critic_config = config["critic_optimizer"]
        critic_params = list(self._model.get_critic_params())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(critic_config, critic_params)
        return
    
    def _sync_optimizer(self):
        self._actor_optimizer.sync()
        self._critic_optimizer.sync()
        return

    def _get_exp_buffer_length(self):
        return self._steps_per_iter
    
    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
            
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("a_logp", action_info["a_logp"])
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return
    
    def _build_train_data(self):
        return self._algorithm.build_train_data(self)

    def _build_phase_weights(self):
        if (not self._exp_buffer.has_buffer("diag_phase")):
            weights = torch.ones(
                [self._steps_per_iter, self.get_num_envs()],
                device=self._device, dtype=torch.float32)
            return weights, {
                "phase_weight_mean": weights.mean(),
                "phase_weight_max": weights.max(),
                "phase_bins_present": torch.tensor(
                    1.0, device=self._device)
            }

        phase = self._exp_buffer.get_data("diag_phase").to(torch.float32)
        phase = torch.nan_to_num(phase, nan=0.0, posinf=1.0, neginf=0.0)
        phase = torch.clamp(phase, 0.0, 1.0 - torch.finfo(phase.dtype).eps)
        phase_bin = (phase * self._phase_balance_bins).to(torch.long)
        flat_bin = phase_bin.flatten()
        counts = torch.bincount(
            flat_bin, minlength=self._phase_balance_bins).to(torch.float32)
        present = counts > 0.0
        num_present = present.sum().to(torch.float32)
        total = torch.tensor(
            float(flat_bin.numel()), device=phase.device, dtype=torch.float32)
        weights_by_bin = torch.zeros_like(counts)
        weights_by_bin[present] = total / (num_present * counts[present])
        weights = weights_by_bin[flat_bin].reshape(phase.shape)
        return weights, {
            "phase_weight_mean": weights.mean(),
            "phase_weight_max": weights.max(),
            "phase_bins_present": num_present,
        }

    def _weighted_mean_std(self, values, weights):
        weights = weights.to(dtype=values.dtype)
        sum_w = mp_util.reduce_sum(weights.sum())
        sum_wx = mp_util.reduce_sum(torch.sum(weights * values))
        sum_wx2 = mp_util.reduce_sum(torch.sum(weights * values.square()))
        mean = sum_wx / torch.clamp_min(sum_w, 1e-8)
        var = sum_wx2 / torch.clamp_min(sum_w, 1e-8) - mean.square()
        return mean, torch.sqrt(torch.clamp_min(var, 0.0))

    @staticmethod
    def _weighted_mean(values, weights):
        return torch.sum(values * weights) / torch.clamp_min(
            torch.sum(weights), 1e-8)
    
    def _get_exp_prob(self):
        if (np.isfinite(self._exp_anneal_samples)):
            samples = self._sample_count
            l = float(samples) / self._exp_anneal_samples
            l = np.clip(l, 0.0, 1.0)
            prob = (1.0 - l) * self._exp_prob_beg + l * self._exp_prob_end
        else:
            prob = self._exp_prob_beg
        return prob

    def _update_model(self):
        self.train()
        
        num_envs = self.get_num_envs()
        num_samples = self._exp_buffer.get_sample_count()

        critic_batch_size = int(np.ceil(self._critic_batch_size * num_envs))
        num_critic_batches = int(np.ceil(float(num_samples) / critic_batch_size))
        num_critic_steps = num_critic_batches * self._critic_epochs
        critic_info = self._update_critic(critic_batch_size, num_critic_steps)
        
        actor_batch_size = int(np.ceil(self._actor_batch_size * num_envs))
        num_actor_batches = int(np.ceil(float(num_samples) / actor_batch_size))
        num_actor_steps = num_actor_batches * self._actor_epochs
        actor_info = self._update_actor(actor_batch_size, num_actor_steps)
        
        train_info = {**critic_info, **actor_info}
        return train_info

    def _update_critic(self, batch_size, steps):
        info = dict()
        device_type = torch.device(self._device).type

        for i in range(steps):
            batch = self._exp_buffer.sample(batch_size)

            with torch.amp.autocast(device_type=device_type, enabled=self._use_mixed_precision, dtype=torch.bfloat16):
                loss_info = self._compute_critic_loss(batch)
                loss = loss_info["critic_loss"]

            self._critic_optimizer.step(loss)

            loss_info["critic_grad_norm"] = torch.tensor(
                self._critic_optimizer.get_last_grad_norm(),
                device=self._device,
                dtype=torch.float32)

            torch_util.add_torch_dict(loss_info, info)
        
        torch_util.scale_torch_dict(1.0 / steps, info)
        return info

    def _update_actor(self, batch_size, num_steps):
        info = dict()
        device_type = torch.device(self._device).type

        # A fixed slice of the just-collected rollout gives an exact, cheap
        # pre/post-update policy diagnostic.  Unlike the PPO importance ratio,
        # this compares the complete Gaussian action distributions on the same
        # observations and is therefore suitable for cross-run analysis.
        audit_obs, audit_old_dist, audit_phase = self._build_policy_audit_batch()

        for i in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)

            with torch.amp.autocast(device_type=device_type, enabled=self._use_mixed_precision, dtype=torch.bfloat16):
                loss_info = self._compute_actor_loss(batch)
                loss = loss_info["actor_loss"]

            self._actor_optimizer.step(loss)

            loss_info["actor_grad_norm"] = torch.tensor(
                self._actor_optimizer.get_last_grad_norm(),
                device=self._device,
                dtype=torch.float32)

            torch_util.add_torch_dict(loss_info, info)
        
        torch_util.scale_torch_dict(1.0 / num_steps, info)
        info.update(self._compute_policy_audit(
            audit_obs, audit_old_dist, audit_phase))
        info.update(self._compute_first_layer_grad_info())
        return info

    def _build_policy_audit_batch(self, max_samples=4096):
        obs = self._exp_buffer.get_data_flat("obs")
        num_samples = min(int(obs.shape[0]), int(max_samples))
        norm_obs = self._obs_norm.normalize(obs[:num_samples]).detach()
        phase = None
        if self._exp_buffer.has_buffer("diag_phase"):
            phase = self._exp_buffer.get_data_flat("diag_phase")[:num_samples]

        with torch.no_grad():
            old_dist = self._model.eval_actor(norm_obs)
            old_dist_data = {
                "mean": old_dist.mean.detach().clone(),
                "logstd": old_dist.logstd.detach().clone(),
            }
        return norm_obs, old_dist_data, phase

    def _compute_policy_audit(self, norm_obs, old_dist_data, phase=None):
        with torch.no_grad():
            new_dist = self._model.eval_actor(norm_obs)
            old_dist = distribution_gaussian_diag.DistributionGaussianDiag(
                mean=old_dist_data["mean"], logstd=old_dist_data["logstd"])
            policy_kl = torch.mean(old_dist.kl(new_dist))
            action_mean_delta = torch.sqrt(torch.mean(torch.square(
                new_dist.mean - old_dist.mean)))
            action_bound_frac = torch.mean(
                (torch.abs(new_dist.mean) > 1.0).to(dtype=torch.float32))

        info = {
            "policy_kl": policy_kl,
            "action_mean_update_rms": action_mean_delta,
            "action_mean_bound_frac": action_bound_frac,
        }
        if phase is not None:
            phase = phase.flatten()
            kl = old_dist.kl(new_dist)
            num_bins = 4
            phase_bin = torch.clamp((phase * num_bins).long(), 0, num_bins - 1)
            for i in range(num_bins):
                mask = phase_bin == i
                info["policy_kl_phase{}".format(i)] = (
                    torch.mean(kl[mask]) if torch.any(mask)
                    else torch.full((), float("nan"), device=kl.device))
        return info

    def _compute_first_layer_grad_info(self):
        first_linear = None
        for module in self._model._actor_layers.modules():
            if isinstance(module, torch.nn.Linear):
                first_linear = module
                break

        if first_linear is None or first_linear.weight.grad is None:
            return {}

        grad = first_linear.weight.grad.detach()
        info = {"actor_first_layer_grad_norm": torch.linalg.vector_norm(grad)}

        # RCCI's paired interfaces have a known [self, x, command1, command2]
        # layout.  Per-block norms test whether learning actually uses each
        # coordinate block; no corresponding loss or gradient is introduced.
        if (hasattr(self._env, "get_rcci_self_obs_dim")
                and hasattr(self._env, "get_rcci_phi_dim")):
            self_dim = self._env.get_rcci_self_obs_dim()
            phi_dim = self._env.get_rcci_phi_dim()
            slices = {
                "actor_grad_self_block": slice(0, self_dim),
                "actor_grad_x_block": slice(self_dim, self_dim + phi_dim),
                "actor_grad_command1_block": slice(
                    self_dim + phi_dim, self_dim + 2 * phi_dim),
                "actor_grad_command2_block": slice(
                    self_dim + 2 * phi_dim, self_dim + 3 * phi_dim),
            }
            for key, block_slice in slices.items():
                info[key] = torch.linalg.vector_norm(grad[:, block_slice])

        return info
    
    def _compute_critic_loss(self, batch):
        return self._algorithm.compute_critic_loss(self, batch)

    def _compute_actor_loss(self, batch):
        return self._algorithm.compute_actor_loss(self, batch)

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time,
                        test_random_info=None):
        super()._log_train_info(
            train_info, test_info, env_diag_info, start_time,
            test_random_info=test_random_info)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        return
