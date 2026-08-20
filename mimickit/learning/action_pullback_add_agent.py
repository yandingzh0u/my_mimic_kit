import numpy as np
import torch
import torch.nn.functional as functional

import learning.action_pullback_add_model as action_pullback_add_model
import learning.aligned_add_agent as aligned_add_agent
import learning.diff_normalizer as diff_normalizer
import learning.mp_optimizer as mp_optimizer
import learning.normalizer as normalizer
import util.torch_util as torch_util


def split_aligned_command(obs, self_dim, command_dim):
    """Split [self, current error, one-step reference increment]."""
    expected_dim = int(self_dim + 2 * command_dim)
    if obs.shape[-1] != expected_dim:
        raise ValueError(
            "aligned observation has size {}, expected {}".format(
                obs.shape[-1], expected_dim))
    i1 = int(self_dim)
    i2 = i1 + int(command_dim)
    return obs[..., :i1], obs[..., i1:i2], obs[..., i2:]


def response_target_from_closure(obs, next_obs, self_dim, command_dim):
    """Return the realized feature increment using the exact closure identity.

    e_{t+1} = e_t + m_t - delta_t implies
    delta_t = e_t + m_t - e_{t+1}.  The target therefore requires no expert
    action, inverse dynamics, differentiable simulator, or additional rollout
    observation.
    """
    _, error, motion = split_aligned_command(obs, self_dim, command_dim)
    _, next_error, _ = split_aligned_command(
        next_obs, self_dim, command_dim)
    return error + motion - next_error


def linearized_pullback_loss(
        action_mean, action_reward_grad, reference_action=None,
        action_delta_clip=0.0):
    """First-order actor objective whose descent moves along reward ascent."""
    if action_mean.shape != action_reward_grad.shape:
        raise ValueError("action mean and pullback gradient shapes differ")
    if reference_action is None:
        reference_action = torch.zeros_like(action_mean)
    if action_mean.shape != reference_action.shape:
        raise ValueError("action mean and reference action shapes differ")
    action_delta = action_mean - reference_action.detach()
    if action_delta_clip > 0:
        action_delta = torch.clamp(
            action_delta, -action_delta_clip, action_delta_clip)
    return -torch.mean(torch.sum(
        action_delta * action_reward_grad.detach(), dim=-1))


def differentiable_add_reward(logits, reward_scale):
    """ADD reward with the same saturation as the rollout implementation."""
    prob = torch.sigmoid(logits)
    min_complement = torch.tensor(
        1e-4, device=logits.device, dtype=logits.dtype)
    return (-torch.log(torch.maximum(1.0 - prob, min_complement))
            * reward_scale)


def build_private_generator(device, seed):
    """Build a deterministic generator without advancing the global RNG."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def update_response_normalizers_from_train(
        self_obs, target_delta, train_idx, self_normalizer,
        delta_normalizer):
    """Update response coordinates from training samples only."""
    if train_idx.numel() == 0:
        raise ValueError("response normalizers require nonempty training data")
    self_normalizer.record(self_obs[train_idx])
    delta_normalizer.record(target_delta[train_idx])
    self_normalizer.update()
    delta_normalizer.update()


def zero_motion_skill(response_loss, zero_baseline_loss):
    """Prediction skill relative to the zero-motion response baseline."""
    return 1.0 - response_loss / torch.clamp_min(
        zero_baseline_loss, 1e-8)


class ActionPullbackADDAgent(aligned_add_agent.AlignedADDAgent):
    """Pull ADD's learned differential objective back into action space.

    A self-supervised forward response model predicts the feature increment
    caused by an action.  The frozen-for-this-update response model and ADD
    discriminator define a local action gradient

      grad_a R(e + m - F(s, a)).

    PPO remains responsible for long-horizon credit assignment.  The added
    linearized objective supplies the locally executable direction without
    imitation labels, a teacher, inverse dynamics, or staged training.
    """

    def _load_params(self, config):
        super()._load_params(config)
        self._response_epochs = int(config.get("response_epochs", 2))
        self._response_batch_size = int(config.get("response_batch_size", 4))
        self._pullback_weight = float(config.get("pullback_weight", 0.1))
        self._pullback_grad_clip = float(
            config.get("pullback_grad_clip", 1.0))
        self._pullback_normalize_direction = bool(
            config.get("pullback_normalize_direction", True))
        self._pullback_min_grad_norm = float(
            config.get("pullback_min_grad_norm", 1e-6))
        self._pullback_action_delta_clip = float(
            config.get("pullback_action_delta_clip", 0.05))
        self._pullback_eval_batch_size = int(
            config.get("pullback_eval_batch_size", 4096))
        self._pullback_audit_batch_size = int(
            config.get("pullback_audit_batch_size", 2048))
        self._response_validation_size = int(
            config.get("response_validation_size", 4096))

        if self._response_epochs <= 0:
            raise ValueError("response_epochs must be positive")
        if self._response_batch_size <= 0:
            raise ValueError("response_batch_size must be positive")
        if self._pullback_weight < 0:
            raise ValueError("pullback_weight must be nonnegative")
        if self._pullback_grad_clip < 0:
            raise ValueError("pullback_grad_clip must be nonnegative")
        if self._pullback_min_grad_norm < 0:
            raise ValueError("pullback_min_grad_norm must be nonnegative")
        if self._pullback_action_delta_clip < 0:
            raise ValueError(
                "pullback_action_delta_clip must be nonnegative")
        if self._pullback_eval_batch_size <= 0:
            raise ValueError("pullback_eval_batch_size must be positive")
        if self._pullback_audit_batch_size <= 0:
            raise ValueError("pullback_audit_batch_size must be positive")
        if self._response_validation_size <= 0:
            raise ValueError("response_validation_size must be positive")

    def _build_normalizers(self):
        super()._build_normalizers()
        self_dim = int(self._env.get_aligned_self_obs_dim())
        command_dim = int(self._env.get_aligned_command_dim())
        obs_dtype = torch_util.numpy_dtype_to_torch(
            self._env.get_obs_space().dtype)

        # F has its own cumulative input/output coordinates.  Updating them
        # from the actual response targets before fitting the first batch
        # avoids the iteration-0 coordinate jump of the actor and ADD
        # normalizers.  Neither normalizer changes the policy or reward.
        self._response_self_norm = normalizer.Normalizer(
            [self_dim], clip=10.0, device=self._device, dtype=obs_dtype)
        self._response_delta_norm = diff_normalizer.DiffNormalizer(
            [command_dim], device=self._device, dtype=obs_dtype)

    def _build_model(self, config):
        self._model = action_pullback_add_model.ActionPullbackADDModel(
            config["model"], self._env)

    def _build_optimizer(self, config):
        super()._build_optimizer(config)
        response_params = [
            param for param in self._model.get_response_params()
            if param.requires_grad
        ]
        self._response_optimizer = mp_optimizer.MPOptimizer(
            config["response_optimizer"], response_params)

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._response_optimizer.sync()

    def _update_model(self):
        # Fit the current local controlled response before using its Jacobian
        # for the actor update.  ADD's discriminator update order remains
        # unchanged, so PPO rewards and pullback gradients use the same critic
        # snapshot collected at the start of the iteration.
        if self._pullback_weight == 0:
            return super()._update_model()

        response_info = self._update_response_model()
        pullback_info = self._cache_pullback_data()
        train_info = super()._update_model()
        return {**train_info, **response_info, **pullback_info}

    def _update_response_model(self):
        obs = self._exp_buffer.get_data_flat("obs")
        next_obs = self._exp_buffer.get_data_flat("next_obs")
        action = self._exp_buffer.get_data_flat("action")
        num_samples = int(obs.shape[0])
        if num_samples < 2:
            raise RuntimeError("response fitting requires at least two samples")

        self_dim = self._env.get_aligned_self_obs_dim()
        command_dim = self._env.get_aligned_command_dim()
        # The response learner has a deterministic, private sampling stream,
        # so it cannot silently alter PPO's random minibatch order.
        generator = build_private_generator(
            self._device, 0xA11CE + int(self._iter))
        permutation = torch.randperm(
            num_samples, device=self._device, dtype=torch.long,
            generator=generator)
        validation_size = min(
            self._response_validation_size,
            max(1, num_samples // 8))
        validation_idx = permutation[:validation_size]
        train_idx = permutation[validation_size:]

        self_obs, _, _ = split_aligned_command(
            obs, self_dim, command_dim)
        target_delta = response_target_from_closure(
            obs, next_obs, self_dim, command_dim)
        if self._need_normalizer_update():
            # Validation targets must not determine coordinates used to fit F.
            update_response_normalizers_from_train(
                self_obs, target_delta, train_idx,
                self._response_self_norm, self._response_delta_norm)

        batch_size = min(
            int(train_idx.shape[0]),
            int(np.ceil(self._response_batch_size * self.get_num_envs())))

        train_loss_sum = torch.zeros((), device=self._device)
        train_sample_count = 0
        response_grad_norm_sum = torch.zeros((), device=self._device)
        response_step_count = 0
        device_type = torch.device(self._device).type
        for _ in range(self._response_epochs):
            order = train_idx[torch.randperm(
                int(train_idx.shape[0]), device=self._device,
                dtype=torch.long, generator=generator)]
            for start in range(0, int(order.shape[0]), batch_size):
                idx = order[start:min(start + batch_size, int(order.shape[0]))]
                batch = {
                    "obs": obs[idx],
                    "next_obs": next_obs[idx],
                    "action": action[idx],
                }
                with torch.amp.autocast(
                        device_type=device_type,
                        enabled=self._use_mixed_precision,
                        dtype=torch.bfloat16):
                    loss_info = self._compute_response_loss(batch)
                    loss = loss_info["response_loss"]

                self._response_optimizer.step(loss)
                grad_norm = torch.tensor(
                    self._response_optimizer.get_last_grad_norm(),
                    device=self._device, dtype=torch.float32)
                train_loss_sum += loss_info["response_loss"].detach() * len(idx)
                train_sample_count += len(idx)
                response_grad_norm_sum += grad_norm
                response_step_count += 1

        validation_batch = {
            "obs": obs[validation_idx],
            "next_obs": next_obs[validation_idx],
            "action": action[validation_idx],
        }
        validation_info = self._evaluate_response_model(validation_batch)
        validation_info["response_train_loss"] = (
            train_loss_sum / max(train_sample_count, 1))
        validation_info["response_grad_norm"] = (
            response_grad_norm_sum / max(response_step_count, 1))
        return validation_info

    def _compute_response_loss(self, batch):
        self_obs, _, _ = split_aligned_command(
            batch["obs"], self._env.get_aligned_self_obs_dim(),
            self._env.get_aligned_command_dim())
        norm_self_obs = self._response_self_norm.normalize(self_obs)
        norm_action = torch.clamp(
            self._a_norm.normalize(batch["action"]), -1.0, 1.0)
        target_delta = response_target_from_closure(
            batch["obs"], batch["next_obs"],
            self._env.get_aligned_self_obs_dim(),
            self._env.get_aligned_command_dim())

        # F predicts a raw physical feature increment.  Scaling is applied to
        # the supervised residual only, so online normalizer updates cannot
        # silently change the semantics of the model output.
        pred_delta = self._model.eval_response(norm_self_obs, norm_action)
        norm_error = self._response_delta_norm.normalize(
            pred_delta - target_delta)
        norm_target_delta = self._response_delta_norm.normalize(target_delta)
        response_loss = torch.mean(torch.square(norm_error))
        zero_baseline_loss = torch.mean(torch.square(norm_target_delta))
        response_skill = zero_motion_skill(
            response_loss.detach(), zero_baseline_loss.detach())

        pred_flat = pred_delta.detach().flatten(start_dim=1)
        target_flat = target_delta.detach().flatten(start_dim=1)
        response_cosine = torch.mean(functional.cosine_similarity(
            pred_flat, target_flat, dim=-1, eps=1e-8))
        return {
            "response_loss": response_loss,
            "response_zero_loss": zero_baseline_loss.detach(),
            "response_skill": response_skill,
            # Backward-compatible alias for existing logs. This is skill
            # against a zero-motion predictor, not variance-explained R^2.
            "response_r2": response_skill,
            "response_cosine": response_cosine,
        }

    def _evaluate_response_model(self, batch):
        with torch.no_grad():
            info = self._compute_response_loss(batch)
            self_obs, error, motion = split_aligned_command(
                batch["obs"], self._env.get_aligned_self_obs_dim(),
                self._env.get_aligned_command_dim())
            _, actual_next_error, _ = split_aligned_command(
                batch["next_obs"], self._env.get_aligned_self_obs_dim(),
                self._env.get_aligned_command_dim())
            norm_self_obs = self._response_self_norm.normalize(self_obs)
            norm_action = torch.clamp(
                self._a_norm.normalize(batch["action"]), -1.0, 1.0)
            pred_delta = self._model.eval_response(
                norm_self_obs, norm_action)
            pred_next_error = error + motion - pred_delta

            pred_logits = self._model.eval_disc(
                self._disc_obs_norm.normalize(pred_next_error)).squeeze(-1)
            actual_logits = self._model.eval_disc(
                self._disc_obs_norm.normalize(
                    actual_next_error)).squeeze(-1)
            pred_reward = differentiable_add_reward(
                pred_logits, self._disc_reward_scale)
            actual_reward = differentiable_add_reward(
                actual_logits, self._disc_reward_scale)
            pred_centered = pred_reward - torch.mean(pred_reward)
            actual_centered = actual_reward - torch.mean(actual_reward)
            correlation = torch.sum(pred_centered * actual_centered) / (
                torch.linalg.vector_norm(pred_centered)
                * torch.linalg.vector_norm(actual_centered) + 1e-8)
            residual_rmse = torch.sqrt(torch.mean(torch.square(
                self._disc_obs_norm.normalize(
                    pred_next_error - actual_next_error))))

            info["response_reward_corr"] = correlation
            info["response_next_residual_rmse"] = residual_rmse
        return info

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        actor_loss = info["actor_loss"]

        if self._pullback_weight == 0:
            info["pullback_loss"] = torch.zeros(
                (), device=self._device, dtype=actor_loss.dtype)
            return info

        rand_action_mask = (batch["rand_action_mask"] == 1.0)
        obs = batch["obs"][rand_action_mask]
        norm_obs = self._obs_norm.normalize(obs)
        action_dist = self._model.eval_actor(norm_obs)
        action_mean = action_dist.mode

        action_grad = batch["pullback_action_grad"][rand_action_mask]
        reference_action = batch["pullback_action_mean"][rand_action_mask]
        pullback_loss = linearized_pullback_loss(
            action_mean, action_grad, reference_action,
            self._pullback_action_delta_clip)
        actor_loss = actor_loss + self._pullback_weight * pullback_loss

        info["actor_loss"] = actor_loss
        info["pullback_loss"] = pullback_loss.detach()
        info["pullback_action_grad_rms"] = torch.sqrt(torch.mean(
            torch.square(action_grad)))
        info["pullback_action_grad_norm"] = torch.mean(
            torch.linalg.vector_norm(action_grad, dim=-1))
        action_delta = action_mean - reference_action
        info["pullback_action_delta_rms"] = torch.sqrt(torch.mean(
            torch.square(action_delta)))
        if self._pullback_action_delta_clip > 0:
            info["pullback_surrogate_clip_frac"] = torch.mean(
                (torch.abs(action_delta)
                 >= self._pullback_action_delta_clip).float())
        return info

    def _cache_pullback_data(self):
        """Linearize the frozen one-step learned reward once per rollout."""
        obs = self._exp_buffer.get_data_flat("obs")
        norm_obs = self._obs_norm.normalize(obs).detach()
        self_obs, _, _ = split_aligned_command(
            obs, self._env.get_aligned_self_obs_dim(),
            self._env.get_aligned_command_dim())
        norm_response_self = self._response_self_norm.normalize(
            self_obs).detach()
        action_means = []
        with torch.no_grad():
            for start in range(0, obs.shape[0],
                               self._pullback_eval_batch_size):
                end = min(start + self._pullback_eval_batch_size,
                          obs.shape[0])
                action_means.append(
                    self._model.eval_actor(norm_obs[start:end]).mode)
        action_mean = torch.cat(action_means, dim=0).detach()

        if self._pullback_weight == 0:
            action_grad = torch.zeros_like(action_mean)
            predicted_reward = torch.zeros((), device=self._device)
            clipped_fraction = torch.zeros((), device=self._device)
            raw_grad_rms = torch.zeros((), device=self._device)
            raw_grad_norm = torch.zeros((), device=self._device)
            active_fraction = torch.zeros((), device=self._device)
        else:
            (action_grad, predicted_reward, clipped_fraction,
             raw_grad_rms, raw_grad_norm, active_fraction) = (
                self._compute_pullback_action_gradient(
                    obs.detach(), norm_response_self, action_mean))

        self._exp_buffer.set_data_flat(
            "pullback_action_grad", action_grad)
        self._exp_buffer.set_data_flat(
            "pullback_action_mean", action_mean)
        action_grad_norm = torch.linalg.vector_norm(action_grad, dim=-1)
        info = {
            "pullback_pred_reward": predicted_reward,
            "pullback_action_grad_rms_cached": torch.sqrt(torch.mean(
                torch.square(action_grad))),
            "pullback_action_grad_norm_cached": torch.mean(action_grad_norm),
            "pullback_grad_nonzero_frac": torch.mean(
                (action_grad_norm > 1e-8).float()),
            "pullback_grad_clipped_frac": clipped_fraction,
            "pullback_raw_action_grad_rms": raw_grad_rms,
            "pullback_raw_action_grad_norm": raw_grad_norm,
            "pullback_direction_active_frac": active_fraction,
        }
        info.update(self._compute_pullback_parameter_diagnostics())
        return info

    def _compute_pullback_parameter_diagnostics(self):
        """Compare the auxiliary and PPO gradients on the same audit batch."""
        num_samples = min(
            self._pullback_audit_batch_size,
            int(self._exp_buffer.get_data_flat("obs").shape[0]))
        keys = [
            "obs", "action", "a_logp", "adv", "rand_action_mask",
            "pullback_action_grad", "pullback_action_mean",
        ]
        total_samples = int(
            self._exp_buffer.get_data_flat("obs").shape[0])
        # Uniformly audit the complete [time, environment] rollout using a
        # private stream. In particular, do not audit only time step zero and
        # do not perturb PPO's global random stream.
        generator = build_private_generator(
            self._device, 0xA0D17 + int(self._iter))
        audit_idx = torch.randperm(
            total_samples, device=self._device, dtype=torch.long,
            generator=generator)[:num_samples]
        batch = {
            key: self._exp_buffer.get_data_flat(key)[audit_idx]
            for key in keys
        }

        ppo_loss = super()._compute_actor_loss(batch)["actor_loss"]
        actor_params = [
            param for param in self._model.get_actor_params()
            if param.requires_grad
        ]
        ppo_grads = torch.autograd.grad(
            ppo_loss, actor_params, allow_unused=True,
            retain_graph=False, create_graph=False)

        rand_action_mask = (batch["rand_action_mask"] == 1.0)
        norm_obs = self._obs_norm.normalize(
            batch["obs"][rand_action_mask])
        action_mean = self._model.eval_actor(norm_obs).mode
        pullback_loss = linearized_pullback_loss(
            action_mean,
            batch["pullback_action_grad"][rand_action_mask],
            batch["pullback_action_mean"][rand_action_mask],
            self._pullback_action_delta_clip)
        weighted_pullback_loss = self._pullback_weight * pullback_loss
        pullback_grads = torch.autograd.grad(
            weighted_pullback_loss, actor_params, allow_unused=True,
            retain_graph=False, create_graph=False)

        ppo_sq = torch.zeros((), device=self._device)
        pullback_sq = torch.zeros((), device=self._device)
        dot = torch.zeros((), device=self._device)
        for ppo_grad, pullback_grad in zip(ppo_grads, pullback_grads):
            if ppo_grad is not None:
                ppo_sq += torch.sum(torch.square(ppo_grad.detach()))
            if pullback_grad is not None:
                pullback_sq += torch.sum(
                    torch.square(pullback_grad.detach()))
            if ppo_grad is not None and pullback_grad is not None:
                dot += torch.sum(
                    ppo_grad.detach() * pullback_grad.detach())

        ppo_norm = torch.sqrt(ppo_sq)
        pullback_norm = torch.sqrt(pullback_sq)
        ratio = pullback_norm / torch.clamp_min(ppo_norm, 1e-12)
        cosine = dot / torch.clamp_min(
            ppo_norm * pullback_norm, 1e-12)
        return {
            "ppo_param_grad_norm_audit": ppo_norm,
            "pullback_param_grad_norm": pullback_norm,
            "pullback_to_ppo_grad_ratio": ratio,
            "pullback_ppo_grad_cosine": cosine,
        }

    def _compute_pullback_action_gradient(
            self, obs, norm_response_self, norm_action_mean):
        grads = []
        reward_sum = torch.zeros((), device=self._device)
        clipped_count = torch.zeros((), device=self._device)
        raw_grad_sq_sum = torch.zeros((), device=self._device)
        raw_grad_norm_sum = torch.zeros((), device=self._device)
        active_count = torch.zeros((), device=self._device)
        sample_count = 0
        chunk_size = self._pullback_eval_batch_size
        self_dim = self._env.get_aligned_self_obs_dim()
        command_dim = self._env.get_aligned_command_dim()

        for start in range(0, obs.shape[0], chunk_size):
            end = min(start + chunk_size, obs.shape[0])
            obs_chunk = obs[start:end]
            norm_self_chunk = norm_response_self[start:end]
            action_probe = norm_action_mean[start:end].clone().requires_grad_(True)
            executed_action_probe = torch.clamp(action_probe, -1.0, 1.0)

            _, error, motion = split_aligned_command(
                obs_chunk, self_dim, command_dim)
            pred_delta = self._model.eval_response(
                norm_self_chunk, executed_action_probe)
            pred_next_error = self._disc_obs_norm.normalize(
                error + motion - pred_delta)
            logits = self._model.eval_disc(pred_next_error).squeeze(-1)
            rewards = differentiable_add_reward(
                logits, self._disc_reward_scale)
            action_grad = torch.autograd.grad(
                rewards.sum(), action_probe, create_graph=False,
                retain_graph=False, only_inputs=True)[0].detach()

            raw_grad_sq_sum += torch.sum(torch.square(action_grad))
            grad_norm = torch.linalg.vector_norm(
                action_grad, dim=-1, keepdim=True)
            raw_grad_norm_sum += torch.sum(grad_norm)
            active = grad_norm >= self._pullback_min_grad_norm
            active_count += torch.sum(active.float())

            if self._pullback_normalize_direction:
                # Unit Euclidean steepest-ascent direction. The discriminator
                # supplies direction while its saturating output scale cannot
                # silently switch the module off. The separately truncated
                # local actor surrogate is not an executed-action constraint.
                action_grad = torch.where(
                    active,
                    action_grad / torch.clamp_min(grad_norm, 1e-12),
                    torch.zeros_like(action_grad))
            elif self._pullback_grad_clip > 0:
                scale = torch.clamp(
                    self._pullback_grad_clip /
                    torch.clamp_min(grad_norm, 1e-8), max=1.0)
                clipped_count += torch.sum((scale < 1.0).float())
                action_grad = action_grad * scale

            grads.append(action_grad)
            reward_sum += torch.sum(rewards.detach())
            sample_count += end - start

        action_grad = torch.cat(grads, dim=0)
        denom = max(sample_count, 1)
        return (action_grad,
                reward_sum / denom,
                clipped_count / denom,
                torch.sqrt(raw_grad_sq_sum / (denom * action_grad.shape[-1])),
                raw_grad_norm_sum / denom,
                active_count / denom)
