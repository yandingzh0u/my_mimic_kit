import math

import torch

import learning.aligned_add_agent as aligned_add_agent
import learning.phase_transition_model as phase_transition_model
import util.torch_util as torch_util


class PhaseTransitionCriticAgent(aligned_add_agent.AlignedADDAgent):
    """Aligned actor with a phase-matched Wasserstein transition critic.

    Every policy transition is paired with the reference transition at the
    exact same rollout step.  The actor still observes ``[self, e_t, m_t]``;
    only the learned reward changes.
    """

    def _load_params(self, config):
        super()._load_params(config)
        if self._disc_logit_reg != 0:
            raise ValueError(
                "phase transition critic is offset-invariant; "
                "disc_logit_reg must be zero")

        self._transition_gp_target = float(
            config.get("transition_gp_target", 1.0))
        reward_tau = config.get("transition_reward_tau", None)
        transition_dim = 2 * self._env.get_aligned_command_dim()
        # A standardized D-dimensional residual has Euclidean radius O(sqrt
        # D).  Normalize the Wasserstein term to an RMS distance so its scale
        # remains commensurate with a dimension-independent unit-gradient GP.
        self._transition_wasserstein_scale = math.sqrt(transition_dim)
        if reward_tau is None:
            reward_tau = self._transition_wasserstein_scale
        self._transition_reward_tau = float(reward_tau)
        self._transition_input_clip = float(
            config.get("transition_input_clip", 10.0))
        self._transition_private_seed = int(
            config.get("transition_private_seed", 918273))
        self._phase_shuffle_min_distance = float(
            config.get("phase_shuffle_min_distance", 0.1))
        self._phase_shuffle_min_rms = float(
            config.get("phase_shuffle_min_rms", 1e-3))
        if not math.isfinite(self._disc_grad_penalty) \
                or self._disc_grad_penalty < 0:
            raise ValueError(
                "disc_grad_penalty must be finite and nonnegative")
        if not math.isfinite(self._transition_gp_target) \
                or self._transition_gp_target <= 0:
            raise ValueError(
                "transition_gp_target must be finite and positive")
        if not math.isfinite(self._transition_reward_tau) \
                or self._transition_reward_tau <= 0:
            raise ValueError(
                "transition_reward_tau must be finite and positive")
        if not math.isfinite(self._transition_input_clip) \
                or self._transition_input_clip <= 0:
            raise ValueError(
                "transition_input_clip must be finite and positive")
        if self._transition_private_seed < 0:
            raise ValueError("transition_private_seed must be nonnegative")
        if not 0 < self._phase_shuffle_min_distance <= 0.5:
            raise ValueError(
                "phase_shuffle_min_distance must lie in (0, 0.5]")
        if not math.isfinite(self._phase_shuffle_min_rms) \
                or self._phase_shuffle_min_rms <= 0:
            raise ValueError(
                "phase_shuffle_min_rms must be finite and positive")
        if not math.isfinite(self._disc_reward_scale) \
                or self._disc_reward_scale <= 0:
            raise ValueError("disc_reward_scale must be finite and positive")

    def _build_normalizers(self):
        # Preserve Aligned ADD's actor normalizers, including the shared
        # online error scale.  The critic itself uses fixed reference-only
        # statistics registered below.
        super()._build_normalizers()
        stats = self._env.get_phase_transition_reference_stats()
        if len(stats) != 4:
            raise ValueError(
                "reference transition statistics must contain four tensors")
        state_mean, state_scale, motion_mean, motion_scale = stats

        expected_dim = self._env.get_aligned_command_dim()
        for name, value in (
            ("state_mean", state_mean),
            ("state_scale", state_scale),
            ("motion_mean", motion_mean),
            ("motion_scale", motion_scale),
        ):
            if tuple(value.shape) != (expected_dim,):
                raise ValueError(
                    "{} must have shape ({},), got {}".format(
                        name, expected_dim, tuple(value.shape)))
            if not torch.all(torch.isfinite(value)):
                raise ValueError("{} contains non-finite values".format(name))
        if torch.any(state_scale <= 0) or torch.any(motion_scale <= 0):
            raise ValueError("reference transition scales must be positive")

        self.register_buffer(
            "_transition_state_mean", state_mean.detach().clone(),
            persistent=True)
        self.register_buffer(
            "_transition_state_scale", state_scale.detach().clone(),
            persistent=True)
        self.register_buffer(
            "_transition_motion_mean", motion_mean.detach().clone(),
            persistent=True)
        self.register_buffer(
            "_transition_motion_scale", motion_scale.detach().clone(),
            persistent=True)
        # A checkpointed counter seeds a critic-private generator.  GP
        # interpolation never consumes the global RNG stream used by PPO
        # exploration, and strict resume continues the same sequence.
        self.register_buffer(
            "_transition_private_counter",
            torch.zeros((), device=state_mean.device, dtype=torch.int64),
            persistent=True,
        )

    def _build_model(self, config):
        self._model = phase_transition_model.PhaseTransitionCriticModel(
            config["model"], self._env)

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        metadata = self._env.get_phase_transition_metadata()
        for name, value in metadata.items():
            self._exp_buffer.record(name, value)

    def _store_disc_replay_data(self):
        transition = self._build_rollout_transition()
        idx = self._sample_disc_replay_indices(
            transition["sim_state"].shape[0])
        replay_data = {
            name: value[idx].unsqueeze(1)
            for name, value in transition.items()
        }
        self._disc_buffer.push(replay_data)

    def _compute_rewards(self):
        task_reward = self._exp_buffer.get_data_flat("reward")
        transition = self._build_rollout_transition()
        transition_error, reference_context = self._normalize_transition(
            **transition)
        disc_reward, anchored_score = self._calc_transition_rewards(
            transition_error, reference_context)

        reward = (
            self._task_reward_weight * task_reward
            + self._disc_reward_weight * disc_reward
        )
        self._exp_buffer.set_data_flat("reward", reward)

        # This remains the sole online scale used by the aligned actor's
        # feedback block.  Critic normalization is fixed and independent.
        if self._need_normalizer_update():
            next_error = (
                self._exp_buffer.get_data_flat("disc_obs_demo")
                - self._exp_buffer.get_data_flat("disc_obs")
            )
            self._disc_obs_norm.record(next_error)

        reward_std, reward_mean = torch.std_mean(disc_reward)
        score_std, score_mean = torch.std_mean(anchored_score)
        return {
            "disc_reward_mean": reward_mean,
            "disc_reward_std": reward_std,
            "transition_advantage_mean": score_mean,
            "transition_advantage_std": score_std,
            "transition_anchor_hit_frac": torch.mean(
                (torch.abs(anchored_score) < 1e-6).float()),
        }

    def _compute_disc_loss(self, batch):
        current = self._build_transition_from_batch(batch)
        replay = self._disc_buffer.sample(current["sim_state"].shape[0])
        transition = {
            name: torch.cat([value, replay[name]], dim=0)
            for name, value in current.items()
        }
        policy_error, reference_context = self._normalize_transition(
            **transition)
        (shuffle_error, shuffle_context, shuffle_indices,
         phase_distance, shuffle_valid_frac) = (
            self._build_phase_shuffle_error(
                transition, reference_context)
        )

        zero_error = torch.zeros_like(policy_error)
        ref_score = self._model.eval_transition_score(
            zero_error, reference_context).squeeze(-1)
        policy_score = self._model.eval_transition_score(
            policy_error, reference_context).squeeze(-1)
        policy_advantage = policy_score - ref_score
        has_shuffle = shuffle_error.shape[0] > 0
        if has_shuffle:
            shuffle_score = self._model.eval_transition_score(
                shuffle_error, shuffle_context).squeeze(-1)
            shuffle_advantage = (
                shuffle_score - ref_score[shuffle_indices]
            )
            wasserstein_loss = 0.5 * (
                torch.mean(policy_advantage)
                + torch.mean(shuffle_advantage)
            )
        else:
            shuffle_advantage = policy_advantage.new_empty([0])
            wasserstein_loss = torch.mean(policy_advantage)
        wasserstein_objective = (
            wasserstein_loss / self._transition_wasserstein_scale)

        if self._disc_grad_penalty > 0:
            policy_alpha, shuffle_alpha = self._private_gp_alphas(
                policy_error, shuffle_error)
            policy_grad_norm = self._calc_interp_grad_norm(
                policy_error, reference_context, policy_alpha)
            policy_gp = torch.mean(torch.square(
                policy_grad_norm - self._transition_gp_target))
            if has_shuffle:
                shuffle_grad_norm = self._calc_interp_grad_norm(
                    shuffle_error, shuffle_context, shuffle_alpha)
                shuffle_gp = torch.mean(torch.square(
                    shuffle_grad_norm - self._transition_gp_target))
                grad_penalty = 0.5 * (policy_gp + shuffle_gp)
                all_grad_norm = torch.cat(
                    [policy_grad_norm, shuffle_grad_norm], dim=0)
            else:
                grad_penalty = policy_gp
                all_grad_norm = policy_grad_norm
            disc_loss = (
                wasserstein_objective
                + self._disc_grad_penalty * grad_penalty
            )
            gp_norm_mean = torch.mean(all_grad_norm)
            gp_norm_std = torch.std(all_grad_norm)
        else:
            # The zero-weight ablation must not build a second-order graph.
            grad_penalty = torch.zeros(
                (), device=policy_error.device, dtype=policy_error.dtype)
            disc_loss = wasserstein_objective
            gp_norm_mean = torch.zeros_like(grad_penalty)
            gp_norm_std = torch.zeros_like(grad_penalty)

        with torch.no_grad():
            detached_ref_score = ref_score.detach()
            fake_score = detached_ref_score + policy_advantage.detach()
            if has_shuffle:
                shuffled_score = (
                    detached_ref_score[shuffle_indices]
                    + shuffle_advantage.detach()
                )
                shuffle_score_mean = torch.mean(shuffled_score)
                shuffle_advantage_mean = torch.mean(shuffle_advantage)
                shuffle_acc = torch.mean(
                    (shuffle_advantage < 0).float())
                phase_margin = torch.mean(
                    policy_advantage[shuffle_indices]
                    - shuffle_advantage)
                phase_distance_mean = torch.mean(phase_distance)
            else:
                zero = torch.zeros_like(torch.mean(policy_advantage))
                shuffle_score_mean = zero
                shuffle_advantage_mean = zero
                shuffle_acc = zero
                phase_margin = zero
                phase_distance_mean = zero
        return {
            "disc_loss": disc_loss,
            "disc_wasserstein_loss": wasserstein_loss.detach(),
            "disc_wasserstein_objective": (
                wasserstein_objective.detach()),
            "disc_grad_penalty": grad_penalty.detach(),
            "disc_gp_grad_norm": gp_norm_mean.detach(),
            "disc_gp_grad_norm_std": gp_norm_std.detach(),
            "disc_ref_score": torch.mean(detached_ref_score),
            "disc_agent_score": torch.mean(fake_score).detach(),
            "disc_shuffle_score": shuffle_score_mean.detach(),
            "disc_policy_advantage": torch.mean(
                policy_advantage).detach(),
            "disc_shuffle_advantage": shuffle_advantage_mean.detach(),
            "disc_agent_acc": torch.mean(
                (policy_advantage < 0).float()).detach(),
            "disc_shuffle_acc": shuffle_acc.detach(),
            "disc_phase_margin": phase_margin.detach(),
            "disc_shuffle_valid_frac": shuffle_valid_frac.detach(),
            "disc_shuffle_phase_distance": phase_distance_mean.detach(),
        }

    def _build_phase_shuffle_error(self, transition, reference_context):
        partner, valid, phase_distance = build_phase_derangement(
            motion_id=transition["motion_id"],
            motion_phase=transition["motion_phase"],
            motion_is_wrap=transition["motion_is_wrap"],
            min_phase_distance=self._phase_shuffle_min_distance,
        )
        shuffle_error, _ = self._normalize_transition(
            sim_state=transition["ref_state"][partner],
            sim_motion=transition["ref_motion"][partner],
            ref_state=transition["ref_state"],
            ref_motion=transition["ref_motion"],
        )
        shuffle_rms = torch.sqrt(torch.mean(
            torch.square(shuffle_error), dim=-1))
        valid = valid & (shuffle_rms > self._phase_shuffle_min_rms)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        valid_frac = torch.mean(valid.float())
        return (
            shuffle_error[valid_indices],
            reference_context[valid_indices],
            valid_indices,
            phase_distance[valid_indices],
            valid_frac,
        )

    def _private_gp_alphas(self, policy_error, shuffle_error):
        generator = torch.Generator(device=policy_error.device)
        counter = int(self._transition_private_counter.item())
        generator.manual_seed(self._transition_private_seed + counter)
        alpha_shape = [policy_error.shape[0]] + [
            1 for _ in policy_error.shape[1:]
        ]
        policy_alpha = torch.rand(
            alpha_shape,
            device=policy_error.device,
            dtype=policy_error.dtype,
            generator=generator,
        )
        shuffle_alpha_shape = [shuffle_error.shape[0]] + [
            1 for _ in shuffle_error.shape[1:]
        ]
        shuffle_alpha = torch.rand(
            shuffle_alpha_shape,
            device=shuffle_error.device,
            dtype=shuffle_error.dtype,
            generator=generator,
        )
        self._transition_private_counter.add_(1)
        return policy_alpha, shuffle_alpha

    def _calc_interp_grad_norm(self, transition_error, reference_context,
                               alpha):
        interp_error = (alpha * transition_error).detach()
        interp_error.requires_grad_(True)
        interp_score = self._model.eval_transition_score(
            interp_error, reference_context).squeeze(-1)
        interp_grad = torch.autograd.grad(
            interp_score,
            interp_error,
            grad_outputs=torch.ones_like(interp_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return torch.linalg.vector_norm(interp_grad, dim=-1)

    def _build_rollout_transition(self):
        transition = reconstruct_phase_matched_transition(
            obs=self._exp_buffer.get_data_flat("obs"),
            next_sim_state=self._exp_buffer.get_data_flat("disc_obs"),
            next_ref_state=self._exp_buffer.get_data_flat("disc_obs_demo"),
            self_dim=self._env.get_aligned_self_obs_dim(),
            command_dim=self._env.get_aligned_command_dim(),
        )
        for name in ("motion_id", "motion_phase", "motion_is_wrap"):
            transition[name] = self._exp_buffer.get_data_flat(name)
        return transition

    def _build_transition_from_batch(self, batch):
        transition = reconstruct_phase_matched_transition(
            obs=batch["obs"],
            next_sim_state=batch["disc_obs"],
            next_ref_state=batch["disc_obs_demo"],
            self_dim=self._env.get_aligned_self_obs_dim(),
            command_dim=self._env.get_aligned_command_dim(),
        )
        for name in ("motion_id", "motion_phase", "motion_is_wrap"):
            transition[name] = batch[name]
        return transition

    def _normalize_transition(self, sim_state, sim_motion, ref_state,
                              ref_motion, motion_id=None, motion_phase=None,
                              motion_is_wrap=None):
        return normalize_phase_matched_transition(
            sim_state=sim_state,
            sim_motion=sim_motion,
            ref_state=ref_state,
            ref_motion=ref_motion,
            state_mean=self._transition_state_mean,
            state_scale=self._transition_state_scale,
            motion_mean=self._transition_motion_mean,
            motion_scale=self._transition_motion_scale,
            clip=self._transition_input_clip,
        )

    def _calc_transition_rewards(self, transition_error,
                                 reference_context):
        with torch.no_grad():
            inputs = {
                "transition_error": transition_error,
                "reference_context": reference_context,
            }
            anchored_score = torch_util.eval_minibatch(
                self._model.eval_anchored_score,
                inputs,
                self._disc_eval_batch_size,
            ).squeeze(-1)
            reward = anchored_transition_reward(
                anchored_score,
                scale=self._disc_reward_scale,
                tau=self._transition_reward_tau,
            )
        return reward, anchored_score

    def _calc_disc_rewards(self, transition_error, reference_context=None):
        """Compatibility entry point that rejects missing phase context."""
        if reference_context is None:
            raise ValueError(
                "phase transition reward requires the paired reference context")
        reward, _ = self._calc_transition_rewards(
            transition_error, reference_context)
        return reward

    def calc_policy_reward_from_transition(self, obs, next_info, env_reward):
        """Reconstruct the exact optimized reward for offline evaluation."""
        transition = reconstruct_phase_matched_transition(
            obs=obs,
            next_sim_state=next_info["disc_obs"],
            next_ref_state=next_info["disc_obs_demo"],
            self_dim=self._env.get_aligned_self_obs_dim(),
            command_dim=self._env.get_aligned_command_dim(),
        )
        transition_error, reference_context = self._normalize_transition(
            **transition)
        disc_reward, _ = self._calc_transition_rewards(
            transition_error, reference_context)
        return (
            self._task_reward_weight * env_reward
            + self._disc_reward_weight * disc_reward
        )


def reconstruct_phase_matched_transition(obs, next_sim_state,
                                         next_ref_state, self_dim,
                                         command_dim):
    """Recover the exact pre/post policy and reference transition.

    ``obs`` is the raw pre-step aligned observation.  ``next_*`` tensors are
    recorded from ``info`` immediately after physics and before any reset.
    """
    self_dim = int(self_dim)
    command_dim = int(command_dim)
    expected_obs_dim = self_dim + 2 * command_dim
    if obs.shape[-1] != expected_obs_dim:
        raise ValueError(
            "expected aligned observation size {}, got {}".format(
                expected_obs_dim, obs.shape[-1]))
    if next_sim_state.shape != next_ref_state.shape:
        raise ValueError("next simulation and reference states must match")
    if next_sim_state.shape[:-1] != obs.shape[:-1] \
            or next_sim_state.shape[-1] != command_dim:
        raise ValueError("step tensors do not match the aligned observation")

    error_start = self_dim
    motion_start = error_start + command_dim
    curr_error = obs[..., error_start:motion_start]
    ref_motion = obs[..., motion_start:]

    ref_state = next_ref_state - ref_motion
    sim_state = ref_state - curr_error
    sim_motion = next_sim_state - sim_state
    return {
        "sim_state": sim_state,
        "sim_motion": sim_motion,
        "ref_state": ref_state,
        "ref_motion": ref_motion,
    }


def normalize_phase_matched_transition(sim_state, sim_motion, ref_state,
                                       ref_motion, state_mean, state_scale,
                                       motion_mean, motion_scale, clip=10.0):
    """Construct the normalized transition error ``u`` and context ``c``."""
    tensors = (sim_state, sim_motion, ref_state, ref_motion)
    if any(value.shape != ref_state.shape for value in tensors):
        raise ValueError("all paired transition tensors must share one shape")
    if torch.any(state_scale <= 0) or torch.any(motion_scale <= 0):
        raise ValueError("transition scales must be strictly positive")
    if not math.isfinite(float(clip)) or float(clip) <= 0:
        raise ValueError("clip must be finite and positive")

    curr_state_error = ref_state - sim_state
    motion_error = ref_motion - sim_motion
    # This closure form is algebraically identical to
    # (ref_t + m_t) - (sim_t + delta_t), while preserving an exact floating
    # point zero for an exactly matched reference transition.
    next_state_error = curr_state_error + motion_error
    transition_error = torch.cat([
        next_state_error / state_scale,
        motion_error / motion_scale,
    ], dim=-1)
    reference_context = torch.cat([
        (ref_state - state_mean) / state_scale,
        (ref_motion - motion_mean) / motion_scale,
    ], dim=-1)
    transition_error = torch.clamp(
        transition_error, min=-float(clip), max=float(clip))
    reference_context = torch.clamp(
        reference_context, min=-float(clip), max=float(clip))
    return transition_error, reference_context


def build_phase_derangement(motion_id, motion_phase, motion_is_wrap,
                            min_phase_distance=0.1):
    """Pair each row with a far phase of the same reference motion.

    Rows are grouped by motion, ordered by phase, and circularly shifted by
    half the group.  Invalid small groups or pairs below the requested phase
    distance remain marked false and are excluded from the shuffle loss/GP.
    No random-number generator is touched.
    """
    if motion_id.ndim != 1 or motion_phase.ndim != 1 \
            or motion_is_wrap.ndim != 1:
        raise ValueError("phase metadata must be one-dimensional")
    if not (motion_id.shape == motion_phase.shape == motion_is_wrap.shape):
        raise ValueError("phase metadata tensors must share a shape")
    if not 0 < float(min_phase_distance) <= 0.5:
        raise ValueError("min_phase_distance must lie in (0, 0.5]")

    num_samples = motion_id.shape[0]
    row = torch.arange(num_samples, device=motion_id.device)
    partner = row.clone()
    valid = torch.zeros(
        num_samples, device=motion_id.device, dtype=torch.bool)
    distance = torch.zeros_like(motion_phase)

    for curr_motion_id in torch.unique(motion_id):
        group = torch.nonzero(
            motion_id == curr_motion_id, as_tuple=False).flatten()
        group_size = group.numel()
        if group_size < 2:
            continue

        ordered = group[torch.argsort(motion_phase[group])]
        shift = max(1, group_size // 2)
        ordered_partner = torch.roll(ordered, shifts=-shift, dims=0)
        partner[ordered] = ordered_partner

        phase_delta = torch.abs(
            motion_phase[ordered] - motion_phase[ordered_partner])
        wrap_rows = motion_is_wrap[ordered].bool()
        phase_delta = torch.where(
            wrap_rows,
            torch.minimum(phase_delta, 1.0 - phase_delta),
            phase_delta,
        )
        distance[ordered] = phase_delta
        valid[ordered] = (
            (ordered_partner != ordered)
            & (phase_delta >= float(min_phase_distance))
        )

    return partner, valid, distance


def anchored_transition_reward(anchored_score, scale=2.0, tau=1.0):
    """Bounded, offset-invariant reward from ``A=F(u,c)-F(0,c)``."""
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError("tau must be finite and positive")
    return float(scale) / (
        1.0 + torch.abs(anchored_score) / float(tau)
    )


def centered_wasserstein_gp_loss(anchored_score, interp_grad_norm,
                                 gp_weight=10.0, gp_target=1.0,
                                 dimension_scale=1.0):
    """Minimized centered-Wasserstein loss with an error-only GP norm."""
    gp_weight = float(gp_weight)
    gp_target = float(gp_target)
    dimension_scale = float(dimension_scale)
    if not math.isfinite(gp_weight) or gp_weight < 0:
        raise ValueError("gp_weight must be finite and nonnegative")
    if not math.isfinite(gp_target) or gp_target <= 0:
        raise ValueError("gp_target must be finite and positive")
    if not math.isfinite(dimension_scale) or dimension_scale <= 0:
        raise ValueError("dimension_scale must be finite and positive")
    wasserstein_loss = torch.mean(anchored_score)
    grad_penalty = torch.mean(torch.square(
        interp_grad_norm - gp_target))
    return (
        wasserstein_loss / dimension_scale
        + gp_weight * grad_penalty
    )
