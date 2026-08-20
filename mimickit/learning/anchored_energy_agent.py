import torch
import torch.nn.functional as functional

import learning.aligned_add_agent as aligned_add_agent
import learning.anchored_energy_model as anchored_energy_model
import util.torch_util as torch_util


class AnchoredEnergyADDAgent(aligned_add_agent.AlignedADDAgent):
    """Aligned policy conditioning with a zero-anchored learned energy.

    The policy observation is unchanged from Aligned ADD: ``[self, e_t,
    m_t]``.  Only the learned scalar objective changes.  The objective sees
    the post-action residual ``e_{t+1}`` together with the exactly paired
    pre-action reference state and reference increment.  Its metric is
    positive definite and trace normalized for every reference context, so
    zero residual is the unique reward maximum throughout training.
    """

    def _load_params(self, config):
        super()._load_params(config)
        if self._disc_grad_penalty != 0:
            raise ValueError(
                "anchored energy has a structural gradient bound; "
                "disc_grad_penalty must be zero")
        if self._disc_logit_reg != 0:
            raise ValueError(
                "anchored energy fixes its metric scale by construction; "
                "disc_logit_reg must be zero")
        if self._disc_reward_scale <= 0:
            raise ValueError("disc_reward_scale must be positive")
        return

    def _build_normalizers(self):
        # Keep the aligned actor normalizers exactly as before.  The energy
        # uses a separate, fixed, demonstration-only scale so its zero anchor
        # cannot drift with the policy distribution.
        super()._build_normalizers()
        ref_mean, ref_scale = self._env.get_reference_phi_stats()
        if ref_mean.shape != ref_scale.shape:
            raise ValueError("reference phi statistics have different shapes")
        if torch.any(ref_scale <= 0):
            raise ValueError("reference phi scale must be strictly positive")

        self.register_buffer(
            "_energy_ref_mean", ref_mean.detach().clone(), persistent=True)
        self.register_buffer(
            "_energy_ref_scale", ref_scale.detach().clone(), persistent=True)
        self._energy_phi_dim = int(ref_mean.numel())
        return

    def _build_model(self, config):
        self._model = anchored_energy_model.AnchoredEnergyModel(
            config["model"], self._env)
        return

    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        obs = self._exp_buffer.get_data_flat("obs")
        ref_motion = self._extract_ref_motion(obs)

        idx = self._sample_disc_replay_indices(disc_obs.shape[0])
        disc_data = {
            "disc_obs": disc_obs[idx].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[idx].unsqueeze(1),
            "ref_motion": ref_motion[idx].unsqueeze(1),
        }
        self._disc_buffer.push(disc_data)
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        obs = self._exp_buffer.get_data_flat("obs")
        ref_motion = self._extract_ref_motion(obs)

        error = disc_obs_demo - disc_obs
        residual, context = self._normalize_energy_inputs(
            error=error, next_ref_obs=disc_obs_demo,
            ref_motion=ref_motion)
        disc_r, energy = self._calc_energy_rewards(residual, context)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        energy_std, energy_mean = torch.std_mean(energy)

        reward = (self._task_reward_weight * task_r
                  + self._disc_reward_weight * disc_r)
        self._exp_buffer.set_data_flat("reward", reward)

        # This online differential scale is retained solely for the aligned
        # actor's error block.  It never enters the anchored energy.
        if self._need_normalizer_update():
            self._disc_obs_norm.record(error)

        return {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        }

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        next_ref_obs = batch["disc_obs_demo"]
        ref_motion = self._extract_ref_motion(batch["obs"])

        replay = self._disc_buffer.sample(disc_obs.shape[0])
        disc_obs = torch.cat([disc_obs, replay["disc_obs"]], dim=0)
        next_ref_obs = torch.cat(
            [next_ref_obs, replay["disc_obs_demo"]], dim=0)
        ref_motion = torch.cat(
            [ref_motion, replay["ref_motion"]], dim=0)

        error = next_ref_obs - disc_obs
        residual, context = self._normalize_energy_inputs(
            error=error, next_ref_obs=next_ref_obs,
            ref_motion=ref_motion)

        energy = self._model.eval_energy(residual, context).squeeze(-1)
        disc_neg_logit = self._model.eval_disc(residual, context)
        disc_neg_logit = disc_neg_logit.squeeze(-1)
        disc_pos_logit = self._model.get_energy_bias().expand_as(
            disc_neg_logit)

        disc_loss_pos = functional.binary_cross_entropy_with_logits(
            disc_pos_logit, torch.ones_like(disc_pos_logit))
        disc_loss_neg = functional.binary_cross_entropy_with_logits(
            disc_neg_logit, torch.zeros_like(disc_neg_logit))
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        residual_sq = torch.sum(torch.square(residual), dim=-1)
        lower_bound = (
            self._model.get_energy_epsilon()
            * residual_sq / (2.0 * self._energy_phi_dim))
        bound_slack = energy - lower_bound

        return {
            "disc_loss": disc_loss,
            "disc_loss_pos": disc_loss_pos.detach(),
            "disc_loss_neg": disc_loss_neg.detach(),
            "disc_grad_penalty": torch.zeros(
                (), device=disc_loss.device, dtype=disc_loss.dtype),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach(),
            "disc_energy": torch.mean(energy).detach(),
            "disc_energy_std": torch.std(energy).detach(),
            "energy_lower_bound_min_slack": torch.min(
                bound_slack).detach(),
            # ``add_torch_dict`` accumulates diagnostics in-place.  A plain
            # detach would still alias the scalar parameter and corrupt it by
            # repeatedly adding the log value back into the model.
            "energy_bias": self._model.get_energy_bias().detach().clone(),
        }

    def _extract_ref_motion(self, obs):
        command_dim = self._env.get_aligned_command_dim()
        self_dim = self._env.get_aligned_self_obs_dim()
        motion_start = self_dim + command_dim
        motion_end = motion_start + command_dim
        if obs.shape[-1] != motion_end:
            raise ValueError(
                "anchored energy expected aligned [self,e,m] observation")
        return obs[..., motion_start:motion_end]

    def _normalize_energy_inputs(self, error, next_ref_obs, ref_motion):
        """Map a paired transition to fixed zero-preserving coordinates.

        ``next_ref_obs`` is the post-action demonstration feature.  Since the
        pre-action command stores ``m_t = ref_{t+1} - ref_t``, subtracting it
        reconstructs the exactly paired current reference without another
        rollout buffer.
        """
        return normalize_energy_inputs(
            error=error, next_ref_obs=next_ref_obs,
            ref_motion=ref_motion, ref_mean=self._energy_ref_mean,
            ref_scale=self._energy_ref_scale)

    def _calc_energy_rewards(self, residual, context):
        with torch.no_grad():
            inputs = {"residual": residual, "context": context}
            energy = torch_util.eval_minibatch(
                self._model.eval_energy, inputs,
                self._disc_eval_batch_size).squeeze(-1)
            disc_r = self._disc_reward_scale / (1.0 + energy)
        return disc_r, energy

    def _calc_disc_rewards(self, norm_disc_obs, norm_context=None):
        """Compatibility entry point for tools that provide the context.

        A context-free call would silently evaluate a different objective, so
        it is rejected instead of fabricating a zero reference command.
        """
        if norm_context is None:
            raise ValueError(
                "anchored energy reward requires the paired reference context")
        reward, _ = self._calc_energy_rewards(norm_disc_obs, norm_context)
        return reward

    def calc_policy_reward_from_transition(self, obs, next_info, env_reward):
        """Reconstruct the optimized reward for an offline evaluator."""
        next_ref_obs = next_info["disc_obs_demo"]
        error = next_ref_obs - next_info["disc_obs"]
        ref_motion = self._extract_ref_motion(obs)
        residual, context = self._normalize_energy_inputs(
            error=error, next_ref_obs=next_ref_obs,
            ref_motion=ref_motion)
        disc_r, _ = self._calc_energy_rewards(residual, context)
        return (self._task_reward_weight * env_reward
                + self._disc_reward_weight * disc_r)


def normalize_energy_inputs(error, next_ref_obs, ref_motion, ref_mean,
                            ref_scale):
    """Pure helper used by tests and offline evaluators."""
    if torch.any(ref_scale <= 0):
        raise ValueError("reference scale must be strictly positive")
    ref_obs = next_ref_obs - ref_motion
    residual = error / ref_scale
    context = torch.cat([
        (ref_obs - ref_mean) / ref_scale,
        ref_motion / ref_scale,
    ], dim=-1)
    return residual, context
