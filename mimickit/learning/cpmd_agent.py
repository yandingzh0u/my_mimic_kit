import torch

import learning.add_agent as add_agent
from util.logger import Logger

class CPMDAgent(add_agent.ADDAgent):
    """ADD agent for the CPMD differential: diagnostics only.

    Training is deliberately stock ADD and nothing here may change it: one
    discriminator (ADDModel), one zero-vs-policy BCE with logit regularization
    and gradient penalty, one reward r = scale * softplus(z). No branches, no
    fusion, no auxiliary losses, no extra reward terms, and a single training
    run: the state differential, the motion summary and the context
    interactions all enter the same discriminator on every step.

    The only addition is instrumentation of the three differential blocks
    [state | summary | interactions], so a failure can be attributed to the
    representation rather than guessed at: block RMS before and after the
    scale-only normalizer, the effective support of the interaction block, and
    how much of the normalizer sits on its floor.
    """

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._state_dim = env.get_disc_state_obs_dim()
        self._summary_dim = env.get_cpmd_summary_dim()
        self._interaction_dim = env.get_cpmd_interaction_dim()
        self._dof_dim = env.get_cpmd_dof_dim()
        self._state_vel_dim = 6 + self._dof_dim
        self._memory_seconds = env.get_cpmd_memory_seconds()
        self._mean_motion_length = env.get_cpmd_mean_motion_length()
        self._memory_to_motion_ratio = self._memory_seconds / self._mean_motion_length

        total_dim = env.get_disc_obs_space().shape[0]
        assert self._state_dim + self._summary_dim + self._interaction_dim == total_dim

        Logger.print("CPMD differential: state {} + summary {} + interactions {} = {} (rho {:.5f})".format(
            self._state_dim, self._summary_dim, self._interaction_dim, total_dim,
            env.get_cpmd_memory_decay()))
        Logger.print("CPMD memory: {:.5f}s / mean motion {:.5f}s = {:.5f} cycles".format(
            self._memory_seconds, self._mean_motion_length,
            self._memory_to_motion_ratio))
        return

    def _compute_disc_loss(self, batch):
        disc_info = super()._compute_disc_loss(batch)
        with torch.no_grad():
            disc_info.update(self._compute_disc_diagnostics(batch["disc_diff"]))
        return disc_info

    def _compute_disc_diagnostics(self, diff_obs):
        """Block statistics of the differential. Pure measurement: it never
        touches the loss, the reward, or the normalizer state."""
        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)

        s = self._state_dim
        d = self._summary_dim

        def rms(x):
            return torch.sqrt(torch.mean(torch.square(x)))

        interactions = diff_obs[..., s + d:]
        interactions_rms = rms(interactions)

        norm_state = norm_diff_obs[..., :s]
        norm_summary = norm_diff_obs[..., s:s + d]
        norm_interactions = norm_diff_obs[..., s + d:]
        state_energy = torch.sum(torch.square(norm_state))
        summary_energy = torch.sum(torch.square(norm_summary))
        interaction_energy = torch.sum(torch.square(norm_interactions))
        total_energy = torch.clamp_min(
            state_energy + summary_energy + interaction_energy, 1e-12)

        info = {
            "disc_state_rms": rms(diff_obs[..., :s]),
            "disc_state_rms_norm": rms(norm_diff_obs[..., :s]),
            "disc_summary_rms": rms(diff_obs[..., s:s + d]),
            "disc_summary_rms_norm": rms(norm_diff_obs[..., s:s + d]),
            "disc_interaction_rms": interactions_rms,
            "disc_interaction_rms_norm": rms(norm_diff_obs[..., s + d:]),
            "disc_state_energy_frac": state_energy / total_energy,
            "disc_summary_energy_frac": summary_energy / total_energy,
            "disc_interaction_energy_frac": interaction_energy / total_energy,
            "disc_state_abs_max_norm": torch.max(torch.abs(norm_state)),
            "disc_summary_abs_max_norm": torch.max(torch.abs(norm_summary)),
            "disc_interaction_abs_max_norm": torch.max(torch.abs(norm_interactions)),
            "cpmd_memory_seconds": torch.tensor(self._memory_seconds, device=diff_obs.device),
            "cpmd_motion_length_mean": torch.tensor(self._mean_motion_length, device=diff_obs.device),
            "cpmd_memory_to_motion_ratio": torch.tensor(self._memory_to_motion_ratio, device=diff_obs.device),
        }

        # effective support: entries above 10% of the block RMS. Near zero
        # means the interaction block rides on a handful of coordinates.
        thresh = 0.1 * torch.clamp_min(interactions_rms, 1e-8)
        info["disc_interaction_nonzero_frac"] = torch.mean((torch.abs(interactions) > thresh).float())

        # how much of the scale-only normalizer sits on its floor: those
        # coordinates are effectively unnormalized
        abs_mean = self._disc_obs_norm.get_abs_mean()
        min_diff = self._disc_obs_norm._min_diff
        info["disc_norm_min_scale_frac"] = torch.mean((abs_mean <= min_diff).float())

        return info

    def _compute_disc_input_diagnostics(self, norm_diff_obs, disc_neg_grad,
                                        disc_neg_logit):
        """Attribute ADD's existing input gradient to CPMD's three blocks.

        This reuses ``disc_neg_grad`` already required by the gradient penalty;
        it performs no extra forward/backward pass and cannot change the loss.
        Fractions sum to one (up to floating-point error).  Velocity fractions
        identify the original ADD instantaneous velocity coordinates inside
        the 172-D state block.
        """
        with torch.no_grad():
            s = self._state_dim
            d = self._summary_dim

            grad_sq = torch.square(disc_neg_grad.detach())
            state_grad = torch.sum(grad_sq[..., :s])
            summary_grad = torch.sum(grad_sq[..., s:s + d])
            interaction_grad = torch.sum(grad_sq[..., s + d:])
            total_grad = torch.clamp_min(
                state_grad + summary_grad + interaction_grad, 1e-12)

            state_vel_start = s - self._state_vel_dim
            dof_vel_start = s - self._dof_dim
            state_vel_grad = torch.sum(grad_sq[..., state_vel_start:s])
            dof_vel_grad = torch.sum(grad_sq[..., dof_vel_start:s])

            logits = disc_neg_logit.detach()
            # Before the numerical reward cap, dr/dz = scale * sigmoid(z).
            reward_slope = self._disc_reward_scale * torch.sigmoid(logits)

            return {
                "disc_grad_state_frac": state_grad / total_grad,
                "disc_grad_summary_frac": summary_grad / total_grad,
                "disc_grad_interaction_frac": interaction_grad / total_grad,
                "disc_grad_state_vel_frac": state_vel_grad / total_grad,
                "disc_grad_dof_vel_frac": dof_vel_grad / total_grad,
                "disc_reward_logit_slope_mean": torch.mean(reward_slope),
                "disc_neg_logit_lt_m5_frac": torch.mean((logits < -5.0).float()),
            }
