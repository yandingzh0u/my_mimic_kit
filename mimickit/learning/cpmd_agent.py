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

        total_dim = env.get_disc_obs_space().shape[0]
        assert self._state_dim + self._summary_dim + self._interaction_dim == total_dim

        Logger.print("CPMD differential: state {} + summary {} + interactions {} = {} (rho {:.5f})".format(
            self._state_dim, self._summary_dim, self._interaction_dim, total_dim,
            env.get_cpmd_memory_decay()))
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

        info = {
            "disc_state_rms": rms(diff_obs[..., :s]),
            "disc_state_rms_norm": rms(norm_diff_obs[..., :s]),
            "disc_summary_rms": rms(diff_obs[..., s:s + d]),
            "disc_summary_rms_norm": rms(norm_diff_obs[..., s:s + d]),
            "disc_interaction_rms": interactions_rms,
            "disc_interaction_rms_norm": rms(norm_diff_obs[..., s + d:]),
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
