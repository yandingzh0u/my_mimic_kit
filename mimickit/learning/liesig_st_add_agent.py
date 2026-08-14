import torch

import learning.add_agent as add_agent
from util.logger import Logger

class LieSigSTADDAgent(add_agent.ADDAgent):
    """ADD agent for the LieSig differential: diagnostics only.

    Training is deliberately stock ADD and nothing here may change it: one
    discriminator (ADDModel), one zero-vs-policy BCE with logit regularization
    and gradient penalty, one reward r = scale * softplus(z). No branches, no
    fusion, no auxiliary losses, no extra reward terms.

    The only addition is instrumentation of the new differential blocks
    [state | level-1 | level-2], so a failure can be attributed to the
    representation rather than guessed at: block RMS before and after the
    scale-only normalizer, the effective support of the level-2 block, and
    how much of the normalizer sits on its floor.
    """

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._state_dim = env.get_disc_state_obs_dim()
        self._tangent_dim = env.get_liesig_tangent_dim()
        self._area_dim = env.get_liesig_area_dim()

        total_dim = env.get_disc_obs_space().shape[0]
        assert self._state_dim + self._tangent_dim + self._area_dim == total_dim

        Logger.print("LieSig differential: state {} + level1 {} + level2 {} = {} (order {}, rho {:.5f})".format(
            self._state_dim, self._tangent_dim, self._area_dim, total_dim,
            env.get_liesig_order(), env.get_liesig_memory_decay()))
        return

    def _compute_disc_loss(self, batch):
        disc_info = super()._compute_disc_loss(batch)
        with torch.no_grad():
            diff_obs = batch["disc_obs_demo"] - batch["disc_obs"]
            disc_info.update(self._compute_disc_diagnostics(diff_obs))
        return disc_info

    def _compute_disc_diagnostics(self, diff_obs):
        """Block statistics of the differential. Pure measurement: it never
        touches the loss, the reward, or the normalizer state."""
        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)

        s = self._state_dim
        d = self._tangent_dim

        def rms(x):
            return torch.sqrt(torch.mean(torch.square(x)))

        info = {
            "disc_state_rms": rms(diff_obs[..., :s]),
            "disc_state_rms_norm": rms(norm_diff_obs[..., :s]),
            "disc_level1_rms": rms(diff_obs[..., s:s + d]),
            "disc_level1_rms_norm": rms(norm_diff_obs[..., s:s + d]),
        }

        if (self._area_dim > 0):
            level2 = diff_obs[..., s + d:]
            level2_rms = rms(level2)
            info["disc_level2_rms"] = level2_rms
            info["disc_level2_rms_norm"] = rms(norm_diff_obs[..., s + d:])
            # effective support: entries above 10% of the block RMS. Near zero
            # means the level-2 block rides on a handful of coordinates.
            thresh = 0.1 * torch.clamp_min(level2_rms, 1e-8)
            info["disc_level2_nonzero_frac"] = torch.mean((torch.abs(level2) > thresh).float())

        # how much of the scale-only normalizer sits on its floor: those
        # coordinates are effectively unnormalized
        abs_mean = self._disc_obs_norm.get_abs_mean()
        min_diff = self._disc_obs_norm._min_diff
        info["disc_norm_min_scale_frac"] = torch.mean((abs_mean <= min_diff).float())

        return info
