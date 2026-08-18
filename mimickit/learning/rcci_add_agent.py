import learning.add_agent as add_agent
import learning.rcci_obs_normalizer as rcci_obs_normalizer
import util.torch_util as torch_util


class RCCIADDAgent(add_agent.ADDAgent):
    """Stock ADD with one of the strict information-equivalent RCCI inputs."""

    def _build_normalizers(self):
        # Keep ADD's action and discriminator normalizers unchanged.  Replay
        # capacity handling lives in the common AMP/ADD implementation.
        add_agent.ADDAgent._build_normalizers(self)

        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        phi_mean, phi_std = self._env.get_rcci_phi_stats()
        self._obs_norm = rcci_obs_normalizer.RCCIObsNormalizer(
            self_dim=self._env.get_rcci_self_obs_dim(),
            phi_mean=phi_mean,
            phi_std=phi_std,
            representation=self._env.get_rcci_representation(),
            device=self._device,
            dtype=obs_dtype)
