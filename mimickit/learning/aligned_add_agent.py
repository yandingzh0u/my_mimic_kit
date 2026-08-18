import learning.add_agent as add_agent
import learning.aligned_obs_normalizer as aligned_obs_normalizer
import util.torch_util as torch_util


class AlignedADDAgent(add_agent.ADDAgent):
    """Stock ADD with a factorized feedback-feedforward policy command."""

    def _build_normalizers(self):
        # Build the untouched ADD DiffNormalizer first.
        super()._build_normalizers()

        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self_dim = self._env.get_aligned_self_obs_dim()
        command_dim = self._env.get_aligned_command_dim()

        obs_norm = aligned_obs_normalizer.AlignedObsNormalizer(
            self_dim=self_dim, command_dim=command_dim,
            device=self._device, dtype=obs_dtype)
        obs_norm.set_error_normalizer(self._disc_obs_norm)
        self._obs_norm = obs_norm
