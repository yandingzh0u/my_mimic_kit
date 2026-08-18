import torch

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

    def _store_disc_replay_data(self):
        """Stock ADD replay sampling with a safe first push at 8192 envs."""
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")

        n = disc_obs.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if self._disc_buffer.is_full():
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = min(n, self._disc_buffer.get_capacity())

        idx = rand_idx[:num_samples]
        disc_data = {
            "disc_obs": disc_obs[idx].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[idx].unsqueeze(1),
        }
        self._disc_buffer.push(disc_data)
