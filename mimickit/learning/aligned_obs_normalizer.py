import torch

import learning.normalizer as normalizer


class AlignedObsNormalizer(torch.nn.Module):
    """Block normalizer for [self observation, PRDA command].

    The command block deliberately reuses the ADD discriminator's
    DiffNormalizer.  This object does not register or update that shared
    normalizer; ADDAgent remains its single owner.  Therefore the actor command
    and the reward residual use identical per-coordinate scales.
    """

    def __init__(self, self_dim, command_dim, device, dtype=torch.float):
        super().__init__()
        self._self_dim = int(self_dim)
        self._command_dim = int(command_dim)
        self._total_dim = self._self_dim + self._command_dim

        self._self_norm = normalizer.Normalizer(
            [self._self_dim], clip=10.0, device=device, dtype=dtype)
        self._command_norm_ref = None

    def set_command_normalizer(self, command_normalizer):
        if tuple(command_normalizer.get_shape()) != (self._command_dim,):
            raise ValueError("ADD normalizer shape does not match PRDA command size")
        # Avoid registering a second alias of the discriminator normalizer in
        # state_dict.  ADDAgent owns, saves, loads, and updates it.
        object.__setattr__(self, "_command_norm_ref", command_normalizer)

    def record(self, obs):
        self_obs, _ = self._split(obs)
        self._self_norm.record(self_obs)

    def update(self):
        self._self_norm.update()

    def normalize(self, obs):
        self_obs, command = self._split(obs)
        if self._command_norm_ref is None:
            raise RuntimeError("ADD command normalizer has not been attached")

        norm_self = self._self_norm.normalize(self_obs)
        norm_command = self._command_norm_ref.normalize(command)
        return torch.cat([norm_self, norm_command], dim=-1)

    def get_mean(self):
        zeros_command = torch.zeros(
            [self._command_dim], device=self._self_norm.get_mean().device,
            dtype=self._self_norm.get_mean().dtype)
        return torch.cat([self._self_norm.get_mean(), zeros_command], dim=-1)

    def get_std(self):
        if self._command_norm_ref is None:
            raise RuntimeError("ADD command normalizer has not been attached")
        command_scale = torch.clamp_min(self._command_norm_ref.get_abs_mean(), 1e-4)
        return torch.cat([self._self_norm.get_std(), command_scale], dim=-1)

    def get_shape(self):
        return torch.Size([self._total_dim])

    def _split(self, obs):
        if obs.shape[-1] != self._total_dim:
            raise ValueError(
                "aligned observation has size {}, expected {}".format(
                    obs.shape[-1], self._total_dim))
        i0 = self._self_dim
        return obs[..., :i0], obs[..., i0:]
