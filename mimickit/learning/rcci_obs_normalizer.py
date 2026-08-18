import torch

import learning.normalizer as normalizer


class RCCIObsNormalizer(torch.nn.Module):
    """Shared fixed affine scaling for strict RCCI paired controls."""

    def __init__(self, self_dim, phi_mean, phi_std, representation, device,
                 dtype=torch.float):
        super().__init__()
        self._self_dim = int(self_dim)
        self._phi_dim = int(phi_mean.numel())
        self._total_dim = self._self_dim + 3 * self._phi_dim
        self._representation = representation

        self._self_norm = normalizer.Normalizer(
            [self._self_dim], clip=10.0, device=device, dtype=dtype)
        self.register_buffer("_phi_mean", phi_mean.to(device=device, dtype=dtype).clone())
        self.register_buffer("_phi_std", phi_std.to(device=device, dtype=dtype).clone())

        if not torch.all(torch.isfinite(self._phi_mean)):
            raise ValueError("RCCI phi mean must be finite")
        if not torch.all(torch.isfinite(self._phi_std)) or not torch.all(self._phi_std > 0):
            raise ValueError("RCCI phi std must be finite and positive")
        if representation not in ("absolute", "residual"):
            raise ValueError("unsupported RCCI representation")

    def record(self, obs):
        self_obs, _, _, _ = self._split(obs)
        self._self_norm.record(self_obs)

    def update(self):
        self._self_norm.update()

    def normalize(self, obs):
        self_obs, block1, block2, block3 = self._split(obs)
        norm_self = self._self_norm.normalize(self_obs)
        norm_x = (block1 - self._phi_mean) / self._phi_std

        if self._representation == "absolute":
            norm_block2 = (block2 - self._phi_mean) / self._phi_std
            norm_block3 = (block3 - self._phi_mean) / self._phi_std
        else:
            norm_block2 = block2 / self._phi_std
            norm_block3 = block3 / self._phi_std

        return torch.cat([norm_self, norm_x, norm_block2, norm_block3], dim=-1)

    def get_mean(self):
        zeros = torch.zeros_like(self._phi_mean)
        if self._representation == "absolute":
            command_mean = torch.cat(
                [self._phi_mean, self._phi_mean, self._phi_mean], dim=-1)
        else:
            command_mean = torch.cat([self._phi_mean, zeros, zeros], dim=-1)
        return torch.cat([self._self_norm.get_mean(), command_mean], dim=-1)

    def get_std(self):
        command_std = self._phi_std.repeat(3)
        return torch.cat([self._self_norm.get_std(), command_std], dim=-1)

    def get_shape(self):
        return torch.Size([self._total_dim])

    def get_phi_mean(self):
        return self._phi_mean

    def get_phi_std(self):
        return self._phi_std

    def _split(self, obs):
        if obs.shape[-1] != self._total_dim:
            raise ValueError(
                "RCCI observation has size {}, expected {}".format(
                    obs.shape[-1], self._total_dim))
        i0 = self._self_dim
        i1 = i0 + self._phi_dim
        i2 = i1 + self._phi_dim
        return obs[..., :i0], obs[..., i0:i1], obs[..., i1:i2], obs[..., i2:]
