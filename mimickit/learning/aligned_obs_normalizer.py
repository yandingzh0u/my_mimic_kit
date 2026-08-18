import torch

import learning.diff_normalizer as diff_normalizer
import learning.normalizer as normalizer


class AlignedObsNormalizer(torch.nn.Module):
    """Block normalizer for [self observation, ADD error, ref motion].

    The error block deliberately reuses the ADD discriminator's
    DiffNormalizer.  This object does not register or update that shared
    normalizer; ADDAgent remains its single owner.  Reference motion has a
    separate scale-only normalizer so a zero physical motion remains zero.
    """

    def __init__(self, self_dim, command_dim, device, dtype=torch.float):
        super().__init__()
        self._self_dim = int(self_dim)
        self._command_dim = int(command_dim)
        self._total_dim = self._self_dim + 2 * self._command_dim

        self._self_norm = normalizer.Normalizer(
            [self._self_dim], clip=10.0, device=device, dtype=dtype)
        self._motion_norm = diff_normalizer.DiffNormalizer(
            [self._command_dim], clip=10.0, device=device, dtype=dtype)
        self._error_norm_ref = None

    def set_error_normalizer(self, error_normalizer):
        if tuple(error_normalizer.get_shape()) != (self._command_dim,):
            raise ValueError("ADD error normalizer shape does not match aligned command size")
        # Avoid registering a second alias of the discriminator normalizer in
        # state_dict.  ADDAgent owns, saves, loads, and updates it.
        object.__setattr__(self, "_error_norm_ref", error_normalizer)

    def record(self, obs):
        self_obs, _, ref_motion = self._split(obs)
        self._self_norm.record(self_obs)
        self._motion_norm.record(ref_motion)

    def update(self):
        self._self_norm.update()
        self._motion_norm.update()

    def normalize(self, obs):
        self_obs, curr_error, ref_motion = self._split(obs)
        if self._error_norm_ref is None:
            raise RuntimeError("aligned error normalizer has not been attached")

        norm_self = self._self_norm.normalize(self_obs)
        norm_error = self._error_norm_ref.normalize(curr_error)
        norm_motion = self._motion_norm.normalize(ref_motion)
        return torch.cat([norm_self, norm_error, norm_motion], dim=-1)

    def get_mean(self):
        zeros_error = torch.zeros(
            [self._command_dim], device=self._self_norm.get_mean().device,
            dtype=self._self_norm.get_mean().dtype)
        zeros_motion = torch.zeros_like(zeros_error)
        return torch.cat([self._self_norm.get_mean(), zeros_error, zeros_motion], dim=-1)

    def get_std(self):
        if self._error_norm_ref is None:
            raise RuntimeError("aligned error normalizer has not been attached")
        error_scale = torch.clamp_min(self._error_norm_ref.get_abs_mean(), 1e-4)
        motion_scale = torch.clamp_min(self._motion_norm.get_abs_mean(), 1e-4)
        return torch.cat([self._self_norm.get_std(), error_scale, motion_scale], dim=-1)

    def get_shape(self):
        return torch.Size([self._total_dim])

    def get_motion_normalizer(self):
        return self._motion_norm

    def _split(self, obs):
        if obs.shape[-1] != self._total_dim:
            raise ValueError(
                "aligned observation has size {}, expected {}".format(
                    obs.shape[-1], self._total_dim))
        i0 = self._self_dim
        i1 = i0 + self._command_dim
        return obs[..., :i0], obs[..., i0:i1], obs[..., i1:]
