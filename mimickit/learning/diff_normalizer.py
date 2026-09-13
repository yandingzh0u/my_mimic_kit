import numpy as np
import torch

import util.mp_util as mp_util
from util.logger import Logger

class DiffNormalizer(torch.nn.Module):
    def __init__(self, shape, device, init_mean=None, min_diff=1e-4,
                 clip=np.inf, dtype=torch.float, groups=None):
        super().__init__()

        self._min_diff = min_diff
        self._clip = clip
        self.dtype = dtype
        self._groups = tuple(groups) if groups is not None else None
        self._build_params(shape, device, init_mean)
        if self._groups is not None:
            self._build_group_stats(shape, device)
        return

    def record(self, x):
        shape = self.get_shape()
        assert len(x.shape) > len(shape)

        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)

        self._new_count += x.shape[0]
        self._new_sum_abs += torch.sum(torch.abs(x), axis=0)
        if self._groups is not None:
            self._new_sum_sq += torch.sum(torch.square(x), axis=0)
        return

    def update(self):
        self._new_count = mp_util.reduce_sum(self._new_count)
        mp_util.reduce_inplace_sum(self._new_sum_abs)

        new_count = self._new_count
        if int(new_count) == 0:
            return
        new_mean_abs = self._new_sum_abs / new_count

        new_total = self._count + new_count
        w_old = self._count.type(torch.float) / new_total.type(torch.float)
        w_new = float(new_count) / new_total.type(torch.float)

        self._mean_abs[:] = w_old * self._mean_abs + w_new * new_mean_abs
        self._count[:] = new_total

        if self._groups is not None:
            mp_util.reduce_inplace_sum(self._new_sum_sq)
            new_mean_sq = self._new_sum_sq / new_count
            self._mean_sq[:] = (w_old * self._mean_sq
                                + w_new * new_mean_sq)
            self._update_group_scales()

        self._new_count = 0
        self._new_sum_abs[:] = 0
        if self._groups is not None:
            self._new_sum_sq[:] = 0
        return

    def get_shape(self):
        return self._mean_abs.shape

    def get_count(self):
        return self._count

    def get_abs_mean(self):
        return self._mean_abs

    def get_group_scales(self):
        if self._groups is None:
            return None
        return self._group_scales

    def normalize(self, x):
        diff = torch.clamp_min(self._mean_abs, self._min_diff)
        norm_x = x / diff
        norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        if self._groups is not None:
            inv_scales = torch.reciprocal(
                torch.clamp_min(self._group_scales, self._min_diff))
            norm_x = norm_x * inv_scales[self._group_ids]
        return norm_x.type(self.dtype)

    def unnormalize(self, norm_x):
        diff = torch.clamp_min(self._mean_abs, self._min_diff)
        x = norm_x * diff
        if self._groups is not None:
            x = x * self._group_scales[self._group_ids]
        return x.type(self.dtype)

    def training_state_dict(self):
        """Return pending online statistics needed for exact resume."""
        state = {
            "new_count": int(self._new_count),
            "new_sum_abs": self._new_sum_abs.detach().cpu(),
        }
        if self._groups is not None:
            state["new_sum_sq"] = self._new_sum_sq.detach().cpu()
        return state

    def load_training_state_dict(self, state_dict):
        self._new_count = int(state_dict.get("new_count", 0))
        if "new_sum_abs" in state_dict:
            self._new_sum_abs.copy_(state_dict["new_sum_abs"].to(
                device=self._new_sum_abs.device,
                dtype=self._new_sum_abs.dtype))
        if self._groups is not None and "new_sum_sq" in state_dict:
            self._new_sum_sq.copy_(state_dict["new_sum_sq"].to(
                device=self._new_sum_sq.device,
                dtype=self._new_sum_sq.dtype))
        return

    def _build_params(self, shape, device, init_mean):
        self._count = torch.nn.Parameter(torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), requires_grad=False)
        self._mean_abs = torch.nn.Parameter(torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)

        if init_mean is not None:
            assert init_mean.shape == shape, \
            Logger.print('Normalizer init mean shape mismatch, expecting {:d}, but got {:d}'.shape(shape, init_mean.shape))
            self._mean_abs[:] = init_mean

        self._new_count = 0
        self._new_sum_abs = torch.zeros_like(self._mean_abs)
        return

    def _build_group_stats(self, shape, device):
        dim = int(np.prod(shape))
        group_ids = torch.full((dim,), -1, device=device, dtype=torch.long)
        for group_id, (_, indices) in enumerate(self._groups):
            index = torch.as_tensor(indices, device=device, dtype=torch.long)
            if index.numel() == 0 or torch.any(index < 0) \
                    or torch.any(index >= dim) \
                    or torch.any(group_ids[index] != -1):
                raise ValueError("Differential groups must be disjoint valid indices")
            group_ids[index] = group_id
        if torch.any(group_ids == -1):
            raise ValueError("Differential groups must cover the full input")

        self.register_buffer("_group_ids", group_ids)
        self._mean_sq = torch.nn.Parameter(
            torch.ones(shape, device=device, dtype=self.dtype),
            requires_grad=False)
        self._group_scales = torch.nn.Parameter(
            torch.ones(len(self._groups), device=device, dtype=self.dtype),
            requires_grad=False)
        self._new_sum_sq = torch.zeros(shape, device=device, dtype=self.dtype)

    @torch.no_grad()
    def _update_group_scales(self):
        if self._groups is None:
            return
        denom = torch.clamp_min(self._mean_abs, self._min_diff)
        normalized_second = self._mean_sq / torch.square(denom)
        scales = []
        for group_id, (_, indices) in enumerate(self._groups):
            index = self._group_ids.new_tensor(indices)
            scales.append(torch.sqrt(torch.clamp_min(
                torch.sum(normalized_second[index]), self._min_diff)))
        self._group_scales[:] = torch.stack(scales)
