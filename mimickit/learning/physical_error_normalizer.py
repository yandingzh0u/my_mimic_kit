"""Seven historical scales with fixed per-group RMS geometry.

The discriminator still receives all 64 physical components.  Historical
statistics are kept at the seven physical-group level, while each component
is divided by its group's scale and ``sqrt(group_dim)``.  This makes a group's
RMS error contribute one unit of squared radius regardless of its number of
coordinates, without introducing a tunable reward weight.
"""

import torch

from learning.dare_monotone_model import (
    FEATURE_DIM, FEATURE_SLICES, METRIC_NAMES, aggregate_error_metrics)
from learning.diff_normalizer import DiffNormalizer


class PhysicalErrorNormalizer(DiffNormalizer):
    """Keep baseline group statistics without averaging network inputs.

    Inheriting DiffNormalizer also preserves pending training statistics in
    BaseAgent checkpoints. The stored shape is seven; inputs/outputs are 64.
    """

    def __init__(self, device, dtype=torch.float):
        super().__init__((len(METRIC_NAMES),), device=device, dtype=dtype)
        feature_groups = torch.empty(FEATURE_DIM, device=device, dtype=torch.long)
        for index, name in enumerate(METRIC_NAMES):
            feature_groups[FEATURE_SLICES[name]] = index
        self.register_buffer("_feature_groups", feature_groups)
        group_dims = torch.tensor(
            [FEATURE_SLICES[name].stop - FEATURE_SLICES[name].start
             for name in METRIC_NAMES], device=device, dtype=dtype)
        self.register_buffer("_group_sqrt_dims", group_dims.sqrt())

    def record(self, features):
        super().record(aggregate_error_metrics(features))

    def _feature_scales(self):
        return (self._mean_abs.clamp_min(self._min_diff)[self._feature_groups]
                * self._group_sqrt_dims[self._feature_groups])

    def normalize(self, features):
        if features.shape[-1] != FEATURE_DIM:
            raise ValueError("PhysicalErrorNormalizer expects 64 physical errors")
        normalized = features / self._feature_scales()
        return normalized.clamp(-self._clip, self._clip).type(self.dtype)

    def unnormalize(self, normalized):
        if normalized.shape[-1] != FEATURE_DIM:
            raise ValueError("PhysicalErrorNormalizer expects 64 physical errors")
        return (normalized * self._feature_scales()).type(self.dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "_feature_groups")
        if saved is None or not torch.equal(saved.cpu(), self._feature_groups.cpu()):
            error_msgs.append("Cannot load a different physical normalizer feature schema")
        saved_dims = state_dict.get(prefix + "_group_sqrt_dims")
        if saved_dims is None or not torch.equal(saved_dims.cpu(), self._group_sqrt_dims.cpu()):
            error_msgs.append("Cannot load a different physical normalizer group geometry")
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)
