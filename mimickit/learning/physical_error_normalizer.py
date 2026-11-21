"""Historical error scales with fixed per-group RMS geometry.

The discriminator receives every physical component. Each component is
divided by its group's historical scale and, by default, ``sqrt(group_dim)``,
so a group's squared radius depends on its RMS error instead of its width.
"""

import torch

from learning.mira_monotone_model import (
    FEATURE_DIM, FEATURE_SLICES, aggregate_error_metrics)
from learning.diff_normalizer import DiffNormalizer


class PhysicalErrorNormalizer(DiffNormalizer):
    """Keep baseline group statistics without averaging network inputs.

    DiffNormalizer also preserves pending training statistics in checkpoints.
    """

    def __init__(self, device, dtype=torch.float, feature_slices=FEATURE_SLICES, *, group_size_scaling=True):
        self._feature_dim = max(part.stop for part in feature_slices.values())
        metric_names = tuple(feature_slices)
        super().__init__((len(metric_names),), device=device, dtype=dtype)
        feature_groups = torch.empty(self._feature_dim, device=device, dtype=torch.long)
        for index, part in enumerate(feature_slices.values()):
            feature_groups[part] = index
        self.register_buffer("_feature_groups", feature_groups)
        group_dims = torch.tensor(
            [part.stop - part.start for part in feature_slices.values()], device=device, dtype=dtype)
        self.register_buffer("_group_sqrt_dims", group_dims.sqrt() if group_size_scaling
                             else torch.ones_like(group_dims))

    def record(self, features):
        super().record(aggregate_physical_groups(features))

    def _feature_scales(self):
        return (self._mean_abs.clamp_min(self._min_diff)[self._feature_groups]
                * self._group_sqrt_dims[self._feature_groups])

    def normalize(self, features):
        if features.shape[-1] != self._feature_dim:
            raise ValueError(f"Expected {self._feature_dim} physical errors")
        normalized = features / self._feature_scales()
        return normalized.clamp(-self._clip, self._clip).type(self.dtype)

    def unnormalize(self, normalized):
        if normalized.shape[-1] != self._feature_dim:
            raise ValueError(f"Expected {self._feature_dim} physical errors")
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


def aggregate_physical_groups(features):
    human = aggregate_error_metrics(features[..., :FEATURE_DIM])
    if features.shape[-1] == FEATURE_DIM:
        return human
    return torch.cat((human, features[..., FEATURE_DIM:]), dim=-1)
