"""Ordinary-SN Body split, preserving the original Root/Velocity schemas."""

import torch

from learning.dare_multi_model import DAREMultiModel, IndependentDiscriminators


BODY_SPLIT_SN_SCHEMA_VERSION = 2
HIDDEN_WIDTHS = (341, 512)
BODY_SPLIT_GROUPS = (
    ("root_pose", ("root_pos", "root_rot")),
    ("body_pos", ("body_pos",)),
    ("body_rot", ("body_rot",)),
    ("velocity", ("root_vel", "root_ang_vel", "dof_vel")),
)


def build_body_split_groups(error_groups, input_dim):
    source = {}
    for name, indices in error_groups:
        if name in source:
            raise ValueError("Duplicate discriminator error group: " + name)
        source[name] = tuple(indices)
    expected = {part for _, parts in BODY_SPLIT_GROUPS for part in parts}
    if set(source) != expected:
        raise ValueError("Body-split discriminators require the seven ADD error groups")
    flat = [index for indices in source.values() for index in indices]
    if (any(not indices for indices in source.values())
            or sorted(flat) != list(range(input_dim))):
        raise ValueError("Body-split groups must partition all discriminator coordinates")
    return tuple((name, tuple(index for part in parts for index in source[part]))
                 for name, parts in BODY_SPLIT_GROUPS)


def hierarchical_parent_mean(values):
    """Three parent objectives; split only Body's contribution in equal halves."""
    if values.shape[-1] != 4:
        raise ValueError("Body-split aggregation requires four ordered branch values")
    return (values[..., 0] + 0.5 * (values[..., 1] + values[..., 2])
            + values[..., 3]) / 3.0


class DAREBodySplitSNModel(DAREMultiModel):
    """Root9/BodyPos45/BodyRot84/Velocity34 for the audited one-frame task.

    Each branch keeps the original 341->512 widths, independent of the number
    of branches. Actor/critic construction remains the baseline implementation.
    """
    def _build_disc(self, config, env):
        space = env.get_disc_obs_space()
        self._disc_error_groups = build_body_split_groups(
            env.get_disc_error_groups(), int(space.shape[0]))
        self._disc_branches = IndependentDiscriminators(
            self._disc_error_groups, HIDDEN_WIDTHS[0] * len(BODY_SPLIT_GROUPS),
            [HIDDEN_WIDTHS[1]], self._activation)

