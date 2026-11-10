"""Body-split discriminators with a tied-weight convex-potential trunk.

CPL preserves its input dimension.  The two rectangular input projections and
scalar heads use ordinary Linear, without SN.  CPL replaces only the last
hidden ReLU, at width 512; its internal tied weight retains historical SN.
"""

import torch
import torch.nn.functional as F

from learning.dare_body_split_sn_model import (
    HIDDEN_WIDTHS, DAREBodySplitSNModel,
    build_body_split_groups,
)
from learning.dare_multi_model import IndependentDiscriminators


BODY_SPLIT_CPL_SCHEMA_VERSION = 1
CPL_CONSTRAINT = "ordinary_linear_projections_tied_sn_cpl_ordinary_linear_head"
CPL_PLACEMENT = "replace_final_hidden_relu"


class ConvexPotentialBlock(torch.nn.Module):
    """Historical CPL: x - 2 W.T ReLU(W x + b), with one SN weight read.

    The tied transpose uses the same weight tensor as the forward projection.
    Reading ``linear.weight`` twice would run SN's power iteration twice in
    training mode and needlessly change that historical execution contract.
    """

    def __init__(self, width):
        super().__init__()
        linear = torch.nn.Linear(width, width)
        torch.nn.init.zeros_(linear.bias)
        self.linear = torch.nn.utils.parametrizations.spectral_norm(linear)

    def forward(self, inputs):
        weight = self.linear.weight
        pre = F.linear(inputs, weight, self.linear.bias)
        return inputs - 2.0 * F.linear(F.relu(pre), weight.transpose(-1, -2))


class IndependentCPLDiscriminators(IndependentDiscriminators):
    def __init__(self, groups, depth, activation):
        # Draw every common projection/head before drawing CPL parameters.  A
        # common seed gives identical actor/critic and projection/head weights
        # for CPL1 and CPL7.  Preserve the baseline's effective initial weights
        # when removing SN; this is initialization only, not a train-time norm.
        super().__init__(groups, HIDDEN_WIDTHS[0] * len(groups),
                         [HIDDEN_WIDTHS[1]], activation)
        for branch in self.branches:
            for layer in (branch[0], branch[2], branch[-1]):
                layer.eval()
                torch.nn.utils.parametrize.remove_parametrizations(
                    layer, "weight", leave_parametrized=True)
                layer.train()
        for group_id, branch in enumerate(self.branches):
            layers = list(branch.children())
            self.branches[group_id] = torch.nn.Sequential(
                *layers[:3],
                *(ConvexPotentialBlock(HIDDEN_WIDTHS[1]) for _ in range(depth)),
                layers[-1],
            )


class DAREBodySplitCPLModel(DAREBodySplitSNModel):
    """Same four groups/widths; CPL trunk, unconstrained projections/head."""

    def __init__(self, config, env):
        super().__init__(config, env)
        self.register_buffer("_body_split_cpl_schema", torch.tensor(
            [BODY_SPLIT_CPL_SCHEMA_VERSION, self._disc_cpl_depth, *HIDDEN_WIDTHS],
            dtype=torch.int64))

    def _build_disc(self, config, env):
        depth = config.get("disc_cpl_depth")
        if type(depth) is not int or depth not in (1, 7):
            raise ValueError("Body-split CPL requires disc_cpl_depth equal to 1 or 7")
        self._disc_cpl_depth = depth
        space = env.get_disc_obs_space()
        self._disc_error_groups = build_body_split_groups(
            env.get_disc_error_groups(), int(space.shape[0]))
        self._disc_branches = IndependentCPLDiscriminators(
            self._disc_error_groups, depth, self._activation)

    def get_disc_cpl_depth(self):
        return self._disc_cpl_depth

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        expected = self._body_split_cpl_schema
        saved = state_dict.get(prefix + "_body_split_cpl_schema")
        if (saved is None or saved.dtype != torch.int64
                or not torch.equal(saved.cpu(), expected.cpu())):
            error_msgs.append("Cannot load a different Body-split CPL architecture: "
                              + prefix + "_body_split_cpl_schema")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)
