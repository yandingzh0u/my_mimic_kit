"""The three predeclared A/B ablations of the physical-error scorer.

A includes the physical 64-component representation and its nonnegative
weight ordering prior. B is the learned diagonal operator-norm bound.
Removing A restores ADD's signed reference-minus-policy coordinates, so
positivity is removed together with the representation.
"""

import torch
from torch import nn
from torch.nn.utils import parametrize

from learning.add_model import ADDModel
from learning.dare_monotone_model import (
    ADDPairErrorMetrics, FEATURE_DIM, HIDDEN_WIDTHS, MixedELU, PositiveWeight,
    aggregate_error_metrics)
from learning.diagonal_normalization import PositiveDiagonalRescaling, SignedDiagonalRescaling


ABLATION_SCHEMA_VERSION = 2
ABLATION_VARIANTS = {"no_a": 1, "no_b": 2, "no_ab": 3}


def ablation_linear(in_features, out_features, variant, bias=True):
    """Match initial effective weights, then retain only the chosen constraints.

    Every variant starts with the same ordinary Linear random draw as its
    B-on counterpart. B-off weights materialize that counterpart's initial
    diagonal normalization once. Subsequent optimizer updates have no
    diagonal normalization, so this isolates its training-time effect rather
    than introducing a different starting score/reward distribution.
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    if variant == "no_b":
        parametrize.register_parametrization(layer, "weight", PositiveWeight())
        normalization = PositiveDiagonalRescaling(
            in_features, device=layer.weight.device, dtype=layer.weight.dtype)
        normalization.initialize_from_weight(layer.weight)
        parametrize.register_parametrization(layer, "weight", normalization)
        parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)
        parametrize.register_parametrization(layer, "weight", PositiveWeight())
    elif variant in ("no_a", "no_ab"):
        normalization = SignedDiagonalRescaling(
            in_features, device=layer.weight.device, dtype=layer.weight.dtype)
        normalization.initialize_from_weight(layer.weight)
        parametrize.register_parametrization(layer, "weight", normalization)
        if variant == "no_ab":
            parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)
    else:
        raise ValueError("Unknown DARE ablation variant: " + str(variant))
    return layer


class DAREAblationModel(ADDModel):
    """One scorer with the same Actor/Critic and MixedELU hidden widths."""

    def _build_disc(self, config, env):
        self.variant = config.get("ablation_variant")
        if self.variant not in ABLATION_VARIANTS:
            raise ValueError("ablation_variant must be no_a, no_b or no_ab")
        self.physical_errors = self.variant == "no_b"
        self.diagonal_enabled = self.variant == "no_a"
        self.input_dim = FEATURE_DIM if self.physical_errors else 172
        # The physical decoder remains available for identical ERR diagnostics
        # in signed-input runs; signed scorer inputs never pass through it.
        self._pair_metrics = ADDPairErrorMetrics(env)
        self._disc_energy = nn.Sequential(
            ablation_linear(self.input_dim, HIDDEN_WIDTHS[0], self.variant), MixedELU(),
            ablation_linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1], self.variant), MixedELU(),
            ablation_linear(HIDDEN_WIDTHS[1], 1, self.variant, bias=False))
        self._anchor_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("_ablation_schema_version", torch.tensor(
            ABLATION_SCHEMA_VERSION, dtype=torch.int64))
        self.register_buffer("_ablation_variant_code", torch.tensor(
            ABLATION_VARIANTS[self.variant], dtype=torch.int64))

    def pair_to_input(self, policy, reference):
        if policy.ndim != 2 or policy.shape[-1] != 172 or reference.shape != policy.shape:
            raise ValueError("Ablation input requires matching [N,172] policy/reference observations")
        if self.physical_errors:
            return self.pair_to_error_metrics(policy, reference)
        return reference - policy

    def pair_to_error_metrics(self, policy, reference):
        return self._pair_metrics(policy, reference)

    @staticmethod
    def aggregate_error_metrics(features):
        return aggregate_error_metrics(features)

    def eval_disc_raw(self, disc_obs):
        if disc_obs.ndim != 2 or disc_obs.shape[-1] != self.input_dim:
            raise ValueError("DARE ablation discriminator expects " + str(self.input_dim)
                             + " normalized input components")
        anchor = disc_obs.new_zeros((1, self.input_dim))
        energy = self._disc_energy(torch.cat((disc_obs, anchor), 0))
        return self._anchor_logit - (energy[:-1] - energy[-1:])

    def eval_disc(self, disc_obs):
        return self.eval_disc_raw(disc_obs)

    def get_disc_params(self):
        return list(self._disc_energy.parameters()) + [self._anchor_logit]

    def get_disc_logit_weights(self):
        return self._disc_energy[-1].weight.flatten()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        expected = {"_ablation_schema_version": ABLATION_SCHEMA_VERSION,
                    "_ablation_variant_code": ABLATION_VARIANTS[self.variant]}
        for name, value in expected.items():
            saved = state_dict.get(prefix + name)
            if (saved is None or saved.dtype != torch.int64 or saved.numel() != 1
                    or int(saved.item()) != value):
                error_msgs.append("Cannot load a different DARE ablation variant/schema: " + name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)
