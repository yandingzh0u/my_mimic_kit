import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_monotone_information_audit import gradient_audit, mapping_audit
from dare_monotone_formal_rescore import FrozenFormalDisc, GROUP_NAMES
from learning.monotone_cpl import AnchorLogit, MonotoneGroupCost


def models():
    groups = [(name, [i, i + 7]) for i, name in enumerate(GROUP_NAMES)]
    cost = MonotoneGroupCost(groups, width=8, depth=4)
    head = AnchorLogit()
    state = {f'_model._disc_layers.{key}': value for key, value in cost.state_dict().items()}
    state['_model._disc_logits.bias'] = head.bias.detach().clone()
    state['_disc_obs_norm._mean_abs'] = torch.ones(14)
    state['_disc_obs_norm._count'] = torch.tensor([12])
    candidate = FrozenFormalDisc(state)

    class FullCoordinates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('scale', torch.ones(14))
            self.register_buffer('weights', torch.arange(1, 15).float() / 14)

        def forward(self, values):
            return -(values * self.weights).sum(-1)

    return FullCoordinates(), candidate


def test_mapping_probes_identify_pooling_invariance_without_mutation():
    shared, candidate = models()
    raw = torch.randn(32, 14)
    inputs = raw.clone()
    before = {key: value.clone() for key, value in candidate.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = mapping_audit(shared, candidate, raw)
    for row in result['groups']:
        assert row['candidate_permutation_abs_logit_max'] < 1e-6
        assert row['candidate_sign_flip_abs_logit_max'] < 1e-6
        assert row['original_permutation_abs_logit_mean'] > 0
    torch.testing.assert_close(raw, inputs, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[key]) for key, value in candidate.state_dict().items())


def test_group_rms_gradient_has_no_tangential_component():
    shared, candidate = models()
    result = gradient_audit(shared, candidate, torch.randn(32, 14))
    for row in result['groups']:
        assert row['candidate_tangent_gradient_norm_mean'] < 1e-6
        assert row['original_tangent_gradient_norm_mean'] > .01
    lower, upper = candidate.cost.slope_bounds()
    for row in result['magnitude_slopes']:
        assert lower <= row['mean'] <= upper
