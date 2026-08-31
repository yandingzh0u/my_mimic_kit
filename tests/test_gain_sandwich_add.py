import math
import pathlib

import gymnasium.spaces as spaces
import torch

from learning.gain_sandwich_add_model import DirectSumSandwich, GainSandwichADDModel


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    _dims = (3, 6, 45, 84, 3, 3, 28)

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(sum(self._dims),))

    def get_disc_error_groups(self):
        groups = []
        start = 0
        for group_id, dim in enumerate(self._dims):
            groups.append(("group{}".format(group_id), list(range(start, start + dim))))
            start += dim
        return groups


def _config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def test_rms_metric_does_not_change_relative_group_geometry():
    scale = 1.0 / math.sqrt(172)
    root = torch.randn(128, 3)
    body = torch.randn(128, 84)
    before = root.norm(dim=-1) / body.norm(dim=-1)
    after = (scale * root).norm(dim=-1) / (scale * body).norm(dim=-1)
    torch.testing.assert_close(after, before)


def test_direct_sum_local_encoder_has_zero_cross_group_jacobian():
    groups = [("left", [0, 1]), ("right", [2, 3, 4])]
    direct_sum = DirectSumSandwich(groups, first_width=8, trunk_widths=[])
    inputs = torch.randn(5, 5, requires_grad=True)
    left_indices = direct_sum.group_indices_0
    left_features = direct_sum.encoders[0](
        torch.index_select(inputs, -1, left_indices))
    gradient = torch.autograd.grad(left_features.sum(), inputs)[0]
    torch.testing.assert_close(
        gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))


def test_end_to_end_rms_lipschitz_certificate():
    torch.manual_seed(0)
    model = GainSandwichADDModel(_config(), _Env()).eval()
    x = torch.randn(256, 172)
    y = torch.randn(256, 172)
    lhs = (model.eval_disc(x) - model.eval_disc(y)).abs().squeeze(-1)
    rhs = model.get_disc_gain() * (x - y).norm(dim=-1) / math.sqrt(172)
    assert torch.max(lhs - rhs).item() <= 2e-5


def test_local_gradient_respects_rms_bound():
    torch.manual_seed(1)
    model = GainSandwichADDModel(_config(), _Env()).eval()
    x = torch.randn(32, 172, requires_grad=True)
    logits = model.eval_disc(x).sum()
    grad = torch.autograd.grad(logits, x)[0]
    bound = model.get_disc_gain().item() / math.sqrt(172)
    assert grad.norm(dim=-1).max().item() <= bound + 2e-5


def test_zero_differential_is_anchored_at_learned_bias():
    model = GainSandwichADDModel(_config(), _Env()).eval()
    zero = torch.zeros(7, 172)
    expected = model.get_disc_bias().expand(7, 1)
    torch.testing.assert_close(model.eval_disc(zero), expected, atol=1e-6, rtol=0)


def test_existing_logit_regularizer_controls_gain():
    model = GainSandwichADDModel(_config(), _Env()).eval()
    base_loss = model.get_disc_logit_weights().square().sum()
    with torch.no_grad():
        model._disc_log_gain.fill_(math.log(3.0))
    scaled_loss = model.get_disc_logit_weights().square().sum()
    torch.testing.assert_close(scaled_loss, 9.0 * base_loss)
    torch.testing.assert_close(
        model.get_disc_lipschitz_bound(),
        torch.tensor(3.0 / math.sqrt(172)), atol=1e-6, rtol=0)


def test_classification_and_logit_reg_both_reach_gain():
    model = GainSandwichADDModel(_config(), _Env()).train()
    inputs = torch.randn(32, 172)
    logits = model.eval_disc(inputs).squeeze(-1)
    loss = torch.nn.functional.softplus(logits).mean()
    loss = loss + 0.01 * model.get_disc_logit_weights().square().sum()
    loss.backward()
    assert model._disc_log_gain.grad is not None
    assert torch.isfinite(model._disc_log_gain.grad)
    assert model._disc_log_gain.grad.abs().item() > 0


def test_candidate_config_has_no_gp_or_sn_switch():
    text = (ROOT / "data/agents/gain_sandwich_add_humanoid_agent.yaml").read_text()
    assert 'agent_name: "GAIN_SANDWICH_ADD"' in text
    assert "disc_grad_penalty: 0" in text
    assert "spectral_norm" not in text
    assert "iters_per_output: 100" in text
