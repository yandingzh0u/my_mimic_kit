import math
import pathlib

import gymnasium.spaces as spaces
import torch

from learning.rd_fsn_add_model import (
    GroupSeparableFullSNLayers,
    RDFSNADDModel,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    _dims = (3, 6, 45, 84, 3, 3, 28)

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        groups = []
        start = 0
        for group_id, dim in enumerate(self._dims):
            groups.append((
                "group{}".format(group_id),
                list(range(start, start + dim))))
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


def test_group_frontend_is_the_exact_a30_shape():
    model = RDFSNADDModel(_config(), _Env())
    assert isinstance(model._disc_layers, GroupSeparableFullSNLayers)
    assert model._disc_layers.group_width == 128 // 7
    assert model._disc_layers.total_width == 7 * (128 // 7)
    disc_linears = [
        *[encoder[0] for encoder in model._disc_layers.encoders],
        *[m for m in model._disc_layers.trunk if isinstance(m, torch.nn.Linear)],
        model._disc_logits,
    ]
    assert len(disc_linears) == 9
    assert all(hasattr(layer, "parametrizations") for layer in disc_linears)
    assert all(hasattr(layer.parametrizations, "weight") for layer in disc_linears)


def test_base_score_is_globally_one_lipschitz():
    torch.manual_seed(0)
    model = RDFSNADDModel(_config(), _Env()).eval()
    x = torch.randn(256, 172)
    y = torch.randn(256, 172)
    lhs = (model.eval_disc_score(x) - model.eval_disc_score(y)).abs().squeeze(-1)
    rhs = (x - y).norm(dim=-1)
    assert torch.max(lhs - rhs).item() <= 2e-5


def test_classification_scale_never_enters_reward_score():
    torch.manual_seed(1)
    model = RDFSNADDModel(_config(), _Env()).eval()
    x = torch.randn(32, 172)
    score_before = model.eval_disc_score(x)
    with torch.no_grad():
        model._disc_log_class_scale.fill_(math.log(5.0))
    score_after = model.eval_disc_score(x)
    class_logit = model.eval_disc_classification(x)
    torch.testing.assert_close(score_after, score_before)
    torch.testing.assert_close(class_logit, 5.0 * score_before)


def test_existing_logit_regularizer_is_exactly_scale_squared():
    model = RDFSNADDModel(_config(), _Env()).eval()
    base_penalty = model.get_disc_logit_weights().square().sum()
    with torch.no_grad():
        model._disc_log_class_scale.fill_(math.log(3.0))
    scaled_penalty = model.get_disc_logit_weights().square().sum()
    torch.testing.assert_close(base_penalty, torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scaled_penalty, torch.tensor(9.0), atol=1e-5, rtol=1e-5)


def test_regularizer_breaks_score_temperature_scale_degeneracy():
    scale = torch.tensor(3.0)
    score = torch.tensor([-1.0, 1.0])
    shrunken_score = 0.25 * score
    compensating_scale = scale / 0.25
    torch.testing.assert_close(scale * score, compensating_scale * shrunken_score)
    assert compensating_scale.square() > scale.square()


def test_regularized_class_scale_has_a_finite_optimum():
    log_scale = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_scale], lr=0.05)
    for _ in range(400):
        scale = torch.exp(log_scale)
        loss = torch.nn.functional.softplus(-scale) + 0.01 * scale.square()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scale = torch.exp(log_scale).item()
    assert 2.0 < scale < 4.0


def test_reward_score_gradient_cannot_update_class_scale():
    model = RDFSNADDModel(_config(), _Env()).train()
    score = model.eval_disc_score(torch.randn(16, 172)).mean()
    score.backward()
    assert model._disc_log_class_scale.grad is None


def test_classification_and_logit_reg_update_class_scale():
    model = RDFSNADDModel(_config(), _Env()).train()
    logits = model.eval_disc_classification(torch.randn(16, 172)).squeeze(-1)
    loss = torch.nn.functional.softplus(logits).mean()
    loss = loss + 0.01 * model.get_disc_logit_weights().square().sum()
    loss.backward()
    grad = model._disc_log_class_scale.grad
    assert grad is not None
    assert torch.isfinite(grad)
    assert grad.abs().item() > 0


def test_final_config_has_no_gp_or_new_loss_weight():
    text = (ROOT / "data/agents/rd_fsn_add_humanoid_agent.yaml").read_text()
    assert 'agent_name: "RD_FSN_ADD"' in text
    assert "disc_grad_penalty: 0" in text
    assert "disc_logit_reg: 0.01" in text
    assert "disc_spectral_norm" not in text
    assert "sandwich" not in text.lower()
