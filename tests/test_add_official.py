import pathlib

import torch

from learning.add_agent import calc_unscaled_disc_reward


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_official_add_config_uses_gp2():
    text = (ROOT / "data/agents/add_humanoid_agent.yaml").read_text()
    assert "disc_grad_penalty: 2" in text
    assert "spectral_norm" not in text
    assert "group_separable" not in text


def test_official_add_model_is_dense_amp_discriminator():
    text = (ROOT / "mimickit/learning/add_model.py").read_text()
    assert "class ADDModel(amp_model.AMPModel)" in text
    assert "spectral_norm" not in text
    assert "GroupSeparable" not in text


def test_add_reward_matches_official_definition():
    logits = torch.tensor([-2.0, 0.0, 2.0])
    expected = -torch.log(torch.clamp_min(1 - torch.sigmoid(logits), 1e-4))
    torch.testing.assert_close(calc_unscaled_disc_reward(logits), expected)


def test_add_loss_contains_positive_and_negative_gp():
    text = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in text
