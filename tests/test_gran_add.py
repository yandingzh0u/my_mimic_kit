import pathlib
import sys

import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from learning.gran_add_model import ScaleSeparatedGraNHead


def test_gran_head_separates_direction_and_regularized_gain():
    direction = torch.tensor([[3.0, 4.0]])
    head = ScaleSeparatedGraNHead(2, direction)
    with torch.no_grad():
        head.weight.mul_(2.5)

    x = torch.tensor([[0.7, -0.2]], requires_grad=True)
    logits, stats = head(
        x, torch.zeros_like(x), x, create_graph=True)
    input_grad = torch.autograd.grad(logits.sum(), x)[0]

    torch.testing.assert_close(
        torch.linalg.vector_norm(head.weight), torch.tensor(2.5))
    torch.testing.assert_close(
        torch.linalg.vector_norm(input_grad), torch.tensor(2.5),
        rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats["alpha"], torch.tensor(2.5))
    torch.testing.assert_close(
        torch.sum(torch.square(head.weight)), torch.tensor(6.25))


def test_gran_head_is_anchored_at_the_add_positive_sample():
    head = ScaleSeparatedGraNHead(2, torch.tensor([[3.0, 4.0]]))
    x = torch.zeros((4, 2), requires_grad=True)
    logits, stats = head(
        x, torch.zeros((1, 2)), x, create_graph=True)

    torch.testing.assert_close(logits, torch.zeros_like(logits))
    torch.testing.assert_close(
        stats["normalized_shape_mean"], torch.tensor(0.0))


def test_gran_config_has_no_gradient_penalty_parameter():
    path = ROOT / "data" / "agents" / "gran_add_humanoid_climb_agent.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)

    assert config["agent_name"] == "GraNADD"
    assert "disc_grad_penalty" not in config
    assert config["disc_logit_reg"] == 0.01


def test_obsolete_gadd_configs_are_removed():
    for filename in ("gadd_humanoid_agent.yaml",
                     "gadd_refconcat_humanoid_agent.yaml"):
        assert not (ROOT / "data" / "agents" / filename).exists()
