from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.aligned_add_agent as aligned_add_agent
import learning.ray_ordinal_aligned_add_agent as ray_ordinal_agent


def test_agent_changes_discriminator_objective_only():
    cls = ray_ordinal_agent.RayOrdinalAlignedADDAgent
    assert issubclass(cls, aligned_add_agent.AlignedADDAgent)
    assert "_compute_disc_loss" in cls.__dict__
    assert "_calc_disc_rewards" not in cls.__dict__
    assert "_build_model" not in cls.__dict__
    assert "_build_normalizers" not in cls.__dict__


def test_ray_samples_contract_toward_zero():
    residual = torch.tensor([[2.0, -4.0], [1.0, 3.0]])
    alpha = torch.tensor([[0.25], [0.75]])
    ray, returned_alpha = ray_ordinal_agent.build_ray_samples(
        residual, alpha=alpha)

    assert torch.equal(returned_alpha, alpha)
    assert torch.allclose(ray, alpha * residual)
    assert torch.all(torch.linalg.vector_norm(ray, dim=-1)
                     < torch.linalg.vector_norm(residual, dim=-1))


def test_ray_samples_reject_wrong_alpha_shape():
    with pytest.raises(ValueError, match="alpha must have shape"):
        ray_ordinal_agent.build_ray_samples(
            torch.zeros(4, 3), alpha=torch.zeros(4))


def test_initial_endpoint_gradients_match_stock_add_bce():
    anchor = torch.tensor([0.0], requires_grad=True)
    ray = torch.zeros(8, requires_grad=True)
    negative = torch.zeros(8, requires_grad=True)
    loss, _ = ray_ordinal_agent.calc_ray_ordinal_objective(
        anchor, ray, negative)
    loss.backward()

    assert anchor.grad.item() == pytest.approx(-0.25)
    assert negative.grad.sum().item() == pytest.approx(0.25)
    assert ray.grad.sum().item() == pytest.approx(0.0, abs=1e-7)


def test_correct_ray_order_reduces_ordinal_terms():
    anchor = torch.tensor([2.0])
    ray = torch.tensor([0.0, 0.0])
    negative = torch.tensor([-2.0, -2.0])
    _, ordered = ray_ordinal_agent.calc_ray_ordinal_objective(
        anchor, ray, negative)

    tied = torch.zeros(2)
    _, unordered = ray_ordinal_agent.calc_ray_ordinal_objective(
        torch.zeros(1), tied, tied)
    assert ordered["ordinal_near"] < unordered["ordinal_near"]
    assert ordered["ordinal_far"] < unordered["ordinal_far"]


def test_config_is_a_strict_aligned_add_match():
    baseline = yaml.safe_load(
        (ROOT / "data/agents/aligned_add_humanoid_agent.yaml").read_text())
    ray_ordinal = yaml.safe_load(
        (ROOT / "data/agents/ray_ordinal_aligned_add_humanoid_agent.yaml").read_text())

    assert baseline.pop("agent_name") == "ALIGNED_ADD"
    assert ray_ordinal.pop("agent_name") == "RAY_ORDINAL_ALIGNED_ADD"
    assert ray_ordinal == baseline


def test_builder_and_training_budgets():
    builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'agent_name == "RAY_ORDINAL_ALIGNED_ADD"' in builder

    smoke = (
        ROOT / "args/ray_ordinal_aligned_add/roll_smoke_args.txt"
    ).read_text()
    assert "--num_envs 64" in smoke
    assert "--max_samples 10240" in smoke
    assert "--rand_seed 0" in smoke

    formal = (
        ROOT / "args/ray_ordinal_aligned_add/roll_2k_8192_args.txt"
    ).read_text()
    assert "--num_envs 8192" in formal
    assert "--max_samples 524288000" in formal
    assert "--rand_seed 0" in formal
