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
import learning.relative_logistic_aligned_add_agent as relative_agent


def test_agent_changes_loss_and_reward_only():
    cls = relative_agent.RelativeLogisticAlignedADDAgent
    assert issubclass(cls, aligned_add_agent.AlignedADDAgent)
    assert "_compute_disc_loss" in cls.__dict__
    assert "_calc_disc_rewards" in cls.__dict__
    assert "_build_model" not in cls.__dict__
    assert "_build_normalizers" not in cls.__dict__


def test_relative_loss_matches_stock_add_initial_gradient_scale():
    anchor = torch.tensor(0.0, requires_grad=True)
    negative = torch.zeros(8, requires_grad=True)
    relative = relative_agent.calc_relative_logit(negative, anchor)
    loss = relative_agent.calc_relative_logistic_loss(relative)
    loss.backward()

    assert anchor.grad.item() == pytest.approx(-0.25)
    assert negative.grad.sum().item() == pytest.approx(0.25)


def test_symmetric_reward_is_bounded_and_two_sided():
    q = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    reward = relative_agent.calc_symmetric_relative_reward(q, 2.0)

    assert torch.all(reward > 0)
    assert torch.all(reward <= 2.0)
    assert reward[2].item() == pytest.approx(2.0)
    assert reward[0].item() == pytest.approx(reward[4].item())
    assert reward[1].item() == pytest.approx(reward[3].item())


def test_config_is_a_strict_aligned_add_match():
    baseline = yaml.safe_load(
        (ROOT / "data/agents/aligned_add_humanoid_agent.yaml").read_text())
    relative = yaml.safe_load(
        (ROOT / "data/agents/relative_logistic_aligned_add_humanoid_agent.yaml").read_text())

    assert baseline.pop("agent_name") == "ALIGNED_ADD"
    assert relative.pop("agent_name") == "RELATIVE_LOGISTIC_ALIGNED_ADD"
    assert relative == baseline


def test_builder_route_and_smoke_budget():
    builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'agent_name == "RELATIVE_LOGISTIC_ALIGNED_ADD"' in builder

    args = (ROOT / "args/relative_logistic_aligned_add/roll_smoke_args.txt").read_text()
    assert "--num_envs 64" in args
    assert "--max_samples 10240" in args
    assert "--rand_seed 0" in args

    formal_args = (
        ROOT / "args/relative_logistic_aligned_add/roll_10k_4096_args.txt"
    ).read_text()
    assert "--num_envs 4096" in formal_args
    assert "--max_samples 1310720000" in formal_args
    assert "--rand_seed 0" in formal_args
