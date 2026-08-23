import math
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.add_agent as add_agent
import learning.lexmm_aligned_add_agent as lexmm_agent
from learning.phase_scalarization import compute_phase_importance_weights


def test_bounded_quality_is_discriminator_probability():
    logits = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    reward_scale = 2.0
    disc_reward, quality = lexmm_agent.add_reward_and_quality_from_logits(
        logits, reward_scale)
    assert torch.allclose(quality, torch.sigmoid(logits), atol=1e-7)
    assert torch.allclose(
        disc_reward, reward_scale * torch.nn.functional.softplus(logits),
        atol=1e-6)


def test_theory_selected_concentration_uses_fixed_horizon():
    concentration = lexmm_agent.lexicographic_concentration(2000)
    assert math.isclose(concentration, 0.25 * math.log(2000))
    assert math.isclose(math.exp(concentration), 2000 ** 0.25)


def test_lexicographic_probabilities_prioritize_worse_phases():
    quality = torch.tensor([0.2, 0.6, 0.9])
    present = torch.ones(3, dtype=torch.bool)
    probability = lexmm_agent.compute_lexicographic_probabilities(
        quality, present, horizon_iters=2000)
    assert torch.allclose(probability.sum(), torch.tensor(1.0))
    assert probability[0] > probability[1] > probability[2]
    assert probability.max() / probability.min() <= 2000 ** 0.25 + 1e-6


def test_missing_phases_receive_no_probability_mass():
    quality = torch.tensor([0.2, 0.0, 0.8])
    present = torch.tensor([True, False, True])
    probability = lexmm_agent.compute_lexicographic_probabilities(
        quality, present, horizon_iters=2000)
    assert probability[1] == 0.0
    assert torch.allclose(probability.sum(), torch.tensor(1.0))


def test_lexmm_importance_weights_have_unit_rollout_mean():
    quality = torch.tensor([0.2, 0.6, 0.9])
    present = torch.ones(3, dtype=torch.bool)
    probability = lexmm_agent.compute_lexicographic_probabilities(
        quality, present, horizon_iters=2000)
    counts = torch.tensor([2.0, 3.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    weights = compute_phase_importance_weights(probability, counts, idx)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_lexmm_agent_keeps_stock_add_discriminator_objective():
    assert issubclass(lexmm_agent.LexMMAlignedADDAgent, add_agent.ADDAgent)
    assert "_compute_disc_loss" not in lexmm_agent.LexMMAlignedADDAgent.__dict__
    assert "_build_model" not in lexmm_agent.LexMMAlignedADDAgent.__dict__


def test_lexmm_configs_and_budget():
    agent = yaml.safe_load(
        (ROOT / "data/agents/lexmm_aligned_add_humanoid_agent.yaml").read_text())
    env = yaml.safe_load(
        (ROOT / "data/envs/lexmm_aligned_add_humanoid_roll_env.yaml").read_text())
    args = (
        ROOT / "args/lexmm_aligned_add_humanoid_roll_2k_4096_args.txt"
    ).read_text()

    assert agent["agent_name"] == "LEXMM_ALIGNED_ADD"
    assert agent["disc_grad_penalty"] == 2
    assert agent["lexmm_horizon_iters"] == 2000
    assert "mm_dual_step_size" not in agent
    assert env["env_name"] == "lexmm_aligned_add"
    assert env["enable_phase_obs"] is False
    assert env["num_disc_obs_steps"] == 1
    assert "--num_envs 4096" in args
    assert "--max_samples 262144000" in args
