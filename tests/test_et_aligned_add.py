from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.add_agent as add_agent
import learning.et_aligned_add_agent as et_agent
from learning.phase_scalarization import compute_phase_importance_weights


def test_phase_utility_uses_the_same_add_reward_as_ppo():
    reward = torch.tensor([0.2, 1.0, 3.0])
    utility = et_agent.normalize_add_reward(reward, reward_scale=2.0)
    assert torch.allclose(utility, reward / 2.0)


def test_softmin_gradient_equals_adversarial_probability():
    utility = torch.tensor([0.2, 0.6, 0.9], requires_grad=True)
    present = torch.ones(3, dtype=torch.bool)
    beta = 1.9
    objective = et_agent.compute_entropic_softmin(utility, present, beta)
    objective.backward()
    probability = et_agent.compute_entropic_phase_probabilities(
        utility.detach(), present, beta)
    assert torch.allclose(utility.grad, probability, atol=1e-7)


def test_worse_phases_receive_more_probability():
    utility = torch.tensor([0.2, 0.6, 0.9])
    present = torch.ones(3, dtype=torch.bool)
    probability = et_agent.compute_entropic_phase_probabilities(
        utility, present, beta=1.9)
    assert torch.allclose(probability.sum(), torch.tensor(1.0))
    assert probability[0] > probability[1] > probability[2]


def test_missing_phases_receive_no_probability_mass():
    utility = torch.tensor([0.2, 0.0, 0.8])
    present = torch.tensor([True, False, True])
    probability = et_agent.compute_entropic_phase_probabilities(
        utility, present, beta=1.9)
    assert probability[1] == 0.0
    assert torch.allclose(probability.sum(), torch.tensor(1.0))


def test_phase_permutation_only_permutes_probability():
    utility = torch.tensor([0.2, 0.6, 0.9])
    present = torch.ones(3, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    original = et_agent.compute_entropic_phase_probabilities(
        utility, present, beta=1.9)
    permuted = et_agent.compute_entropic_phase_probabilities(
        utility[permutation], present[permutation], beta=1.9)
    assert torch.allclose(permuted, original[permutation])


def test_importance_weights_have_unit_rollout_mean():
    utility = torch.tensor([0.2, 0.6, 0.9])
    present = torch.ones(3, dtype=torch.bool)
    probability = et_agent.compute_entropic_phase_probabilities(
        utility, present, beta=1.9)
    counts = torch.tensor([2.0, 3.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    weights = compute_phase_importance_weights(probability, counts, idx)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_agent_keeps_stock_add_discriminator_objective():
    assert issubclass(et_agent.ETAlignedADDAgent, add_agent.ADDAgent)
    assert "_compute_disc_loss" not in et_agent.ETAlignedADDAgent.__dict__
    assert "_build_model" not in et_agent.ETAlignedADDAgent.__dict__


def test_configs_and_three_iteration_smoke_budget():
    agent = yaml.safe_load(
        (ROOT / "data/agents/et_aligned_add_humanoid_agent.yaml").read_text())
    env = yaml.safe_load(
        (ROOT / "data/envs/et_aligned_add_humanoid_roll_env.yaml").read_text())
    smoke_args = (
        ROOT / "args/et_aligned_add_humanoid_roll_smoke4096_args.txt"
    ).read_text()

    assert agent["agent_name"] == "ET_ALIGNED_ADD"
    assert agent["disc_grad_penalty"] == 2
    assert agent["et_beta"] > 0
    assert "lexmm_horizon_iters" not in agent
    assert env["env_name"] == "et_aligned_add"
    assert env["enable_phase_obs"] is False
    assert env["num_disc_obs_steps"] == 1
    assert "--num_envs 4096" in smoke_args
    assert "--max_samples 393216" in smoke_args
