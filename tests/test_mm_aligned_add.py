from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import envs.mm_aligned_add_env as mm_env
import learning.add_agent as add_agent
import learning.mm_aligned_add_agent as mm_agent


def test_control_phase_binning_closes_last_endpoint():
    phase = torch.tensor([0.0, 0.01, 0.5, 0.999999, 1.0])
    bins = mm_env.phase_to_control_bin(phase, 60)
    assert torch.equal(bins, torch.tensor([0, 0, 30, 59, 59]))


def test_phase_importance_weights_have_unit_rollout_mean():
    phase_lambda = torch.tensor([0.2, 0.3, 0.5])
    counts = torch.tensor([2.0, 3.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    weights = mm_agent.compute_phase_importance_weights(
        phase_lambda, counts, idx)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    for phase in range(3):
        contribution = weights[idx == phase].sum() / len(idx)
        assert torch.allclose(contribution, phase_lambda[phase])


def test_dual_update_moves_mass_to_worst_phase():
    phase_lambda = torch.full((3,), 1.0 / 3.0)
    quality = torch.tensor([0.2, 1.0, 1.8])
    present = torch.ones(3, dtype=torch.bool)
    updated = mm_agent.exponentiated_dual_update(
        phase_lambda, quality, present, step_size=0.1,
        quality_scale=2.0)
    assert torch.allclose(updated.sum(), torch.tensor(1.0))
    assert updated[0] > updated[1] > updated[2]


def test_equal_phase_quality_preserves_uniform_dual():
    phase_lambda = torch.full((4,), 0.25)
    quality = torch.ones(4)
    present = torch.ones(4, dtype=torch.bool)
    updated = mm_agent.exponentiated_dual_update(
        phase_lambda, quality, present, step_size=0.1,
        quality_scale=2.0)
    assert torch.allclose(updated, phase_lambda)


def test_mm_agent_keeps_stock_add_discriminator_objective():
    assert issubclass(mm_agent.MMAlignedADDAgent, add_agent.ADDAgent)
    assert "_compute_disc_loss" not in mm_agent.MMAlignedADDAgent.__dict__
    assert "_build_model" not in mm_agent.MMAlignedADDAgent.__dict__


def test_mm_configs_and_budget():
    agent = yaml.safe_load(
        (ROOT / "data/agents/mm_aligned_add_humanoid_agent.yaml").read_text())
    env = yaml.safe_load(
        (ROOT / "data/envs/mm_aligned_add_humanoid_roll_env.yaml").read_text())
    args = (ROOT / "args/mm_aligned_add_humanoid_roll_300_4096_args.txt").read_text()

    assert agent["agent_name"] == "MM_ALIGNED_ADD"
    assert agent["disc_grad_penalty"] == 2
    assert agent["mm_dual_step_size"] == 0.05
    assert env["env_name"] == "mm_aligned_add"
    assert env["enable_phase_obs"] is False
    assert env["num_disc_obs_steps"] == 1
    assert "--num_envs 4096" in args
    assert "--max_samples 39321600" in args
