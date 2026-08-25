from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import envs.plot_env as plot_env  # noqa: E402
import envs.add_env as add_env  # noqa: E402
import learning.add_agent as add_agent  # noqa: E402
import learning.plot_agent as plot_agent  # noqa: E402
from learning.phase_scalarization import (  # noqa: E402
    compute_phase_importance_weights,
)


def test_control_phase_binning_closes_last_endpoint():
    phase = torch.tensor([0.0, 0.01, 0.5, 0.999999, 1.0])
    bins = plot_env.phase_to_control_bin(phase, 60)
    assert torch.equal(bins, torch.tensor([0, 0, 30, 59, 59]))


def test_phase_count_resolution_is_explicit_or_deterministic_half_up():
    explicit, explicit_is_auto = plot_env.resolve_plot_num_phases(
        52, motion_length=1.75, control_dt=1.0 / 30.0)
    assert explicit == 52
    assert explicit_is_auto is False

    # Automatic resolution does not inherit Python's ties-to-even behavior and
    # is stable to the small CPU/CUDA discrepancy seen around x.5.
    for length in (1.75, 1.75 - 1e-8, 1.75 + 1e-8):
        automatic, automatic_is_auto = plot_env.resolve_plot_num_phases(
            "auto", motion_length=length, control_dt=1.0 / 30.0)
        assert automatic == 53
        assert automatic_is_auto is True


@pytest.mark.parametrize("value", [0, -1, True, "52", "nearest"])
def test_invalid_phase_count_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="plot_num_phases"):
        plot_env.resolve_plot_num_phases(
            value, motion_length=1.0, control_dt=1.0 / 30.0)


def test_phase_utility_uses_the_policy_reward_scale():
    reward = torch.tensor([0.2, 1.0, 3.0])
    utility = plot_agent.normalize_adversarial_reward(
        reward, reward_scale=2.0)
    assert torch.allclose(utility, reward / 2.0)


def test_softmin_gradient_equals_phase_probability():
    utility = torch.tensor([0.2, 0.6, 0.9], requires_grad=True)
    occupancy = torch.tensor([0.1, 0.3, 0.6])
    beta = 1.9
    objective = plot_agent.compute_entropic_softmin(
        utility, occupancy, beta)
    objective.backward()
    probability = plot_agent.compute_phase_probabilities(
        utility.detach(), occupancy, beta)
    assert torch.allclose(utility.grad, probability, atol=1e-7)


def test_worse_phases_receive_more_probability_relative_to_occupancy():
    utility = torch.tensor([0.2, 0.6, 0.9])
    occupancy = torch.tensor([0.05, 0.15, 0.8])
    probability = plot_agent.compute_phase_probabilities(
        utility, occupancy, beta=1.9)
    density = probability / occupancy
    assert torch.allclose(probability.sum(), torch.tensor(1.0))
    assert density[0] > density[1] > density[2]


def test_equal_utilities_recover_standard_reward_under_nonuniform_occupancy():
    utility = torch.tensor([0.4, 0.4, 0.4])
    counts = torch.tensor([1.0, 10.0, 1000.0])
    probability = plot_agent.compute_phase_probabilities(
        utility, counts, beta=1.9)
    idx = torch.tensor([0] + [1] * 10 + [2] * 1000)
    weights = compute_phase_importance_weights(probability, counts, idx)
    assert torch.allclose(probability, counts / counts.sum(), atol=1e-7)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_occupancy_anchored_density_ratio_is_independent_of_phase_counts():
    utility = torch.tensor([0.2, 0.6, 0.9])
    counts = torch.tensor([1.0, 100.0, 10000.0])
    beta = 1.9
    probability = plot_agent.compute_phase_probabilities(
        utility, counts, beta)
    occupancy = counts / counts.sum()
    density = plot_agent.compute_phase_density(utility, counts, beta)
    assert torch.allclose(probability, occupancy * density)
    expected_ratio = torch.exp(beta * (utility.max() - utility.min()))
    assert torch.allclose(
        density.max() / density.min(), expected_ratio, rtol=1e-6)


def test_extreme_occupancy_does_not_amplify_equal_utility():
    utility = torch.tensor([0.5, 0.5])
    counts = torch.tensor([999.0, 1.0])
    density = plot_agent.compute_phase_density(utility, counts, beta=1.9)
    assert torch.allclose(density, torch.ones_like(density), atol=1e-6)


def test_empty_phase_is_masked_without_nan_or_infinite_sample_weight():
    utility = torch.tensor([0.2, 0.0, 0.8])
    counts = torch.tensor([5.0, 0.0, 3.0])
    probability = plot_agent.compute_phase_probabilities(
        utility, counts, beta=1.9)
    density = plot_agent.compute_phase_density(utility, counts, beta=1.9)
    assert probability[1] == 0.0
    assert density[1] == 0.0
    assert torch.all(torch.isfinite(probability))
    assert torch.all(torch.isfinite(density))
    idx = torch.tensor([0, 0, 0, 0, 0, 2, 2, 2])
    assert torch.all(torch.isfinite(density[idx]))
    assert torch.allclose(density[idx].mean(), torch.tensor(1.0))


def test_single_observed_phase_has_unit_density():
    utility = torch.tensor([10.0, -5.0, 3.0])
    counts = torch.tensor([0.0, 7.0, 0.0])
    probability = plot_agent.compute_phase_probabilities(
        utility, counts, beta=1.9)
    density = plot_agent.compute_phase_density(utility, counts, beta=1.9)
    assert torch.equal(probability, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(density, torch.tensor([0.0, 1.0, 0.0]))


def test_small_beta_recovers_standard_learned_reward():
    utility = torch.tensor([0.2, 0.6, 0.9], dtype=torch.float64)
    counts = torch.tensor([1.0, 10.0, 1000.0], dtype=torch.float64)
    density = plot_agent.compute_phase_density(
        utility, counts, beta=1.0e-8)
    assert torch.allclose(density, torch.ones_like(density), atol=1e-7)


def test_nonfinite_phase_utility_fails_fast():
    with pytest.raises(ValueError, match="phase_utility must be finite"):
        plot_agent.compute_phase_density(
            torch.tensor([0.2, float("nan")]),
            torch.tensor([1.0, 1.0]), beta=1.9)


def test_importance_weights_have_unit_rollout_mean():
    probability = torch.tensor([0.2, 0.3, 0.5])
    counts = torch.tensor([2.0, 3.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    weights = compute_phase_importance_weights(probability, counts, idx)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_discounted_rollout_return_uses_the_exact_ppo_reward_sequence():
    reward = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    result = plot_agent.compute_discounted_rollout_return(
        reward, discount=0.5)
    assert torch.allclose(
        result, torch.tensor([1.0 + 1.5 + 1.25, 2.0 + 2.0 + 1.5]))


def test_phase_argmax_log_uses_paper_one_based_indexing():
    probability = torch.tensor([0.1, 0.7, 0.2])
    assert plot_agent.one_based_phase_argmax(probability).item() == 2


def test_plot_renames_online_environment_return_logs():
    assert plot_agent.PLOTAgent._get_return_log_keys(None) == (
        "Plot_Environment_Train_Return",
        "Plot_Environment_Test_Return",
    )


def test_plot_keeps_the_learned_differential_discriminator():
    assert plot_agent.PLOTAgent.__bases__ == (add_agent.ADDAgent,)
    assert plot_env.PLOTEnv.__bases__ == (add_env.ADDEnv,)
    assert "_compute_disc_loss" not in plot_agent.PLOTAgent.__dict__
    assert "_build_model" not in plot_agent.PLOTAgent.__dict__


def test_public_plot_configuration_names():
    agent = yaml.safe_load(
        (ROOT / "data/agents/plot_humanoid_agent.yaml").read_text())
    assert agent["agent_name"] == "PLOT"
    assert agent["disc_grad_penalty"] == 2
    assert agent["plot_beta"] == pytest.approx(1.90022564)
    assert agent["plot_phase_prior"] == "occupancy"

    expected_phases = {
        "run": 24,
        "backflip": 52,
        "crawl": 77,
        "getup_facedown": 91,
        "spinkick": 39,
        "climb": 299,
    }
    for motion, num_phases in expected_phases.items():
        env = yaml.safe_load(
            (ROOT / "data/envs/paper_benchmark"
             / f"plot_{motion}_env.yaml").read_text())
        assert env["env_name"] == "plot"
        assert env["enable_phase_obs"] is False
        assert env["enable_tar_obs"] is True
        assert env["tar_obs_steps"] == [1, 2, 3]
        assert env["num_disc_obs_steps"] == 1
        assert "aligned_command_step" not in env
        assert env["plot_num_phases"] == num_phases
