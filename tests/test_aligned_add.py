from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import envs.aligned_add_env as aligned_add_env
import learning.add_agent as add_agent
import learning.aligned_add_agent as aligned_add_agent
import learning.aligned_obs_normalizer as aligned_obs_normalizer
import learning.diff_normalizer as diff_normalizer


def test_config_contract():
    valid = {
        "enable_tar_obs": False,
        "enable_phase_obs": False,
        "num_disc_obs_steps": 1,
        "aligned_command_step": 1,
        "global_obs": True,
    }
    aligned_add_env.validate_aligned_add_config(valid)

    for key, value in (("enable_tar_obs", True),
                       ("enable_phase_obs", True),
                       ("num_disc_obs_steps", 2),
                       ("aligned_command_step", 2),
                       ("global_obs", False)):
        invalid = dict(valid)
        invalid[key] = value
        with pytest.raises(ValueError):
            aligned_add_env.validate_aligned_add_config(invalid)


def test_aligned_agent_keeps_stock_add_objective():
    assert issubclass(aligned_add_agent.AlignedADDAgent, add_agent.ADDAgent)
    assert "_compute_disc_loss" not in aligned_add_agent.AlignedADDAgent.__dict__
    assert "_compute_rewards" not in aligned_add_agent.AlignedADDAgent.__dict__
    assert "_build_model" not in aligned_add_agent.AlignedADDAgent.__dict__


def test_prda_command_uses_shared_add_scale():
    command_norm = diff_normalizer.DiffNormalizer([3], device="cpu")
    samples = torch.tensor([[2.0, -4.0, 8.0], [-2.0, 4.0, -8.0]])
    command_norm.record(samples)
    command_norm.update()

    obs_norm = aligned_obs_normalizer.AlignedObsNormalizer(
        self_dim=2, command_dim=3, device="cpu")
    obs_norm.set_command_normalizer(command_norm)

    obs = torch.tensor([[1.0, -1.0, 2.0, -4.0, 8.0]])
    norm_obs = obs_norm.normalize(obs)
    assert torch.allclose(norm_obs[0, 2:], torch.tensor([1.0, -1.0, 1.0]))

    state_keys = obs_norm.state_dict().keys()
    assert all("command" not in key for key in state_keys)


def test_prda_residual_identity():
    sim_t = torch.randn(8, 172)
    sim_t1 = torch.randn(8, 172)
    ref_t = torch.randn(8, 172)
    ref_t1 = torch.randn(8, 172)

    error_t = ref_t - sim_t
    ref_motion = ref_t1 - ref_t
    command = aligned_add_env.compute_prda_command(ref_t1, sim_t)
    actual_motion = sim_t1 - sim_t
    error_t1 = ref_t1 - sim_t1

    assert torch.allclose(command, error_t + ref_motion, atol=1e-6, rtol=1e-6)
    assert torch.allclose(error_t1, command - actual_motion, atol=1e-6, rtol=1e-6)


def test_method_configs_and_budget():
    agent = yaml.safe_load((ROOT / "data/agents/aligned_add_humanoid_agent.yaml").read_text())
    assert agent["agent_name"] == "ALIGNED_ADD"
    assert agent["steps_per_iter"] == 32
    assert agent["disc_grad_penalty"] == 2

    for name, motion in (("roll", "humanoid_roll.pkl"),
                         ("spinkick", "humanoid_spinkick.pkl")):
        config = yaml.safe_load(
            (ROOT / f"data/envs/aligned_add_humanoid_{name}_env.yaml").read_text())
        assert config["env_name"] == "aligned_add"
        assert config["enable_tar_obs"] is False
        assert config["enable_phase_obs"] is False
        assert config["num_disc_obs_steps"] == 1
        assert config["aligned_command_step"] == 1
        assert config["motion_file"].endswith(motion)

        args = (ROOT / f"args/aligned_add_humanoid_{name}_2k_8192_args.txt").read_text()
        assert "--num_envs 8192" in args
        assert "--max_samples 524288000" in args

    prda_args = (ROOT / "args/prda_add_humanoid_roll_2k_8192_args.txt").read_text()
    assert "--num_envs 8192" in prda_args
    assert "--max_samples 524288000" in prda_args
    assert "--out_dir output/prda_add_roll_2k_8192_seed0" in prda_args


def test_builder_routes_exist():
    env_builder = (ROOT / "mimickit/envs/env_builder.py").read_text()
    agent_builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'env_name == "aligned_add"' in env_builder
    assert 'agent_name == "ALIGNED_ADD"' in agent_builder
