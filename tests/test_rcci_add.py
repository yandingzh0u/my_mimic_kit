from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import envs.rcci_add_env as rcci_add_env
import learning.add_agent as add_agent
import learning.rcci_add_agent as rcci_add_agent
import learning.rcci_obs_normalizer as rcci_obs_normalizer


@pytest.mark.parametrize("representation", ["absolute", "residual"])
def test_config_contract(representation):
    valid = {
        "enable_tar_obs": False,
        "enable_phase_obs": False,
        "num_disc_obs_steps": 1,
        "rcci_command_step": 1,
        "rcci_representation": representation,
        "rcci_stats_samples": 64,
        "global_obs": True,
    }
    rcci_add_env.validate_rcci_add_config(valid)

    for key, value in (("enable_tar_obs", True),
                       ("enable_phase_obs", True),
                       ("num_disc_obs_steps", 2),
                       ("rcci_command_step", 2),
                       ("global_obs", False),
                       ("rcci_stats_samples", 0)):
        invalid = dict(valid)
        invalid[key] = value
        with pytest.raises(ValueError):
            rcci_add_env.validate_rcci_add_config(invalid)


def test_agent_keeps_stock_add_objective():
    assert issubclass(rcci_add_agent.RCCIADDAgent, add_agent.ADDAgent)
    assert "_compute_disc_loss" not in rcci_add_agent.RCCIADDAgent.__dict__
    assert "_compute_rewards" not in rcci_add_agent.RCCIADDAgent.__dict__
    assert "_build_model" not in rcci_add_agent.RCCIADDAgent.__dict__


def test_raw_commands_are_exact_affine_bijection():
    sim = torch.randn(8, 172)
    ref = torch.randn(8, 172)
    next_ref = torch.randn(8, 172)
    abs_x, abs_ref, abs_next = rcci_add_env.compute_rcci_command(
        sim, ref, next_ref, "absolute")
    res_x, error, motion = rcci_add_env.compute_rcci_command(
        sim, ref, next_ref, "residual")

    assert torch.equal(abs_x, res_x)
    assert torch.allclose(abs_ref, res_x + error, atol=1e-6, rtol=1e-6)
    assert torch.allclose(abs_next, res_x + error + motion,
                          atol=1e-6, rtol=1e-6)


def test_fixed_normalized_commands_are_exact_affine_bijection():
    self_dim = 2
    phi_dim = 4
    mean = torch.randn(phi_dim)
    std = torch.rand(phi_dim) + 0.2
    self_obs = torch.randn(16, self_dim)
    sim = torch.randn(16, phi_dim)
    ref = torch.randn(16, phi_dim)
    next_ref = torch.randn(16, phi_dim)

    abs_norm = rcci_obs_normalizer.RCCIObsNormalizer(
        self_dim, mean, std, "absolute", "cpu")
    res_norm = rcci_obs_normalizer.RCCIObsNormalizer(
        self_dim, mean, std, "residual", "cpu")
    abs_obs = torch.cat([self_obs, sim, ref, next_ref], dim=-1)
    res_obs = torch.cat(
        [self_obs, sim, ref - sim, next_ref - ref], dim=-1)
    # Both normalizers see the exact same self observations.
    abs_norm.record(abs_obs)
    res_norm.record(res_obs)
    abs_norm.update()
    res_norm.update()
    norm_abs = abs_norm.normalize(abs_obs)
    norm_res = res_norm.normalize(res_obs)

    i = self_dim
    norm_x = norm_res[:, i:i + phi_dim]
    norm_e = norm_res[:, i + phi_dim:i + 2 * phi_dim]
    norm_m = norm_res[:, i + 2 * phi_dim:]
    assert torch.allclose(norm_abs[:, :i], norm_res[:, :i])
    assert torch.allclose(norm_abs[:, i:i + phi_dim], norm_x)
    assert torch.allclose(norm_abs[:, i + phi_dim:i + 2 * phi_dim],
                          norm_x + norm_e)
    assert torch.allclose(norm_abs[:, i + 2 * phi_dim:],
                          norm_x + norm_e + norm_m)


def test_phi_statistics_never_update_from_policy_observations():
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([3.0, 4.0])
    obs_norm = rcci_obs_normalizer.RCCIObsNormalizer(
        1, mean, std, "residual", "cpu")
    before_mean = obs_norm.get_phi_mean().clone()
    before_std = obs_norm.get_phi_std().clone()
    obs = torch.randn(32, 7)
    obs_norm.record(obs)
    obs_norm.update()
    assert torch.equal(obs_norm.get_phi_mean(), before_mean)
    assert torch.equal(obs_norm.get_phi_std(), before_std)


def test_configs_builders_and_equal_budget():
    agent = yaml.safe_load(
        (ROOT / "data/agents/rcci_add_humanoid_agent.yaml").read_text())
    assert agent["agent_name"] == "RCCI_ADD"
    assert agent["steps_per_iter"] == 32
    assert agent["disc_grad_penalty"] == 2

    for representation in ("absolute", "residual"):
        env = yaml.safe_load(
            (ROOT / f"data/envs/rcci_{representation}_humanoid_roll_env.yaml").read_text())
        args = (
            ROOT / f"args/rcci_{representation}_humanoid_roll_2k_8192_args.txt"
        ).read_text()
        assert env["env_name"] == "rcci_add"
        assert env["rcci_representation"] == representation
        assert env["enable_tar_obs"] is False
        assert env["enable_phase_obs"] is False
        assert env["episode_length"] == 2.0
        assert env["pose_termination"] is False
        assert "--num_envs 8192" in args
        assert "--max_samples 524288000" in args

    env_builder = (ROOT / "mimickit/envs/env_builder.py").read_text()
    agent_builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'env_name == "rcci_add"' in env_builder
    assert 'agent_name == "RCCI_ADD"' in agent_builder
