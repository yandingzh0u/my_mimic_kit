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
import envs.anchored_energy_env as anchored_energy_env
import learning.aligned_add_agent as aligned_add_agent
import learning.anchored_energy_agent as anchored_energy_agent
import learning.anchored_energy_model as anchored_energy_model


def test_reference_schedule_is_deterministic_and_balanced():
    ids0, phase0 = anchored_energy_env.build_uniform_reference_schedule(
        num_motions=3, num_samples=10, device="cpu")
    ids1, phase1 = anchored_energy_env.build_uniform_reference_schedule(
        num_motions=3, num_samples=10, device="cpu")
    assert torch.equal(ids0, ids1)
    assert torch.equal(phase0, phase1)
    assert torch.equal(torch.bincount(ids0), torch.tensor([4, 3, 3]))
    assert torch.all(phase0 > 0)
    assert torch.all(phase0 < 1)


def test_reference_stats_use_population_std_and_stabilize_constants():
    reference = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 7.0],
    ])
    mean, scale = anchored_energy_env.compute_reference_phi_stats(reference)
    assert torch.equal(mean, torch.tensor([2.0, 2.0, 5.0]))
    assert torch.equal(scale, torch.tensor([1.0, 1.0, 2.0]))


def test_energy_coordinates_reconstruct_exact_paired_reference():
    batch = 16
    dim = 7
    ref_t = torch.randn(batch, dim)
    ref_t1 = torch.randn(batch, dim)
    sim_t1 = torch.randn(batch, dim)
    motion = ref_t1 - ref_t
    error = ref_t1 - sim_t1
    mean = torch.randn(dim)
    scale = torch.rand(dim) + 0.2

    residual, context = anchored_energy_agent.normalize_energy_inputs(
        error=error, next_ref_obs=ref_t1, ref_motion=motion,
        ref_mean=mean, ref_scale=scale)
    norm_ref, norm_motion = torch.chunk(context, 2, dim=-1)

    assert torch.allclose(residual, error / scale)
    assert torch.allclose(norm_ref, (ref_t - mean) / scale)
    assert torch.allclose(norm_motion, motion / scale)


def test_energy_residual_normalization_preserves_zero_and_has_no_clip():
    dim = 5
    mean = torch.randn(dim)
    scale = torch.full([dim], 0.5)
    next_ref = torch.randn(2, dim)
    motion = torch.randn(2, dim)
    zero = torch.zeros_like(next_ref)
    residual, _ = anchored_energy_agent.normalize_energy_inputs(
        zero, next_ref, motion, mean, scale)
    assert torch.equal(residual, zero)

    huge = torch.full_like(next_ref, 1e6)
    residual, _ = anchored_energy_agent.normalize_energy_inputs(
        huge, next_ref, motion, mean, scale)
    assert torch.allclose(residual, huge / scale)
    assert torch.max(torch.abs(residual)) > 1e6


def test_invalid_reference_scale_is_rejected():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        anchored_energy_agent.normalize_energy_inputs(
            x, x, x, torch.zeros(2), torch.tensor([1.0, 0.0]))


def test_agent_keeps_aligned_policy_interface_but_replaces_objective():
    assert issubclass(
        anchored_energy_agent.AnchoredEnergyADDAgent,
        aligned_add_agent.AlignedADDAgent)
    assert "_compute_disc_loss" in (
        anchored_energy_agent.AnchoredEnergyADDAgent.__dict__)
    assert "_compute_rewards" in (
        anchored_energy_agent.AnchoredEnergyADDAgent.__dict__)
    assert (anchored_energy_env.AnchoredEnergyEnv._compute_obs
            is aligned_add_env.AlignedADDEnv._compute_obs)


class _AlignedEnvStub:
    def __init__(self, self_dim, command_dim):
        self._self_dim = self_dim
        self._command_dim = command_dim

    def get_aligned_self_obs_dim(self):
        return self._self_dim

    def get_aligned_command_dim(self):
        return self._command_dim


class _ReplayStub:
    def __init__(self, data):
        self._data = data

    def sample(self, count):
        return {key: value[:count] for key, value in self._data.items()}


class _EnergyModelStub(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.energy = anchored_energy_model.ConditionalPositiveDefiniteEnergy(
            residual_dim=dim, context_dim=2 * dim,
            hidden_units=(11,), rank=2, eigen_floor=0.1)
        self.beta = torch.nn.Parameter(torch.tensor(0.0))

    def eval_energy(self, residual, context):
        return self.energy.eval_energy(residual, context)

    def eval_disc(self, residual, context):
        return self.beta - self.eval_energy(residual, context)

    def get_energy_bias(self):
        return self.beta

    def get_energy_epsilon(self):
        return self.energy.eigen_floor


def test_disc_diagnostic_does_not_alias_or_corrupt_energy_bias():
    torch.manual_seed(3)
    dim = 4
    self_dim = 3
    batch_size = 8
    agent = object.__new__(anchored_energy_agent.AnchoredEnergyADDAgent)
    torch.nn.Module.__init__(agent)
    agent._env = _AlignedEnvStub(self_dim, dim)
    agent._model = _EnergyModelStub(dim)
    agent._energy_phi_dim = dim
    agent._energy_ref_mean = torch.zeros(dim)
    agent._energy_ref_scale = torch.ones(dim)

    obs = torch.randn(batch_size, self_dim + 2 * dim)
    disc_obs = torch.randn(batch_size, dim)
    disc_demo = torch.randn(batch_size, dim)
    motion = obs[:, self_dim + dim:]
    agent._disc_buffer = _ReplayStub({
        "disc_obs": disc_obs.clone(),
        "disc_obs_demo": disc_demo.clone(),
        "ref_motion": motion.clone(),
    })
    info = agent._compute_disc_loss({
        "obs": obs,
        "disc_obs": disc_obs,
        "disc_obs_demo": disc_demo,
    })

    beta_before = agent._model.beta.detach().clone()
    info["energy_bias"].add_(123.0)
    assert torch.equal(agent._model.beta.detach(), beta_before)
    info["disc_loss"].backward()
    assert torch.isfinite(agent._model.beta.grad)
    assert torch.abs(agent._model.beta.grad) <= 1.0
    assert info["energy_lower_bound_min_slack"] >= -1e-6


def test_configs_builders_and_roll_budgets():
    agent = yaml.safe_load(
        (ROOT / "data/agents/anchored_energy_add_humanoid_agent.yaml")
        .read_text())
    env = yaml.safe_load(
        (ROOT / "data/envs/anchored_energy_humanoid_roll_env.yaml")
        .read_text())
    aligned_env = yaml.safe_load(
        (ROOT / "data/envs/aligned_add_humanoid_roll_env.yaml").read_text())

    assert agent["agent_name"] == "ANCHORED_ENERGY_ADD"
    assert agent["disc_grad_penalty"] == 0
    assert agent["disc_logit_reg"] == 0
    assert env["env_name"] == "anchored_energy"
    assert env["reference_phi_stats_samples"] == 4096
    comparable_env = dict(env)
    comparable_env.pop("env_name")
    comparable_env.pop("reference_phi_stats_samples")
    aligned_env = dict(aligned_env)
    aligned_env.pop("env_name")
    assert comparable_env == aligned_env

    smoke = (ROOT / "args/anchored_energy_humanoid_roll_smoke_args.txt")
    scale = (
        ROOT / "args/anchored_energy_humanoid_roll_scale_smoke_args.txt")
    formal = (
        ROOT / "args/anchored_energy_humanoid_roll_2k_8192_args.txt")
    assert "--num_envs 64" in smoke.read_text()
    assert "--num_envs 8192" in scale.read_text()
    assert "--max_samples 262144" in scale.read_text()
    assert "--num_envs 8192" in formal.read_text()
    assert "--max_samples 524288000" in formal.read_text()

    env_builder = (ROOT / "mimickit/envs/env_builder.py").read_text()
    agent_builder = (ROOT / "mimickit/learning/agent_builder.py").read_text()
    assert 'env_name == "anchored_energy"' in env_builder
    assert 'agent_name == "ANCHORED_ENERGY_ADD"' in agent_builder
