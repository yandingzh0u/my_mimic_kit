"""Unit tests for CPMD schema 2's fixed-budget contextual metric.

These tests exercise the structured discriminator directly on CPU. The policy
continues to receive only its ordinary robot observation; synchronized
reference features are used solely as intrinsic context for the reward-side
metric.
"""

from pathlib import Path
import os
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "mimickit"))

import envs.cpmd_env as cpmd_env
import envs.cpmd_obs as cpmd_obs
import envs.env_builder as env_builder
import learning.add_model as add_model
import learning.agent_builder as agent_builder
import learning.cpmd_agent as cpmd_agent
import learning.cpmd_model as cpmd_model
import learning.experience_buffer as experience_buffer
import util.torch_util as torch_util


DEVICE = "cpu"
DELTA_DIM = 172
CONTEXT_DIM = 172
SCHEMA_VERSION = 2


class FakeEnv:
    def __init__(self, delta_dim=DELTA_DIM, context_dim=CONTEXT_DIM,
                 schema_version=SCHEMA_VERSION):
        self._delta_dim = delta_dim
        self._context_dim = context_dim
        self._schema_version = schema_version

    def get_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=[64], dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(
            low=-1.0, high=1.0, shape=[28], dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=[self._delta_dim], dtype=np.float32)

    def get_cpmd_context_dim(self):
        return self._context_dim

    def get_cpmd_schema_version(self):
        return self._schema_version


def metric_config(**overrides):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "cpmd_schema_version": SCHEMA_VERSION,
        "metric_rank": 8,
        "metric_base_weight": 0.01,
        "metric_context_budget": 1.0,
        "metric_norm_eps": 1.0e-8,
        "metric_context_hidden": [64, 32],
    }
    config.update(overrides)
    return config


def make_metric_model(seed=0, **config_overrides):
    torch.manual_seed(seed)
    return cpmd_model.CPMDModel(
        metric_config(**config_overrides), FakeEnv())


def add_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def random_unit_quat(num):
    axis = torch.randn([num, 3])
    angle = torch.rand([num]) * 2.0 - 1.0
    return torch_util.axis_angle_to_quat(axis, angle)


# ---------------------------------------------------------------------------
# Structured metric geometry
# ---------------------------------------------------------------------------


def test_zero_anchor_is_exact_and_context_independent():
    model = make_metric_model()
    context_a = torch.randn([13, CONTEXT_DIM])
    context_b = torch.randn([13, CONTEXT_DIM]) * 20.0
    zero = torch.zeros([13, DELTA_DIM])

    logits_a = model.eval_disc(zero, context_a)
    logits_b = model.eval_disc(zero, context_b)
    anchor = model.eval_zero_logit(13, zero.device, zero.dtype)

    assert torch.equal(logits_a, anchor)
    assert torch.equal(logits_b, anchor)
    assert torch.equal(logits_a, logits_b)


def test_nonzero_error_is_radially_monotone():
    model = make_metric_model()
    context = torch.randn([17, CONTEXT_DIM])
    direction = torch.randn([17, DELTA_DIM])

    z0 = model.eval_disc(torch.zeros_like(direction), context)
    z_half = model.eval_disc(0.5 * direction, context)
    z_one = model.eval_disc(direction, context)
    z_two = model.eval_disc(2.0 * direction, context)

    assert torch.all(z0 > z_half)
    assert torch.all(z_half > z_one)
    assert torch.all(z_one > z_two)


def test_metric_has_fixed_trace_budget_and_bounded_context_energy():
    model = make_metric_model()
    delta = torch.randn([64, DELTA_DIM])
    context = torch.randn([64, CONTEXT_DIM])
    terms = model.eval_metric_terms(delta, context)

    # epsilon makes the trace infinitesimally smaller than one, never larger.
    assert torch.all(terms["trace"] <= 1.0 + 1.0e-6)
    assert torch.all(terms["trace"] > 0.9999)
    assert torch.allclose(
        terms["metric_diag"].sum(dim=-1, keepdim=True),
        terms["trace"], atol=1.0e-6, rtol=1.0e-6)

    # A is PSD with trace at most one, hence delta^T A delta <= ||delta||^2.
    bound = model.get_metric_context_budget() * torch.sum(
        torch.square(delta), dim=-1, keepdim=True)
    assert torch.all(terms["context_energy"] <= bound + 1.0e-5)


class ConstantV(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("v", v.reshape(1, -1))

    def forward(self, context):
        return self.v.expand(context.shape[0], -1)


def test_context_metric_is_invariant_to_scaling_v():
    model = make_metric_model(metric_norm_eps=1.0e-12)
    delta = torch.randn([32, DELTA_DIM])
    context = torch.randn([32, CONTEXT_DIM])
    v = torch.randn([model.get_metric_rank(), DELTA_DIM])

    model._metric_context_net = ConstantV(v)
    terms_1 = model.eval_metric_terms(delta, context)
    model._metric_context_net = ConstantV(10.0 * v)
    terms_10 = model.eval_metric_terms(delta, context)

    assert torch.allclose(
        terms_1["context_energy"], terms_10["context_energy"],
        atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(
        terms_1["logit"], terms_10["logit"],
        atol=2.0e-6, rtol=2.0e-6)


def test_metric_is_sign_symmetric_by_design():
    model = make_metric_model()
    delta = torch.randn([31, DELTA_DIM])
    context = torch.randn([31, CONTEXT_DIM])
    assert torch.allclose(
        model.eval_disc(delta, context),
        model.eval_disc(-delta, context),
        atol=1.0e-6, rtol=1.0e-6)


def test_metric_network_receives_finite_nonzero_gradients():
    model = make_metric_model()
    delta = torch.randn([48, DELTA_DIM])
    context = torch.randn([48, CONTEXT_DIM])
    terms = model.eval_metric_terms(delta, context)

    assert torch.all(terms["v_norm_sq"] > 0.0)
    # The training signal is the logit itself.  Do not add context_energy to
    # it here: logit already contains its negative, so that artificial sum
    # would cancel the contextual path exactly.
    loss = -terms["logit"].mean()
    loss.backward()

    grads = [p.grad for p in model.get_disc_params()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() > 0 for g in grads[:-1])
    assert torch.count_nonzero(grads[-1]).item() > 0  # shared anchor b


def test_positive_input_gradient_is_zero_and_negative_is_finite():
    model = make_metric_model()
    context = torch.randn([8, CONTEXT_DIM])
    zero = torch.zeros([8, DELTA_DIM], requires_grad=True)
    zero_logit = model.eval_disc(zero, context)
    zero_grad = torch.autograd.grad(zero_logit.sum(), zero)[0]
    assert torch.equal(zero_grad, torch.zeros_like(zero_grad))

    delta = torch.randn([8, DELTA_DIM], requires_grad=True)
    neg_logit = model.eval_disc(delta, context)
    neg_grad = torch.autograd.grad(neg_logit.sum(), delta)[0]
    assert torch.isfinite(neg_grad).all()
    assert torch.count_nonzero(neg_grad).item() > 0


# ---------------------------------------------------------------------------
# Intrinsic reference context
# ---------------------------------------------------------------------------


def make_reference_state(num_envs=5, num_joints=3, dof_dim=7,
                         num_bodies=6):
    root_pos = torch.randn([num_envs, 3])
    root_pos[:, 2] += 1.0
    root_rot = random_unit_quat(num_envs)
    root_vel = torch.randn([num_envs, 3])
    root_ang_vel = torch.randn([num_envs, 3])
    joint_rot = random_unit_quat(num_envs * num_joints).reshape(
        num_envs, num_joints, 4)
    dof_vel = torch.randn([num_envs, dof_dim])
    body_offsets = torch.randn([num_envs, num_bodies, 3])
    body_pos = root_pos.unsqueeze(1) + body_offsets
    return (root_pos, root_rot, root_vel, root_ang_vel,
            joint_rot, dof_vel, body_pos)


def test_intrinsic_context_removes_global_xy():
    state = make_reference_state()
    context = cpmd_obs.compute_intrinsic_context(*state)
    assert torch.equal(context[:, 0:2], torch.zeros_like(context[:, 0:2]))

    translated = list(state)
    shift = torch.tensor([91.0, -37.0, 0.0])
    translated[0] = translated[0] + shift
    translated[6] = translated[6] + shift
    shifted_context = cpmd_obs.compute_intrinsic_context(*translated)
    # Subtracting large float32 world coordinates can leave a few ulps in the
    # root-relative body positions; the representation is otherwise equal.
    assert torch.allclose(context, shifted_context, atol=1.0e-5, rtol=2.0e-6)


def test_intrinsic_context_is_global_yaw_and_translation_invariant():
    state = make_reference_state()
    (root_pos, root_rot, root_vel, root_ang_vel,
     joint_rot, dof_vel, body_pos) = state
    context = cpmd_obs.compute_intrinsic_context(*state)

    num_envs = root_pos.shape[0]
    yaw_axis = torch.tensor([[0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    yaw = torch_util.axis_angle_to_quat(
        yaw_axis, torch.linspace(-2.2, 2.2, num_envs))
    shift = torch.tensor([8.0, -13.0, 0.0])

    transformed_root_pos = torch_util.quat_rotate(yaw, root_pos) + shift
    transformed_root_rot = torch_util.quat_mul(yaw, root_rot)
    transformed_root_vel = torch_util.quat_rotate(yaw, root_vel)
    transformed_root_ang_vel = torch_util.quat_rotate(yaw, root_ang_vel)
    expanded_yaw = yaw.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    transformed_body_pos = (
        torch_util.quat_rotate(expanded_yaw, body_pos) + shift)

    transformed_context = cpmd_obs.compute_intrinsic_context(
        transformed_root_pos, transformed_root_rot,
        transformed_root_vel, transformed_root_ang_vel,
        joint_rot, dof_vel, transformed_body_pos)
    assert torch.allclose(
        context, transformed_context, atol=1.0e-5, rtol=1.0e-5)


# ---------------------------------------------------------------------------
# Configuration, routing, and checkpoints
# ---------------------------------------------------------------------------


def test_all_cpmd_env_configs_are_schema2_and_actor_reference_blind():
    env_files = sorted((ROOT / "data/envs").glob("cpmd*.yaml"))
    assert env_files
    for path in env_files:
        config = yaml.safe_load(path.read_text())
        assert config["env_name"] == "cpmd"
        assert config["cpmd_schema_version"] == SCHEMA_VERSION
        assert config["enable_tar_obs"] is False
        assert config["enable_phase_obs"] is False
        assert "tar_obs_steps" not in config
        assert not any("memory" in key.lower() for key in config)

    agent_config = yaml.safe_load(
        (ROOT / "data/agents/cpmd_humanoid_agent.yaml").read_text())
    assert agent_config["agent_name"] == "CPMD"
    assert agent_config["model"]["cpmd_schema_version"] == SCHEMA_VERSION
    assert "disc_net" not in agent_config["model"]
    assert "metric_rank" in agent_config["model"]
    assert "metric_context_budget" in agent_config["model"]


def test_schema_constants_and_builders_route_only_to_cpmd():
    assert cpmd_env.CPMDEnv.SCHEMA_VERSION == SCHEMA_VERSION
    assert cpmd_model.CPMDModel.SCHEMA_VERSION == SCHEMA_VERSION

    agent_source = Path(agent_builder.__file__).read_text()
    env_source = Path(env_builder.__file__).read_text()
    assert 'agent_name == "CPMD"' in agent_source
    assert "cpmd_agent.CPMDAgent" in agent_source
    assert 'env_name == "cpmd"' in env_source
    assert "cpmd_env.CPMDEnv" in env_source
    assert "CPMD_COND" not in agent_source
    assert "cpmd_cond" not in env_source

    with pytest.raises(ValueError, match="schema version"):
        cpmd_model.CPMDModel(
            metric_config(cpmd_schema_version=1), FakeEnv(schema_version=1))
    with pytest.raises(ValueError, match="schema mismatch"):
        cpmd_model.CPMDModel(metric_config(), FakeEnv(schema_version=1))


def test_schema2_checkpoint_round_trip_is_exact():
    model = make_metric_model(seed=10)
    other = make_metric_model(seed=11)
    other.load_state_dict(model.state_dict())

    delta = torch.randn([19, DELTA_DIM])
    context = torch.randn([19, CONTEXT_DIM])
    assert torch.equal(
        model.eval_disc(delta, context), other.eval_disc(delta, context))


def test_old_add_checkpoint_is_rejected_loudly():
    torch.manual_seed(0)
    old_model = add_model.ADDModel(add_config(), FakeEnv())
    new_model = make_metric_model()
    with pytest.raises(RuntimeError):
        new_model.load_state_dict(old_model.state_dict(), strict=True)


# ---------------------------------------------------------------------------
# Paired replay
# ---------------------------------------------------------------------------


class ReplayStub:
    _store_disc_replay_data = cpmd_agent.CPMDAgent._store_disc_replay_data


def build_replay_stub(steps, num_envs, capacity, replay_samples=1000):
    agent = ReplayStub()
    agent._device = DEVICE
    agent._disc_replay_samples = replay_samples
    agent._exp_buffer = experience_buffer.ExperienceBuffer(
        steps, num_envs, DEVICE)
    agent._disc_buffer = experience_buffer.ExperienceBuffer(
        capacity, 1, DEVICE)
    return agent


def record_identifiable_pairs(agent, steps, num_envs):
    for step in range(steps):
        ids = torch.arange(
            step * num_envs, (step + 1) * num_envs,
            dtype=torch.float32).unsqueeze(-1)
        delta = torch.cat([ids, ids + 0.25, ids + 0.5], dim=-1)
        context = torch.cat([-ids, 2.0 * ids + 1.0], dim=-1)
        agent._exp_buffer.record("disc_diff", delta)
        agent._exp_buffer.record("cpmd_context", context)
        agent._exp_buffer.inc()


def assert_replay_pairs_are_intact(replay):
    delta = replay["disc_diff"].squeeze(1)
    context = replay["cpmd_context"].squeeze(1)
    ids = delta[:, 0]
    assert torch.equal(delta[:, 1], ids + 0.25)
    assert torch.equal(delta[:, 2], ids + 0.5)
    assert torch.equal(context[:, 0], -ids)
    assert torch.equal(context[:, 1], 2.0 * ids + 1.0)


def test_replay_keeps_differential_and_context_at_the_same_indices():
    steps, num_envs = 4, 16
    agent = build_replay_stub(steps, num_envs, capacity=1000)
    record_identifiable_pairs(agent, steps, num_envs)

    torch.manual_seed(40)
    agent._store_disc_replay_data()
    replay = agent._disc_buffer.sample(steps * num_envs)
    assert_replay_pairs_are_intact(replay)


def test_first_replay_push_caps_an_overcapacity_rollout_without_unpairing():
    steps, num_envs, capacity = 4, 16, 32
    agent = build_replay_stub(
        steps, num_envs, capacity=capacity, replay_samples=8)
    record_identifiable_pairs(agent, steps, num_envs)

    torch.manual_seed(41)
    agent._store_disc_replay_data()
    assert agent._disc_buffer.is_full()
    assert agent._disc_buffer.get_sample_count() == capacity
    assert_replay_pairs_are_intact(agent._disc_buffer.sample(capacity))
