"""Contract and mechanism tests for bilinear CPMD schema 3."""

from pathlib import Path
import shlex
import sys
import types

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

AGENT_CONFIG = REPO_ROOT / "data/agents/cpmd_residual_humanoid_agent.yaml"
TRAIN_ENV_CONFIG = REPO_ROOT / "data/envs/cpmd_residual_humanoid_roll_cycle_env.yaml"
EVAL_ENV_CONFIG = REPO_ROOT / "data/envs/cpmd_residual_humanoid_roll_eval_env.yaml"
ARGS_FILE = REPO_ROOT / "args/cpmd_residual_humanoid_args.txt"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_arg_file(path):
    tokens = shlex.split(path.read_text(encoding="utf-8"), comments=True)
    assert len(tokens) % 2 == 0
    return dict(zip(tokens[::2], tokens[1::2]))


def test_agent_config_preserves_add_base_and_isolates_context_optimizer():
    add = load_yaml(REPO_ROOT / "data/agents/add_humanoid_agent.yaml")
    bilinear = load_yaml(AGENT_CONFIG)

    assert bilinear["agent_name"] == "CPMD_RESIDUAL"
    assert bilinear["context_optimizer"]["type"] == "SGD"
    assert bilinear["context_optimizer"]["learning_rate"] == pytest.approx(2.5e-5)
    assert bilinear["context_optimizer"]["weight_decay"] == pytest.approx(0.0001)
    assert bilinear["disc_optimizer"] == add["disc_optimizer"]
    assert "cpmd_rank" not in bilinear["model"]
    assert "cpmd_residual_bound" not in bilinear["model"]
    assert "cpmd_residual_warmup_iters" not in bilinear
    assert "cpmd_residual_output_reg" not in bilinear

    for key, value in add.items():
        if key == "agent_name":
            continue
        if key == "model":
            for model_key, model_value in value.items():
                assert bilinear["model"][model_key] == model_value
        elif key != "disc_optimizer":
            assert bilinear[key] == value


def test_train_and_eval_envs_share_schema_and_differ_only_in_horizon():
    train_env = load_yaml(TRAIN_ENV_CONFIG)
    eval_env = load_yaml(EVAL_ENV_CONFIG)

    assert train_env["env_name"] == "cpmd_residual"
    assert train_env["cpmd_schema_version"] == 3
    assert train_env["cpmd_memory_seconds"] == pytest.approx(32.0 / 30.0)
    assert train_env["pose_termination"] is False
    assert train_env["enable_early_termination"] is True
    assert train_env["contact_bodies"] == []
    assert train_env["episode_length"] == pytest.approx(2.0)
    assert eval_env["episode_length"] == pytest.approx(10.0)

    train_no_horizon = {k: v for k, v in train_env.items()
                        if k != "episode_length"}
    eval_no_horizon = {k: v for k, v in eval_env.items()
                       if k != "episode_length"}
    assert train_no_horizon == eval_no_horizon


def test_training_args_encode_exact_4096_by_1000_budget():
    args = parse_arg_file(ARGS_FILE)
    assert args["--mode"] == "train"
    assert args["--num_envs"] == "4096"
    assert args["--max_samples"] == str(4096 * 32 * 1000)
    assert args["--rand_seed"] == "0"
    assert args["--visualize"] == "false"
    assert args["--out_dir"] == "output/cpmd_bilinear_roll_cycle_1k_seed0"


def test_builders_route_schema_without_touching_legacy(monkeypatch):
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder

    env_sentinel = object()

    class FakeResidualEnv:
        def __new__(cls, **kwargs):
            assert kwargs["env_config"]["env_name"] == "cpmd_residual"
            return env_sentinel

    fake_env_module = types.ModuleType("envs.cpmd_residual_env")
    fake_env_module.CPMDResidualEnv = FakeResidualEnv
    monkeypatch.setitem(sys.modules, "envs.cpmd_residual_env", fake_env_module)
    monkeypatch.setattr(
        env_builder, "load_configs",
        lambda env_file, engine_file: ({"env_name": "cpmd_residual"}, {}))
    assert env_builder.build_env("unused", "unused", 3, "cpu", False) is env_sentinel

    class FakeResidualAgent:
        def __init__(self, config, env, device):
            assert config["agent_name"] == "CPMD_RESIDUAL"

        def calc_num_params(self):
            return 0

    fake_agent_module = types.ModuleType("learning.cpmd_residual_agent")
    fake_agent_module.CPMDResidualAgent = FakeResidualAgent
    monkeypatch.setitem(
        sys.modules, "learning.cpmd_residual_agent", fake_agent_module)
    monkeypatch.setattr(
        agent_builder, "load_agent_file",
        lambda _: {"agent_name": "CPMD_RESIDUAL"})
    assert isinstance(
        agent_builder.build_agent("unused", env_sentinel, "cpu"),
        FakeResidualAgent)


class DummyModelEnv:
    def get_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(172,), dtype=np.float32)

    def get_cpmd_motion_dim(self):
        return 34


def build_model():
    import learning.cpmd_residual_model as cpmd_residual_model
    return cpmd_residual_model.CPMDResidualModel(
        load_yaml(AGENT_CONFIG)["model"], DummyModelEnv())


def test_zero_initialization_is_exact_add_and_exact_history_match_is_zero():
    torch.manual_seed(0)
    model = build_model()
    disc = torch.randn(16, 172)
    delta = torch.randn(16, 34)
    common = torch.randn(16, 34)

    base = model.eval_disc(disc).squeeze(-1)
    combined = model.eval_combined(disc, delta, common)
    assert torch.equal(combined, base)

    with torch.no_grad():
        model._context_linear.normal_()
        model._context_bilinear.normal_()
    correction = model.eval_context_residual(torch.zeros_like(delta), common)
    assert torch.count_nonzero(correction).item() == 0


def test_bilinear_contraction_matches_explicit_strict_upper_features():
    torch.manual_seed(1)
    model = build_model()
    delta = torch.randn(19, 34)
    common = torch.randn(19, 34)
    weights = torch.randn(model.get_context_num_pairs())
    with torch.no_grad():
        model._context_linear.zero_()
        model._context_bilinear.copy_(weights)

    _, _, actual = model.eval_context(delta, common)
    i = model._context_pair_i
    j = model._context_pair_j
    explicit_features = 0.25 * (
        delta[:, i] * common[:, j] + delta[:, j] * common[:, i])
    expected = torch.sum(explicit_features * weights, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_context_zero_init_has_immediate_gradient_and_is_base_isolated():
    torch.manual_seed(2)
    model = build_model()
    disc = torch.randn(64, 172)
    delta = torch.randn(64, 34)
    common = torch.randn(64, 34)
    base = model.eval_disc(disc).detach().squeeze(-1)
    correction = model.eval_context_residual(delta, common)
    loss = 0.5 * torch.nn.functional.softplus(base + correction).mean()
    loss.backward()

    assert model._context_linear.grad is not None
    assert model._context_bilinear.grad is not None
    total_grad = (torch.linalg.vector_norm(model._context_linear.grad)
                  + torch.linalg.vector_norm(model._context_bilinear.grad))
    assert total_grad.item() > 0.0
    assert all(p.grad is None for p in model.get_disc_params())


def _build_memory(num_envs=3, rho=0.9):
    import anim.mjcf_char_model as mjcf_char_model
    import envs.cpmd_residual_obs as cpmd_residual_obs

    kin = mjcf_char_model.MJCFCharModel("cpu")
    kin.load(str(REPO_ROOT / "data/assets/humanoid/humanoid.xml"))
    return kin, cpmd_residual_obs.CPMDErrorMemory(
        num_envs, kin, rho=rho, device="cpu")


def test_pair_memory_is_in_place_and_matches_separate_summary_algebra():
    kin, memory = _build_memory()
    num_joints = kin.get_num_joints() - 1
    root_pos = torch.zeros(3, 3)
    root_rot = torch.zeros(3, 4)
    root_rot[:, 3] = 1.0
    joint_rot = torch.zeros(3, num_joints, 4)
    joint_rot[..., 3] = 1.0
    anchor = root_rot.clone()
    env_ids = torch.arange(3)
    memory.reset(env_ids, root_pos, root_rot, joint_rot)

    delta = memory.get_delta_motion()
    common = memory.get_sum_motion()
    delta_ptr, common_ptr = delta.data_ptr(), common.data_ptr()

    sim_pos = root_pos + torch.tensor([0.1, 0.0, 0.0])
    ref_pos = root_pos + torch.tensor([0.3, 0.0, 0.0])
    memory.push(sim_pos, root_rot, joint_rot,
                ref_pos, root_rot, joint_rot, anchor)
    assert delta.data_ptr() == delta_ptr
    assert common.data_ptr() == common_ptr
    assert torch.allclose(delta[:, 0], torch.full((3,), 0.2), atol=1e-6)
    assert torch.allclose(common[:, 0], torch.full((3,), 0.4), atol=1e-6)

    rho = memory.get_memory_decay()
    sim_next = sim_pos + torch.tensor([0.05, 0.0, 0.0])
    ref_next = ref_pos + torch.tensor([0.15, 0.0, 0.0])
    memory.push(sim_next, root_rot, joint_rot,
                ref_next, root_rot, joint_rot, anchor)
    expected_ref = rho * 0.3 + 0.15
    expected_sim = rho * 0.1 + 0.05
    assert torch.allclose(
        delta[:, 0], torch.full((3,), expected_ref - expected_sim), atol=1e-6)
    assert torch.allclose(
        common[:, 0], torch.full((3,), expected_ref + expected_sim), atol=1e-6)


def test_pair_memory_partial_reset_is_isolated():
    kin, memory = _build_memory()
    num_joints = kin.get_num_joints() - 1
    pos = torch.zeros(3, 3)
    rot = torch.zeros(3, 4)
    rot[:, 3] = 1.0
    joint = torch.zeros(3, num_joints, 4)
    joint[..., 3] = 1.0
    ids = torch.arange(3)
    memory.reset(ids, pos, rot, joint)
    moved = pos + torch.tensor([0.2, -0.1, 0.0])
    memory.push(pos, rot, joint, moved, rot, joint, rot)

    delta_before = memory.get_delta_motion().clone()
    sum_before = memory.get_sum_motion().clone()
    memory.reset(torch.tensor([1]), moved[1:2], rot[1:2], joint[1:2])
    assert torch.count_nonzero(memory.get_delta_motion()[1]).item() == 0
    assert torch.count_nonzero(memory.get_sum_motion()[1]).item() == 0
    assert torch.equal(memory.get_delta_motion()[[0, 2]], delta_before[[0, 2]])
    assert torch.equal(memory.get_sum_motion()[[0, 2]], sum_before[[0, 2]])


def test_shared_scale_reconstructs_both_sides_and_preserves_zero():
    import learning.cpmd_residual_agent as cpmd_residual_agent
    import learning.diff_normalizer as diff_normalizer

    delta = torch.tensor([[2.0, -1.0], [0.0, 4.0]])
    common = torch.tensor([[6.0, 3.0], [2.0, 4.0]])
    ref, sim = cpmd_residual_agent.CPMDResidualAgent._recover_side_summaries(
        delta, common)
    assert torch.equal(ref - sim, delta)
    assert torch.equal(ref + sim, common)

    norm = diff_normalizer.DiffNormalizer([2], device="cpu")
    norm.record(torch.cat([ref, sim], dim=0))
    norm.update()
    assert torch.count_nonzero(norm.normalize(torch.zeros_like(delta))).item() == 0
    # The same scale is applied to both operands coordinate by coordinate.
    scale = norm.get_abs_mean()
    assert torch.allclose(norm.normalize(delta), delta / scale)
    assert torch.allclose(norm.normalize(common), common / scale)


def test_replay_stores_diff_delta_and_sum_with_one_indexing():
    import learning.cpmd_residual_agent as cpmd_residual_agent
    import learning.experience_buffer as experience_buffer

    class ReplayStub:
        _store_disc_replay_data = (
            cpmd_residual_agent.CPMDResidualAgent._store_disc_replay_data)

    steps, envs, capacity = 3, 16, 32
    agent = ReplayStub()
    agent._device = "cpu"
    agent._disc_replay_samples = 8
    agent._exp_buffer = experience_buffer.ExperienceBuffer(steps, envs, "cpu")
    agent._disc_buffer = experience_buffer.ExperienceBuffer(capacity, 1, "cpu")

    for step in range(steps):
        ids = torch.arange(envs, dtype=torch.float32) + step * envs
        agent._exp_buffer.record("disc_diff", ids[:, None].repeat(1, 172))
        agent._exp_buffer.record(
            "cpmd_delta_motion", ids[:, None].repeat(1, 34))
        agent._exp_buffer.record(
            "cpmd_sum_motion", (1000 + ids)[:, None].repeat(1, 34))
        agent._exp_buffer.inc()

    agent._store_disc_replay_data()
    sample = agent._disc_buffer.sample(capacity)
    diff_id = sample["disc_diff"][:, 0]
    delta_id = sample["cpmd_delta_motion"][:, 0]
    sum_id = sample["cpmd_sum_motion"][:, 0]
    assert torch.equal(diff_id, delta_id)
    assert torch.equal(sum_id, 1000 + diff_id)
