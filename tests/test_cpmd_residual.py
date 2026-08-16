"""Contract tests for the residual-context CPMD schema.

These tests intentionally keep the new method separate from the frozen 767-D
CPMD implementation. They exercise configuration provenance, builder routing,
and the public interfaces shared by the environment, model, and agent.
"""

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
    residual = load_yaml(AGENT_CONFIG)

    assert residual["agent_name"] == "CPMD_RESIDUAL"
    assert residual["model"]["cpmd_rank"] == 16
    assert residual["model"]["cpmd_residual_bound"] == pytest.approx(1.0)
    assert residual["cpmd_residual_warmup_iters"] == 100
    assert residual["cpmd_residual_output_reg"] == pytest.approx(0.01)
    assert residual["context_optimizer"] == residual["disc_optimizer"]

    # Every stock training setting remains the ADD value. Only the registered
    # name and explicitly listed contextual fields are new.
    for key, value in add.items():
        if key == "agent_name":
            continue
        if key == "model":
            for model_key, model_value in value.items():
                assert residual["model"][model_key] == model_value
        else:
            assert residual[key] == value


def test_train_and_eval_envs_share_schema_and_differ_only_in_horizon():
    train_env = load_yaml(TRAIN_ENV_CONFIG)
    eval_env = load_yaml(EVAL_ENV_CONFIG)

    assert train_env["env_name"] == "cpmd_residual"
    assert train_env["cpmd_schema_version"] == 2
    assert train_env["cpmd_memory_seconds"] == pytest.approx(32.0 / 30.0)
    assert train_env["pose_termination"] is False
    assert train_env["enable_early_termination"] is True
    assert train_env["contact_bodies"] == []
    assert train_env["episode_length"] == pytest.approx(2.0)
    assert eval_env["episode_length"] == pytest.approx(10.0)

    train_without_horizon = {k: v for k, v in train_env.items() if k != "episode_length"}
    eval_without_horizon = {k: v for k, v in eval_env.items() if k != "episode_length"}
    assert train_without_horizon == eval_without_horizon


def test_training_args_encode_exact_4096_by_1000_budget():
    args = parse_arg_file(ARGS_FILE)

    assert args["--mode"] == "train"
    assert args["--num_envs"] == "4096"
    assert args["--max_samples"] == str(4096 * 32 * 1000)
    assert args["--rand_seed"] == "0"
    assert args["--visualize"] == "false"
    assert args["--env_config"] == str(TRAIN_ENV_CONFIG.relative_to(REPO_ROOT))
    assert args["--agent_config"] == str(AGENT_CONFIG.relative_to(REPO_ROOT))
    assert args["--out_dir"] == "output/cpmd_residual_roll_cycle_1k_seed0"


def test_env_builder_routes_the_new_name_without_touching_legacy(monkeypatch):
    import envs.env_builder as env_builder

    sentinel = object()

    class FakeResidualEnv:
        def __new__(cls, **kwargs):
            assert kwargs["env_config"]["env_name"] == "cpmd_residual"
            return sentinel

    fake_module = types.ModuleType("envs.cpmd_residual_env")
    fake_module.CPMDResidualEnv = FakeResidualEnv
    monkeypatch.setitem(sys.modules, "envs.cpmd_residual_env", fake_module)
    monkeypatch.setattr(
        env_builder,
        "load_configs",
        lambda env_file, engine_file: ({"env_name": "cpmd_residual"}, {}),
    )

    built = env_builder.build_env("unused", "unused", 3, "cpu", False)
    assert built is sentinel


def test_agent_builder_routes_the_new_name_without_touching_legacy(monkeypatch):
    import learning.agent_builder as agent_builder

    sentinel = object()

    class FakeResidualAgent:
        def __init__(self, config, env, device):
            assert config["agent_name"] == "CPMD_RESIDUAL"

        def calc_num_params(self):
            return 0

    fake_module = types.ModuleType("learning.cpmd_residual_agent")
    fake_module.CPMDResidualAgent = FakeResidualAgent
    monkeypatch.setitem(sys.modules, "learning.cpmd_residual_agent", fake_module)
    monkeypatch.setattr(
        agent_builder,
        "load_agent_file",
        lambda _: {"agent_name": "CPMD_RESIDUAL"},
    )

    built = agent_builder.build_agent("unused", sentinel, "cpu")
    assert isinstance(built, FakeResidualAgent)


def test_residual_public_class_contracts():
    import envs.add_env as add_env
    import envs.cpmd_residual_env as cpmd_residual_env
    import learning.add_agent as add_agent
    import learning.add_model as add_model
    import learning.cpmd_residual_agent as cpmd_residual_agent
    import learning.cpmd_residual_model as cpmd_residual_model

    assert issubclass(cpmd_residual_env.CPMDResidualEnv, add_env.ADDEnv)
    assert issubclass(cpmd_residual_agent.CPMDResidualAgent, add_agent.ADDAgent)
    assert issubclass(cpmd_residual_model.CPMDResidualModel, add_model.ADDModel)

    for method in ["get_cpmd_history_dim", "get_cpmd_ref_motion_dim"]:
        assert hasattr(cpmd_residual_env.CPMDResidualEnv, method)
    for method in ["eval_context_residual", "get_context_params"]:
        assert hasattr(cpmd_residual_model.CPMDResidualModel, method)


class DummyModelEnv:
    def get_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(172,), dtype=np.float32)

    def get_cpmd_history_dim(self):
        return 34

    def get_cpmd_ref_motion_dim(self):
        return 34


def build_residual_model():
    import learning.cpmd_residual_model as cpmd_residual_model

    config = load_yaml(AGENT_CONFIG)["model"]
    return cpmd_residual_model.CPMDResidualModel(config, DummyModelEnv())


def test_zero_initialization_is_exact_add_and_residual_is_bounded():
    torch.manual_seed(0)
    model = build_residual_model()
    disc = torch.randn(16, 172)
    hist = torch.randn(16, 34)
    ref = torch.randn(16, 34)

    base = model.eval_disc(disc)
    combined = model.eval_combined(disc, hist, ref)
    assert torch.equal(combined, base)

    # Exact tracking history disables context for every reference motion.
    residual, raw = model.eval_context(torch.zeros_like(hist), ref)
    assert torch.count_nonzero(residual).item() == 0
    assert torch.count_nonzero(raw).item() == 0

    with torch.no_grad():
        model._context_logits.weight.fill_(100.0)
    residual, _ = model.eval_context(hist, ref)
    assert torch.max(torch.abs(residual)).item() <= 1.0


def test_context_backward_is_parameter_isolated_and_w_learns_first():
    torch.manual_seed(1)
    model = build_residual_model()
    disc = torch.randn(32, 172)
    hist = torch.randn(32, 34)
    ref = torch.randn(32, 34)

    base = model.eval_disc(disc).detach().squeeze(-1)
    residual, _ = model.eval_context(hist, ref)
    gate = torch.sigmoid(base).detach()
    loss = torch.nn.functional.softplus(base + gate * residual.squeeze(-1)).mean()
    loss.backward()

    assert model._context_logits.weight.grad is not None
    assert torch.linalg.vector_norm(model._context_logits.weight.grad).item() > 0
    # w=0 blocks U/V on the first step, but does not deadlock w itself.
    assert torch.count_nonzero(model._context_hist_proj.weight.grad).item() == 0
    assert torch.count_nonzero(model._context_ref_proj.weight.grad).item() == 0
    assert all(p.grad is None for p in model.get_disc_params())


def test_error_memory_is_in_place_and_perfect_tracking_stays_zero():
    import anim.mjcf_char_model as mjcf_char_model
    import envs.cpmd_residual_obs as cpmd_residual_obs

    kin = mjcf_char_model.MJCFCharModel("cpu")
    kin.load(str(REPO_ROOT / "data/assets/humanoid/humanoid.xml"))
    memory = cpmd_residual_obs.CPMDErrorMemory(3, kin, rho=0.9, device="cpu")

    num_joints = kin.get_num_joints() - 1
    root_pos = torch.zeros(3, 3)
    root_rot = torch.zeros(3, 4)
    root_rot[:, 3] = 1.0
    joint_rot = torch.zeros(3, num_joints, 4)
    joint_rot[..., 3] = 1.0
    anchor = root_rot.clone()
    env_ids = torch.arange(3)
    memory.reset(env_ids, root_pos, root_rot, joint_rot)

    history = memory.get_history()
    ptr = history.data_ptr()
    next_pos = root_pos + torch.tensor([0.1, -0.2, 0.0])
    memory.push(next_pos, root_rot, joint_rot,
                next_pos, root_rot, joint_rot, anchor)
    assert history.data_ptr() == ptr
    assert torch.count_nonzero(history).item() == 0

    # A reference-only displacement is accumulated with ref-minus-sim sign.
    ref_pos = next_pos.clone()
    ref_pos[:, 0] += 0.25
    memory.push(next_pos, root_rot, joint_rot,
                ref_pos, root_rot, joint_rot, anchor)
    assert torch.allclose(history[:, 0], torch.full((3,), 0.25), atol=1e-6)

    before = history.clone()
    memory.reset(torch.tensor([1]), ref_pos[1:2], root_rot[1:2], joint_rot[1:2])
    assert torch.count_nonzero(history[1]).item() == 0
    assert torch.equal(history[[0, 2]], before[[0, 2]])


def test_reference_motion_context_uses_fixed_anchor():
    import envs.cpmd_residual_obs as cpmd_residual_obs
    import util.torch_util as torch_util

    yaw = torch.tensor([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]],
                       dtype=torch.float32)
    anchor = torch_util.quat_conjugate(yaw)
    world_vel = torch.tensor([[0.0, 1.0, 0.0]])
    world_ang = torch.tensor([[-1.0, 0.0, 0.0]])
    dof_vel = torch.arange(28, dtype=torch.float32).unsqueeze(0)
    context = cpmd_residual_obs.compute_ref_motion_context(
        world_vel, world_ang, dof_vel, anchor)
    assert context.shape == (1, 34)
    assert torch.allclose(context[:, 6:], dof_vel)
    assert torch.all(torch.isfinite(context))


def test_replay_stores_diff_history_and_context_with_one_indexing():
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
        agent._exp_buffer.record("cpmd_hist_err", ids[:, None].repeat(1, 34))
        agent._exp_buffer.record("cpmd_ref_motion", (1000 + ids)[:, None].repeat(1, 34))
        agent._exp_buffer.inc()

    agent._store_disc_replay_data()
    sample = agent._disc_buffer.sample(capacity)
    diff_id = sample["disc_diff"][:, 0]
    hist_id = sample["cpmd_hist_err"][:, 0]
    ref_id = sample["cpmd_ref_motion"][:, 0]
    assert torch.equal(diff_id, hist_id)
    assert torch.equal(ref_id, 1000 + diff_id)
