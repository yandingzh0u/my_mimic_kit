"""Focused tests for the context-paired CPMD route."""

import os
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import anim.mjcf_char_model as mjcf_char_model
import anim.motion_lib as motion_lib
import envs.cpmd_cond_obs as cpmd_cond_obs
import learning.add_model as add_model
import learning.cpmd_cond_agent as cpmd_cond_agent
import learning.cpmd_cond_model as cpmd_cond_model
import learning.diff_normalizer as diff_normalizer
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import util.torch_util as torch_util


DEVICE = "cpu"
STATE_DIM = 172
MOTION_DIM = 34
ERROR_DIM = STATE_DIM + MOTION_DIM
COND_DIM = ERROR_DIM + MOTION_DIM
CHAR_FILE = os.path.join(
    REPO_ROOT, "data/assets/humanoid/humanoid.xml")
MOTION_FILE = os.path.join(
    REPO_ROOT, "data/motions/humanoid/humanoid_roll.pkl")


@pytest.fixture(scope="module")
def kin_model():
    model = mjcf_char_model.MJCFCharModel(DEVICE)
    model.load(CHAR_FILE)
    return model


class FakeEnv:
    def get_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=[64], dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(
            low=-1.0, high=1.0, shape=[28], dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=[STATE_DIM], dtype=np.float32)

    def get_disc_state_obs_dim(self):
        return STATE_DIM

    def get_cpmd_error_dim(self):
        return MOTION_DIM

    def get_cpmd_context_dim(self):
        return MOTION_DIM


def model_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def identity_quat(n=1, joints=None):
    shape = [n, 4] if joints is None else [n, joints, 4]
    value = torch.zeros(shape)
    value[..., 3] = 1.0
    return value


def test_conditional_model_is_exact_add_at_initialization():
    env = FakeEnv()
    torch.manual_seed(7)
    base = add_model.ADDModel(model_config(), env)
    torch.manual_seed(7)
    cond = cpmd_cond_model.CPMDConditionalModel(model_config(), env)

    state = torch.randn([32, STATE_DIM])
    assert torch.equal(base.eval_disc(state), cond.eval_disc(state))
    assert torch.count_nonzero(cond.get_added_input_weights()).item() == 0
    assert cond.get_error_dim() == ERROR_DIM
    assert cond.get_conditional_dim() == COND_DIM


def test_first_paired_update_reaches_added_columns():
    torch.manual_seed(8)
    model = cpmd_cond_model.CPMDConditionalModel(model_config(), FakeEnv())
    neg_error = torch.randn([64, ERROR_DIM])
    pos_error = torch.zeros_like(neg_error)
    context = torch.randn([64, MOTION_DIM])
    pos = model.eval_cond(pos_error, context).squeeze(-1)
    neg = model.eval_cond(neg_error, context).squeeze(-1)
    loss = 0.5 * (
        torch.nn.functional.softplus(-pos).mean()
        + torch.nn.functional.softplus(neg).mean())
    loss.backward()

    first_grad = model._disc_layers[0].weight.grad
    assert first_grad is not None
    assert torch.linalg.vector_norm(
        first_grad[:, STATE_DIM:ERROR_DIM]).item() > 0.0
    assert torch.linalg.vector_norm(
        first_grad[:, ERROR_DIM:]).item() > 0.0


def test_positive_and_negative_rows_share_context_exactly():
    class IdentityNorm:
        @staticmethod
        def normalize(value):
            return value

    class Stub:
        _normalize_conditional_inputs = (
            cpmd_cond_agent.CPMDConditionalAgent.
            _normalize_conditional_inputs)
        _build_paired_rows = (
            cpmd_cond_agent.CPMDConditionalAgent._build_paired_rows)

    stub = Stub()
    stub._disc_obs_norm = IdentityNorm()
    stub._hist_error_norm = IdentityNorm()
    stub._ref_context_norm = IdentityNorm()
    state = torch.randn([13, STATE_DIM])
    hist = torch.randn([13, MOTION_DIM])
    context = torch.randn([13, MOTION_DIM])
    pos, neg, paired_context = stub._build_paired_rows(
        state, hist, context)

    assert torch.equal(pos, torch.zeros_like(pos))
    assert torch.equal(neg, torch.cat([state, hist], dim=-1))
    assert paired_context.data_ptr() == context.data_ptr()


def test_scale_normalizers_preserve_zero():
    hist_norm = diff_normalizer.DiffNormalizer([MOTION_DIM], DEVICE)
    context_norm = diff_normalizer.DiffNormalizer([MOTION_DIM], DEVICE)
    hist_norm.record(torch.randn([32, MOTION_DIM]))
    context_norm.record(torch.randn([32, MOTION_DIM]))
    hist_norm.update()
    context_norm.update()
    zero = torch.zeros([4, MOTION_DIM])
    assert torch.equal(hist_norm.normalize(zero), zero)
    assert torch.equal(context_norm.normalize(zero), zero)


def test_motion_error_memory_recurrence_and_stable_buffers(kin_model):
    rho = 0.9
    memory = cpmd_cond_obs.CPMDConditionalMemory(
        2, kin_model, rho, DEVICE)
    joints = kin_model.get_num_joints() - 1
    root = torch.zeros([2, 3])
    rot = identity_quat(2)
    joint = identity_quat(2, joints)
    env_ids = torch.arange(2)
    context = torch.randn([2, MOTION_DIM])
    memory.reset(env_ids, root, rot, joint, context)
    error_ptr = memory.get_error_memory().data_ptr()
    context_ptr = memory.get_ref_context().data_ptr()

    sim = root.clone()
    ref = root.clone()
    ref[:, 0] = torch.tensor([0.1, 0.2])
    memory.push(sim, rot, joint, ref, rot, joint, identity_quat(2))
    expected = torch.zeros([2, MOTION_DIM])
    expected[:, 0] = torch.tensor([0.1, 0.2])
    assert torch.allclose(memory.get_error_memory(), expected, atol=1e-6)
    assert memory.get_error_memory().data_ptr() == error_ptr
    assert memory.get_ref_context().data_ptr() == context_ptr

    ref[:, 0] += torch.tensor([0.05, -0.03])
    memory.push(sim, rot, joint, ref, rot, joint, identity_quat(2))
    expected[:, 0] = rho * expected[:, 0] + torch.tensor([0.05, -0.03])
    assert torch.allclose(memory.get_error_memory(), expected, atol=1e-6)


def test_partial_reset_does_not_touch_other_environment(kin_model):
    memory = cpmd_cond_obs.CPMDConditionalMemory(
        2, kin_model, 0.9, DEVICE)
    joints = kin_model.get_num_joints() - 1
    root = torch.zeros([2, 3])
    rot = identity_quat(2)
    joint = identity_quat(2, joints)
    memory.reset(torch.arange(2), root, rot, joint,
                 torch.ones([2, MOTION_DIM]))
    ref = root.clone()
    ref[:, 1] = torch.tensor([0.2, 0.4])
    memory.push(root, rot, joint, ref, rot, joint, identity_quat(2))
    untouched_error = memory.get_error_memory()[1].clone()
    untouched_context = memory.get_ref_context()[1].clone()
    memory.reset(torch.tensor([0]), root[:1], rot[:1], joint[:1],
                 torch.zeros([1, MOTION_DIM]))
    assert torch.equal(memory.get_error_memory()[1], untouched_error)
    assert torch.equal(memory.get_ref_context()[1], untouched_context)


def test_phase_context_obeys_recurrence_for_arbitrary_phase(kin_model):
    library = motion_lib.MotionLib(MOTION_FILE, kin_model, DEVICE)
    dt = 1.0 / 30.0
    rho = cpmd_cond_obs.calc_memory_decay(32.0 / 30.0, dt)
    table = cpmd_cond_obs.PhaseReferenceContext(
        library, kin_model, rho, dt, grid_size=512,
        tail_tolerance=1e-5, device=DEVICE)

    motion_ids = torch.zeros([17], dtype=torch.long)
    length = library.get_motion_length(motion_ids[:1])[0]
    times = torch.linspace(0.013, float(length) - 0.017, 17)
    prev = table.lookup(motion_ids, times)
    nxt = table.lookup(motion_ids, times + dt)

    pose0 = library.calc_motion_frame(motion_ids, times)
    pose1 = library.calc_motion_frame(motion_ids, times + dt)
    ids0 = torch.zeros([17], dtype=torch.long)
    zero = torch.zeros([17])
    root_rot0 = library.calc_motion_frame(ids0, zero)[1]
    anchor = cpmd_cond_obs.calc_motion_anchor_quat_inv(root_rot0)
    increment = cpmd_cond_obs.calc_motion_increment(
        kin_model,
        pose1[0], pose1[1], pose1[4],
        pose0[0], pose0[1], pose0[4], anchor)
    residual = nxt - (rho * prev + increment)
    assert torch.sqrt(torch.mean(torch.square(residual))).item() < 5e-3

    # WRAP lookup is periodic even when the clip length is not assumed to be
    # an integer number of control steps.
    assert torch.allclose(
        table.lookup(torch.tensor([0]), torch.tensor([0.0])),
        table.lookup(torch.tensor([0]), length.reshape(1)),
        atol=1e-5,
    )


def test_replay_triplets_remain_row_aligned():
    class Stub:
        _store_disc_replay_data = (
            cpmd_cond_agent.CPMDConditionalAgent._store_disc_replay_data)

    stub = Stub()
    stub._device = DEVICE
    stub._disc_replay_samples = 1000
    stub._exp_buffer = experience_buffer.ExperienceBuffer(8, 4, DEVICE)
    stub._disc_buffer = experience_buffer.ExperienceBuffer(32, 1, DEVICE)
    for step in range(8):
        row = torch.arange(4, dtype=torch.float32) + 4 * step
        stub._exp_buffer.record("disc_diff", row[:, None])
        stub._exp_buffer.record("cpmd_error_memory", (10 * row)[:, None])
        stub._exp_buffer.record("cpmd_ref_context", (100 * row)[:, None])
        stub._exp_buffer.inc()
    stub._store_disc_replay_data()
    replay = stub._disc_buffer.sample(32)
    assert torch.equal(
        replay["cpmd_error_memory"], 10 * replay["disc_diff"])
    assert torch.equal(
        replay["cpmd_ref_context"], 100 * replay["disc_diff"])


def test_mp_optimizer_gradient_accumulation_matches_full_batch():
    config = {"type": "SGD", "learning_rate": 1e-2}
    full = torch.nn.Linear(3, 1, bias=False)
    split = torch.nn.Linear(3, 1, bias=False)
    split.load_state_dict(full.state_dict())
    opt_full = mp_optimizer.MPOptimizer(config, list(full.parameters()))
    opt_split = mp_optimizer.MPOptimizer(config, list(split.parameters()))
    x = torch.randn([20, 3])
    target = torch.randn([20, 1])

    opt_full.step(torch.mean(torch.square(full(x) - target)))
    opt_split.zero_grad()
    opt_split.backward(
        0.5 * torch.mean(torch.square(split(x[:10]) - target[:10])))
    opt_split.backward(
        0.5 * torch.mean(torch.square(split(x[10:]) - target[10:])))
    opt_split.apply_step()
    assert torch.allclose(full.weight, split.weight, atol=1e-7)


def test_conditional_train_and_eval_configs_match():
    with open(os.path.join(
            REPO_ROOT, "data/agents/cpmd_cond_humanoid_agent.yaml")) as file:
        agent = yaml.safe_load(file)
    with open(os.path.join(
            REPO_ROOT,
            "data/envs/cpmd_cond_humanoid_roll_cycle_env.yaml")) as file:
        train_env = yaml.safe_load(file)
    with open(os.path.join(
            REPO_ROOT,
            "data/envs/cpmd_cond_humanoid_roll_eval_env.yaml")) as file:
        eval_env = yaml.safe_load(file)

    assert agent["agent_name"] == "CPMD_COND"
    assert agent["disc_pair_microbatch_size"] == 4096
    assert train_env["env_name"] == "cpmd_cond"
    assert train_env["episode_length"] == 2.0
    assert eval_env["episode_length"] == 10.0
    for key in train_env:
        if key != "episode_length":
            assert train_env[key] == eval_env[key]
