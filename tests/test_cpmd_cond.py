"""Focused tests for isolated ADD plus the paired context veto."""

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


DEVICE = "cpu"
STATE_DIM = 172
MOTION_DIM = 34
CONTEXT_INPUT_DIM = 2 * MOTION_DIM
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
        "context_net": "fc_2layers_128units",
    }


def identity_quat(n=1, joints=None):
    shape = [n, 4] if joints is None else [n, joints, 4]
    value = torch.zeros(shape)
    value[..., 3] = 1.0
    return value


def clone_params(params):
    return [p.detach().clone() for p in params]


def params_equal(params, snapshot):
    return all(torch.equal(p.detach(), old)
               for p, old in zip(params, snapshot))


def test_add_branch_is_exact_stock_add_and_parameter_disjoint():
    env = FakeEnv()
    torch.manual_seed(7)
    base = add_model.ADDModel(model_config(), env)
    torch.manual_seed(7)
    model = cpmd_cond_model.CPMDConditionalModel(model_config(), env)

    state = torch.randn([32, STATE_DIM])
    assert torch.equal(base.eval_disc(state), model.eval_disc(state))
    for base_param, model_param in zip(
            base.get_disc_params(), model.get_disc_params()):
        assert torch.equal(base_param, model_param)

    base_ids = {id(p) for p in model.get_disc_params()}
    context_ids = {id(p) for p in model.get_context_params()}
    assert base_ids.isdisjoint(context_ids)
    assert model.get_context_input_dim() == CONTEXT_INPUT_DIM


def test_context_veto_is_exactly_inactive_at_initialization():
    model = cpmd_cond_model.CPMDConditionalModel(
        model_config(), FakeEnv())
    hist = torch.randn([64, MOTION_DIM])
    context = torch.randn([64, MOTION_DIM])
    pos = model.eval_context(torch.zeros_like(hist), context)
    neg = model.eval_context(hist, context)
    assert torch.equal(pos, torch.zeros_like(pos))
    assert torch.equal(neg, torch.zeros_like(neg))

    add_reward = torch.rand([64]) * 4.0
    reward, veto, ratio = (
        cpmd_cond_agent.CPMDConditionalAgent._apply_context_veto(
            add_reward, pos.squeeze(-1), neg.squeeze(-1)))
    assert torch.equal(reward, add_reward)
    assert torch.count_nonzero(veto).item() == 0
    assert torch.equal(ratio, torch.ones_like(ratio))


def test_first_context_update_has_signal_and_does_not_change_add():
    torch.manual_seed(8)
    model = cpmd_cond_model.CPMDConditionalModel(
        model_config(), FakeEnv())
    base_before = clone_params(model.get_disc_params())
    context_before = clone_params(model.get_context_params())

    hist = torch.randn([128, MOTION_DIM])
    context = torch.randn([128, MOTION_DIM])
    pos = model.eval_context(torch.zeros_like(hist), context).squeeze(-1)
    neg = model.eval_context(hist, context).squeeze(-1)
    loss = 0.5 * (
        torch.nn.functional.softplus(-pos).mean()
        + torch.nn.functional.softplus(neg).mean())
    optimizer = torch.optim.SGD(model.get_context_params(), lr=1e-2)
    optimizer.zero_grad()
    loss.backward()
    assert model._context_logits.weight.grad is not None
    assert torch.linalg.vector_norm(
        model._context_logits.weight.grad).item() > 0.0
    optimizer.step()

    assert params_equal(model.get_disc_params(), base_before)
    assert not params_equal(model.get_context_params(), context_before)


def test_add_update_does_not_change_context():
    torch.manual_seed(9)
    model = cpmd_cond_model.CPMDConditionalModel(
        model_config(), FakeEnv())
    context_before = clone_params(model.get_context_params())
    base_before = clone_params(model.get_disc_params())

    neg = model.eval_disc(torch.randn([128, STATE_DIM])).squeeze(-1)
    pos = model.eval_disc(torch.zeros([1, STATE_DIM])).squeeze(-1)
    loss = 0.5 * (
        torch.nn.functional.softplus(-pos).mean()
        + torch.nn.functional.softplus(neg).mean())
    optimizer = torch.optim.SGD(model.get_disc_params(), lr=1e-2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not params_equal(model.get_disc_params(), base_before)
    assert params_equal(model.get_context_params(), context_before)


def test_veto_reward_bounds_and_zero_history_identity():
    add_reward = torch.rand([256]) * 8.0
    pos_logit = torch.randn([256])
    neg_logit = torch.randn([256])
    reward, veto, ratio = (
        cpmd_cond_agent.CPMDConditionalAgent._apply_context_veto(
            add_reward, pos_logit, neg_logit))
    assert torch.all(veto >= 0.0)
    assert torch.all(veto <= 1.0)
    assert torch.all(reward <= add_reward + 1e-7)
    assert torch.all(reward >= 0.5 * add_reward - 1e-7)
    assert torch.all(ratio <= 1.0)
    assert torch.all(ratio >= 0.5)

    same = torch.randn([256])
    exact, exact_veto, exact_ratio = (
        cpmd_cond_agent.CPMDConditionalAgent._apply_context_veto(
            add_reward, same, same))
    assert torch.equal(exact, add_reward)
    assert torch.count_nonzero(exact_veto).item() == 0
    assert torch.equal(exact_ratio, torch.ones_like(exact_ratio))


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
    assert torch.allclose(
        table.lookup(torch.tensor([0]), torch.tensor([0.0])),
        table.lookup(torch.tensor([0]), length.reshape(1)),
        atol=1e-5,
    )


def test_base_and_context_replays_are_separate_and_context_rows_align():
    class Stub:
        _num_replay_rows = staticmethod(
            cpmd_cond_agent.CPMDConditionalAgent._num_replay_rows)
        _store_disc_replay_data = (
            cpmd_cond_agent.CPMDConditionalAgent._store_disc_replay_data)

    stub = Stub()
    stub._device = DEVICE
    stub._disc_replay_samples = 1000
    stub._context_replay_samples = 1000
    stub._exp_buffer = experience_buffer.ExperienceBuffer(8, 4, DEVICE)
    stub._disc_buffer = experience_buffer.ExperienceBuffer(32, 1, DEVICE)
    stub._context_buffer = experience_buffer.ExperienceBuffer(32, 1, DEVICE)
    for step in range(8):
        row = torch.arange(4, dtype=torch.float32) + 4 * step
        stub._exp_buffer.record("disc_diff", row[:, None])
        stub._exp_buffer.record("cpmd_error_memory", (10 * row)[:, None])
        stub._exp_buffer.record("cpmd_ref_context", (100 * row)[:, None])
        stub._exp_buffer.inc()
    stub._store_disc_replay_data()

    assert set(stub._disc_buffer._buffers) == {"disc_diff"}
    assert set(stub._context_buffer._buffers) == {
        "cpmd_error_memory", "cpmd_ref_context"}
    replay = stub._context_buffer.sample(32)
    assert torch.equal(
        replay["cpmd_ref_context"],
        10 * replay["cpmd_error_memory"],
    )


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


def test_schema_one_checkpoint_is_rejected():
    model = cpmd_cond_model.CPMDConditionalModel(
        model_config(), FakeEnv())
    state = model.state_dict()
    state["_cpmd_cond_schema"] = torch.tensor(1, dtype=torch.int64)
    target = cpmd_cond_model.CPMDConditionalModel(
        model_config(), FakeEnv())
    with pytest.raises(RuntimeError, match="checkpoint schema 1"):
        target.load_state_dict(state)


def test_veto_train_and_eval_configs_match_and_outputs_are_new():
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
    with open(os.path.join(
            REPO_ROOT, "args/cpmd_cond_humanoid_jog_args.txt")) as file:
        jog_args = file.read()
    with open(os.path.join(
            REPO_ROOT, "args/cpmd_cond_humanoid_args.txt")) as file:
        roll_args = file.read()

    assert agent["agent_name"] == "CPMD_COND"
    assert agent["context_pair_microbatch_size"] == 4096
    assert "disc_pair_microbatch_size" not in agent
    assert train_env["cpmd_schema_version"] == 2
    assert eval_env["cpmd_schema_version"] == 2
    assert train_env["episode_length"] == 2.0
    assert eval_env["episode_length"] == 10.0
    for key in train_env:
        if key != "episode_length":
            assert train_env[key] == eval_env[key]
    assert "output/cpmd_veto_jog_1k_seed0" in jog_args
    assert "output/cpmd_veto_roll_cycle_1k_seed0" in roll_args
