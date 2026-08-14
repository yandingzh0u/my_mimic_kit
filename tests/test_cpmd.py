"""Unit tests for CPMD (context-preserving motion differentials).

Everything here runs on the real humanoid kinematic model (28 dof, hinge +
spherical + fixed joints) on CPU, without a simulator, so the operator is
tested with the exact widths the training run uses:

    summary dim D     = 3 (root pos) + 3 (root rot) + 28 (dof) = 34
    interaction dim   = D (D - 1) / 2 = 561
    differential      = 172 (ADD state) + 34 + 561 = 767

Both blocks are produced every step for both sides and enter the same
discriminator together; there is no staging and no second network.

Coverage:

operator math
  - m matches the explicit exponentially weighted increment sum
  - the interaction block equals 1/2 m_i m_j and carries no recursive state
  - its differential is exactly 1/4 (dm sm^T + sm dm^T), i.e. the tracking
    error modulated by the two sides' common-mode motion
  - that differential is not a function of dm, so the interactions are not a
    re-encoding of the summary error
  - rho is pinned to a physical memory time

geometry
  - a full 2*pi root roll does not fold back to zero (per-step logs)
  - quaternion hemisphere flips (q vs -q) change nothing
  - a global yaw + translation of the whole scene changes nothing

streaming / lifecycle
  - perfect tracking gives an exactly zero differential for the whole episode
  - a partial reset touches only its own envs
  - a reset does not inherit the previous episode's summary
  - each side accumulates exactly one increment per push

training-side invariants
  - the differential dim is 767 and the scale-only normalizer maps 0 to 0
  - storing only the differential is bit-identical to keeping both endpoints
    and subtracting later, and the replay buffer keeps the pairs intact
  - the agent does not override the ADD loss or the ADD reward
  - the diagnostics are pure measurement and cannot shadow the loss
  - fetch_disc_obs_demo fails loudly instead of returning [state, zeros]
  - a checkpoint round-trips within a config and fails loudly across widths
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import gymnasium.spaces as spaces

import anim.mjcf_char_model as mjcf_char_model
import envs.amp_env as amp_env
import envs.cpmd_env as cpmd_env
import envs.cpmd_obs as cpmd_obs
import learning.add_agent as add_agent
import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.cpmd_agent as cpmd_agent
import learning.diff_normalizer as diff_normalizer
import learning.experience_buffer as experience_buffer
import util.torch_util as torch_util

DEVICE = "cpu"
CHAR_FILE = os.path.join(REPO_ROOT, "data/assets/humanoid/humanoid.xml")

# ADD state differential dim for the humanoid roll config (verified against a
# live env in the Isaac smoke run)
STATE_DIM = 172
SUMMARY_DIM = 34
INTERACTION_DIM = 561
TOTAL_DIM = STATE_DIM + SUMMARY_DIM + INTERACTION_DIM

RHO = 0.96923

@pytest.fixture(scope="module")
def kin_model():
    model = mjcf_char_model.MJCFCharModel(DEVICE)
    model.load(CHAR_FILE)
    return model

def make_hist(kin_model, num_envs=1, rho=RHO):
    return cpmd_obs.CPMDHistory(num_envs=num_envs, kin_char_model=kin_model,
                                rho=rho, device=DEVICE)

def identity_quat(n=1, num_joints=None):
    shape = [n, 4] if (num_joints is None) else [n, num_joints, 4]
    q = torch.zeros(shape)
    q[..., 3] = 1.0
    return q

def axis_quat(axis, angle, n=1):
    axis = torch.tensor([axis], dtype=torch.float32).repeat(n, 1)
    return torch_util.axis_angle_to_quat(axis, torch.full([n], float(angle)))

def rest_state(kin_model, n=1):
    num_joints = kin_model.get_num_joints() - 1
    return (torch.zeros([n, 3]), identity_quat(n), identity_quat(n, num_joints))

def push_pos_path(hist, kin_model, positions, anchor_inv=None):
    """Stream a pure root-translation path."""
    n = positions[0].shape[0]
    if (anchor_inv is None):
        anchor_inv = identity_quat(n)
    root_rot = identity_quat(n)
    joint_rot = identity_quat(n, kin_model.get_num_joints() - 1)

    hist.reset(torch.arange(n), positions[0], root_rot, joint_rot)
    for p in positions[1:]:
        hist.push(p, root_rot, joint_rot, anchor_inv)
    return

def random_walk(steps, seed, scale=0.2):
    torch.manual_seed(seed)
    return [torch.randn([1, 3]) * scale for _ in range(steps)]

def stream_increments(kin_model, increments, rho=RHO):
    pos = [torch.zeros([1, 3])]
    for step in increments:
        pos.append(pos[-1] + step)
    hist = make_hist(kin_model, rho=rho)
    push_pos_path(hist, kin_model, pos)
    return hist.extract()

# ---------------------------------------------------------------------------
# operator math
# ---------------------------------------------------------------------------

def test_m_matches_explicit_discounted_sum(kin_model):
    """m_T = sum_k rho^(T-k) xi_k, computed independently from the raw path."""
    torch.manual_seed(0)
    steps = 12
    pos = [torch.zeros([1, 3])]
    for _ in range(steps):
        pos.append(pos[-1] + torch.randn([1, 3]) * 0.1)

    hist = make_hist(kin_model)
    push_pos_path(hist, kin_model, pos)

    expected = torch.zeros([1, 3])
    for k in range(steps):
        expected += (RHO ** (steps - 1 - k)) * (pos[k + 1] - pos[k])

    m = hist.calc_motion_summary()
    assert torch.allclose(m[:, 0:3], expected, atol=1e-5)
    assert torch.allclose(m[:, 3:], torch.zeros([1, SUMMARY_DIM - 3]), atol=1e-6)

def test_interactions_are_half_the_outer_product_of_m(kin_model):
    """Per side the block is exactly {1/2 m_i m_j}_{i<j}, so it needs no
    recursive state at all."""
    out = stream_increments(kin_model, random_walk(9, seed=11))
    m = out[0, :SUMMARY_DIM]
    block = out[0, SUMMARY_DIM:]

    idx = torch.triu_indices(SUMMARY_DIM, SUMMARY_DIM, offset=1)
    expected = 0.5 * torch.outer(m, m)[idx[0], idx[1]]
    assert torch.allclose(block, expected, atol=1e-6)
    assert torch.max(torch.abs(block)).item() > 1e-4

def test_interaction_differential_is_the_common_mode_modulation(kin_model):
    """The identity the method rests on:

        dc = 1/4 (dm sm^T + sm dm^T),  dm = m_ref - m_sim, sm = m_ref + m_sim

    so what reaches the discriminator is the tracking error scaled by how much
    absolute motion the two sides are doing."""
    base = random_walk(11, seed=14)
    extra = random_walk(11, seed=15, scale=0.05)

    sim = stream_increments(kin_model, base)
    ref = stream_increments(kin_model, [a + b for a, b in zip(base, extra)])

    dm = ref[:, :SUMMARY_DIM] - sim[:, :SUMMARY_DIM]
    sm = ref[:, :SUMMARY_DIM] + sim[:, :SUMMARY_DIM]
    i, j = torch.triu_indices(SUMMARY_DIM, SUMMARY_DIM, offset=1)

    dc = ref[:, SUMMARY_DIM:] - sim[:, SUMMARY_DIM:]
    expected = 0.25 * (dm[:, i] * sm[:, j] + sm[:, i] * dm[:, j])
    assert torch.allclose(dc, expected, atol=1e-5)

def test_interaction_differential_is_not_a_function_of_the_summary_error(kin_model):
    """The expansion is per side and the subtraction happens after it, so the
    same summary error under different absolute motion produces a different
    interaction differential. This is exactly the context a differential of
    raw errors cannot carry."""
    extra = random_walk(11, seed=16, scale=0.05)

    def differential(base_seed):
        base = random_walk(11, seed=base_seed)
        sim = stream_increments(kin_model, base)
        ref = stream_increments(kin_model, [a + b for a, b in zip(base, extra)])
        return ref - sim

    d_slow = differential(17)
    d_fast = differential(18)

    assert torch.allclose(d_slow[:, :SUMMARY_DIM], d_fast[:, :SUMMARY_DIM], atol=1e-6)
    assert torch.max(torch.abs(d_slow[:, SUMMARY_DIM:] - d_fast[:, SUMMARY_DIM:])).item() > 1e-3

def test_memory_decay_is_physical():
    """rho is pinned to a physical memory time, so halving the control step
    keeps the same effective memory."""
    rho_30 = cpmd_obs.calc_memory_decay(32.0 / 30.0, 1.0 / 30.0)
    rho_60 = cpmd_obs.calc_memory_decay(32.0 / 30.0, 1.0 / 60.0)
    assert abs(rho_30 - np.exp(-1.0 / 32.0)) < 1e-9
    assert abs(rho_60 ** 2 - rho_30) < 1e-9

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_full_roll_does_not_fold_to_zero(kin_model):
    """A complete 2*pi root rotation accumulates 2*pi: the increments are
    per-step logs, so the endpoint quaternion alias (q ~ identity) cannot
    hide a completed roll."""
    n = 1
    steps = 36
    d_angle = 2.0 * np.pi / steps

    hist = make_hist(kin_model, rho=1.0)
    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    hist.reset(torch.arange(n), root_pos, root_rot, joint_rot)

    anchor_inv = identity_quat(n)
    for k in range(1, steps + 1):
        hist.push(root_pos, axis_quat([0.0, 1.0, 0.0], d_angle * k, n), joint_rot, anchor_inv)

    winding = hist.calc_motion_summary()[:, 3:6]
    assert abs(winding[0, 1].item() - 2.0 * np.pi) < 1e-3
    assert torch.max(torch.abs(winding[:, [0, 2]])).item() < 1e-4

def test_quaternion_hemisphere_invariance(kin_model):
    """q and -q describe the same rotation and must give the same blocks."""
    n = 1
    steps = 10
    d_angle = 0.3

    def run(flip):
        hist = make_hist(kin_model)
        root_pos, root_rot, joint_rot = rest_state(kin_model, n)
        hist.reset(torch.arange(n), root_pos, root_rot, joint_rot)
        anchor_inv = identity_quat(n)
        for k in range(1, steps + 1):
            rot = axis_quat([0.0, 1.0, 0.0], d_angle * k, n)
            jr = joint_rot.clone()
            if (flip and k % 2 == 0):
                rot = -rot
                jr = -jr
            hist.push(root_pos + 0.01 * k, rot, jr, anchor_inv)
        return hist.extract()

    assert torch.allclose(run(False), run(True), atol=1e-5)

def test_global_yaw_and_translation_invariance(kin_model):
    """Rotating and translating the whole scene changes nothing: increments
    kill the translation, the phase-0 heading anchor kills the yaw."""
    n = 1
    steps = 12
    yaw = 0.9
    offset = torch.tensor([[3.0, -2.0, 0.0]])

    def run(rotated):
        num_joints = kin_model.get_num_joints() - 1
        joint_rot = identity_quat(n, num_joints)
        yaw_q = axis_quat([0.0, 0.0, 1.0], yaw, n)

        # the anchor always comes from the reference root rotation at phase 0
        root_rot0 = identity_quat(n)
        if (rotated):
            root_rot0 = torch_util.quat_mul(yaw_q, root_rot0)
        anchor_inv = cpmd_obs.calc_motion_anchor_quat_inv(root_rot0)

        hist = make_hist(kin_model)

        positions, rotations = [], []
        for k in range(steps + 1):
            p = torch.tensor([[0.05 * k, 0.02 * k, 0.9 + 0.01 * k]])
            r = axis_quat([0.0, 1.0, 0.0], 0.25 * k, n)
            if (rotated):
                p = torch_util.quat_rotate(yaw_q, p) + offset
                r = torch_util.quat_mul(yaw_q, r)
            positions.append(p)
            rotations.append(r)

        hist.reset(torch.arange(n), positions[0], rotations[0], joint_rot)
        for k in range(1, steps + 1):
            hist.push(positions[k], rotations[k], joint_rot, anchor_inv)
        return hist.extract()

    assert torch.allclose(run(False), run(True), atol=1e-4)

# ---------------------------------------------------------------------------
# streaming / lifecycle
# ---------------------------------------------------------------------------

def test_perfect_tracking_gives_zero_differential(kin_model):
    """Two operators fed identical states stay identical: the ADD ideal point
    Delta = 0 survives the quadratic expansion, for the whole episode and not
    just at reset."""
    torch.manual_seed(4)
    n = 3
    steps = 15
    num_joints = kin_model.get_num_joints() - 1

    sim = make_hist(kin_model, num_envs=n)
    ref = make_hist(kin_model, num_envs=n)

    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    env_ids = torch.arange(n)
    sim.reset(env_ids, root_pos, root_rot, joint_rot)
    ref.reset(env_ids, root_pos, root_rot, joint_rot)
    assert torch.max(torch.abs(ref.extract() - sim.extract())).item() == 0.0

    anchor_inv = identity_quat(n)
    for _ in range(steps):
        root_pos = root_pos + torch.randn([n, 3]) * 0.05
        root_rot = torch_util.quat_mul(axis_quat([0.0, 1.0, 0.0], 0.2, n), root_rot)
        joint_rot = torch_util.quat_normalize(joint_rot + torch.randn([n, num_joints, 4]) * 0.02)

        sim.push(root_pos, root_rot, joint_rot, anchor_inv)
        ref.push(root_pos, root_rot, joint_rot, anchor_inv)
        assert torch.max(torch.abs(ref.extract() - sim.extract())).item() == 0.0

def test_partial_reset_does_not_pollute_other_envs(kin_model):
    n = 4
    steps = 6
    hist = make_hist(kin_model, num_envs=n)

    pos = [torch.zeros([n, 3])]
    torch.manual_seed(5)
    for _ in range(steps):
        pos.append(pos[-1] + torch.randn([n, 3]) * 0.2)
    push_pos_path(hist, kin_model, pos)

    before = hist.extract().clone()

    reset_ids = torch.tensor([1, 3])
    root_pos, root_rot, joint_rot = rest_state(kin_model, len(reset_ids))
    hist.reset(reset_ids, root_pos, root_rot, joint_rot)

    after = hist.extract()
    keep = torch.tensor([0, 2])
    assert torch.allclose(after[keep], before[keep], atol=0.0)
    assert torch.max(torch.abs(after[reset_ids])).item() == 0.0
    assert torch.all(hist.get_push_count()[reset_ids] == 0)
    assert torch.all(hist.get_push_count()[keep] == steps)

def test_reset_does_not_inherit_previous_episode(kin_model):
    """After a reset the operator restarts from zero, and the first increment
    of the new episode is measured from the reset pose (not from the last
    pose of the previous episode)."""
    n = 1
    hist = make_hist(kin_model)

    far = [torch.zeros([n, 3]), torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([[5.0, 5.0, 0.0]])]
    push_pos_path(hist, kin_model, far)
    assert torch.max(torch.abs(hist.extract())).item() > 1.0

    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    reset_pos = torch.tensor([[100.0, -100.0, 1.0]])
    hist.reset(torch.arange(n), reset_pos, root_rot, joint_rot)
    assert torch.max(torch.abs(hist.extract())).item() == 0.0

    hist.push(torch.tensor([[100.1, -100.0, 1.0]]), root_rot, joint_rot, identity_quat(n))
    out = hist.extract()
    assert abs(out[0, 0].item() - 0.1) < 1e-5

def test_push_count_tracks_increments(kin_model):
    """Each push contributes exactly one increment; both sides are pushed the
    same number of times when streamed together (the env-level guarantee is
    re-checked at runtime in the Isaac smoke run)."""
    n = 2
    sim = make_hist(kin_model, num_envs=n)
    ref = make_hist(kin_model, num_envs=n)
    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    sim.reset(torch.arange(n), root_pos, root_rot, joint_rot)
    ref.reset(torch.arange(n), root_pos, root_rot, joint_rot)

    for k in range(7):
        sim.push(root_pos + 0.1 * k, root_rot, joint_rot, identity_quat(n))
        ref.push(root_pos + 0.1 * k, root_rot, joint_rot, identity_quat(n))
        assert torch.all(sim.get_push_count() == k + 1)
        assert torch.all(ref.get_push_count() == sim.get_push_count())

# ---------------------------------------------------------------------------
# training-side invariants
# ---------------------------------------------------------------------------

def test_differential_dims(kin_model):
    hist = make_hist(kin_model)
    assert hist.get_summary_dim() == SUMMARY_DIM
    assert hist.get_interaction_dim() == INTERACTION_DIM
    assert hist.get_obs_dim() == SUMMARY_DIM + INTERACTION_DIM
    assert STATE_DIM + hist.get_obs_dim() == TOTAL_DIM == 767

def test_normalizer_keeps_zero_at_zero():
    """The scale-only DiffNormalizer maps the ADD ideal point to itself, for
    any recorded statistics."""
    norm = diff_normalizer.DiffNormalizer([TOTAL_DIM], device=DEVICE)
    torch.manual_seed(6)
    norm.record(torch.randn([256, TOTAL_DIM]) * 7.0)
    norm.update()

    zeros = torch.zeros([4, TOTAL_DIM])
    assert torch.max(torch.abs(norm.normalize(zeros))).item() == 0.0

def make_disc_data(steps, num_envs, seed):
    torch.manual_seed(seed)
    obs = torch.randn([steps, num_envs, TOTAL_DIM])
    demo = torch.randn([steps, num_envs, TOTAL_DIM])
    return obs, demo

def test_storing_only_the_differential_is_bit_identical():
    """The endpoints are no longer kept: the difference every consumer used to
    form is formed once at record time instead. Same arithmetic on the same
    values, so the discriminator input is bit-identical -- and everything
    downstream is a deterministic function of that input."""
    steps, num_envs = 5, 8
    obs, demo = make_disc_data(steps, num_envs, seed=20)

    old = experience_buffer.ExperienceBuffer(steps, num_envs, DEVICE)
    new = experience_buffer.ExperienceBuffer(steps, num_envs, DEVICE)
    for k in range(steps):
        old.record("disc_obs", obs[k])
        old.record("disc_obs_demo", demo[k])
        new.record("disc_diff", demo[k] - obs[k])
        old.inc()
        new.inc()

    old_diff = old.get_data_flat("disc_obs_demo") - old.get_data_flat("disc_obs")
    assert torch.equal(new.get_data_flat("disc_diff"), old_diff)

    # and the endpoints are genuinely gone, which is the point of the change
    assert set(new._buffers.keys()) == {"disc_diff"}

class ReplayStub:
    """Exercises the real replay path without constructing an agent."""
    _store_disc_replay_data = add_agent.ADDAgent._store_disc_replay_data

def test_replay_buffer_keeps_the_pairs_intact():
    """Replay rows must be differentials of matching sim/ref pairs. With the
    endpoints stored separately this was an indexing invariant; storing the
    difference makes it structural."""
    steps, num_envs = 4, 16
    obs, demo = make_disc_data(steps, num_envs, seed=21)

    agent = ReplayStub()
    agent._device = DEVICE
    agent._disc_replay_samples = 1000
    agent._exp_buffer = experience_buffer.ExperienceBuffer(steps, num_envs, DEVICE)
    agent._disc_buffer = experience_buffer.ExperienceBuffer(1000, 1, DEVICE)

    for k in range(steps):
        agent._exp_buffer.record("disc_diff", demo[k] - obs[k])
        agent._exp_buffer.inc()

    torch.manual_seed(22)
    agent._store_disc_replay_data()

    truth = (demo - obs).reshape(-1, TOTAL_DIM)
    replay = agent._disc_buffer.sample(64)["disc_diff"].squeeze(1)
    assert replay.shape[0] == 64

    def is_a_true_pair(rows):
        return torch.eq(rows.unsqueeze(1), truth.unsqueeze(0)).all(dim=-1).any(dim=-1)

    assert torch.all(is_a_true_pair(replay))

    # negative control: a mismatched pair would have been caught
    cross = demo[0, 0] - obs[0, 1]
    assert not is_a_true_pair(cross.unsqueeze(0)).item()

def test_agent_does_not_change_add_training():
    """The method is a differential, not a new training rule: the reward path
    and the model builder are inherited untouched."""
    cls = cpmd_agent.CPMDAgent
    assert cls._compute_rewards is add_agent.ADDAgent._compute_rewards
    assert cls._calc_disc_rewards is amp_agent.AMPAgent._calc_disc_rewards
    assert cls._build_model is add_agent.ADDAgent._build_model
    assert cls._build_normalizers is add_agent.ADDAgent._build_normalizers

def test_diagnostics_are_pure_measurement():
    """The only override, _compute_disc_loss, adds diagnostic keys and can
    never shadow the ADD loss or its terms."""
    # the agent is an nn.Module, so exercise the exact same function object on
    # a light stand-in instead of half-constructing an agent
    class DiagStub:
        _compute_disc_diagnostics = cpmd_agent.CPMDAgent._compute_disc_diagnostics

    agent = DiagStub()
    agent._state_dim = STATE_DIM
    agent._summary_dim = SUMMARY_DIM
    agent._interaction_dim = INTERACTION_DIM
    agent._disc_obs_norm = diff_normalizer.DiffNormalizer([TOTAL_DIM], device=DEVICE)

    torch.manual_seed(7)
    info = agent._compute_disc_diagnostics(torch.randn([32, TOTAL_DIM]))

    add_keys = {"disc_loss", "disc_grad_penalty", "disc_logit_loss", "disc_pos_acc",
                "disc_neg_acc", "disc_pos_logit", "disc_neg_logit"}
    assert add_keys.isdisjoint(info.keys())
    for key in ["disc_state_rms", "disc_summary_rms", "disc_interaction_rms",
                "disc_interaction_nonzero_frac", "disc_norm_min_scale_frac"]:
        assert key in info
        assert torch.isfinite(info[key]).all()

    zero_info = agent._compute_disc_diagnostics(torch.zeros([8, TOTAL_DIM]))
    assert zero_info["disc_state_rms"].item() == 0.0
    assert zero_info["disc_interaction_rms"].item() == 0.0

def test_demo_api_fails_loudly():
    """The reference blocks depend on episode age, so there is no honest way
    to answer fetch_disc_obs_demo from a sampled motion frame. It must raise
    rather than hand out [state, zeros]."""
    with pytest.raises(RuntimeError, match="paired differential"):
        cpmd_env.CPMDEnv.fetch_disc_obs_demo(None, 4)
    with pytest.raises(RuntimeError):
        cpmd_env.CPMDEnv._compute_disc_obs_demo(None, None, None)

    # the base class builds the obs space by calling it, so CPMD has to derive
    # the shape some other way or nothing would come up at all
    assert cpmd_env.CPMDEnv.get_disc_obs_space is not amp_env.AMPEnv.get_disc_obs_space
    assert "fetch_disc_obs_demo" not in cpmd_env.CPMDEnv.get_disc_obs_space.__code__.co_names

# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

class FakeEnv:
    def __init__(self, disc_dim):
        self._disc_dim = disc_dim

    def get_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=[64], dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=[28], dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=[self._disc_dim], dtype=np.float32)

def make_add_model(disc_dim):
    config = {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }
    torch.manual_seed(0)
    return add_model.ADDModel(config, FakeEnv(disc_dim))

def test_checkpoint_round_trip():
    model = make_add_model(TOTAL_DIM)
    other = make_add_model(TOTAL_DIM)
    other.load_state_dict(model.state_dict())

    x = torch.randn([16, TOTAL_DIM])
    assert torch.allclose(model.eval_disc(x), other.eval_disc(x), atol=0.0)

@pytest.mark.parametrize("src,dst", [(STATE_DIM + SUMMARY_DIM, TOTAL_DIM),
                                     (TOTAL_DIM, STATE_DIM + SUMMARY_DIM),
                                     (STATE_DIM, TOTAL_DIM)])
def test_mismatched_checkpoints_fail_loudly(src, dst):
    """A differential of a different width can never be silently loaded."""
    src_model = make_add_model(src)
    dst_model = make_add_model(dst)
    with pytest.raises(RuntimeError):
        dst_model.load_state_dict(src_model.state_dict())
