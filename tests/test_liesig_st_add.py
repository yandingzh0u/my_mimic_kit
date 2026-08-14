"""Unit tests for LieSig-STADD.

Everything here runs on the real humanoid kinematic model (28 dof, hinge +
spherical + fixed joints) on CPU, without a simulator, so the operator is
tested with the exact tangent dimension the training runs use:

    tangent dim D = 3 (root pos) + 3 (root rot) + 28 (dof) = 34
    area dim      = D (D - 1) / 2 = 561
    Level-1 Phi   = 34,      Level-2 Phi = 595
    differential  = 172 (ADD state) + 34 = 206  |  + 561 = 767

Coverage:

operator math
  - m matches the explicit exponentially weighted increment sum
  - the packed area matches a brute-force full-matrix recursion
  - A is exactly antisymmetric
  - a path confined to one tangent direction has zero area
  - reversing the traversal order leaves m unchanged and flips A's sign
  - rho = 1 reproduces the ordinary finite-sequence level-2 (Levy) area

geometry
  - a full 2*pi root roll does not fold back to zero (per-step logs)
  - quaternion hemisphere flips (q vs -q) change nothing
  - a global yaw + translation of the whole scene changes nothing

streaming / lifecycle
  - perfect tracking gives an exactly zero differential for the whole episode
  - a partial reset touches only its own envs
  - a reset does not inherit the previous episode's m or A
  - each side accumulates exactly one increment per push

training-side invariants
  - the differential dims are 206 (L1) and 767 (L2)
  - the scale-only normalizer maps 0 to 0 (the ADD ideal point survives)
  - the agent does not override the ADD loss or the ADD reward
  - the diagnostics are pure measurement and cannot shadow the loss
  - a checkpoint round-trips within a config and fails loudly across
    L1 / L2 / ST-ADD-v1 configs
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
import envs.lie_signature_obs as lie_signature_obs
import learning.add_agent as add_agent
import learning.add_model as add_model
import learning.amp_agent as amp_agent
import learning.diff_normalizer as diff_normalizer
import learning.liesig_st_add_agent as liesig_st_add_agent
import util.torch_util as torch_util

DEVICE = "cpu"
CHAR_FILE = os.path.join(REPO_ROOT, "data/assets/humanoid/humanoid.xml")

# ADD state differential dim for the humanoid roll config (verified against a
# live env in the Isaac smoke run)
STATE_DIM = 172
TANGENT_DIM = 34
AREA_DIM = 561
L1_TOTAL = STATE_DIM + TANGENT_DIM
L2_TOTAL = STATE_DIM + TANGENT_DIM + AREA_DIM
V1_TOTAL = 409

RHO = 0.96923

@pytest.fixture(scope="module")
def kin_model():
    model = mjcf_char_model.MJCFCharModel(DEVICE)
    model.load(CHAR_FILE)
    return model

def make_hist(kin_model, num_envs=1, order=2, rho=RHO):
    return lie_signature_obs.LieSigHistory(num_envs=num_envs, kin_char_model=kin_model,
                                           order=order, rho=rho, device=DEVICE)

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
    """Stream a pure root-translation path; returns the increments used."""
    n = positions[0].shape[0]
    if (anchor_inv is None):
        anchor_inv = identity_quat(n)
    root_rot = identity_quat(n)
    joint_rot = identity_quat(n, kin_model.get_num_joints() - 1)

    hist.reset(torch.arange(n), positions[0], root_rot, joint_rot)
    for p in positions[1:]:
        hist.push(p, root_rot, joint_rot, anchor_inv)
    return

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

    hist = make_hist(kin_model, order=1)
    push_pos_path(hist, kin_model, pos)

    xis = [pos[k + 1] - pos[k] for k in range(steps)]
    expected = torch.zeros([1, 3])
    for k, xi in enumerate(xis):
        expected += (RHO ** (steps - 1 - k)) * xi

    m = hist.extract()
    assert torch.allclose(m[:, 0:3], expected, atol=1e-5)
    assert torch.allclose(m[:, 3:], torch.zeros([1, TANGENT_DIM - 3]), atol=1e-6)

def test_packed_area_matches_full_matrix(kin_model):
    """The packed strict upper triangle equals a brute-force recursion on the
    full D x D matrix."""
    torch.manual_seed(1)
    steps = 10
    pos = [torch.zeros([1, 3])]
    for _ in range(steps):
        pos.append(pos[-1] + torch.randn([1, 3]) * 0.2)

    hist = make_hist(kin_model, order=2)
    push_pos_path(hist, kin_model, pos)

    D = TANGENT_DIM
    m = torch.zeros([D])
    A = torch.zeros([D, D])
    for k in range(steps):
        xi = torch.zeros([D])
        xi[0:3] = (pos[k + 1] - pos[k])[0]
        A = RHO * RHO * A + 0.5 * RHO * (torch.outer(m, xi) - torch.outer(xi, m))
        m = RHO * m + xi

    packed = hist.extract()[:, TANGENT_DIM:]
    full = lie_signature_obs.unpack_area(packed, D)[0]
    assert torch.allclose(full, A, atol=1e-5)

def test_area_is_antisymmetric(kin_model):
    torch.manual_seed(2)
    steps = 8
    pos = [torch.zeros([1, 3])]
    for _ in range(steps):
        pos.append(pos[-1] + torch.randn([1, 3]) * 0.3)

    hist = make_hist(kin_model, order=2)
    push_pos_path(hist, kin_model, pos)

    full = lie_signature_obs.unpack_area(hist.extract()[:, TANGENT_DIM:], TANGENT_DIM)[0]
    assert torch.allclose(full, -full.T, atol=0.0)
    assert torch.allclose(torch.diagonal(full), torch.zeros(TANGENT_DIM), atol=0.0)

def test_single_axis_path_has_zero_area(kin_model):
    """A path confined to one tangent direction sweeps no area, however long
    or fast it is."""
    steps = 20
    pos = [torch.zeros([1, 3])]
    for k in range(steps):
        step = torch.zeros([1, 3])
        step[0, 0] = 0.1 * (k + 1)
        pos.append(pos[-1] + step)

    hist = make_hist(kin_model, order=2)
    push_pos_path(hist, kin_model, pos)

    area = hist.extract()[:, TANGENT_DIM:]
    assert torch.max(torch.abs(area)).item() < 1e-6

def test_order_reversal_flips_area_but_not_m(kin_model):
    """e_i then e_j versus e_j then e_i: same level 1, opposite level 2."""
    e_x = torch.tensor([[1.0, 0.0, 0.0]])
    e_y = torch.tensor([[0.0, 1.0, 0.0]])

    fwd = make_hist(kin_model, order=2, rho=1.0)
    push_pos_path(fwd, kin_model, [torch.zeros([1, 3]), e_x, e_x + e_y])

    rev = make_hist(kin_model, order=2, rho=1.0)
    push_pos_path(rev, kin_model, [torch.zeros([1, 3]), e_y, e_x + e_y])

    phi_f, phi_r = fwd.extract(), rev.extract()
    assert torch.allclose(phi_f[:, :TANGENT_DIM], phi_r[:, :TANGENT_DIM], atol=1e-6)

    area_f = phi_f[:, TANGENT_DIM:]
    area_r = phi_r[:, TANGENT_DIM:]
    assert torch.allclose(area_f, -area_r, atol=1e-6)
    assert torch.max(torch.abs(area_f)).item() > 0.1

def test_rho_one_recovers_plain_levy_area(kin_model):
    """With rho = 1 the recursion is the antisymmetric part of the ordinary
    step-2 signature of the finite increment sequence."""
    torch.manual_seed(3)
    steps = 9
    pos = [torch.zeros([1, 3])]
    for _ in range(steps):
        pos.append(pos[-1] + torch.randn([1, 3]) * 0.25)

    hist = make_hist(kin_model, order=2, rho=1.0)
    push_pos_path(hist, kin_model, pos)

    D = TANGENT_DIM
    xis = []
    for k in range(steps):
        xi = torch.zeros([D])
        xi[0:3] = (pos[k + 1] - pos[k])[0]
        xis.append(xi)

    # plain level-2 signature tensor, then its antisymmetric part
    sig2 = torch.zeros([D, D])
    partial = torch.zeros([D])
    for xi in xis:
        sig2 += torch.outer(partial, xi) + 0.5 * torch.outer(xi, xi)
        partial += xi
    levy = 0.5 * (sig2 - sig2.T)

    full = lie_signature_obs.unpack_area(hist.extract()[:, TANGENT_DIM:], D)[0]
    assert torch.allclose(full, levy, atol=1e-5)

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

    hist = make_hist(kin_model, order=1, rho=1.0)
    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    hist.reset(torch.arange(n), root_pos, root_rot, joint_rot)

    anchor_inv = identity_quat(n)
    for k in range(1, steps + 1):
        rot = axis_quat([0.0, 1.0, 0.0], d_angle * k, n)
        hist.push(root_pos, rot, joint_rot, anchor_inv)

    winding = hist.extract()[:, 3:6]
    assert abs(winding[0, 1].item() - 2.0 * np.pi) < 1e-3
    assert torch.max(torch.abs(winding[:, [0, 2]])).item() < 1e-4

def test_quaternion_hemisphere_invariance(kin_model):
    """q and -q describe the same rotation and must give the same signature."""
    n = 1
    steps = 10
    d_angle = 0.3

    def run(flip):
        hist = make_hist(kin_model, order=2)
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
        anchor_inv = lie_signature_obs.calc_motion_anchor_quat_inv(root_rot0)

        hist = make_hist(kin_model, order=2)

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
    Delta = 0 is preserved for the whole episode, not just at reset."""
    torch.manual_seed(4)
    n = 3
    steps = 15
    num_joints = kin_model.get_num_joints() - 1

    sim = make_hist(kin_model, num_envs=n, order=2)
    ref = make_hist(kin_model, num_envs=n, order=2)

    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    env_ids = torch.arange(n)
    sim.reset(env_ids, root_pos, root_rot, joint_rot)
    ref.reset(env_ids, root_pos, root_rot, joint_rot)

    anchor_inv = identity_quat(n)
    for k in range(steps):
        root_pos = root_pos + torch.randn([n, 3]) * 0.05
        root_rot = torch_util.quat_mul(axis_quat([0.0, 1.0, 0.0], 0.2, n), root_rot)
        joint_rot = torch_util.quat_normalize(joint_rot + torch.randn([n, num_joints, 4]) * 0.02)

        sim.push(root_pos, root_rot, joint_rot, anchor_inv)
        ref.push(root_pos, root_rot, joint_rot, anchor_inv)

        diff = ref.extract() - sim.extract()
        assert torch.max(torch.abs(diff)).item() == 0.0

def test_partial_reset_does_not_pollute_other_envs(kin_model):
    n = 4
    steps = 6
    hist = make_hist(kin_model, num_envs=n, order=2)

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
    hist = make_hist(kin_model, order=2)

    far = [torch.zeros([n, 3]), torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([[5.0, 5.0, 0.0]])]
    push_pos_path(hist, kin_model, far)
    assert torch.max(torch.abs(hist.extract())).item() > 1.0

    root_pos, root_rot, joint_rot = rest_state(kin_model, n)
    reset_pos = torch.tensor([[100.0, -100.0, 1.0]])
    hist.reset(torch.arange(n), reset_pos, root_rot, joint_rot)
    assert torch.max(torch.abs(hist.extract())).item() == 0.0

    step = torch.tensor([[100.1, -100.0, 1.0]])
    hist.push(step, root_rot, joint_rot, identity_quat(n))
    phi = hist.extract()
    assert abs(phi[0, 0].item() - 0.1) < 1e-5
    # a single increment cannot sweep area
    assert torch.max(torch.abs(phi[:, TANGENT_DIM:])).item() < 1e-8

def test_push_count_tracks_increments(kin_model):
    """Each push contributes exactly one increment; both sides are pushed the
    same number of times when streamed together (the env-level guarantee is
    re-checked at runtime in the Isaac smoke run)."""
    n = 2
    sim = make_hist(kin_model, num_envs=n, order=2)
    ref = make_hist(kin_model, num_envs=n, order=2)
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
    l1 = make_hist(kin_model, order=1)
    l2 = make_hist(kin_model, order=2)

    assert l1.get_tangent_dim() == TANGENT_DIM
    assert l1.get_area_dim() == 0
    assert l1.get_obs_dim() == TANGENT_DIM
    assert l2.get_area_dim() == AREA_DIM
    assert l2.get_obs_dim() == TANGENT_DIM + AREA_DIM

    assert STATE_DIM + l1.get_obs_dim() == L1_TOTAL == 206
    assert STATE_DIM + l2.get_obs_dim() == L2_TOTAL == 767

def test_memory_decay_is_physical():
    """rho is pinned to a physical memory time, so halving the control step
    keeps the same effective memory."""
    rho_30 = lie_signature_obs.calc_memory_decay(32.0 / 30.0, 1.0 / 30.0)
    rho_60 = lie_signature_obs.calc_memory_decay(32.0 / 30.0, 1.0 / 60.0)
    assert abs(rho_30 - np.exp(-1.0 / 32.0)) < 1e-9
    assert abs(rho_60 ** 2 - rho_30) < 1e-9

def test_normalizer_keeps_zero_at_zero():
    """The scale-only DiffNormalizer maps the ADD ideal point to itself, for
    any recorded statistics."""
    norm = diff_normalizer.DiffNormalizer([L2_TOTAL], device=DEVICE)
    torch.manual_seed(6)
    norm.record(torch.randn([256, L2_TOTAL]) * 7.0)
    norm.update()

    zeros = torch.zeros([4, L2_TOTAL])
    assert torch.max(torch.abs(norm.normalize(zeros))).item() == 0.0

def test_agent_does_not_change_add_training():
    """The route is a differential, not a new training rule: the reward path
    and the model builder are inherited untouched."""
    cls = liesig_st_add_agent.LieSigSTADDAgent
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
        _compute_disc_diagnostics = liesig_st_add_agent.LieSigSTADDAgent._compute_disc_diagnostics

    agent = DiagStub()
    agent._state_dim = STATE_DIM
    agent._tangent_dim = TANGENT_DIM
    agent._area_dim = AREA_DIM
    agent._disc_obs_norm = diff_normalizer.DiffNormalizer([L2_TOTAL], device=DEVICE)

    torch.manual_seed(7)
    diff = torch.randn([32, L2_TOTAL])
    info = agent._compute_disc_diagnostics(diff)

    add_keys = {"disc_loss", "disc_grad_penalty", "disc_logit_loss", "disc_pos_acc",
                "disc_neg_acc", "disc_pos_logit", "disc_neg_logit"}
    assert add_keys.isdisjoint(info.keys())
    for key in ["disc_state_rms", "disc_level1_rms", "disc_level2_rms",
                "disc_area_nonzero_frac", "disc_norm_min_scale_frac"]:
        assert key in info
        assert torch.isfinite(info[key]).all()

    zero_info = agent._compute_disc_diagnostics(torch.zeros([8, L2_TOTAL]))
    assert zero_info["disc_state_rms"].item() == 0.0
    assert zero_info["disc_level2_rms"].item() == 0.0

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
    model = make_add_model(L2_TOTAL)
    other = make_add_model(L2_TOTAL)
    other.load_state_dict(model.state_dict())

    x = torch.randn([16, L2_TOTAL])
    assert torch.allclose(model.eval_disc(x), other.eval_disc(x), atol=0.0)

@pytest.mark.parametrize("src,dst", [(L1_TOTAL, L2_TOTAL), (L2_TOTAL, L1_TOTAL),
                                     (V1_TOTAL, L2_TOTAL), (L2_TOTAL, V1_TOTAL)])
def test_mismatched_checkpoints_fail_loudly(src, dst):
    """L1, L2 and the frozen ST-ADD v1 differentials have different widths, so
    a wrong checkpoint can never be silently loaded."""
    src_model = make_add_model(src)
    dst_model = make_add_model(dst)
    with pytest.raises(RuntimeError):
        dst_model.load_state_dict(src_model.state_dict())
