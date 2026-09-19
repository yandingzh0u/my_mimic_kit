import inspect
import pathlib

import gymnasium.spaces as spaces
import torch

import learning.diff_normalizer as diff_normalizer
from learning.dare_agent import (DAREAgent, logit_standardization,
                                 solve_reward_scale)
from learning.dare_model import DAREModel, GroupSeparableDiscLayers
import util.torch_util as torch_util

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    dims = (3, 6, 45, 84, 3, 3, 28)

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        groups, start = [], 0
        for group_id, dim in enumerate(self.dims):
            groups.append(("group{}".format(group_id),
                           tuple(range(start, start + dim))))
            start += dim
        return tuple(groups)


def _config():
    return {"actor_net": "fc_2layers_128units",
            "actor_init_output_scale": 0.01,
            "actor_std_type": "FIXED", "action_std": 0.05,
            "critic_net": "fc_2layers_128units",
            "disc_net": "fc_2layers_128units"}


def _sn_linears(module):
    return [child for child in module.modules()
            if isinstance(child, torch.nn.Linear)]


def test_coordinate_normalizer_zero_and_state_round_trip():
    norm = diff_normalizer.DiffNormalizer((4,), device="cpu")
    data = torch.tensor([[1., -2., 3., -4.], [-1., 2., -3., 4.]])
    norm.record(data); norm.update()
    torch.testing.assert_close(norm.get_abs_mean(), torch.tensor([1., 2., 3., 4.]))
    torch.testing.assert_close(norm.normalize(torch.zeros(4)), torch.zeros(4))
    torch.testing.assert_close(norm.unnormalize(norm.normalize(data)), data)
    state = norm.training_state_dict()
    restored = diff_normalizer.DiffNormalizer((4,), device="cpu")
    restored.load_training_state_dict(state)
    assert restored._new_count == norm._new_count


def test_model_restores_explicit_a30_group_frontend():
    model = DAREModel(_config(), _Env()).train()
    layers = model._disc_layers
    assert isinstance(layers, GroupSeparableDiscLayers)
    assert len(layers.encoders) == len(_Env.dims)
    assert layers.group_width == 18 and layers.total_width == 126
    assert all(hasattr(encoder[0].parametrizations, "weight")
               for encoder in layers.encoders)


def test_all_discriminator_linears_use_spectral_norm():
    model = DAREModel(_config(), _Env()).train()
    linears = _sn_linears(model._disc_layers) + [model._disc_logits]
    assert len(linears) == 9
    assert all(hasattr(layer.parametrizations, "weight") for layer in linears)


def test_flat_ablation_only_removes_group_frontend():
    config = _config(); config["disc_group_embedding"] = False
    model = DAREModel(config, _Env()).train()
    assert not model.uses_disc_group_embedding()
    assert float(model.get_disc_group_width()) == 0.0
    logits = model.eval_disc(torch.randn(16, 172)).squeeze(-1)
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()


def test_uncalibrated_model_is_bitwise_v6():
    model = DAREModel(_config(), _Env()).eval()
    inputs = torch.randn(64, 172)
    torch.testing.assert_close(model.eval_disc(inputs), model.eval_disc_raw(inputs),
                               rtol=0., atol=0.)


def test_calibrated_classifier_is_affine_standardized_logit():
    model = DAREModel(_config(), _Env()).eval()
    inputs = torch.randn(32, 172); raw = model.eval_disc_raw(inputs)
    model.set_disc_logit_calibration(-0.7, 3.5)
    torch.testing.assert_close(model.eval_disc(inputs), 3.5 * (raw + 0.7),
                               rtol=0., atol=1e-6)


def test_logit_standardization_gives_zero_mean_unit_variance():
    torch.manual_seed(0)
    model = DAREModel(_config(), _Env()).eval()
    norm = diff_normalizer.DiffNormalizer((172,), device="cpu")
    pos = torch.zeros(172)
    current, replay = torch.randn(2048, 172), torch.randn(1024, 172)
    center, std, gap = logit_standardization(model, norm, pos, current,
                                             replay, 512)
    assert gap > 0.0
    model.set_disc_logit_calibration(center, 1.0 / std)

    def eval_scaled(disc_obs):
        return model.eval_disc(norm.normalize(disc_obs))

    pos_z = model.eval_disc(pos.unsqueeze(0)).mean()
    cur_logits = torch_util.eval_minibatch(
        eval_scaled, {"disc_obs": current}, 512)
    rep_logits = torch_util.eval_minibatch(
        eval_scaled, {"disc_obs": replay}, 512)
    balanced_mean = (0.5 * pos_z + 0.25 * cur_logits.mean()
                    + 0.25 * rep_logits.mean())
    balanced_second = (0.5 * pos_z.square() + 0.25 * cur_logits.square().mean()
                      + 0.25 * rep_logits.square().mean())
    torch.testing.assert_close(balanced_mean, torch.tensor(0.0),
                               rtol=0., atol=1e-5)
    torch.testing.assert_close(balanced_second, torch.tensor(1.0),
                               rtol=0., atol=1e-5)


def test_anchor_gap_does_not_update_spectral_norm_state():
    model = DAREModel(_config(), _Env()).eval()
    norm = diff_normalizer.DiffNormalizer((172,), device="cpu")
    before = {n: v.clone() for n, v in model.named_buffers()
              if n.endswith("._u") or n.endswith("._v")}
    logit_standardization(model, norm, torch.zeros(172), torch.randn(256, 172),
                          torch.randn(256, 172), 128)
    after = dict(model.named_buffers())
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_reward_scale_solver_hits_target_probability():
    delta = torch.tensor([-0.5, -0.8, -1.2, -2.0])
    scale = solve_reward_scale(delta, 1.0, 0.2)
    assert scale is not None and scale > 0
    prob = torch.sigmoid(1.0 + scale * delta).mean()
    torch.testing.assert_close(prob, torch.tensor(.2), atol=1e-6, rtol=0.)


def test_reward_scale_solver_reports_no_root():
    assert solve_reward_scale(torch.tensor([.1, .2]), 1.0, .2) is None


def test_reward_forward_uses_training_mode_sn_updates_like_a30():
    agent = object.__new__(DAREAgent); torch.nn.Module.__init__(agent)
    # Historical SN semantics: pin the legacy geometry, since the default
    # (semi-orthogonal) backbone has no spectral-norm layers to update.
    config = _config()
    config["disc_hidden_geometry"] = "full_sn"
    agent._model = DAREModel(config, _Env()).train()
    agent._disc_reward_scale = 2.; agent._disc_eval_batch_size = 0
    agent._reward_calibrated = False
    before = {n: v.clone() for n, v in agent._model.named_buffers()
              if n.endswith("._u") or n.endswith("._v")}
    reward = agent._calc_disc_rewards(torch.randn(31, 172))
    after = dict(agent._model.named_buffers())
    assert reward.shape == (31,) and torch.isfinite(reward).all()
    assert any(not torch.equal(before[n], after[n]) for n in before)


def test_disc_loss_evaluates_positive_before_negative():
    source = inspect.getsource(DAREAgent._compute_disc_loss)
    assert source.index("pos_logit =") < source.index("neg_logit =")


def test_bc_separation_disc_loss_raw_reward_calibrated():
    """B/C decoupling: discriminator trains on the RAW logit, reward on z.

     B = spectral norm -> normalized geometry (inside the raw network)
     C = affine calibration z = (f_raw - c_f) / s_f -> reward origin/scale
    The calibration must therefore never enter the BCE, otherwise kappa = 1/s_f
    silently rescales the classifier loss too.
    """
    disc_src = inspect.getsource(DAREAgent._compute_disc_loss)
    assert "eval_disc_raw" in disc_src
    assert "eval_disc(" not in disc_src
    reward_src = inspect.getsource(DAREAgent._calc_disc_rewards)
    assert "eval_disc" in reward_src


def test_dare_logit_reg_is_removed_not_silently_inert():
    """DARE's `_disc_logits` is spectral-normalized, so the effective weight
    always satisfies ||W_eff||_2 == 1 and sum(W_eff^2) is a constant with an
    exactly zero gradient.  The objective must not add that term (it would be
    dead weight), and a non-zero config value must be reported rather than
    silently ignored, otherwise an ablation arm that only toggles it compares
    two identical models.
    """
    disc_src = inspect.getsource(DAREAgent._compute_disc_loss)
    assert "logit_reg_loss" in disc_src          # still reported in the info dict
    assert "_disc_logit_reg * logit_loss" not in disc_src
    assert "has no effect for DARE" in disc_src
    assert "get_disc_logit_weights" not in disc_src


def _build_disc_model(geometry, net="fc_2layers_128units"):
    config = _config()
    config["disc_net"] = net
    config["disc_group_embedding"] = True
    config["disc_hidden_geometry"] = geometry
    return DAREModel(config, _Env())


# Enough width for the isometry tests: the largest error group has 84 dims and
# there are 7 groups, so the per-group encoder only stays a tall (isometric
# embedding) matrix when width // 7 >= 84.  A narrower discriminator turns the
# big encoder into a wide, direction-dropping map - which is a real property of
# the construction, not a bug, but it makes the isometry untestable.
# fc_2layers_1024units (1024 // 7 = 146) is the narrowest registered net that
# satisfies this; fc_2layers_512units (73 per group) does not.
ISO_NET = "fc_2layers_1024units"


def _hidden_isometry(model, seeds=(0,)):
    """sum(sigma^2(J_h)) / dim of the hidden discriminator map."""
    values = []
    for seed in seeds:
        torch.manual_seed(seed)
        probe = torch.randn(1, sum(_Env.dims)) * 0.5
        jacobian = torch.autograd.functional.jacobian(
            model.eval_disc_hidden, probe).detach()
        # jacobian has shape (1, out, 1, in).  Feeding the 4-D array to
        # numpy.linalg.svd silently stacks the wrong axes and returns garbage
        # (it reported 1.0000 for a clearly non-isometric map), so reshape.
        jacobian = jacobian.reshape(jacobian.shape[1], jacobian.shape[3])
        singular = torch.linalg.svdvals(jacobian)
        values.append(float(singular.square().sum() / singular.numel()))
    return sum(values) / len(values)


def test_hidden_discriminator_is_isometric():
    """The new geometry must be norm-preserving as a structural guarantee.

    Every hidden linear is (semi-)orthogonal (all singular values 1) and
    GroupSort(2) has a permutation Jacobian, so the composite hidden map keeps
    ||J_h v|| = ||v|| for every v.  This is a property of the construction, not
    something the checkpoint happens to learn.
    """
    assert _hidden_isometry(_build_disc_model("semi_orthogonal", ISO_NET)) > 0.99


def test_full_sn_hidden_discriminator_contracts():
    """The historical backbone must stay reproducible and is NOT isometric.

    Spectral normalization only pins the largest singular value and the ReLU
    zeroes roughly half the units, so this geometry contracts most directions.
    Pinning the value documents why the new geometry exists.
    """
    assert _hidden_isometry(_build_disc_model("full_sn", ISO_NET)) < 0.5


def test_square_trunk_keeps_every_encoder_direction():
    """The trunk must be square, or the isometry is capped at width / total.

    A rectangular R^total -> R^w trunk is a partial isometry whose row space
    meets the encoder image in about w / total of its dimensions (measured
    0.500 for 512/1022 and 1.000 for 1022/1022).
    """
    for geometry in ("semi_orthogonal",):
        layers = _build_disc_model(geometry, ISO_NET)._disc_layers
        rows, cols = layers.trunk[0].weight.shape
        assert len(layers.encoders) == 7
        assert (rows, cols) == (layers.total_width, layers.total_width)
        assert rows == 7 * layers.group_width
    # The legacy geometry must remain rectangular (the committed ablations).
    legacy = _build_disc_model("full_sn", ISO_NET)._disc_layers
    assert legacy.trunk[0].weight.shape == (512, 1022)


def test_dare_configs_state_calibration_switches_explicitly():
    """disc_anchor_calibration defaults to True while reward_calibration_mode
    defaults to "legacy" (reward calibration off).  A config that sets neither
    runs half calibrated with no error, so every DARE config must state both.
    """
    import yaml
    for name in ["dare_humanoid_agent.yaml",
                 "ablations/dare_climb_base_agent.yaml",
                 "ablations/dare_climb_wocalibration_agent.yaml",
                 "ablations/dare_climb_wogroup_agent.yaml"]:
        path = ROOT / "data/agents" / name
        config = yaml.safe_load(path.read_text())
        assert "disc_anchor_calibration" in config, name
        assert "reward_calibration_mode" in config, name
        assert config.get("disc_logit_reg") == 0, name


def test_config_restores_clean_v6_semantics():
    text = (ROOT / "data/agents/dare_humanoid_agent.yaml").read_text()
    assert 'agent_name: "DARE"' in text
    assert "disc_grad_penalty: 0" in text
    assert "disc_eval_batch_size: 0" in text
    assert "disc_raw_bce" not in text and "group_energy" not in text


def test_getup_ablation_configs_form_two_by_two_grid():
    root = ROOT / "data/agents/ablations"
    expected = {"dare_getup_wogroup_agent.yaml":
                ("disc_group_embedding: false", "disc_anchor_calibration: true"),
                "dare_getup_wocalibration_agent.yaml":
                ("disc_group_embedding: true", "disc_anchor_calibration: false"),
                "dare_getup_base_agent.yaml":
                ("disc_group_embedding: false", "disc_anchor_calibration: false")}
    for filename, flags in expected.items():
        text = (root / filename).read_text()
        assert 'agent_name: "DARE"' in text
        assert all(flag in text for flag in flags)


def test_official_add_is_not_modified_by_dare():
    add_model = (ROOT / "mimickit/learning/add_model.py").read_text()
    add_agent = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "DARE" not in add_model and "GroupSeparableDiscLayers" not in add_model
    assert "DARE" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent
