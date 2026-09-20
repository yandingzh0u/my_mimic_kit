import inspect
import pathlib

import gymnasium.spaces as spaces
import torch

import learning.diff_normalizer as diff_normalizer
from learning.dare_agent import DAREAgent, logit_standardization
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
    # Historical a30 frontend: pin the legacy geometry, since the default
    # isometric backbone uses proportional widths and no spectral norm.
    config = _config()
    config["disc_hidden_geometry"] = "full_sn"
    model = DAREModel(config, _Env()).train()
    layers = model._disc_layers
    assert isinstance(layers, GroupSeparableDiscLayers)
    assert len(layers.encoders) == len(_Env.dims)
    assert layers.group_width == 18 and layers.total_width == 126
    assert all(hasattr(encoder[0].parametrizations, "weight")
               for encoder in layers.encoders)


def test_all_discriminator_linears_use_spectral_norm():
    # Legacy geometry only: the isometric backbone constrains the hidden layers
    # by QR retraction instead, and keeps spectral norm on the output map.
    config = _config()
    config["disc_hidden_geometry"] = "full_sn"
    model = DAREModel(config, _Env()).train()
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
    # The sign of the initial zero-vs-noise gap is arbitrary: measured at
    # initialisation it is negative for 6/8 seeds with the legacy geometry and
    # 2/8 with the isometric one.  DAREAgent defers calibration until the gap is
    # positive, so the test sweeps seeds instead of pinning one.
    for seed in range(32):
        torch.manual_seed(seed)
        model = DAREModel(_config(), _Env()).eval()
        norm = diff_normalizer.DiffNormalizer((172,), device="cpu")
        pos = torch.zeros(172)
        current, replay = torch.randn(2048, 172), torch.randn(1024, 172)
        center, std, gap = logit_standardization(model, norm, pos, current,
                                                 replay, 512)
        if gap > 0.0:
            break
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


def test_reward_forward_uses_training_mode_sn_updates_like_a30():
    agent = object.__new__(DAREAgent); torch.nn.Module.__init__(agent)
    agent._model = DAREModel(_config(), _Env()).train()
    agent._disc_reward_scale = 2.; agent._disc_eval_batch_size = 0
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


def test_dare_configs_state_calibration_switches_explicitly():
    """Every DARE config states whether the affine reward calibration is used."""
    import yaml
    for name in ["dare_humanoid_agent.yaml",
                 "ablations/dare_climb_base_agent.yaml",
                 "ablations/dare_climb_wocalibration_agent.yaml",
                 "ablations/dare_climb_wogroup_agent.yaml"]:
        path = ROOT / "data/agents" / name
        config = yaml.safe_load(path.read_text())
        assert "disc_anchor_calibration" in config, name
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
