import inspect
import pathlib

import gymnasium.spaces as spaces
import pytest
import torch
import yaml

from envs.add_env import build_disc_error_groups
import learning.diff_normalizer as diff_normalizer
from learning.dare_agent import DAREAgent, anchor_gap
from learning.dare_model import (ANCHOR_GAP_TARGET, DAREModel,
                                 GroupSeparableDiscLayers)
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
        return build_disc_error_groups(1, 15, 15, 28, 172)


def _config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def _sn_linears(module):
    return [child for child in module.modules()
            if isinstance(child, torch.nn.Linear)]


def test_model_restores_explicit_a30_group_frontend():
    model = DAREModel(_config(), _Env()).train()
    layers = model._disc_layers
    assert isinstance(layers, GroupSeparableDiscLayers)
    assert len(layers.encoders) == len(_Env.dims)
    assert layers.group_width == 18
    assert layers.total_width == 126
    for dim, encoder in zip(_Env.dims, layers.encoders):
        linear = encoder[0]
        assert linear.in_features == dim
        assert linear.out_features == layers.group_width
        assert hasattr(linear.parametrizations, "weight")
        torch.testing.assert_close(
            linear.bias, torch.zeros_like(linear.bias))


def test_all_discriminator_linears_use_pytorch_spectral_norm():
    model = DAREModel(_config(), _Env()).train()
    linears = _sn_linears(model._disc_layers) + [model._disc_logits]
    assert len(linears) == 9
    assert all(hasattr(layer.parametrizations, "weight")
               for layer in linears)


def test_model_forward_backward_is_finite():
    model = DAREModel(_config(), _Env()).train()
    logits = model.eval_disc(torch.randn(32, 172)).squeeze(-1)
    loss = (torch.nn.functional.softplus(logits).mean()
            + torch.nn.functional.softplus(
                -model.eval_disc(torch.zeros(1, 172))).mean())
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(parameter.grad is None
               or torch.isfinite(parameter.grad).all()
               for parameter in model.get_disc_params())


def test_uncalibrated_model_is_bitwise_v6():
    model = DAREModel(_config(), _Env()).eval()
    inputs = torch.randn(64, 172)
    assert not model.is_disc_logit_calibrated()
    assert float(model.get_disc_logit_scale()) == 1.0
    torch.testing.assert_close(
        model.eval_disc(inputs), model.eval_disc_raw(inputs),
        rtol=0.0, atol=0.0)


def test_anchor_gap_calibration_hits_ln16():
    torch.manual_seed(0)
    model = DAREModel(_config(), _Env()).eval()
    norm = diff_normalizer.DiffNormalizer((172,), device="cpu")
    pos = torch.zeros(172)
    current = torch.randn(2048, 172)
    replay = torch.randn(1024, 172)

    gap = anchor_gap(model, norm, pos, current, replay, 512)
    assert gap != 0.0
    model.set_disc_logit_scale(ANCHOR_GAP_TARGET / gap)

    def mean_scaled(raw_diff):
        def eval_scaled(disc_obs):
            return model.eval_disc(norm.normalize(disc_obs))

        return torch_util.eval_minibatch(
            eval_scaled, {"disc_obs": raw_diff}, 512).mean()

    scaled_gap = (model.eval_disc(pos.unsqueeze(0)).mean()
                  - 0.5 * (mean_scaled(current) + mean_scaled(replay)))
    torch.testing.assert_close(
        scaled_gap, torch.tensor(ANCHOR_GAP_TARGET), rtol=0.0, atol=1e-5)


def test_anchor_gap_leaves_spectral_norm_state_untouched():
    model = DAREModel(_config(), _Env()).eval()
    norm = diff_normalizer.DiffNormalizer((172,), device="cpu")
    before = {
        name: value.clone()
        for name, value in model.named_buffers()
        if name.endswith("._u") or name.endswith("._v")
    }
    anchor_gap(model, norm, torch.zeros(172),
               torch.randn(256, 172), torch.randn(256, 172), 128)
    after = dict(model.named_buffers())
    assert before
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_legacy_state_dict_without_calibration_loads_as_v6():
    source = DAREModel(_config(), _Env())
    legacy_state = source.state_dict()
    del legacy_state["_disc_logit_scale"]
    del legacy_state["_disc_logit_calibrated"]

    restored = DAREModel(_config(), _Env())
    restored.load_state_dict(legacy_state, strict=True)
    assert not restored.is_disc_logit_calibrated()
    assert float(restored.get_disc_logit_scale()) == 1.0


def test_reward_forward_uses_training_mode_sn_updates_like_a30():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = DAREModel(_config(), _Env()).train()
    agent._disc_reward_scale = 2.0
    agent._disc_eval_batch_size = 0

    before = {
        name: value.clone()
        for name, value in agent._model.named_buffers()
        if name.endswith("._u") or name.endswith("._v")
    }
    reward = agent._calc_disc_rewards(torch.randn(31, 172))
    after = dict(agent._model.named_buffers())

    assert reward.shape == (31,)
    assert torch.isfinite(reward).all()
    assert any(not torch.equal(before[name], after[name])
               for name in before)
    assert agent._model._disc_layers.training
    assert agent._model._disc_logits.training


def test_disc_loss_evaluates_positive_before_negative():
    source = inspect.getsource(DAREAgent._compute_disc_loss)
    assert source.index("pos_logit =") < source.index("neg_logit =")


def test_config_restores_a30_gp_and_reward_batch_semantics():
    text = (ROOT / "data/agents/dare_humanoid_agent.yaml").read_text()
    assert 'agent_name: "DARE"' in text
    assert "disc_grad_penalty: 0" in text
    assert "disc_eval_batch_size: 0" in text
    assert "iters_per_output: 100" in text


def test_official_add_is_not_modified_by_dare():
    add_model = (ROOT / "mimickit/learning/add_model.py").read_text()
    add_agent = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "DARE" not in add_model
    assert "GroupSeparableDiscLayers" not in add_model
    assert "DARE" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent


def test_coordinate_normalizer_round_trip():
    norm = diff_normalizer.DiffNormalizer((4,), device="cpu")
    data = torch.tensor([[1., -2., 3., -4.], [-1., 2., -3., 4.]])
    norm.record(data)
    norm.update()
    torch.testing.assert_close(norm.get_abs_mean(),
                               torch.tensor([1., 2., 3., 4.]))
    torch.testing.assert_close(norm.unnormalize(norm.normalize(data)), data)


def test_disc_groups_match_observation_layout_across_frames():
    names = ("root_pos", "root_rot", "body_pos", "body_rot",
             "root_vel", "root_ang_vel", "dof_vel")
    spans = ((0, 3), (3, 9), (93, 138), (9, 93),
             (138, 141), (141, 144), (144, 172))
    for steps in (1, 3):
        groups = build_disc_error_groups(steps, 15, 15, 28, steps * 172)
        assert tuple(name for name, _ in groups) == names
        for (_, indices), (begin, end) in zip(groups, spans):
            expected = tuple(step * 172 + coordinate
                             for step in range(steps)
                             for coordinate in range(begin, end))
            assert indices == expected
        assert sorted(i for _, indices in groups for i in indices) == list(
            range(steps * 172))
    with pytest.raises(ValueError, match="layout mismatch"):
        build_disc_error_groups(1, 15, 15, 28, 171)


def test_paper_model_has_seven_146_wide_encoders_and_no_cpl():
    config = yaml.safe_load(
        (ROOT / "data/agents/dare_humanoid_agent.yaml").read_text())
    model = DAREModel(config["model"], _Env()).eval()
    layers = model._disc_layers
    assert len(layers.encoders) == 7
    assert layers.group_width == 146 and layers.total_width == 1022
    assert layers.trunk[0].in_features == 1022
    assert layers.trunk[0].out_features == 512
    assert model._disc_logits.in_features == 512
    assert model._disc_logits.out_features == 1
    assert all("ConvexPotential" not in type(child).__name__
               for child in model.modules())


class _LogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.residual = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []
        self.register_buffer("scale", torch.ones(()))

    def eval_disc_raw(self, disc_obs):
        self.calls.append(disc_obs.detach().clone())
        logit = torch.where(disc_obs.square().sum(dim=-1) == 0,
                            self.anchor, self.residual)
        return logit.unsqueeze(-1)

    def eval_disc(self, disc_obs):
        return self.scale * self.eval_disc_raw(disc_obs)

    def get_disc_logit_weights(self):
        return self.anchor.new_zeros(1)

    def get_disc_group_width(self):
        return self.anchor.new_tensor(146.0)

    def get_disc_group_total_width(self):
        return self.anchor.new_tensor(1022.0)

    def get_disc_logit_scale(self):
        return self.scale.clone()


class _Replay:
    def sample(self, count):
        return {"disc_obs_demo": torch.full((count, 4), 2.0),
                "disc_obs": torch.zeros(count, 4)}


def _logit_agent():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._model = _LogitModel()
    agent._pos_diff = torch.zeros(4)
    agent._disc_obs_norm = diff_normalizer.DiffNormalizer((4,), "cpu")
    agent._disc_buffer = _Replay()
    agent._disc_logit_reg = 0.0
    agent._disc_eval_batch_size = 0
    agent._disc_reward_scale = 2.0
    return agent


def test_dare_bce_uses_current_replay_and_correct_gradient_directions():
    agent = _logit_agent()
    batch = {"disc_obs_demo": torch.ones(3, 4),
             "disc_obs": torch.zeros(3, 4)}
    info = agent._compute_disc_loss(batch)
    torch.testing.assert_close(info["disc_loss"], torch.log(torch.tensor(2.0)))
    assert len(agent._model.calls) == 2
    torch.testing.assert_close(agent._model.calls[0], torch.zeros(1, 4))
    torch.testing.assert_close(agent._model.calls[1], torch.cat((
        torch.ones(3, 4), torch.full((3, 4), 2.0))))
    info["disc_loss"].backward()
    assert agent._model.anchor.grad < 0
    assert agent._model.residual.grad > 0
    assert "disc_quality_loss" not in info and "disc_pair_acc" not in info


def test_dare_loss_logging_preserves_the_mean_across_disc_updates():
    agent = _logit_agent()
    batch = {"disc_obs_demo": torch.ones(3, 4),
             "disc_obs": torch.zeros(3, 4)}
    accumulated, expected = {}, []
    for step in range(32):
        with torch.no_grad():
            agent._model.anchor.fill_(step / 16.0)
            agent._model.residual.fill_(-step / 8.0)
        info = agent._compute_disc_loss(batch)
        loss = info["disc_loss"]
        expected.append(loss.detach().clone())
        torch_util.add_torch_dict(info, accumulated)
        torch.testing.assert_close(loss.detach(), expected[-1])
    torch_util.scale_torch_dict(1.0 / 32, accumulated)
    mean = torch.stack(expected).mean()
    torch.testing.assert_close(accumulated["disc_loss"], mean)
    torch.testing.assert_close(accumulated["disc_cls_loss"], mean)


def test_dare_calibration_scale_is_used_for_both_bce_and_reward():
    agent = _logit_agent()
    with torch.no_grad():
        agent._model.anchor.fill_(0.25)
        agent._model.residual.fill_(-0.5)
        agent._model.scale.fill_(2.0)
    batch = {"disc_obs_demo": torch.ones(3, 4),
             "disc_obs": torch.zeros(3, 4)}
    info = agent._compute_disc_loss(batch)
    expected_bce = 0.5 * (torch.nn.functional.softplus(torch.tensor(-0.5))
                          + torch.nn.functional.softplus(torch.tensor(-1.0)))
    torch.testing.assert_close(info["disc_cls_loss"], expected_bce)
    reward = agent._calc_disc_rewards(torch.ones(3, 4))
    torch.testing.assert_close(
        reward, torch.full((3,), 2 * torch.nn.functional.softplus(
            torch.tensor(-1.0))))


def test_dare_bounded_reward_is_finite_monotone_and_detached():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = torch.nn.Module()
    agent._model.eval_disc = lambda disc_obs: disc_obs[:, :1]
    agent._disc_eval_batch_size = 0
    agent._disc_reward_scale = 1.0
    logits = torch.tensor([[-1000.], [-1.], [0.], [1.], [1000.]],
                          requires_grad=True)
    reward = agent._calc_disc_rewards(logits)
    assert torch.isfinite(reward).all()
    assert torch.all(reward[1:] >= reward[:-1])
    torch.testing.assert_close(reward[2], torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(reward[-1], torch.tensor(9.2103405))
    assert not reward.requires_grad


def test_all_dare_config_entries_build_the_restored_model():
    paths = [ROOT / "data/agents/dare_humanoid_agent.yaml",
             ROOT / "data/agents/dare_g1_climb_agent.yaml",
             ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml",
             ROOT / "data/agents/g1_real_deployment/dare_g1_sim2real_agent.yaml"]
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert "disc_hidden_sizes" not in config["model"]
        assert config.get("disc_grad_penalty", 0.0) == 0.0
        model = DAREModel(config["model"], _Env()).eval()
        assert len(model._disc_layers.encoders) == 7
        assert model.eval_disc(torch.zeros(2, 172)).shape == (2, 1)
