import inspect
import io
import pathlib
from types import SimpleNamespace

import gymnasium.spaces as spaces
import pytest
import torch
import torch.nn.functional as F

import learning.diff_normalizer as diff_normalizer
from learning.dare_agent import DAREAgent
from learning.dare_model import DAREModel, ConvexPotentialBlock, LearnableCReLUStem

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        raise AssertionError("Learnable CReLU must not request semantic groups")


def _config():
    return {"actor_net": "fc_2layers_128units",
            "actor_init_output_scale": 0.01,
            "actor_std_type": "FIXED", "action_std": 0.05,
            "critic_net": "fc_2layers_128units",
            "disc_net": "fc_10layers_1024units"}


def test_coordinate_normalizer_round_trip():
    norm = diff_normalizer.DiffNormalizer((4,), device="cpu")
    data = torch.tensor([[1., -2., 3., -4.], [-1., 2., -3., 4.]])
    norm.record(data)
    norm.update()
    torch.testing.assert_close(norm.get_abs_mean(),
                               torch.tensor([1., 2., 3., 4.]))
    torch.testing.assert_close(norm.unnormalize(norm.normalize(data)), data)


def test_cpl_discriminator_is_finite_and_spectrally_normalized():
    model = DAREModel(_config(), _Env()).train()
    blocks = [m for m in model._disc_layers.modules()
              if isinstance(m, ConvexPotentialBlock)]
    assert len(blocks) == 7
    assert all(hasattr(block.linear.parametrizations, "weight")
               for block in blocks)
    logits = model.eval_disc(torch.randn(32, 172))
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_dare_reward_is_bounded_raw_logit_reward():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = DAREModel(_config(), _Env()).eval()
    agent._disc_eval_batch_size = 0
    agent._disc_reward_scale = 2.0
    agent._pos_diff = torch.zeros(172)
    inputs = torch.randn(31, 172)
    reward = agent._calc_disc_rewards(inputs)
    anchor_input = agent._pos_diff.unsqueeze(0).expand(31, -1)
    anchor_reward = agent._calc_disc_rewards(anchor_input)
    assert reward.shape == (31,)
    assert torch.isfinite(reward).all()
    anchor_logits = agent._model.eval_disc(anchor_input).squeeze(-1)
    expected_anchor_reward = agent._disc_reward_scale * (
        -torch.log(torch.clamp_min(1.0 - torch.sigmoid(anchor_logits), 1e-4)))
    torch.testing.assert_close(anchor_reward, expected_anchor_reward,
                               atol=1e-6, rtol=0.0)
    assert torch.all(reward >= 0.0)
    logits = agent._model.eval_disc(inputs).squeeze(-1)
    expected = agent._disc_reward_scale * (
        -torch.log(torch.clamp_min(1.0 - torch.sigmoid(logits), 1e-4)))
    torch.testing.assert_close(reward, expected)
    assert not reward.requires_grad


def test_dare_discriminator_uses_only_the_differential():
    model = DAREModel(_config(), _Env()).eval()
    logits = model.eval_disc(torch.zeros(8, 172))
    assert logits.shape == (8, 1)
    stem = model._disc_layers.stem
    assert isinstance(stem, LearnableCReLUStem)
    assert stem.lift.shape == (511, 172)
    assert stem.output_width == 1022
    assert model._disc_layers.trunk[0].in_features == 1022
    assert model._disc_layers.trunk[0].out_features == 512
    assert not hasattr(model._disc_layers, "encoders")
    assert not any("group_indices" in key for key in model.state_dict())


def test_learnable_crelu_initializes_from_seeded_qr_without_consuming_global_rng():
    random_state = torch.get_rng_state()
    stem = LearnableCReLUStem(172)
    assert torch.equal(torch.get_rng_state(), random_state)
    generator = torch.Generator(device="cpu").manual_seed(0)
    raw = torch.randn(511, 172, generator=generator, device="cpu",
                      dtype=torch.float64)
    expected, _ = torch.linalg.qr(raw, mode="reduced")
    torch.testing.assert_close(stem.lift, expected.to(stem.lift.dtype),
                               atol=3e-7, rtol=1e-5)
    assert len(list(stem.parameters())) == 1
    assert not list(stem.named_buffers())
    assert stem.lift.requires_grad
    assert "parametrizations.lift.original" in stem.state_dict()


def test_learnable_crelu_preserves_zero_anchor_norm_and_input_information():
    stem = LearnableCReLUStem(172)
    inputs = torch.randn(12, 172, requires_grad=True)
    features = stem(inputs)
    positive, negative = features.chunk(2, dim=-1)

    torch.testing.assert_close(stem.lift.T @ stem.lift, torch.eye(172),
                               atol=1e-6, rtol=1e-5)
    reconstructed = F.linear(positive - negative, stem.lift.T)
    torch.testing.assert_close(reconstructed, inputs, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(stem(torch.zeros_like(inputs)),
                               torch.zeros_like(features))
    torch.testing.assert_close(features.norm(dim=-1), inputs.norm(dim=-1),
                               atol=3e-6, rtol=1e-5)

    # The constrained projection also passes gradients back to its input.
    features.square().sum().backward()
    torch.testing.assert_close(inputs.grad, 2.0 * inputs,
                               atol=6e-6, rtol=1e-5)


def _agent_with_real_optimizers():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = DAREModel(_config(), _Env()).train()
    agent._build_optimizer({
        "actor_optimizer": {"type": "SGD", "learning_rate": 1e-4},
        "critic_optimizer": {"type": "SGD", "learning_rate": 1e-4},
        "disc_optimizer": {"type": "SGD", "learning_rate": 2.5e-4,
                           "weight_decay": 1e-4}})
    return agent


def test_learnable_lift_updates_with_real_optimizer_and_stays_orthogonal():
    agent = _agent_with_real_optimizers()
    stem = agent._model._disc_layers.stem
    initial_lift = stem.lift.detach().clone()
    raw_parameter = stem.parametrizations.lift.original
    initial_diagonal = raw_parameter.detach().diagonal().clone()
    groups = agent._disc_optimizer._optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["weight_decay"] == 0.0
    assert groups[0]["lr"] == groups[1]["lr"] == 2.5e-4
    assert groups[0]["momentum"] == groups[1]["momentum"] == 0.9
    assert groups[1]["params"] == [raw_parameter]

    inputs = torch.randn(8, 172)
    for _ in range(5):
        logits = agent._model.eval_disc(inputs)
        agent._disc_optimizer.step((logits + 2.0).square().mean())

    assert (stem.lift.detach() - initial_lift).norm().item() > 1e-7
    assert torch.equal(raw_parameter.detach().diagonal(), initial_diagonal)
    assert torch.all(raw_parameter.detach().diagonal().abs() == 1)
    identity_error = (stem.lift.T @ stem.lift - torch.eye(172)).abs().max()
    assert identity_error.item() <= 1e-5


def test_learnable_crelu_model_and_optimizer_resume_without_drift():
    original = _agent_with_real_optimizers()
    inputs = torch.randn(8, 172)
    for _ in range(2):
        loss = (original._model.eval_disc(inputs) + 1.0).square().mean()
        original._disc_optimizer.step(loss)
    checkpoint = io.BytesIO()
    torch.save({"model": original._model.state_dict(),
                "optimizer": original._disc_optimizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = _agent_with_real_optimizers()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored._model.load_state_dict(saved["model"], strict=True)
    restored._disc_optimizer.load_state_dict(saved["optimizer"])
    original._model.eval()
    restored._model.eval()
    torch.testing.assert_close(restored._model.eval_disc(inputs),
                               original._model.eval_disc(inputs),
                               atol=0, rtol=0)
    assert restored._disc_optimizer.get_steps() == 2
    for agent in (original, restored):
        agent._model.train()
        loss = (agent._model.eval_disc(inputs) + 1.0).square().mean()
        agent._disc_optimizer.step(loss)
    for name, value in original._model.state_dict().items():
        torch.testing.assert_close(restored._model.state_dict()[name], value,
                                   atol=0, rtol=0)
    assert restored._disc_optimizer.get_steps() == 3
    lift = restored._model._disc_layers.stem.lift
    assert (lift.T @ lift - torch.eye(172)).abs().max().item() <= 1e-5


@pytest.mark.parametrize("input_dim,output_width", [(0, 1022), (512, 1022),
                                                   (7, 0), (7, 15)])
def test_learnable_crelu_rejects_invalid_dimensions(input_dim, output_width):
    with pytest.raises(ValueError):
        LearnableCReLUStem(input_dim, output_width)


def test_dare_disc_path_uses_zero_vs_residual_bce():
    disc_src = inspect.getsource(DAREAgent._compute_disc_loss)
    reward_src = inspect.getsource(DAREAgent._calc_disc_rewards)
    assert "eval_disc" in disc_src
    assert "replay_diff" in disc_src
    assert "_disc_loss_pos" in disc_src
    assert "_disc_loss_neg" in disc_src
    assert "quality_target" not in disc_src
    assert "mse_loss" not in disc_src
    assert "randperm" not in disc_src
    assert "calc_unscaled_disc_reward" in reward_src
    assert "calibr" not in disc_src.lower()


class _LogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.residual = torch.nn.Parameter(torch.tensor(0.0))
        self._disc_layers = SimpleNamespace(stem=SimpleNamespace(
            lift_width=511, output_width=1022))
        self.calls = []

    def eval_disc(self, disc_obs):
        self.calls.append(disc_obs.detach().clone())
        logit = torch.where(disc_obs.square().sum(dim=-1) == 0,
                            self.anchor, self.residual)
        return logit.unsqueeze(-1)


class _Replay:
    def sample(self, count):
        return {"disc_obs_demo": torch.full((count, 4), 2.0),
                "disc_obs": torch.zeros(count, 4)}


def test_dare_bce_uses_current_replay_and_correct_gradient_directions():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = _LogitModel()
    agent._pos_diff = torch.zeros(4)
    agent._disc_obs_norm = diff_normalizer.DiffNormalizer((4,), "cpu")
    agent._disc_buffer = _Replay()
    batch = {"disc_obs_demo": torch.ones(3, 4),
             "disc_obs": torch.zeros(3, 4)}
    info = agent._compute_disc_loss(batch)
    torch.testing.assert_close(info["disc_loss"], torch.log(torch.tensor(2.0)))
    assert len(agent._model.calls) == 2
    torch.testing.assert_close(agent._model.calls[0], torch.zeros(1, 4))
    torch.testing.assert_close(agent._model.calls[1], torch.cat((
        torch.ones(3, 4), torch.full((3, 4), 2.0))))
    info["disc_loss"].backward()
    assert agent._model.anchor.grad < 0   # Gradient descent raises anchor logit.
    assert agent._model.residual.grad > 0 # Gradient descent lowers residual logit.
    assert "disc_quality_loss" not in info and "disc_pair_acc" not in info
    assert info["disc_lift_width"].item() == 511
    assert info["disc_frontend_width"].item() == 1022
    assert "disc_token_dim" not in info


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


def test_dare_configs_have_single_adversarial_path():
    import yaml

    config = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    assert config["disc_buffer_size"] == 200000
    assert config["disc_replay_samples"] == 1000
    assert "disc_anchor_calibration" not in config
    assert config["disc_reward_scale"] == 2.0
    assert config["normalizer_samples"] == 100000000
    assert "disc_grad_penalty" not in config
    assert "disc_logit_reg" not in config
    assert config["agent_name"] == "DARE"


def test_official_add_is_not_modified_by_dare():
    add_model = (ROOT / "mimickit/learning/add_model.py").read_text()
    add_agent = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "DARE" not in add_model
    assert "DARE" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent
