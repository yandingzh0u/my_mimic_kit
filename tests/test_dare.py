import inspect
import pathlib

import gymnasium.spaces as spaces
import torch

import learning.diff_normalizer as diff_normalizer
from learning.dare_agent import DAREAgent
from learning.dare_model import DAREModel, ConvexPotentialBlock

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
    logits = model.eval_disc_raw(torch.randn(32, 172))
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
    anchor_logits = agent._model.eval_disc_raw(anchor_input).squeeze(-1)
    expected_anchor_reward = agent._disc_reward_scale * (
        -torch.log(torch.clamp_min(1.0 - torch.sigmoid(anchor_logits), 1e-4)))
    torch.testing.assert_close(anchor_reward, expected_anchor_reward,
                               atol=1e-6, rtol=0.0)
    assert torch.all(reward >= 0.0)
    logits = agent._model.eval_disc_raw(inputs).squeeze(-1)
    expected = agent._disc_reward_scale * (
        -torch.log(torch.clamp_min(1.0 - torch.sigmoid(logits), 1e-4)))
    torch.testing.assert_close(reward, expected)
    assert not reward.requires_grad


def test_dare_discriminator_uses_only_the_differential():
    model = DAREModel(_config(), _Env()).eval()
    logits = model.eval_disc_raw(torch.zeros(8, 172))
    assert logits.shape == (8, 1)
    assert model.get_disc_group_width().item() == 146
    assert model.get_disc_group_total_width().item() == 1022


def test_dare_disc_path_uses_zero_vs_residual_bce():
    disc_src = inspect.getsource(DAREAgent._compute_disc_loss)
    reward_src = inspect.getsource(DAREAgent._calc_disc_rewards)
    assert "eval_disc_raw" in disc_src
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
        self.calls = []

    def eval_disc_raw(self, disc_obs):
        self.calls.append(disc_obs.detach().clone())
        logit = torch.where(disc_obs.square().sum(dim=-1) == 0,
                            self.anchor, self.residual)
        return logit.unsqueeze(-1)

    def get_disc_group_width(self):
        return self.anchor.new_tensor(146.0)

    def get_disc_group_total_width(self):
        return self.anchor.new_tensor(1022.0)


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


def test_dare_bounded_reward_is_finite_monotone_and_detached():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = torch.nn.Module()
    agent._model.eval_disc_raw = lambda disc_obs: disc_obs[:, :1]
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

    paths = [
        ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml",
        ROOT / "data/agents/dare_10layer_cpl_climb_smoke_agent.yaml",
    ]
    for path in paths:
        config = yaml.safe_load(path.read_text())
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
    assert "DARE" not in add_model and "GroupSeparableDiscLayers" not in add_model
    assert "DARE" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent
