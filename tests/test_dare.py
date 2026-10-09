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


def test_dare_reward_is_quality_probability_and_bounded():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._model = DAREModel(_config(), _Env()).eval()
    agent._disc_eval_batch_size = 0
    agent._pos_diff = torch.zeros(172)
    inputs = torch.randn(31, 172)
    reward = agent._calc_disc_rewards(inputs)
    anchor_input = agent._pos_diff.unsqueeze(0).expand(31, -1)
    anchor_reward = agent._calc_disc_rewards(anchor_input)
    assert reward.shape == (31,)
    assert torch.isfinite(reward).all()
    expected_anchor_reward = torch.sigmoid(
        agent._model.eval_disc_raw(anchor_input)).squeeze(-1)
    torch.testing.assert_close(anchor_reward, expected_anchor_reward,
                               atol=1e-6, rtol=0.0)
    assert torch.all(reward >= 0.0)
    assert torch.all(reward <= 1.0)


def test_dare_discriminator_uses_only_the_differential():
    model = DAREModel(_config(), _Env()).eval()
    logits = model.eval_disc_raw(torch.zeros(8, 172))
    assert logits.shape == (8, 1)
    assert model.get_disc_group_width().item() == 146
    assert model.get_disc_group_total_width().item() == 1022


def test_dare_disc_path_uses_fixed_quality_target():
    disc_src = inspect.getsource(DAREAgent._compute_disc_loss)
    reward_src = inspect.getsource(DAREAgent._calc_disc_rewards)
    assert "eval_disc_raw" in disc_src
    assert "replay_diff" in disc_src
    assert "quality_target" in disc_src
    assert "quality_loss" in disc_src
    assert "F.mse_loss" in disc_src
    assert "torch.sigmoid" in reward_src
    assert "sigmoid(logits)" in reward_src
    assert "calibr" not in disc_src.lower()


def test_dare_configs_have_single_current_rollout_path():
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
        assert "disc_reward_scale" not in config
        assert config["agent_name"] == "DARE"


def test_official_add_is_not_modified_by_dare():
    add_model = (ROOT / "mimickit/learning/add_model.py").read_text()
    add_agent = (ROOT / "mimickit/learning/add_agent.py").read_text()
    assert "DARE" not in add_model and "GroupSeparableDiscLayers" not in add_model
    assert "DARE" not in add_agent
    assert "grad_penalty = 0.5 * (neg_gp + pos_gp)" in add_agent
