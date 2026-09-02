import inspect
import pathlib

import gymnasium.spaces as spaces
import torch

from learning.dare_agent import DAREAgent
from learning.dare_model import DAREModel, GroupSeparableDiscLayers


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
        groups = []
        start = 0
        for group_id, dim in enumerate(self.dims):
            groups.append(("group{}".format(group_id),
                           tuple(range(start, start + dim))))
            start += dim
        return tuple(groups)


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
