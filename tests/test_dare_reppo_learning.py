import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_reppo_learning import REPPOAgent, ADAM
from learning.reppo import CategoricalQ, Duals, frozen_parameters
from learning.distribution_gaussian_diag import DistributionGaussianDiagBuilder, StdType


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._actor_layers = nn.Linear(4, 8)
        self._action_dist = DistributionGaussianDiagBuilder(8, 2, StdType.CONSTANT, .05)
        self.critic = nn.Linear(4, 1)
        self.disc = nn.Linear(4, 1)

    def eval_actor(self, obs):
        return self._action_dist(self._actor_layers(obs))

    def get_actor_params(self):
        return list(self._actor_layers.parameters()) + list(self._action_dist.parameters())

    def get_critic_params(self):
        return list(self.critic.parameters())

    def get_disc_params(self):
        return list(self.disc.parameters())


class Identity(nn.Module):
    def normalize(self, x):
        return x


def agent_stub():
    agent = REPPOAgent.__new__(REPPOAgent)
    nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._model = Model()
    agent._reppo_q = CategoricalQ(4, 2, -5, 5, width=16)
    agent._reppo_duals = Duals()
    agent._obs_norm, agent._a_norm = Identity(), Identity()
    agent._reppo_rng = torch.Generator().manual_seed(12)
    agent._reppo_enabled = True
    object.__setattr__(agent, "_old_policy", nn.Sequential(
        copy.deepcopy(agent._model._actor_layers), copy.deepcopy(agent._model._action_dist)
    ).eval().requires_grad_(False))
    return agent


def test_optimizer_contains_q_not_archived_v_and_actor_step_does_not_update_q():
    agent = agent_stub()
    agent._model.critic.requires_grad_(False)
    agent._build_optimizer({name:copy.deepcopy(ADAM) for name in
                            ("actor_optimizer", "critic_optimizer", "disc_optimizer")})
    assert {id(p) for p in agent._critic_optimizer._param_list} == {id(p) for p in agent._reppo_q.parameters()}
    assert not ({id(p) for p in agent._model.get_critic_params()} &
                {id(p) for p in agent._critic_optimizer._param_list})
    qbefore = copy.deepcopy(agent._reppo_q.state_dict())
    actor_before = copy.deepcopy(agent._model._actor_layers.state_dict())
    with frozen_parameters(agent._reppo_q):
        info = agent._compute_actor_loss(dict(obs=torch.randn(16,4), phase_weight=torch.ones(16)))
        agent._actor_optimizer.step(info["actor_loss"])
    assert any(not torch.equal(v, actor_before[k]) for k,v in agent._model._actor_layers.state_dict().items())
    assert all(torch.equal(v, qbefore[k]) for k,v in agent._reppo_q.state_dict().items())


def test_legacy_load_is_strict_on_original_keys_and_roundtrips_new_q():
    agent = agent_stub()
    full = copy.deepcopy(agent.state_dict())
    legacy = {k:v for k,v in full.items() if not k.startswith("_reppo_")}
    agent.load_state_dict(legacy)
    broken = dict(legacy)
    broken.pop("_model.critic.weight")
    with pytest.raises(RuntimeError):
        agent.load_state_dict(broken)
    agent.load_state_dict(full)
    assert all(torch.equal(v, full[k]) for k,v in agent.state_dict().items())
