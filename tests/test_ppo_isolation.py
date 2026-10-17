"""Behavioral parity with the immutable pre-refactor DARE/PPO implementation."""
from pathlib import Path
import subprocess
import sys
import types

import gymnasium.spaces as spaces
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
from learning.algorithms import build_algorithm
from learning.algorithms.ppo import PPO
from learning.dare_agent import DAREAgent
from learning.on_policy_agent import OnPolicyAgent
from learning.ppo_agent import PPOAgent

REVISION = "58373240451d6ab093c7e3a154164a94f9dabab3"


class Env:
    def get_num_envs(self):
        return 4

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-2., 3., shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        offset, groups = 0, []
        for dim in (3, 6, 45, 84, 3, 3, 28):
            groups.append((str(offset), tuple(range(offset, offset + dim))))
            offset += dim
        return groups

    def get_reward_succ(self):
        return 0.

    def get_reward_fail(self):
        return 0.


def legacy_agent_class():
    """Load old classes without replacing any live learning.* modules."""
    previous = None
    for name, parent in (("ppo", None), ("amp", "ppo"),
                         ("add", "amp"), ("dare", "add")):
        source = subprocess.check_output(
            ["git", "show", f"{REVISION}:mimickit/learning/{name}_agent.py"],
            cwd=ROOT, text=True)
        module = types.ModuleType(f"legacy_{name}")
        if parent:
            source = source.replace(f"import learning.{parent}_agent as {parent}_agent",
                                    f"{parent}_agent = legacy_parent")
            module.legacy_parent = previous
        exec(compile(source, f"{REVISION}/{name}_agent.py", "exec"), module.__dict__)
        previous = module
    return previous.DAREAgent


def assert_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def config():
    cfg = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    cfg["model"]["actor_net"] = "fc_2layers_128units"
    cfg["model"]["critic_net"] = "fc_2layers_128units"
    cfg.update(steps_per_iter=4, disc_buffer_size=8, actor_epochs=2,
               critic_epochs=2, disc_epochs=2)
    return cfg


@pytest.fixture
def pair():
    torch.manual_seed(734)
    before = torch.get_rng_state()
    old = legacy_agent_class()(config(), Env(), "cpu")
    old_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    new = DAREAgent(config(), Env(), "cpu")
    assert_equal(old_rng, torch.get_rng_state())
    assert_equal(old.state_dict(), new.state_dict())
    return old, new


def fill_rollout(agents, seed):
    rng = torch.Generator().manual_seed(seed)
    obs = torch.randn(4, 4, 10, generator=rng)
    action = torch.randn(4, 4, 4, generator=rng) * .05
    with torch.no_grad():
        logp = agents[0]._model.eval_actor(
            agents[0]._obs_norm.normalize(obs)).log_prob(action)
    fields = dict(obs=obs, next_obs=torch.randn(4, 4, 10, generator=rng),
                  action=agents[0]._a_norm.unnormalize(action), a_logp=logp,
                  reward=torch.rand(4, 4, generator=rng),
                  done=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 0],
                                     [3, 0, 0, 2], [0, 1, 0, 0]]),
                  rand_action_mask=torch.tensor([[1., 1., 1., 0.]]).expand(4, 4),
                  diag_phase=torch.rand(4, 4, generator=rng),
                  disc_obs=torch.randn(4, 4, 172, generator=rng),
                  disc_obs_demo=torch.randn(4, 4, 172, generator=rng))
    for agent in agents:
        for name, value in fields.items():
            agent._exp_buffer.set_data(name, value)
        agent._exp_buffer.set_total_samples(16)


def test_dare_composes_ppo_without_inheriting_it(pair):
    old, new = pair
    assert not issubclass(DAREAgent, PPOAgent)
    assert isinstance(new, OnPolicyAgent)
    assert isinstance(new._algorithm, PPO)
    assert not hasattr(new, "_ppo_clip_ratio")
    assert_equal(old._actor_optimizer.state_dict(), new._actor_optimizer.state_dict())
    assert_equal(old._critic_optimizer.state_dict(), new._critic_optimizer.state_dict())


def test_plain_ppo_compatibility_entry_keeps_original_updates():
    old_class = next(cls for cls in legacy_agent_class().__mro__
                     if cls.__name__ == "PPOAgent")
    torch.manual_seed(953)
    before = torch.get_rng_state()
    old = old_class(config(), Env(), "cpu")
    torch.set_rng_state(before)
    new = PPOAgent(config(), Env(), "cpu")
    assert_equal(old.state_dict(), new.state_dict())
    fill_rollout((old, new), 908)
    before = torch.get_rng_state()
    old_info, old_update = old._build_train_data(), old._update_model()
    old_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    new_info, new_update = new._build_train_data(), new._update_model()
    assert_equal(old_info, new_info)
    assert_equal(old_update, new_update)
    assert_equal(old_rng, torch.get_rng_state())
    assert_equal(old.state_dict(), new.state_dict())


def test_three_fixed_rollout_updates_are_bitwise_identical(pair):
    old, new = pair
    for iteration in range(3):
        fill_rollout(pair, 800 + iteration)
        before = torch.get_rng_state()
        old_info = old._build_train_data()
        old_train = old._update_model()
        old_rng = torch.get_rng_state()
        torch.set_rng_state(before)
        new_info = new._build_train_data()
        new_train = new._update_model()
        assert_equal(old_info, new_info)
        assert_equal(old_train, new_train)
        assert_equal(old_rng, torch.get_rng_state())
        assert_equal(old.state_dict(), new.state_dict())
        assert_equal(old._exp_buffer._buffers, new._exp_buffer._buffers)
        assert_equal(old._disc_buffer.state_dict(), new._disc_buffer.state_dict())
        for name in ("_actor_optimizer", "_critic_optimizer", "_disc_optimizer"):
            assert_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())


def test_legacy_full_checkpoint_resumes_without_key_or_optimizer_migration(pair, tmp_path):
    old, new = pair
    fill_rollout(pair, 802)
    old._build_train_data()
    old._update_model()
    path = tmp_path / "legacy.pt"
    old.save_checkpoint(path, next_iter=3)
    new.resume(path)
    assert new._iter == 3
    assert_equal(old.state_dict(), new.state_dict())
    assert_equal(old._disc_buffer.state_dict(), new._disc_buffer.state_dict())
    for name in ("_actor_optimizer", "_critic_optimizer", "_disc_optimizer"):
        assert_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())


def test_algorithm_dispatch_does_not_change_reward_agent_or_need_ppo_settings():
    class Objective:
        def build_train_data(self, agent):
            return {"target": 7}

        def compute_actor_loss(self, agent, batch):
            return batch

        def compute_critic_loss(self, agent, batch):
            return batch

    class OtherDARE(DAREAgent):
        def _build_algorithm(self, config):
            return Objective()

    cfg = config()
    del cfg["ppo_clip_ratio"], cfg["norm_adv_clip"]
    agent = OtherDARE(cfg, Env(), "cpu")
    assert OnPolicyAgent._build_train_data(agent) == {"target": 7}
    marker = {"batch": object()}
    assert agent._compute_actor_loss(marker) is marker
    assert agent._compute_critic_loss(marker) is marker


def test_unknown_algorithm_fails_instead_of_silently_using_ppo():
    with pytest.raises(ValueError, match="Unsupported on-policy algorithm"):
        build_algorithm({"algorithm": "not_an_algorithm"})


def test_default_dare_import_does_not_load_reppo():
    subprocess.run([sys.executable, "-B", "-c",
                    "import learning.dare_agent, sys; "
                    "assert 'learning.reppo_agent' not in sys.modules; "
                    "assert 'learning.reppo' not in sys.modules"],
                   cwd=ROOT, check=True)
