"""Robot migration invariants, independent of IsaacLab and GPU availability."""
import copy
from pathlib import Path
import sys

import gymnasium.spaces as spaces
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
from learning.dare_agent import DAREAgent
from learning.reppo_agent import REPPOAgent
from learning.reppo import CategoricalQ, soft_lambda_returns


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


def config():
    result = yaml.safe_load((ROOT / "data/agents/dare_reppo_humanoid_agent.yaml").read_text())
    result["model"]["actor_net"] = "fc_2layers_128units"
    result["steps_per_iter"] = 4
    result["disc_buffer_size"] = 8
    return result


@pytest.fixture
def migrated(tmp_path):
    cfg = config()
    old_cfg = copy.deepcopy(cfg)
    old_cfg["model"]["actor_std_type"] = "FIXED"
    old = DAREAgent(old_cfg, Env(), "cpu")
    old._actor_optimizer.step(sum(p.square().sum() for p in old._actor_optimizer._param_list))
    old._disc_buffer.push({"disc_obs": torch.randn(6, 1, 172),
                          "disc_obs_demo": torch.randn(6, 1, 172)})
    path = tmp_path / "checkpoint.pt"
    old.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    candidate = REPPOAgent(cfg, Env(), "cpu")
    candidate.migrate(checkpoint, CategoricalQ(10, 4, -100, 100).state_dict())
    return old, candidate, checkpoint


def test_migration_preserves_actor_discriminator_and_normalizers(migrated):
    old, candidate, _ = migrated
    x = torch.randn(12, 10)
    a, b = old._model.eval_actor(x), candidate._model.eval_actor(x)
    torch.testing.assert_close(a.mean, b.mean, rtol=0, atol=0)
    torch.testing.assert_close(a.stddev, b.stddev, rtol=0, atol=0)
    old.eval(); candidate.eval()
    delta = torch.randn(12, 172)
    torch.testing.assert_close(old._calc_disc_rewards(delta), candidate._calc_disc_rewards(delta), rtol=0, atol=0)
    assert candidate._model._action_dist._logstd_net.requires_grad
    assert not any("_critic_layers" in k or "_critic_out" in k for k in candidate.state_dict())
    assert not candidate._need_normalizer_update()
    torch.testing.assert_close(candidate._entropy_ref, b.entropy().mean())
    action = torch.randn(12, 4)
    torch.testing.assert_close(candidate._a_norm.normalize(candidate._a_norm.unnormalize(action)), action)


def test_momentum_mapping_skips_only_new_std_and_keeps_replay(migrated):
    old, candidate, _ = migrated
    params = [p for p in candidate._actor_optimizer._param_list
              if p is not candidate._model._action_dist._logstd_net]
    for a, b in zip(old._actor_optimizer._param_list, params):
        torch.testing.assert_close(old._actor_optimizer._optimizer.state[a]["momentum_buffer"],
            candidate._actor_optimizer._optimizer.state[b]["momentum_buffer"], rtol=0, atol=0)
    assert candidate._actor_optimizer.get_steps() == old._actor_optimizer.get_steps()
    torch.testing.assert_close(candidate._disc_buffer.get_data("disc_obs"), old._disc_buffer.get_data("disc_obs"))


def test_strict_migration_rejects_missing_legacy_key(migrated):
    _, candidate, checkpoint = migrated
    broken = copy.deepcopy(checkpoint)
    del broken["model_state_dict"]["_model._actor_layers.0.bias"]
    with pytest.raises(ValueError, match="tensor keys"):
        candidate.migrate(broken, candidate._model.q.state_dict())


def test_registered_agent_loads_legacy_training_checkpoint(migrated, tmp_path):
    from learning.agent_builder import build_agent
    old, candidate, checkpoint = migrated
    source, warm, cfg_path = (tmp_path / name for name in ("legacy.pt", "q.pt", "agent.yaml"))
    torch.save(checkpoint, source)
    torch.save({"state_dict": candidate._model.q.state_dict()}, warm)
    cfg = config()
    cfg["migration_q"] = str(warm)
    cfg_path.write_text(yaml.safe_dump(cfg))
    restored = build_agent(str(cfg_path), Env(), "cpu")
    restored.load(source)
    obs = torch.randn(8, 10)
    torch.testing.assert_close(old._model.eval_actor(obs).mean,
                               restored._model.eval_actor(obs).mean, rtol=0, atol=0)


def test_critic_uses_normalized_robot_action(migrated, monkeypatch):
    _, candidate, _ = migrated
    obs, action = torch.randn(8, 10), torch.randn(8, 4)
    captured = {}

    def loss(x, a, target, weights):
        captured.update(obs=x, action=a)
        return target.mean()

    monkeypatch.setattr(candidate._model.q, "loss", loss)
    candidate._compute_critic_loss(dict(obs=obs, action=candidate._a_norm.unnormalize(action),
        tar_val=torch.zeros(8), phase_weight=torch.ones(8)))
    torch.testing.assert_close(captured["action"], action)
    torch.testing.assert_close(captured["obs"], candidate._obs_norm.normalize(obs))


def test_centered_entropy_at_reference_matches_original_returns():
    from learning.rl_util import compute_td_lambda_return
    reward, q = torch.rand(8, 3), torch.rand(8, 3)
    done = torch.randint(0, 4, (8, 3))
    q[(done == 1) | (done == 2)] = 0
    expected = compute_td_lambda_return(reward, q, done, .99, .95)
    torch.testing.assert_close(soft_lambda_returns(reward, q, torch.zeros_like(q), done, alpha=.01), expected)
    shifted = soft_lambda_returns(torch.ones(3, 2), torch.full((3, 2), 5.),
        torch.full((3, 2), -.5), torch.zeros(3, 2), gamma=.9, lam=1., alpha=1.)
    torch.testing.assert_close(shifted[-1], torch.full((2,), 5.95))
    torch.testing.assert_close(shifted[-2], 1 + .9 * (shifted[-1] + .5))


def test_real_actor_update_uses_q_not_value_and_saves_four_optimizers(migrated, tmp_path):
    _, candidate, _ = migrated
    batch = dict(obs=torch.randn(16, 10), old_mean=torch.zeros(16, 4),
                 old_logstd=torch.full((16, 4), -2.9957323), phase_weight=torch.ones(16),
                 diag_phase=torch.linspace(0, 1, 16))
    with torch.no_grad():
        dist = candidate._model.eval_actor(batch["obs"])
        batch.update(old_mean=dist.mean.clone(), old_logstd=dist.logstd.clone())
    for key, val in batch.items():
        candidate._exp_buffer.set_data_flat(key, val)
    candidate._exp_buffer.set_total_samples(16)
    before = {k:v.clone() for k,v in candidate._model.state_dict().items()}
    metrics = candidate._update_actor(8, 2)
    assert metrics["reppo_max_post_step_mean_kl"] >= 0
    assert any(not torch.equal(v,before[k]) for k,v in candidate._model.state_dict().items() if "actor" in k)
    assert all(torch.equal(v,before[k]) for k,v in candidate._model.state_dict().items() if k.startswith("q."))
    path = tmp_path / "reppo.pt"
    candidate.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    assert set(saved["optimizer_state_dicts"]) == {"_actor_optimizer", "_critic_optimizer", "_disc_optimizer", "_dual_optimizer"}
    restored = REPPOAgent(config(), Env(), "cpu")
    restored.load(path)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in candidate.state_dict().items())
    restored.resume(path)
    assert not restored._need_normalizer_update()
    assert {k:v.get_steps() for k,v in restored._get_optimizers().items()} == {
        k:v.get_steps() for k,v in candidate._get_optimizers().items()}
