from pathlib import Path
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import envs.phase_transition_env as phase_transition_env
import learning.phase_transition_agent as phase_transition_agent
import learning.phase_transition_model as phase_transition_model
import learning.experience_buffer as experience_buffer


class _TinyEnv:
    def __init__(self, num_envs=2):
        self._num_envs = num_envs

    def get_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_num_envs(self):
        return self._num_envs

    def get_aligned_self_obs_dim(self):
        return 5

    def get_aligned_command_dim(self):
        return 3

    def get_phase_transition_reference_stats(self):
        zeros = torch.zeros(3)
        ones = torch.ones(3)
        return zeros, ones, zeros.clone(), ones.clone()


@pytest.fixture
def preserve_torch_rng():
    # Regression tests that instantiate extra networks must not change the
    # random stream observed by unrelated tests collected after this module.
    with torch.random.fork_rng(devices=[]):
        yield


def _model_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "transition_init_output_scale": 0.01,
    }


def test_config_contract_extends_aligned_contract():
    valid = {
        "enable_tar_obs": False,
        "enable_phase_obs": False,
        "num_disc_obs_steps": 1,
        "aligned_command_step": 1,
        "global_obs": True,
        "phase_transition_stats_samples": 128,
    }
    phase_transition_env.validate_phase_transition_config(valid)

    invalid = dict(valid)
    invalid["phase_transition_stats_samples"] = 0
    with pytest.raises(ValueError):
        phase_transition_env.validate_phase_transition_config(invalid)


def test_reconstruction_is_algebraically_exact_and_reset_safe():
    batch = 7
    dim = 4
    self_dim = 3
    sim_state = torch.randn(batch, dim)
    sim_motion = torch.randn(batch, dim)
    ref_state = torch.randn(batch, dim)
    ref_motion = torch.randn(batch, dim)
    curr_error = ref_state - sim_state
    obs = torch.cat([
        torch.randn(batch, self_dim), curr_error, ref_motion
    ], dim=-1)

    transition = phase_transition_agent.reconstruct_phase_matched_transition(
        obs=obs,
        next_sim_state=sim_state + sim_motion,
        next_ref_state=ref_state + ref_motion,
        self_dim=self_dim,
        command_dim=dim,
    )
    for name, expected in (
        ("sim_state", sim_state),
        ("sim_motion", sim_motion),
        ("ref_state", ref_state),
        ("ref_motion", ref_motion),
    ):
        # Reconstructing a difference after float32 additions can differ by
        # one rounding unit; this still identifies the same pre-reset row.
        assert torch.allclose(
            transition[name], expected, atol=1e-6, rtol=1e-5)


def test_transition_error_contains_next_residual_and_motion_error():
    sim_state = torch.tensor([[1.0, 2.0]])
    sim_motion = torch.tensor([[0.25, -0.5]])
    ref_state = torch.tensor([[2.0, 4.0]])
    ref_motion = torch.tensor([[1.0, 0.5]])
    state_mean = torch.tensor([0.5, 1.0])
    state_scale = torch.tensor([2.0, 4.0])
    motion_mean = torch.tensor([0.1, -0.1])
    motion_scale = torch.tensor([0.5, 2.0])

    transition_error, context = (
        phase_transition_agent.normalize_phase_matched_transition(
            sim_state=sim_state,
            sim_motion=sim_motion,
            ref_state=ref_state,
            ref_motion=ref_motion,
            state_mean=state_mean,
            state_scale=state_scale,
            motion_mean=motion_mean,
            motion_scale=motion_scale,
        )
    )
    next_error = ref_state + ref_motion - sim_state - sim_motion
    motion_error = ref_motion - sim_motion
    assert torch.allclose(
        transition_error,
        torch.cat([
            next_error / state_scale,
            motion_error / motion_scale,
        ], dim=-1),
    )
    assert torch.allclose(
        context,
        torch.cat([
            (ref_state - state_mean) / state_scale,
            (ref_motion - motion_mean) / motion_scale,
        ], dim=-1),
    )

    # q_t = m_t - delta_t = e_{t+1} - e_t.
    curr_error = ref_state - sim_state
    assert torch.allclose(motion_error, next_error - curr_error)


def test_reference_transition_maps_to_zero_error():
    ref_state = torch.randn(5, 3)
    ref_motion = torch.randn(5, 3)
    ones = torch.ones(3)
    zeros = torch.zeros(3)
    transition_error, _ = (
        phase_transition_agent.normalize_phase_matched_transition(
            sim_state=ref_state,
            sim_motion=ref_motion,
            ref_state=ref_state,
            ref_motion=ref_motion,
            state_mean=zeros,
            state_scale=ones,
            motion_mean=zeros,
            motion_scale=ones,
        )
    )
    assert torch.equal(transition_error, torch.zeros_like(transition_error))


def test_anchored_model_score_is_exactly_zero_at_reference():
    torch.manual_seed(3)
    model = phase_transition_model.PhaseTransitionCriticModel(
        _model_config(), _TinyEnv())
    context = torch.randn(8, 6)
    zero_error = torch.zeros_like(context)
    anchored_score = model.eval_anchored_score(zero_error, context)
    assert torch.equal(anchored_score, torch.zeros_like(anchored_score))

    error = torch.randn_like(context, requires_grad=True)
    score = model.eval_transition_score(error, context)
    score.sum().backward()
    assert error.grad is not None
    assert torch.all(torch.isfinite(error.grad))


def test_centered_score_cancels_output_offset():
    torch.manual_seed(11)
    model = phase_transition_model.PhaseTransitionCriticModel(
        _model_config(), _TinyEnv())
    error = torch.randn(6, 6)
    context = torch.randn(6, 6)
    score_before = model.eval_anchored_score(error, context)
    with torch.no_grad():
        model._disc_logits.bias.add_(123.0)
    score_after = model.eval_anchored_score(error, context)
    assert torch.allclose(score_before, score_after, atol=1e-5, rtol=1e-5)


def test_centered_score_cancels_phase_dependent_context_offset(
        preserve_torch_rng):
    class _ContextOffsetModel(
            phase_transition_model.PhaseTransitionCriticModel):
        def __init__(self, config, env):
            super().__init__(config, env)
            self.context_offset_scale = 0.0

        def eval_transition_score(self, transition_error,
                                  reference_context):
            score = super().eval_transition_score(
                transition_error, reference_context)
            # This offset changes with the phase/reference context but not
            # with the candidate transition error u.  Anchoring must remove
            # it exactly up to floating-point subtraction error.
            context_offset = (
                torch.sin(reference_context[..., :1])
                + torch.square(reference_context[..., 1:2])
            )
            return score + self.context_offset_scale * context_offset

    torch.manual_seed(17)
    model = _ContextOffsetModel(_model_config(), _TinyEnv())
    error = torch.randn(9, 6)
    context = torch.randn(9, 6)
    score_before = model.eval_anchored_score(error, context)
    model.context_offset_scale = 7.0
    score_after = model.eval_anchored_score(error, context)
    assert torch.allclose(score_before, score_after, atol=2e-6, rtol=1e-5)


def test_wasserstein_loss_sign_and_zero_gp_ablation():
    anchored_score = torch.tensor([-2.0, -1.0], requires_grad=True)
    grad_norm = torch.tensor([0.2, 3.0])
    loss = phase_transition_agent.centered_wasserstein_gp_loss(
        anchored_score, grad_norm, gp_weight=0.0)
    assert torch.allclose(loss, torch.tensor(-1.5))
    loss.backward()
    # Gradient descent therefore makes policy-transition scores more
    # negative relative to their exact phase-matched reference score.
    assert torch.all(anchored_score.grad > 0)


def test_wasserstein_objective_uses_rms_dimension_scale():
    score = torch.tensor([-18.0, -18.0])
    grad_norm = torch.ones(2)
    loss = phase_transition_agent.centered_wasserstein_gp_loss(
        score,
        grad_norm,
        gp_weight=10.0,
        dimension_scale=18.0,
    )
    assert torch.allclose(loss, torch.tensor(-1.0))


def test_gradient_penalty_norm_is_taken_only_over_transition_error():
    torch.manual_seed(13)
    model = phase_transition_model.PhaseTransitionCriticModel(
        _model_config(), _TinyEnv())
    error = torch.randn(4, 6, requires_grad=True)
    context = torch.randn(4, 6, requires_grad=True)
    score = model.eval_transition_score(error, context).squeeze(-1)
    grad_u = torch.autograd.grad(
        score,
        error,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    assert grad_u.shape == error.shape
    assert context.grad is None
    grad_norm = torch.linalg.vector_norm(grad_u, dim=-1)
    loss = phase_transition_agent.centered_wasserstein_gp_loss(
        score - model.eval_transition_score(
            torch.zeros_like(error), context).squeeze(-1),
        grad_norm,
        gp_weight=10.0,
    )
    assert torch.isfinite(loss)


def test_reward_is_bounded_and_reference_anchored():
    score = torch.tensor([-4.0, -1.0, 0.0, 1.0])
    reward = phase_transition_agent.anchored_transition_reward(
        score, scale=2.0, tau=1.0)
    assert torch.allclose(
        reward, torch.tensor([0.4, 1.0, 2.0, 1.0]))
    assert torch.all(reward > 0)
    assert torch.all(reward <= 2.0)
    assert reward[2] == torch.max(reward)
    assert reward[1] == reward[3]


def test_transition_inputs_are_clipped_symmetrically():
    large = torch.tensor([[100.0, -100.0]])
    zeros = torch.zeros_like(large)
    ones = torch.ones(2)
    transition_error, context = (
        phase_transition_agent.normalize_phase_matched_transition(
            sim_state=-large,
            sim_motion=zeros,
            ref_state=large,
            ref_motion=large,
            state_mean=zeros[0],
            state_scale=ones,
            motion_mean=zeros[0],
            motion_scale=ones,
            clip=10.0,
        )
    )
    assert torch.max(transition_error) <= 10.0
    assert torch.min(transition_error) >= -10.0
    assert torch.max(context) <= 10.0
    assert torch.min(context) >= -10.0


def test_phase_derangement_stays_in_motion_and_respects_loop_distance():
    motion_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2])
    phase = torch.tensor([0.0, 0.2, 0.5, 0.9, 0.05, 0.2, 0.9, 0.4])
    is_wrap = torch.tensor(
        [True, True, True, True, False, False, False, True])
    partner, valid, distance = (
        phase_transition_agent.build_phase_derangement(
            motion_id, phase, is_wrap, min_phase_distance=0.1)
    )
    row = torch.arange(motion_id.shape[0])
    assert torch.all(motion_id[partner] == motion_id)
    assert torch.all(partner[valid] != row[valid])
    assert torch.all(distance[valid] >= 0.1)
    assert not valid[-1]
    assert partner[-1] == row[-1]


def test_replay_round_trip_keeps_transition_and_phase_metadata():
    buffer = experience_buffer.ExperienceBuffer(
        buffer_length=4, batch_size=1, device="cpu")
    data = {
        "sim_state": torch.randn(3, 1, 2),
        "sim_motion": torch.randn(3, 1, 2),
        "ref_state": torch.randn(3, 1, 2),
        "ref_motion": torch.randn(3, 1, 2),
        "motion_id": torch.tensor([[0], [0], [1]]),
        "motion_phase": torch.tensor([[0.1], [0.6], [0.3]]),
        "motion_is_wrap": torch.tensor([[True], [True], [False]]),
    }
    buffer.push(data)
    restored = experience_buffer.ExperienceBuffer(
        buffer_length=4, batch_size=1, device="cpu")
    restored.load_state_dict(buffer.state_dict())
    for name in data:
        assert torch.equal(
            restored.get_data(name)[:3], buffer.get_data(name)[:3])


def _build_tiny_agent_and_batch(gp_weight):
    config = yaml.safe_load((
        ROOT / "data/agents/phase_transition_critic_humanoid_agent.yaml"
    ).read_text())
    config["model"] = _model_config()
    config["disc_grad_penalty"] = gp_weight
    config["disc_buffer_size"] = 16
    config["normalizer_samples"] = 0
    env = _TinyEnv(num_envs=2)
    agent = phase_transition_agent.PhaseTransitionCriticAgent(
        config, env, "cpu")

    phase = torch.tensor([0.0, 0.25, 0.5, 0.75])
    ref_state = torch.stack([phase, 2.0 * phase, -phase], dim=-1)
    ref_motion = torch.stack([
        0.1 + phase, 0.2 - phase, 0.05 * torch.ones_like(phase)
    ], dim=-1)
    curr_error = 0.2 * torch.ones_like(ref_state)
    sim_state = ref_state - curr_error
    sim_motion = 0.5 * ref_motion
    obs = torch.cat([
        torch.zeros(4, 5), curr_error, ref_motion
    ], dim=-1)
    next_sim = sim_state + sim_motion
    next_ref = ref_state + ref_motion
    batch = {
        "obs": obs,
        "disc_obs": next_sim,
        "disc_obs_demo": next_ref,
        "motion_id": torch.zeros(4, dtype=torch.long),
        "motion_phase": phase,
        "motion_is_wrap": torch.ones(4, dtype=torch.bool),
    }
    replay = phase_transition_agent.reconstruct_phase_matched_transition(
        obs, next_sim, next_ref, self_dim=5, command_dim=3)
    replay.update({
        "motion_id": batch["motion_id"],
        "motion_phase": batch["motion_phase"],
        "motion_is_wrap": batch["motion_is_wrap"],
    })
    agent._disc_buffer.push({
        name: value.unsqueeze(1) for name, value in replay.items()
    })
    return agent, batch


def test_zero_gp_ablation_short_circuits_second_order_path():
    agent, batch = _build_tiny_agent_and_batch(gp_weight=0.0)
    loss_info = agent._compute_disc_loss(batch)
    assert torch.isfinite(loss_info["disc_loss"])
    assert loss_info["disc_grad_penalty"] == 0
    assert agent._transition_private_counter == 0
    assert loss_info["disc_shuffle_valid_frac"] > 0
    loss_info["disc_loss"].backward()


def test_gp_uses_private_rng_and_advances_checkpointed_counter():
    agent, batch = _build_tiny_agent_and_batch(gp_weight=10.0)
    rng_before = torch.get_rng_state().clone()
    loss_info = agent._compute_disc_loss(batch)
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after)
    assert agent._transition_private_counter == 1
    assert torch.isfinite(loss_info["disc_loss"])
    assert torch.isfinite(loss_info["disc_grad_penalty"])
    assert loss_info["disc_shuffle_phase_distance"] >= 0.1


def test_gp_alpha_endpoints_and_norm_only_over_transition_error(
        preserve_torch_rng):
    class _KnownScore(torch.nn.Module):
        def eval_transition_score(self, transition_error,
                                  reference_context):
            # dF/du = 2u, while dF/dc = 3.  A full-input GP would therefore
            # be nonzero at u=0; the specified error-only GP must be zero.
            return (
                torch.sum(torch.square(transition_error), dim=-1)
                + 3.0 * torch.sum(reference_context, dim=-1)
            ).unsqueeze(-1)

    agent, _ = _build_tiny_agent_and_batch(gp_weight=10.0)
    agent._model = _KnownScore()
    transition_error = torch.tensor([
        [1.0, -2.0, 0.5, 0.0, 1.5, -0.25],
        [-1.0, 0.25, 2.0, -0.5, 0.0, 1.0],
    ])
    reference_context = torch.randn(
        transition_error.shape, requires_grad=True)
    alpha_zero = torch.zeros(transition_error.shape[0], 1)
    alpha_one = torch.ones(transition_error.shape[0], 1)

    norm_at_reference = agent._calc_interp_grad_norm(
        transition_error, reference_context, alpha_zero)
    norm_at_candidate = agent._calc_interp_grad_norm(
        transition_error, reference_context, alpha_one)

    assert torch.equal(
        norm_at_reference, torch.zeros_like(norm_at_reference))
    assert torch.allclose(
        norm_at_candidate,
        2.0 * torch.linalg.vector_norm(transition_error, dim=-1),
    )
    assert reference_context.grad is None


def test_fixed_reference_stats_survive_policy_update_and_checkpoint(
        tmp_path, preserve_torch_rng):
    agent, _ = _build_tiny_agent_and_batch(gp_weight=10.0)
    stat_names = (
        "_transition_state_mean",
        "_transition_state_scale",
        "_transition_motion_mean",
        "_transition_motion_scale",
    )
    before_update = {
        name: getattr(agent, name).detach().clone()
        for name in stat_names
    }

    actor_params = list(agent._model.get_actor_params())
    actor_loss = sum(torch.sum(torch.square(param)) for param in actor_params)
    agent._actor_optimizer.step(actor_loss)
    for name, expected in before_update.items():
        assert torch.equal(getattr(agent, name), expected)

    checkpoint = tmp_path / "fixed_reference_stats.pt"
    agent.save_checkpoint(checkpoint, next_iter=8)
    restored_config = yaml.safe_load((
        ROOT / "data/agents/phase_transition_critic_humanoid_agent.yaml"
    ).read_text())
    restored_config["model"] = _model_config()
    restored_config["disc_buffer_size"] = 16
    restored_config["normalizer_samples"] = 0
    restored = phase_transition_agent.PhaseTransitionCriticAgent(
        restored_config, _TinyEnv(num_envs=2), "cpu")
    restored.resume(checkpoint)

    assert restored._iter == 8
    for name, expected in before_update.items():
        assert torch.equal(getattr(restored, name), expected)


def test_full_checkpoint_restores_private_rng_and_atomic_replay(tmp_path):
    agent, _ = _build_tiny_agent_and_batch(gp_weight=10.0)
    agent._transition_private_counter.fill_(17)
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(checkpoint, next_iter=4)

    restored_config = yaml.safe_load((
        ROOT / "data/agents/phase_transition_critic_humanoid_agent.yaml"
    ).read_text())
    restored_config["model"] = _model_config()
    restored_config["disc_buffer_size"] = 16
    restored_config["normalizer_samples"] = 0
    restored = phase_transition_agent.PhaseTransitionCriticAgent(
        restored_config, _TinyEnv(num_envs=2), "cpu")
    restored.resume(checkpoint)

    assert restored._iter == 4
    assert restored._transition_private_counter == 17
    expected_keys = {
        "sim_state", "sim_motion", "ref_state", "ref_motion",
        "motion_id", "motion_phase", "motion_is_wrap",
    }
    assert set(restored._disc_buffer._buffers) == expected_keys
    for name in expected_keys:
        assert torch.equal(
            restored._disc_buffer.get_data(name),
            agent._disc_buffer.get_data(name),
        )


def test_method_configs_and_builder_routes():
    agent_config = yaml.safe_load((
        ROOT / "data/agents/phase_transition_critic_humanoid_agent.yaml"
    ).read_text())
    env_config = yaml.safe_load((
        ROOT / "data/envs/phase_transition_critic_humanoid_roll_env.yaml"
    ).read_text())
    assert agent_config["agent_name"] == "PHASE_TRANSITION_CRITIC"
    assert agent_config["disc_grad_penalty"] == 10.0
    assert agent_config["disc_epochs"] == 1
    assert agent_config["disc_optimizer"]["betas"] == [0.0, 0.9]
    assert agent_config["transition_reward_tau"] is None
    assert agent_config["phase_shuffle_min_distance"] == 0.1
    assert env_config["env_name"] == "phase_transition_critic"
    assert env_config["enable_tar_obs"] is False
    assert env_config["enable_phase_obs"] is False

    agent_builder = (
        ROOT / "mimickit/learning/agent_builder.py"
    ).read_text()
    env_builder = (ROOT / "mimickit/envs/env_builder.py").read_text()
    assert 'agent_name == "PHASE_TRANSITION_CRITIC"' in agent_builder
    assert 'env_name == "phase_transition_critic"' in env_builder

    for suffix, budget in (
        ("smoke", "--max_samples 10240"),
        ("scale_smoke", "--max_samples 262144"),
        ("2k_8192", "--max_samples 524288000"),
    ):
        args = (ROOT / (
            "args/phase_transition_critic_humanoid_roll_{}_args.txt".format(
                suffix)
        )).read_text()
        assert budget in args
