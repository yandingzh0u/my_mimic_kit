import copy
import math
from pathlib import Path
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import torch.nn.functional as functional
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.add_model as add_model
import learning.hinge_sn_add_model as hinge_sn_add_model
import learning.hinge_sn_aligned_add_agent as hinge_sn_agent


class _TinyModelEnv:
    def get_obs_space(self):
        return spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)


def _model_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


def _linear_modules(module):
    return [item for item in module.modules()
            if isinstance(item, torch.nn.Linear)]


def _has_spectral_norm(module):
    parametrizations = getattr(module, "parametrizations", None)
    return (parametrizations is not None
            and "weight" in parametrizations
            and len(parametrizations.weight) == 1
            and hasattr(parametrizations.weight[0], "_u")
            and hasattr(parametrizations.weight[0], "_v"))


def _build_bare_agent(*, gp=0.0, consistency=0.0, noise_std=0.01,
                      reward_type="add_softplus", seed=17):
    agent = object.__new__(hinge_sn_agent.HingeSNAlignedADDAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._disc_hinge_margin = 1.0
    agent._disc_grad_penalty = gp
    agent._disc_logit_reg = 0.0
    agent._disc_consistency_weight = consistency
    agent._disc_consistency_noise_std = noise_std
    agent._disc_reward_type = reward_type
    agent._disc_reward_scale = 2.0
    agent._disc_eval_batch_size = 0
    agent._disc_consistency_generator = torch.Generator(device="cpu")
    agent._disc_consistency_generator.manual_seed(seed)
    return agent


class _IdentityNormalizer:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.inputs = []

    def normalize(self, value):
        self.inputs.append(value.detach().clone())
        return value * self.scale


class _ReplayBuffer:
    def __init__(self, disc_obs, disc_obs_demo):
        self.disc_obs = disc_obs
        self.disc_obs_demo = disc_obs_demo

    def sample(self, count):
        assert count == self.disc_obs.shape[0]
        return {
            "disc_obs": self.disc_obs.clone(),
            "disc_obs_demo": self.disc_obs_demo.clone(),
        }


class _SumDiscriminator:
    def __init__(self):
        self.inputs = []
        self.input_requires_grad = []

    def eval_disc(self, disc_obs):
        self.inputs.append(disc_obs.detach().clone())
        self.input_requires_grad.append(disc_obs.requires_grad)
        return torch.sum(disc_obs, dim=-1, keepdim=True)

    def get_disc_logit_weights(self):
        return torch.zeros(1)

    def get_disc_sn_diagnostics(self):
        return {}


class _FirstCoordinateDiscriminator:
    def eval_disc(self, disc_obs):
        return disc_obs[..., :1]


def _attach_loss_fixture(agent, *, normalizer_scale=1.0):
    agent._pos_diff = torch.zeros(1)
    agent._disc_obs_norm = _IdentityNormalizer(scale=normalizer_scale)
    agent._disc_buffer = _ReplayBuffer(
        disc_obs=torch.zeros(2, 1),
        disc_obs_demo=torch.tensor([[2.0], [-1.0]]),
    )
    agent._model = _SumDiscriminator()
    batch = {
        "disc_obs": torch.zeros(2, 1),
        "disc_obs_demo": torch.tensor([[-2.0], [0.0]]),
    }
    return batch


def test_spectral_norm_is_only_on_all_discriminator_linear_layers():
    torch.manual_seed(23)
    model = hinge_sn_add_model.HingeSNADDModel(
        _model_config(), _TinyModelEnv(), spectral_norm=True,
        sn_power_iterations=1)

    disc_linears = _linear_modules(model._disc_layers) + [model._disc_logits]
    assert len(disc_linears) == 3
    assert all(_has_spectral_norm(layer) for layer in disc_linears)
    assert model._disc_sn_layers == disc_linears

    policy_linears = (
        _linear_modules(model._actor_layers)
        + _linear_modules(model._action_dist)
        + _linear_modules(model._critic_layers)
        + [model._critic_out]
    )
    assert policy_linears
    assert all(not _has_spectral_norm(layer) for layer in policy_linears)


def test_spectral_norm_keeps_actor_critic_initialization_and_global_rng():
    torch.manual_seed(29)
    stock = add_model.ADDModel(_model_config(), _TinyModelEnv())
    stock_next_random = torch.rand(8)

    torch.manual_seed(29)
    hinge_sn = hinge_sn_add_model.HingeSNADDModel(
        _model_config(), _TinyModelEnv(), spectral_norm=True,
        sn_power_iterations=1)
    hinge_sn_next_random = torch.rand(8)

    for module_name in (
            "_actor_layers", "_action_dist", "_critic_layers",
            "_critic_out"):
        stock_state = getattr(stock, module_name).state_dict()
        hinge_state = getattr(hinge_sn, module_name).state_dict()
        assert stock_state.keys() == hinge_state.keys()
        for key in stock_state:
            assert torch.equal(stock_state[key], hinge_state[key])
    assert torch.equal(stock_next_random, hinge_sn_next_random)


def test_hinge_loss_sign_margin_and_active_fractions():
    agent = _build_bare_agent()
    batch = _attach_loss_fixture(agent)

    info = agent._compute_disc_loss(batch)

    # Positive score is 0: [1 - 0]+ = 1.  Negative scores are
    # [-2, 0, 2, -1]: mean([0, 1, 3, 0]) = 1.
    assert info["disc_hinge_pos_loss"].item() == pytest.approx(1.0)
    assert info["disc_hinge_neg_loss"].item() == pytest.approx(1.0)
    assert info["disc_hinge_loss"].item() == pytest.approx(2.0)
    assert info["disc_loss"].item() == pytest.approx(2.0)
    assert info["disc_hinge_pos_active_frac"].item() == pytest.approx(1.0)
    # Strictly active only when f(negative) > -margin; the boundary -1 is
    # inactive because its hinge term and subgradient are zero.
    assert info["disc_hinge_neg_active_frac"].item() == pytest.approx(0.5)
    assert info["disc_score_gap"].item() == pytest.approx(0.25)


def test_zero_gp_never_calls_autograd_grad(monkeypatch):
    agent = _build_bare_agent(gp=0.0)
    batch = _attach_loss_fixture(agent)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GP=0 must not construct autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", fail_if_called)
    info = agent._compute_disc_loss(batch)
    assert info["disc_grad_penalty"].item() == 0.0
    # No hidden requires_grad path is left on normalized policy differences.
    assert agent._model.input_requires_grad == [False]


def test_nonzero_gp_uses_negative_normalized_differences():
    agent = _build_bare_agent(gp=2.0)
    batch = _attach_loss_fixture(agent)
    info = agent._compute_disc_loss(batch)

    # The fake discriminator is f(x)=x, hence |df/dx|^2=1 for every negative.
    assert info["disc_grad_penalty"].item() == pytest.approx(1.0)
    assert info["disc_loss"].item() == pytest.approx(4.0)


def test_add_reward_mapping_is_numerically_unchanged():
    agent = _build_bare_agent(reward_type="add_softplus")
    agent._model = _FirstCoordinateDiscriminator()
    logits = torch.tensor([[-4.0], [0.0], [2.0]])

    actual = agent._calc_disc_rewards(logits)
    expected = 2.0 * functional.softplus(logits.squeeze(-1))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_smooth_margin_reward_has_anchor_scale_and_no_hard_plateau():
    agent = _build_bare_agent(reward_type="smooth_margin")
    agent._model = _FirstCoordinateDiscriminator()
    logits = torch.tensor([[-12.0], [-8.0], [-4.0], [-1.0], [0.0],
                           [1.0], [4.0]])

    rewards = agent._calc_disc_rewards(logits)
    assert rewards[5].item() == pytest.approx(2.0, rel=1e-7)
    assert torch.all(rewards > 0)
    assert torch.all(rewards[1:] > rewards[:-1])

    expected = (2.0 * functional.softplus(logits.squeeze(-1) + 1.0)
                / math.log1p(math.exp(2.0)))
    assert torch.allclose(rewards, expected, atol=1e-7, rtol=1e-7)


def test_consistency_perturbs_only_normalized_fresh_negatives():
    agent = _build_bare_agent(consistency=1.0, noise_std=0.01)
    batch = _attach_loss_fixture(agent, normalizer_scale=10.0)

    info = agent._compute_disc_loss(batch)
    assert info["disc_consistency_loss"].item() > 0
    assert len(agent._model.inputs) == 1

    all_inputs = agent._model.inputs[0]
    # Layout is one zero positive, four normalized negatives (two fresh and
    # two replay), then exactly two perturbed views of the fresh negatives.
    assert all_inputs.shape == (7, 1)
    assert torch.equal(all_inputs[:1], torch.zeros(1, 1))
    normalized_negatives = all_inputs[1:5]
    assert torch.equal(
        normalized_negatives,
        torch.tensor([[-20.0], [0.0], [20.0], [-10.0]]))
    perturbed_fresh = all_inputs[5:]
    perturbation = perturbed_fresh - normalized_negatives[:2]
    assert torch.any(perturbation != 0)
    assert torch.max(torch.abs(perturbation)).item() < 0.1


def test_consistency_rng_is_private_and_state_dict_resumable():
    agent = _build_bare_agent(consistency=1.0, noise_std=0.01, seed=37)
    reference = torch.zeros(5, 3)

    torch.manual_seed(101)
    global_state = torch.get_rng_state().clone()
    first = agent._sample_consistency_noise(reference)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.any(first != 0)

    saved_state = copy.deepcopy(agent.state_dict())
    expected_next = agent._sample_consistency_noise(reference)

    restored = _build_bare_agent(
        consistency=1.0, noise_std=0.01, seed=999)
    restored.load_state_dict(saved_state)
    actual_next = restored._sample_consistency_noise(reference)
    assert torch.equal(actual_next, expected_next)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_four_configs_form_a_one_factor_ablation_and_share_sn_model():
    paths = {
        "e1": ROOT / "data/agents/hinge_sn_gp_aligned_add_humanoid_agent.yaml",
        "e2": ROOT / "data/agents/hinge_sn_nogp_aligned_add_humanoid_agent.yaml",
        "e3": ROOT / "data/agents/hinge_sn_margin_reward_aligned_add_humanoid_agent.yaml",
        "e4": ROOT / "data/agents/hinge_sn_cr_aligned_add_humanoid_agent.yaml",
    }
    configs = {key: yaml.safe_load(path.read_text())
               for key, path in paths.items()}

    expected_factors = {
        "e1": (2, "add_softplus", 0.0),
        "e2": (0, "add_softplus", 0.0),
        "e3": (0, "smooth_margin", 0.0),
        "e4": (0, "add_softplus", 1.0),
    }
    for key, config in configs.items():
        assert config["agent_name"] == "HINGE_SN_ALIGNED_ADD"
        assert config["disc_hinge_margin"] == 1.0
        assert config["disc_spectral_norm"] is True
        assert config["disc_sn_power_iterations"] == 1
        assert config["disc_consistency_noise_std"] == 0.01
        assert (config["disc_grad_penalty"], config["disc_reward_type"],
                config["disc_consistency_weight"]) == expected_factors[key]

    factors = {"disc_grad_penalty", "disc_reward_type",
               "disc_consistency_weight"}
    reference = {key: value for key, value in configs["e2"].items()
                 if key not in factors}
    for name in ("e1", "e3", "e4"):
        candidate = {key: value for key, value in configs[name].items()
                     if key not in factors}
        assert candidate == reference


def test_sn_power_iteration_buffers_round_trip_in_model_state():
    torch.manual_seed(43)
    model = hinge_sn_add_model.HingeSNADDModel(
        _model_config(), _TinyModelEnv(), spectral_norm=True,
        sn_power_iterations=1)
    model.eval()
    probe = torch.randn(7, 3)
    expected = model.eval_disc(probe)
    state = copy.deepcopy(model.state_dict())

    sn_u_keys = [key for key in state if key.endswith(".0._u")]
    sn_v_keys = [key for key in state if key.endswith(".0._v")]
    raw_weight_keys = [key for key in state
                       if key.endswith("parametrizations.weight.original")]
    assert len(sn_u_keys) == 3
    assert len(sn_v_keys) == 3
    assert len(raw_weight_keys) == 3

    torch.manual_seed(999)
    restored = hinge_sn_add_model.HingeSNADDModel(
        _model_config(), _TinyModelEnv(), spectral_norm=True,
        sn_power_iterations=1)
    restored.load_state_dict(state)
    restored.eval()
    actual = restored.eval_disc(probe)
    assert torch.equal(actual, expected)
    restored_state = restored.state_dict()
    for key in sn_u_keys + sn_v_keys + raw_weight_keys:
        assert torch.equal(restored_state[key], state[key])


def test_serial_formal_args_use_8192_envs_and_2000_iterations():
    arg_files = sorted((ROOT / "args/hinge_sn_add").glob(
        "*_roll_2k_8192_args.txt"))
    assert len(arg_files) == 4
    for path in arg_files:
        tokens = path.read_text().split()
        args = dict(zip(tokens[0::2], tokens[1::2]))
        assert int(args["--num_envs"]) == 8192
        assert int(args["--max_samples"]) == 8192 * 32 * 2000

    launcher = (ROOT / "tools/hinge_sn_add/run_serial_roll_2k.sh")
    text = launcher.read_text()
    assert "variants=(" in text
    for name in ("e1_hinge_sn_gp", "e2_hinge_sn_nogp",
                 "e3_hinge_sn_margin_reward", "e4_hinge_sn_cr"):
        assert name in text
