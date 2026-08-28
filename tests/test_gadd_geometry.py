import pathlib
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from learning.add_model import ADDModel


class _Env:
    def __init__(self, dim=12):
        self._obs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self._action = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def get_obs_space(self):
        return self._obs

    def get_disc_obs_space(self):
        return self._obs

    def get_action_space(self):
        return self._action


def _config(geometry="add", spectral_norm=False):
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "disc_geometry": geometry,
        "disc_spectral_norm": spectral_norm,
    }


@pytest.mark.parametrize(
    "geometry,input_dim", [("add", 12), ("ref_concat", 24)])
def test_discriminator_geometry_shapes(geometry, input_dim):
    model = ADDModel(_config(geometry), _Env())
    diff = torch.randn(7, 12)
    context = torch.randn(7, 12)
    disc_input = model.build_disc_input(diff, context)
    assert disc_input.shape == (7, input_dim)
    assert model.eval_disc(diff, context).shape == (7, 1)


def test_full_spectral_norm_covers_hidden_and_output_layers():
    model = ADDModel(_config(spectral_norm=True), _Env())
    hidden = [
        layer for layer in model._disc_layers.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert len(hidden) == 2
    layers = hidden + [model._disc_logits]
    assert all(torch.nn.utils.parametrize.is_parametrized(
        layer, "weight") for layer in layers)


def test_gadd_config_is_full_sn_only():
    path = ROOT / "data" / "agents" / "gadd_humanoid_agent.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)
    assert config["disc_geometry"] == "add"
    assert config["disc_spectral_norm"] is True
    assert "disc_influence_allocation" not in config
    assert config["disc_grad_penalty"] == 0


def test_failed_metric_configs_are_removed():
    for filename in (
            "gadd_global_metric_humanoid_agent.yaml",
            "gadd_metric_raw_gp_humanoid_agent.yaml",
            "gadd_metric_z_gp_humanoid_agent.yaml"):
        assert not (ROOT / "data" / "agents" / filename).exists()
