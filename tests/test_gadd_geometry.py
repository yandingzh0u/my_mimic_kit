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


def _config(geometry):
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
        "metric_net": "fc_2layers_128units",
        "disc_geometry": geometry,
        "metric_max": 5.0,
    }


@pytest.mark.parametrize(
    "geometry,input_dim",
    [("add", 12), ("ref_concat", 24),
     ("global_metric", 12), ("conditioned_metric", 12)])
def test_discriminator_geometry_shapes(geometry, input_dim):
    model = ADDModel(_config(geometry), _Env())
    diff = torch.randn(7, 12)
    context = torch.randn(7, 12)
    disc_input = model.build_disc_input(diff, context)
    assert disc_input.shape == (7, input_dim)
    assert model.eval_disc(diff, context).shape == (7, 1)


@pytest.mark.parametrize("geometry", ["global_metric", "conditioned_metric"])
def test_metric_is_positive_bounded_and_unit_mean(geometry):
    model = ADDModel(_config(geometry), _Env())
    context = torch.randn(19, 12)
    weights = model.calc_metric_weights(context)
    assert torch.all(weights >= 0.2 - 1e-6)
    assert torch.all(weights <= 5.0 + 1e-6)
    assert torch.allclose(
        weights.mean(dim=-1), torch.ones_like(weights.mean(dim=-1)),
        atol=1e-6)


def test_conditioned_metric_preserves_zero_positive_sample():
    model = ADDModel(_config("conditioned_metric"), _Env())
    context = torch.randn(5, 12)
    transformed = model.transform_diff(torch.zeros(5, 12), context)
    assert torch.count_nonzero(transformed) == 0


def test_ablation_configs_match_the_intended_matrix():
    expected = {
        "add_humanoid_agent.yaml": ("add", "raw"),
        "gadd_refconcat_humanoid_agent.yaml": ("ref_concat", "raw"),
        "gadd_global_metric_humanoid_agent.yaml": ("global_metric", "raw"),
        "gadd_metric_raw_gp_humanoid_agent.yaml": ("conditioned_metric", "raw"),
        "gadd_metric_z_gp_humanoid_agent.yaml": ("conditioned_metric", "z"),
    }
    for filename, (geometry, gp_space) in expected.items():
        with (ROOT / "data" / "agents" / filename).open() as stream:
            config = yaml.safe_load(stream)
        assert config.get("disc_geometry", "add") == geometry
        assert config.get("disc_gp_space", "raw") == gp_space
