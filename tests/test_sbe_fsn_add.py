import pathlib

import gymnasium.spaces as spaces
import torch

from learning.sbe_fsn_add_model import SBEFSNADDModel
from learning.semantic_block_sn import SemanticBlockEqualizedLinear


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    _dims = (3, 6, 45, 84, 3, 3, 28)

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172,))

    def get_disc_error_groups(self):
        groups = []
        start = 0
        for group_id, dim in enumerate(self._dims):
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


def test_model_is_one_dense_first_layer_without_group_encoders():
    model = SBEFSNADDModel(_config(), _Env())
    assert isinstance(model._disc_layers[0], SemanticBlockEqualizedLinear)
    assert model._disc_layers[0].weight.shape == (128, 172)
    assert not hasattr(model._disc_layers, "encoders")


def test_remaining_layers_and_head_use_full_sn_and_biases_remain():
    model = SBEFSNADDModel(_config(), _Env())
    assert model._disc_layers[0].bias is not None
    assert hasattr(model._disc_layers[2].parametrizations, "weight")
    assert model._disc_layers[2].bias is not None
    assert hasattr(model._disc_logits.parametrizations, "weight")
    assert model._disc_logits.bias is not None


def test_end_to_end_gradient_is_finite():
    model = SBEFSNADDModel(_config(), _Env()).train()
    inputs = torch.cat((torch.zeros(1, 172), torch.randn(32, 172)), dim=0)
    logits = model.eval_disc(inputs).squeeze(-1)
    loss = (torch.nn.functional.softplus(-logits[:1]).mean()
            + torch.nn.functional.softplus(logits[1:]).mean())
    loss.backward()
    first = model.get_semantic_layer()
    assert torch.isfinite(first.weight.grad).all()
    assert torch.count_nonzero(first.weight.grad) > 0


def test_config_has_no_gp_or_extra_capacity_knob():
    text = (ROOT / "data/agents/sbe_fsn_add_humanoid_agent.yaml").read_text()
    assert 'agent_name: "SBE_FSN_ADD"' in text
    assert "disc_grad_penalty: 0" in text
    for rejected in ("group_width", "modes", "gain", "temperature"):
        assert rejected not in text.lower()
