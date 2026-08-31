import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

from learning.sif_add_model import SIFDiscLayers, UnitNormLinear


def _groups():
    dims = [3, 6, 45, 84, 3, 3, 28]
    groups = []
    offset = 0
    for group_id, dim in enumerate(dims):
        groups.append((str(group_id), tuple(range(offset, offset + dim))))
        offset += dim
    return dims, groups


def test_sif_geometry_and_shapes():
    dims, groups = _groups()
    layers = SIFDiscLayers(groups, slot_width=146, global_rank=512)
    diff = torch.randn(5, sum(dims))

    assert layers.encode_semantics(diff).shape == (5, 1022)
    assert layers(diff).shape == (5, 1022)
    local_error, global_error = layers.orthogonality_errors()
    assert local_error < 1e-4
    assert global_error < 1e-4
    assert not any("spectral_norm" in repr(module).lower()
                   for module in layers.modules())


def test_sif_discriminator_is_non_expansive():
    dims, groups = _groups()
    layers = SIFDiscLayers(groups, slot_width=146, global_rank=512)
    head = UnitNormLinear(1022)
    diff = torch.randn(4, sum(dims), requires_grad=True)
    logit = head(layers(diff)).sum()
    gradient = torch.autograd.grad(logit, diff)[0]

    assert torch.all(torch.linalg.vector_norm(gradient, dim=-1) <= 1.0001)
    assert torch.allclose(
        torch.sum(torch.square(head.weight)), torch.ones(()), atol=1e-6)


def test_semantic_slots_do_not_mix_before_fusion():
    dims, groups = _groups()
    layers = SIFDiscLayers(groups, slot_width=146, global_rank=512)
    base = torch.zeros(1, sum(dims))
    changed = base.clone()
    changed[:, groups[2][1]] = 1.0
    delta = layers.encode_semantics(changed) - layers.encode_semantics(base)
    delta = delta.reshape(1, len(groups), 146)

    inactive = torch.cat((delta[:, :2], delta[:, 3:]), dim=1)
    assert torch.count_nonzero(inactive) == 0
    assert torch.count_nonzero(delta[:, 2]) > 0


def test_sif_checkpoint_round_trip():
    dims, groups = _groups()
    source = SIFDiscLayers(groups, slot_width=146, global_rank=512)
    target = SIFDiscLayers(groups, slot_width=146, global_rank=512)
    diff = torch.randn(3, sum(dims))

    target.load_state_dict(source.state_dict())
    assert torch.allclose(source(diff), target(diff), atol=1e-6)
