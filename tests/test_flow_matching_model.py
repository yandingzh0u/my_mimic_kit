from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

from learning.flow_matching.flow_matching_model import FlowMatchingModel


def _config():
    return {
        "arch_name": "DiT",
        "num_disc_obs_steps": 10,
        "input_dim": 20,
        "num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "dropout": 0.0,
        "time_embed_scale": 49.0,
        "model_ema_decay": 0.9,
        "model_ema_steps": 1,
        "model_ema_update_after": 0,
        "normalizer_std_clip": 0.2,
    }


class _RecordingZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.times = None

    def forward(self, x, timestep):
        self.inputs = x.detach().clone()
        self.times = timestep.detach().clone()
        return torch.zeros_like(x)


def test_forward_has_finite_gradients():
    model = FlowMatchingModel(_config(), "cpu")
    x1 = torch.randn(4, 20)
    loss = model(x1)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.dmodel.parameters()
    )


def test_flow_path_endpoints_and_scaled_times():
    model = FlowMatchingModel(_config(), "cpu")
    recorder = _RecordingZero()
    model.ema_dmodel.ema_model = recorder
    x1 = torch.arange(40, dtype=torch.float32).reshape(2, 20) / 10
    x0 = torch.stack((torch.zeros(20), torch.ones(20)))

    mismatch = model.mismatch_per_time(x1, [0.0, 1.0], x0)
    seen = recorder.inputs.reshape(2, 2, 2, 20)
    seen_times = recorder.times.reshape(2, 2, 2)

    assert mismatch.shape == (2, 2, 2)
    torch.testing.assert_close(seen[:, :, 0], x0.unsqueeze(0).expand(2, -1, -1))
    torch.testing.assert_close(seen[:, :, 1], x1[:, None, :].expand(-1, 2, -1))
    torch.testing.assert_close(seen_times[0, 0], torch.tensor([0.0, 49.0]))
    expected = torch.square(x1[:, None, :] - x0[None, :, :]).mean(dim=-1)
    torch.testing.assert_close(mismatch[:, :, 0], expected)
    torch.testing.assert_close(mismatch[:, :, 1], expected)


def test_fixed_noise_is_deterministic_and_accepts_window_shape():
    model = FlowMatchingModel(_config(), "cpu").eval()
    x1 = torch.randn(3, 20)
    base_noise = torch.randn(2, 10, 2)

    first = model.mismatch_per_time(x1, [0.25, 0.5, 0.75], base_noise)
    second = model.mismatch_per_time(x1, [0.25, 0.5, 0.75], base_noise)

    assert first.shape == (3, 2, 3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_aggregate_is_t_squared_weighted_mean_over_noise_and_time():
    model = FlowMatchingModel(_config(), "cpu")
    per_point = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=torch.float32
    )

    def fake_mismatch(self, x1, times, base_noise, *, use_ema=True):
        return per_point.expand(x1.shape[0], -1, -1)

    model.mismatch_per_time = types.MethodType(fake_mismatch, model)
    times = torch.tensor([0.25, 0.5, 0.75])
    result = model.aggregate_mismatch(torch.zeros(2, 20), times, torch.zeros(2, 20))
    expected_per_time = per_point.mean(dim=1).squeeze(0)
    expected = (expected_per_time * times.square()).sum() / times.square().sum()

    torch.testing.assert_close(result, expected.expand(2))


def test_normalizer_round_trip_and_validation():
    model = FlowMatchingModel(_config(), "cpu")
    samples = torch.randn(32, 20) * 2 + 3
    model.update_normalizer(samples)
    windows = samples.reshape(32, 10, 2)
    reconstructed = model.unnormalize(model.normalize(windows))

    torch.testing.assert_close(reconstructed, windows)
    with pytest.raises(ValueError, match="samples must have shape"):
        model(torch.randn(2, 19))
    with pytest.raises(ValueError, match="squared time weights"):
        model.aggregate_mismatch(samples[:2], [0.0], torch.zeros(1, 20))
