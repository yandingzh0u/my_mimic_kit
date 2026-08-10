from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

from learning.flow_smp_agent import FlowSMPAgent


class _KnownMismatchPrior(nn.Module):
    def aggregate_mismatch(self, x1, times, base_noise, *, use_ema=True):
        assert use_ema
        assert times.tolist() == [0.25, 0.5, 0.75]
        assert tuple(base_noise.shape) == (1, 2)
        return x1[:, 0]


def test_flow_reward_uses_frozen_expert_scale_and_reports_distribution():
    agent = FlowSMPAgent.__new__(FlowSMPAgent)
    nn.Module.__init__(agent)
    agent._prior_model = _KnownMismatchPrior()
    agent._smp_eval_batch_size = 2
    agent._flow_times = torch.tensor([0.25, 0.5, 0.75])
    agent._flow_base_noise = torch.zeros(1, 2)
    agent._flow_expert_scale = 2.0
    agent._flow_reward_alpha = 1.5
    agent._smp_reward_scale = 3.0

    normalized_windows = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
    )
    reward, info = agent._calc_smp_rewards(normalized_windows)

    expected = 3.0 * torch.exp(-1.5 * normalized_windows[:, 0] / 2.0)
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(info["flow_mismatch_raw_mean"], torch.tensor(1.75))
    torch.testing.assert_close(info["flow_mismatch_scaled_mean"], torch.tensor(0.875))
    assert set(info).issuperset(
        {
            "flow_mismatch_raw_p05",
            "flow_mismatch_raw_p50",
            "flow_mismatch_raw_p95",
            "flow_mismatch_scaled_p05",
            "flow_mismatch_scaled_p50",
            "flow_mismatch_scaled_p95",
            "flow_reward_saturated_high_frac",
            "flow_reward_saturated_low_frac",
        }
    )


def _checkpoint(gate_passed=True, num_noise=1):
    noise = torch.randn(1, 10, 2)
    if num_noise == 2:
        noise = torch.cat((noise, -noise), dim=0)
    return {
        "format_version": 1,
        "model_type": "unconditional_flow_matching",
        "model_state_dict": {},
        "metadata": {
            "input_dim": 20,
            "frame_dim": 2,
            "window_steps": 10,
            "time_embed_scale": 49.0,
            "aggregation": "t_squared_weighted_mean",
            "reward_noise_samples": num_noise,
        },
        "calibration": {
            "expert_scale": 1.0,
            "times": torch.tensor([0.25, 0.5, 0.75]),
            "base_noise": noise,
        },
        "offline_validation": {"gate_passed": gate_passed},
    }


def test_checkpoint_requires_offline_gate_and_accepts_antithetic_fallback():
    agent = FlowSMPAgent.__new__(FlowSMPAgent)
    prior_config = {
        "time_embed_scale": 49.0,
        "reward_times": [0.25, 0.5, 0.75],
    }

    metadata, _ = agent._validate_checkpoint(
        _checkpoint(gate_passed=True, num_noise=2),
        prior_config,
        input_dim=20,
        frame_dim=2,
        window_steps=10,
    )
    assert metadata["reward_noise_samples"] == 2

    with pytest.raises(ValueError, match="offline mismatch gate"):
        agent._validate_checkpoint(
            _checkpoint(gate_passed=False),
            prior_config,
            input_dim=20,
            frame_dim=2,
            window_steps=10,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_validation_accepts_cuda_mapped_calibration():
    agent = FlowSMPAgent.__new__(FlowSMPAgent)
    checkpoint = _checkpoint()
    checkpoint["calibration"]["times"] = checkpoint["calibration"]["times"].cuda()
    checkpoint["calibration"]["base_noise"] = checkpoint["calibration"][
        "base_noise"
    ].cuda()

    agent._validate_checkpoint(
        checkpoint,
        {"time_embed_scale": 49.0, "reward_times": [0.25, 0.5, 0.75]},
        input_dim=20,
        frame_dim=2,
        window_steps=10,
    )
