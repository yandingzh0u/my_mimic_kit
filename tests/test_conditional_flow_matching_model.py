from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

from learning.flow_matching.conditional_flow_matching_model import (
    CONDITIONAL_FLOW_FORMAT_VERSION,
    CONDITIONAL_FLOW_MODEL_TYPE,
    ConditionalFlowMatchingModel,
    conditional_checkpoint_payload,
)


def _config():
    return {
        "arch_name": "ConditionalDiT",
        "num_disc_obs_steps": 10,
        "input_dim": 20,
        "latent_dim": 8,
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


def test_standard_conditional_flow_loss_has_finite_gradients():
    model = ConditionalFlowMatchingModel(_config(), "cpu")
    x1 = torch.randn(4, 20)
    latent = F.normalize(torch.randn(4, 8), dim=-1)
    x0 = torch.randn_like(x1)
    times = torch.tensor([0.1, 0.3, 0.6, 0.9])
    null_mask = torch.tensor([False, True, False, True])

    loss = model(
        x1,
        latent,
        null_mask=null_mask,
        base_noise=x0,
        times=times,
    )
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.dmodel.null_condition.grad is not None
    assert torch.isfinite(model.dmodel.null_condition.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.dmodel.condition_embedder.parameters()
    )


def test_explicit_conditional_null_and_paired_mismatch_api():
    model = ConditionalFlowMatchingModel(_config(), "cpu").eval()
    x1 = torch.randn(3, 20)
    latent = F.normalize(torch.randn(3, 8), dim=-1)
    times = [0.25, 0.5, 0.75]
    antithetic_noise = torch.randn(1, 10, 2)
    antithetic_noise = torch.cat((antithetic_noise, -antithetic_noise), dim=0)

    conditional = model.conditional_mismatch(x1, latent, times, antithetic_noise)
    null = model.null_mismatch(x1, times, antithetic_noise)
    paired = model.paired_mismatch(x1, latent, times, antithetic_noise)
    per_time = model.paired_mismatch_per_time(
        x1, latent, times, antithetic_noise
    )

    assert conditional.shape == null.shape == (3,)
    assert per_time["conditional"].shape == per_time["null"].shape == (3, 2, 3)
    torch.testing.assert_close(paired["conditional"], conditional, rtol=0, atol=0)
    torch.testing.assert_close(paired["null"], null, rtol=0, atol=0)
    assert torch.isfinite(conditional).all() and torch.isfinite(null).all()


def test_fixed_noise_pair_is_deterministic_and_validates_shapes():
    model = ConditionalFlowMatchingModel(_config(), "cpu").eval()
    x1 = torch.randn(2, 20)
    latent = F.normalize(torch.randn(2, 8), dim=-1)
    noise = torch.randn(1, 20)
    times = [0.25, 0.5, 0.75]

    first = model.paired_mismatch(x1, latent, times, noise)
    second = model.paired_mismatch(x1, latent, times, noise)
    for key in ("conditional", "null"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    with pytest.raises(ValueError, match="latent_condition must have shape"):
        model(x1, torch.randn(2, 7))
    with pytest.raises(ValueError, match="boolean vector"):
        model(x1, latent, null_mask=torch.ones(2))
    with pytest.raises(ValueError, match="K=1 or K=2"):
        model.mismatch_per_time(x1, latent, times, torch.randn(3, 20))

    with pytest.raises(ValueError, match="unit L2 norm"):
        model(x1, torch.randn(2, 8))


def test_window_normalizer_round_trip():
    model = ConditionalFlowMatchingModel(_config(), "cpu")
    windows = torch.randn(32, 10, 2) * 1.7 + 0.4
    model.update_normalizer(windows)
    reconstructed = model.unnormalize(model.normalize(windows))
    torch.testing.assert_close(reconstructed, windows)


def test_checkpoint_payload_describes_encoder_and_condition_contract():
    model = ConditionalFlowMatchingModel(_config(), "cpu")
    encoder = nn.Linear(20, 8)
    encoder_schema = {
        "type": "motion_window_encoder",
        "input_dim": 20,
        "latent_dim": 8,
    }
    calibration = {
        "times": torch.tensor([0.25, 0.5, 0.75]),
        "base_noise": torch.zeros(1, 10, 2),
    }
    payload = conditional_checkpoint_payload(
        model,
        encoder,
        encoder_schema=encoder_schema,
        calibration=calibration,
        offline_validation={"gate_passed": True},
        encoder_gate={"gate_passed": True},
        iteration=12,
    )

    assert payload["format_version"] == CONDITIONAL_FLOW_FORMAT_VERSION
    assert payload["model_type"] == CONDITIONAL_FLOW_MODEL_TYPE
    assert payload["iteration"] == 12
    assert set(payload).issuperset(
        {"model_state_dict", "encoder_state_dict", "metadata", "calibration"}
    )
    metadata = payload["metadata"]
    assert metadata["latent_dim"] == 8
    assert metadata["condition_mode"] == "continuous_or_null"
    assert metadata["encoder_schema"] == encoder_schema
    assert metadata["condition_schema"]["latent_dim"] == 8
    assert metadata["condition_schema"]["null_sampling"] == (
        "trainer_supplied_boolean_mask"
    )
    assert metadata["condition_schema"]["reward_times"] == [0.25, 0.5, 0.75]
