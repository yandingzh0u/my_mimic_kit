import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.paper_eval.aggregate_results import (
    aggregate_summaries,
    extract_paper_metrics,
    write_aggregate_tables,
)
from tools.paper_eval.evaluate_checkpoint import (
    _input_geometry_blocks,
    infer_checkpoint_metadata,
    intervene_observation,
    parse_training_log,
)
from tools.paper_eval.input_geometry import InputGeometryAccumulator


def test_training_log_parser_handles_carriage_returns_and_resumed_headers(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text(
        "Iteration Samples Wall_Time Samples_Per_Second\r"
        "0 262144 0.1 10\r"
        "100 26476544 1.0 20\r"
        "Iteration Samples Wall_Time Samples_Per_Second\n"
        "200 52690944 2.0 30\n",
        encoding="utf-8",
    )
    assert parse_training_log(log_file) == {
        "iteration": 200,
        "samples": 52690944,
        "wall_time_hours": 2.0,
        "samples_per_second": 30.0,
        "source": "log",
        "log_file": str(log_file.resolve()),
    }
    assert parse_training_log(log_file, 100)["samples"] == 26476544


def test_checkpoint_metadata_matches_intermediate_model_iteration(tmp_path):
    output = tmp_path / "run"
    models = output / "int_models"
    models.mkdir(parents=True)
    model = models / "model_100.pt"
    model.touch()
    (output / "log.txt").write_text(
        "Iteration Samples Wall_Time\n100 1234 0.5\n200 5678 1.0\n",
        encoding="utf-8",
    )
    metadata = infer_checkpoint_metadata(model)
    assert metadata["iteration"] == 100
    assert metadata["samples"] == 1234


@pytest.mark.parametrize("layout", ["aligned", "rcci_residual"])
@pytest.mark.parametrize("condition", ["zero_e", "zero_m", "reverse_e", "shuffle_m"])
def test_residual_command_interventions_modify_only_declared_block(layout, condition):
    self_dim, phi_dim = 2, 3
    total_dim = self_dim + (2 if layout == "aligned" else 3) * phi_dim
    obs = torch.arange(2 * total_dim, dtype=torch.float32).reshape(2, total_dim) + 1
    shuffle = torch.tensor([1, 0])
    changed = intervene_observation(
        obs, condition, layout, self_dim, phi_dim, shuffle
    )
    e_start = self_dim if layout == "aligned" else self_dim + phi_dim
    m_start = e_start + phi_dim

    assert torch.equal(changed[:, :e_start], obs[:, :e_start])
    if condition == "zero_e":
        assert torch.count_nonzero(changed[:, e_start:m_start]) == 0
        assert torch.equal(changed[:, m_start:m_start + phi_dim], obs[:, m_start:m_start + phi_dim])
    elif condition == "zero_m":
        assert torch.equal(changed[:, e_start:m_start], obs[:, e_start:m_start])
        assert torch.count_nonzero(changed[:, m_start:m_start + phi_dim]) == 0
    elif condition == "reverse_e":
        assert torch.equal(changed[:, e_start:m_start], -obs[:, e_start:m_start])
        assert torch.equal(changed[:, m_start:m_start + phi_dim], obs[:, m_start:m_start + phi_dim])
    else:
        assert torch.equal(changed[:, e_start:m_start], obs[:, e_start:m_start])
        assert torch.equal(
            changed[:, m_start:m_start + phi_dim],
            obs[shuffle, m_start:m_start + phi_dim],
        )


def test_nonresidual_method_rejects_interventions():
    with pytest.raises(ValueError, match="requires a residual"):
        intervene_observation(torch.zeros(2, 4), "zero_e", "none", 0, 0)


def test_input_geometry_reports_rank_condition_and_paired_correlation():
    blocks, pairs = _input_geometry_blocks("aligned", 1, 2, 5)
    accumulator = InputGeometryAccumulator(blocks, "cpu", pairs)
    observations = torch.tensor(
        [
            [0.0, 1.0, 2.0, 2.0, 4.0],
            [1.0, 2.0, 1.0, 4.0, 2.0],
            [2.0, 3.0, 0.0, 6.0, 0.0],
        ]
    )
    accumulator.update(observations[:2])
    accumulator.update(observations[2:])
    result = accumulator.finalize()
    assert result["sample_count"] == 3
    assert result["blocks"]["feedback_error"]["rank"] == 1
    pair = result["paired_correlation"]["feedback_error__feedforward_motion"]
    assert pair["mean"] == pytest.approx(1.0)
    assert pair["mean_abs"] == pytest.approx(1.0)


def _summary(seed, value, source):
    return {
        "_source_file": source,
        "metadata": {
            "method": "RCCI",
            "motion": "roll",
            "representation": "residual",
            "condition": "nominal",
            "seed": seed,
            "checkpoint": {"iteration": 1999, "samples": 524288000},
        },
        "completion": {
            "available": True,
            "rate": value,
            "components": {"winding": value},
        },
        "metrics": {
            "tracking": {"paper_pos_err": {"mean": 1.0 - value}},
            "behavior": {"winding_ratio": {"mean": value}},
            "reward": {},
            "intervention": {},
        },
        "efficiency": {"batch1_policy_latency_us": 12.0},
    }


def test_aggregator_uses_seed_level_means_and_writes_json_and_csv(tmp_path):
    summaries = [_summary(0, 0.8, "a"), _summary(1, 1.0, "b")]
    assert extract_paper_metrics(summaries[0])["completion.rate"] == 0.8
    groups = aggregate_summaries(summaries)
    assert len(groups) == 1
    stats = groups[0]["metrics"]["completion.rate"]
    assert stats["n_seeds"] == 2
    assert stats["mean"] == pytest.approx(0.9)
    assert stats["std"] == pytest.approx(np.std([0.8, 1.0], ddof=1))

    write_aggregate_tables(groups, tmp_path)
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["groups"][0]["seeds"] == [0, 1]
    assert (tmp_path / "aggregate.csv").is_file()
    assert (tmp_path / "aggregate_long.csv").is_file()


def test_aggregator_rejects_duplicate_seed_in_matched_group():
    with pytest.raises(ValueError, match="duplicate seed"):
        aggregate_summaries([_summary(0, 0.8, "a"), _summary(0, 0.9, "b")])
