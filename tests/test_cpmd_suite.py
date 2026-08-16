"""Fail-fast checks for the serial CPMD experiment runner."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cpmd_suite_runner", ROOT / "tools/cpmd/run_humanoid_table1_suite.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_nonfinite_metric_detection_is_specific_to_logger_values():
    assert RUNNER.has_nonfinite_metric("| Critic_Loss | nan |")
    assert RUNNER.has_nonfinite_metric("| Disc_Reward_Mean | -Inf |")
    assert not RUNNER.has_nonfinite_metric("[INFO] Initializing simulation")
    assert not RUNNER.has_nonfinite_metric("| Critic_Loss | 1.25 |")
