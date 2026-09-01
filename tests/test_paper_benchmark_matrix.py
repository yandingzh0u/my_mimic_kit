from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import anim.motion as motion_lib  # noqa: E402
import envs.amp_env as amp_env  # noqa: E402
import envs.base_env as base_env  # noqa: E402
import envs.deepmimic_env as deepmimic_env  # noqa: E402
import envs.static_objects_env as static_objects_env  # noqa: E402


METHODS = ("deepmimic", "amp", "add")
MOTIONS = (
    "backflip",
    "crawl",
    "roll",
    "getup_facedown",
    "spinkick",
    "climb",
)
ENV_NAMES = {
    "deepmimic": "deepmimic",
    "amp": "amp",
    "add": "add",
}
MOTION_FILES = {
    "backflip": "humanoid_backflip.pkl",
    "crawl": "humanoid_crawl.pkl",
    "roll": "humanoid_roll.pkl",
    "getup_facedown": "humanoid_getup_facedown.pkl",
    "spinkick": "humanoid_spinkick.pkl",
    "climb": "humanoid_climbing_up_down.pkl",
}


def load_env(method, motion):
    path = ROOT / "data/envs/paper_benchmark" / f"{method}_{motion}_env.yaml"
    return path, yaml.safe_load(path.read_text())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("motion", MOTIONS)
def test_all_matrix_configs_exist_and_resolve(method, motion):
    path, config = load_env(method, motion)
    assert path.is_file()
    assert config["env_name"] == ENV_NAMES[method]
    assert config["motion_name"] == motion
    assert config["motion_file"].endswith(MOTION_FILES[motion])
    assert (ROOT / config["motion_file"]).is_file()
    if method == "amp":
        assert config["pose_termination"] is False
    else:
        assert config["pose_termination"] is True
    assert config["rand_reset"] is True
    assert config["log_tracking_error"] is True

    clip = motion_lib.load_motion(ROOT / config["motion_file"])
    assert config["episode_length"] + 1e-6 >= clip.get_length()


@pytest.mark.parametrize("motion", MOTIONS)
def test_physics_protocol_is_identical_across_methods(motion):
    configs = [load_env(method, motion)[1] for method in METHODS]
    shared_keys = (
        "motion_file",
        "episode_length",
        "rand_reset",
        "enable_early_termination",
        "key_bodies",
        "contact_bodies",
        "objects",
    )
    for key in shared_keys:
        vals = [config.get(key) for config in configs]
        assert vals[1:] == vals[:-1], f"{motion}: method mismatch for {key}"


@pytest.mark.parametrize("method", METHODS)
def test_climb_object_is_shared_and_valid(method):
    _, config = load_env(method, "climb")
    specs = deepmimic_env.parse_static_object_specs(config)
    assert len(specs) == 1
    assert specs[0]["file"] == "data/assets/objects/climbing_box.xml"
    assert specs[0]["pos"].tolist() == pytest.approx([-5.46, 0.1, 1.0])
    assert specs[0]["rot"].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert (ROOT / specs[0]["file"]).is_file()
    # Isaac Lab maps an XML path to the sibling USD asset.
    assert (ROOT / "data/assets/objects/climbing_box.usd").is_file()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("motion", [m for m in MOTIONS if m != "climb"])
def test_only_climb_has_static_objects(method, motion):
    _, config = load_env(method, motion)
    assert deepmimic_env.parse_static_object_specs(config) == []


def test_legacy_static_object_env_does_not_build_objects_twice():
    assert "_build_env" not in static_objects_env.StaticObjectsEnv.__dict__


def test_test_mode_always_starts_reference_motion_at_phase_zero():
    env = object.__new__(deepmimic_env.DeepMimicEnv)
    env._device = torch.device("cpu")
    env._rand_reset = True
    env._mode = base_env.EnvMode.TEST
    env._motion_lib = mock.Mock()
    env._motion_lib.sample_motions.return_value = torch.tensor([0, 0])

    _, motion_times = env._sample_motion_times(2)

    assert torch.equal(motion_times, torch.zeros(2))
    env._motion_lib.sample_time.assert_not_called()


def test_train_mode_preserves_random_reference_initialization():
    env = object.__new__(deepmimic_env.DeepMimicEnv)
    env._device = torch.device("cpu")
    env._rand_reset = True
    env._mode = base_env.EnvMode.TRAIN
    env._motion_lib = mock.Mock()
    motion_ids = torch.tensor([0, 0])
    sampled_times = torch.tensor([0.25, 0.75])
    env._motion_lib.sample_motions.return_value = motion_ids
    env._motion_lib.sample_time.return_value = sampled_times

    _, motion_times = env._sample_motion_times(2)

    assert torch.equal(motion_times, sampled_times)
    env._motion_lib.sample_time.assert_called_once_with(motion_ids)


@pytest.mark.parametrize(
    "visualize_ref,log_tracking,expected_calls",
    ((False, False, 0), (True, False, 1), (False, True, 1)),
)
def test_amp_updates_clocked_reference_for_tracking_logs(
        visualize_ref, log_tracking, expected_calls):
    env = object.__new__(amp_env.AMPEnv)
    env._enable_ref_char = lambda: visualize_ref
    env._log_tracking_error = log_tracking
    with mock.patch.object(
            deepmimic_env.DeepMimicEnv, "_update_ref_motion") as update:
        amp_env.AMPEnv._update_ref_motion(env)
        assert update.call_count == expected_calls


def test_method_specific_interfaces_are_locked():
    for motion in MOTIONS:
        _, deepmimic = load_env("deepmimic", motion)
        _, amp = load_env("amp", motion)
        _, add = load_env("add", motion)

        assert deepmimic["enable_tar_obs"] is True
        assert deepmimic["tar_obs_steps"] == [1, 2, 3]
        assert deepmimic["pose_termination"] is True
        assert amp["enable_tar_obs"] is False
        assert amp["num_disc_obs_steps"] == 10
        assert amp["pose_termination"] is False
        assert add["enable_tar_obs"] is True
        assert add["tar_obs_steps"] == [1, 2, 3]
        assert add["num_disc_obs_steps"] == 1
        assert add["pose_termination"] is True


@pytest.mark.parametrize("method", METHODS)
def test_formal_arg_budget(method):
    path = ROOT / "args/paper_benchmark" / f"{method}_2k_8192_args.txt"
    text = path.read_text()
    assert "--num_envs 8192" in text
    assert "--max_samples 524288000" in text
    assert "--save_int_models true" in text
    assert "--engine_config data/engines/isaac_lab_engine.yaml" in text


def test_serial_launcher_syntax_and_contract():
    script = ROOT / "tools/paper_benchmark/run_serial_matrix.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "motions=(backflip crawl roll getup_facedown spinkick climb)" in text
    assert "methods=(deepmimic amp add)" in text
    assert "motion_filters=()" in text
    assert "--motion" in text
    assert 'for motion in "${run_motions[@]}"' in text
    assert "--resume_file" in text
    assert "checkpoint.pt" in text
    assert "checkpoint_reached_budget" in text
    assert 'checkpoint["metadata"]["checkpoint_context"]' in text
    assert "saved_context != expected_context" in text
    assert "FAILED_BUDGET_CHECK" in text
    assert "tools/paper_eval/evaluate_checkpoint.py" in text
    assert "torch.cuda.is_available()" in text
    assert "torch.cuda.device_count()" in text
    assert 'eval_name="final"' in text
    assert 'eval_dir="$out_dir/eval/$eval_name"' in text
    assert "summary.json" in text
    assert "episodes.npz" in text
    assert "timeseries.npz" in text
    assert "eval_steps=300" in text
    assert "eval_steps=2" in text
    assert "eval_num_envs=256" in text
    assert "eval_num_envs=2" in text
    assert "climbing_box.xml" in text
    assert "climbing_box.usd" in text
    assert "run_stage \"smoke\"" in text
    assert 'run_job_with_retries "scale_smoke"' in text
    assert "scale_smoke_envs=8192" in text
    assert "scale_smoke_iters=3" in text
    assert "scale_motions=(backflip)" in text
    assert "run_stage \"formal\"" in text
    assert "run_job_with_retries" in text
    assert "--method" in text
    assert "jump" not in text.lower()
